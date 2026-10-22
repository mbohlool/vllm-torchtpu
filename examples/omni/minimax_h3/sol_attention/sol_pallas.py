"""Pallas TPU block-sparse + centroid Sol attention kernel for TPU7x."""

import functools
import math
from typing import Any

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

# Sol block size in tokens and log2(e); sol_route below is copied verbatim (re-indented) from the Sol
# reference script bench_tpu_sol_attention.py in
# https://github.com/pmotgi/torchtpu-diffusion-serving/issues/2, so this file has no other dependencies.
BLOCK = 64
LOG2E = math.log2(math.e)


def sol_route(
    q: jax.Array, k: jax.Array, v: jax.Array, sink_tokens: int, tau: float
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Block summaries and the per-head block route of Sol; inputs are (tokens, heads, dim).

  Every 64-row block gets a mean query, a mean key and a value sum. Query block i keeps key block j exact when the
  score of their means exceeds mean + tau * std of a Gaussian (diagonal covariance) fitted to row i's block scores,
  when |i - j| <= 1, or when either block lies in the context prefix.

  Returns:
    Block mean keys and value sums (blocks, heads, dim), block lengths (blocks,), and the route
    (heads, query blocks, key blocks), True where attention is exact.
  """
  tokens, heads, dim = q.shape
  blocks = -(-tokens // BLOCK)
  lengths = jnp.minimum(BLOCK, tokens - jnp.arange(blocks) * BLOCK).astype(jnp.float32)

  def block_sums(x: jax.Array) -> jax.Array:
    x = jnp.pad(x.astype(jnp.float32), ((0, blocks * BLOCK - tokens), (0, 0), (0, 0)))
    return x.reshape(blocks, BLOCK, heads, dim).sum(1)

  # The CUDA version stores block means and sums in BF16. A convert pair would not do: under jit, XLA's default
  # --xla_allow_excess_precision drops it, and the unrounded means flip borderline routes.
  to_bf16 = functools.partial(lax.reduce_precision, exponent_bits=8, mantissa_bits=7)
  kc = to_bf16(block_sums(k) / lengths[:, None, None])
  vc = to_bf16(block_sums(v))
  q_bar = block_sums(q) / lengths[:, None, None]
  scale = dim**-0.5 * LOG2E  # base-2 units, as the CUDA kernels compare
  highest = lax.Precision.HIGHEST
  scores = jnp.einsum("ihd,jhd->hij", q_bar, kc, precision=highest) * scale
  mean_kc = kc.mean(0)
  var_kc = jnp.maximum((kc * kc).mean(0) - mean_kc * mean_kc, 0.0)
  mean = jnp.einsum("ihd,hd->hi", q_bar, mean_kc, precision=highest) * scale
  std = jnp.sqrt(jnp.einsum("ihd,hd->hi", q_bar * q_bar, var_kc, precision=highest) * scale**2 + 1e-6)
  index = jnp.arange(blocks)
  sinks = -(-sink_tokens // BLOCK)
  route = (
      (scores > (mean + tau * std)[..., None])
      | (jnp.abs(index[:, None] - index[None, :]) <= 1)
      | (index[None, :] < sinks)
      | (index[:, None] < sinks)
  )
  return kc, vc, lengths, route


BLOCK_SIZE = BLOCK
DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.float32).max)
QK_DIM_NUMBERS = (((1,), (1,)), ((), ()))
SV_DIM_NUMBERS = (((0,), (0,)), ((), ()))
VMEM_LIMIT_BYTES = 64 * (1 << 20)


def align_to(x: int, multiple: int) -> int:
  return ((x + multiple - 1) // multiple) * multiple


def _static_dims(tokens: int, Tq: int, G: int, Bc: int) -> tuple[int, int, int, int, int, int, int]:
  blocks = -(-tokens // BLOCK_SIZE)
  T_pad = blocks * BLOCK_SIZE
  qb_per_tile = Tq // BLOCK_SIZE
  q_tiles = -(-tokens // Tq)
  Q_pad = q_tiles * Tq
  B_pad = align_to(blocks, max(128, Bc, G))
  min_m = blocks + G + 2
  if min_m <= 1024:
    M_pad = max(128, 1 << (min_m - 1).bit_length())
  else:
    M_pad = align_to(min_m, 1024)
  return blocks, T_pad, qb_per_tile, q_tiles, Q_pad, B_pad, M_pad


def prepare_sol_pallas_inputs(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    sink_tokens: int,
    tau: float = 1.0,
    Tq: int = 512,
    G: int = 16,
    Bc: int = 256,
    fuse_kv: bool = False,
    compact_in_smem: bool = False,
) -> tuple[jax.Array, ...]:
  """Stage 1: Compute Sol route, centroids, and tile-union metadata for Pallas kernel.

  Inputs q, k, v have shape (tokens, heads, head_dim) in bfloat16.
  """
  if Tq % 128 != 0:
    raise ValueError(f"Tq must be a multiple of 128, got {Tq}")
  if G < 2 or (G * 64) % 128 != 0:
    raise ValueError(f"G * 64 must be a multiple of 128, got G={G}")
  qb_check = Tq // BLOCK_SIZE
  if 128 % qb_check != 0 or Tq > 1024:
    # Route tiles are 128 query blocks wide (a tile must not straddle a 128-lane boundary) and route bits
    # are packed as block | bits << 12 in int32 (needs Tq // 64 <= 16).
    raise ValueError(f"Tq must be one of 128, 256, 512, 1024, got {Tq}")

  tokens, heads, head_dim = q.shape
  blocks, T_pad, qb_per_tile, q_tiles, Q_pad, B_pad, M_pad = _static_dims(tokens, Tq, G, Bc)
  if blocks > 4096:
    # Block indices are packed into 12 bits (& 0xFFF) and n_act into 15 bits.
    raise ValueError(f"at most 4096 64-token blocks (262144 tokens) supported, got {blocks} blocks")
  padded_q_blocks = q_tiles * qb_per_tile
  Q_blocks_pad = align_to(padded_q_blocks, 128)
  full_blocks = tokens // BLOCK_SIZE

  # 1. Exact Sol route + centroids from sol_route.
  kc, vc, lengths, route = sol_route(q, k, v, sink_tokens, tau)

  # 2. Format centroids ck, cv (heads, B_pad, head_dim) bf16 and log2(lengths) f32.
  ck = jnp.swapaxes(kc.astype(q.dtype), 0, 1)  # (heads, blocks, head_dim)
  cv = jnp.swapaxes((vc / lengths[:, None, None]).astype(v.dtype), 0, 1)  # (heads, blocks, head_dim)
  ck = jnp.pad(ck, ((0, 0), (0, B_pad - blocks), (0, 0)))
  cv = jnp.pad(cv, ((0, 0), (0, B_pad - blocks), (0, 0)))

  log2_len_1d = jnp.pad(jnp.log2(lengths), (0, B_pad - blocks), constant_values=0.0)
  log2_len_2d = jnp.broadcast_to(log2_len_1d[:, None], (B_pad, 128)).astype(jnp.float32)

  # 3. Transpose Q, K, V to (heads, seq, head_dim) and pad to tile multiples.
  q_htd = jnp.swapaxes(q, 0, 1)
  k_htd = jnp.swapaxes(k, 0, 1)
  v_htd = jnp.swapaxes(v, 0, 1)

  q_hbm = jnp.pad(q_htd, ((0, 0), (0, Q_pad - tokens), (0, 0))) if Q_pad > tokens else q_htd
  k_hbm = jnp.pad(k_htd, ((0, 0), (0, T_pad - tokens), (0, 0))) if T_pad > tokens else k_htd
  v_hbm = jnp.pad(v_htd, ((0, 0), (0, T_pad - tokens), (0, 0))) if T_pad > tokens else v_htd

  # 4. Pad route along query blocks by repeating the last valid query block.
  if padded_q_blocks > blocks:
    last_qb = jnp.broadcast_to(
        route[:, blocks - 1 : blocks, :], (heads, padded_q_blocks - blocks, blocks)
    )
    route_q_pad = jnp.concatenate([route, last_qb], axis=1)
  else:
    route_q_pad = route

  route_reshaped = route_q_pad.reshape(heads, q_tiles, qb_per_tile, blocks)
  bit_weights = (1 << jnp.arange(qb_per_tile, dtype=jnp.int32))[None, None, :, None]
  route_bits = jnp.sum(
      route_reshaped.astype(jnp.int32) * bit_weights, axis=2, dtype=jnp.int32
  )  # (heads, q_tiles, blocks)
  full_bits = (1 << qb_per_tile) - 1
  has_centroids = jnp.any(route_bits < full_bits, axis=-1, keepdims=True).astype(
      jnp.int32
  )  # (heads, q_tiles, 1)

  # Transpose route to (heads, B_pad, Q_blocks_pad) so each tile slices (B_pad, 128).
  route_t = jnp.swapaxes(route_q_pad, 1, 2).astype(jnp.bfloat16)
  route_bf16 = jnp.pad(
      route_t, ((0, 0), (0, B_pad - blocks), (0, Q_blocks_pad - padded_q_blocks)), constant_values=1.0
  )

  block_arange = jnp.arange(blocks, dtype=jnp.int32)[None, None, :]
  is_inter = (route_bits == full_bits) & (block_arange < full_blocks)
  n_inter = jnp.sum(is_inter.astype(jnp.int32), axis=-1, keepdims=True)  # (heads, q_tiles, 1)

  if compact_in_smem:
    # Pack route_bits[0..blocks-1], has_centroids at [blocks], and n_inter at [blocks+1].
    rbits_with_flags = jnp.concatenate([route_bits, has_centroids, n_inter], axis=-1)
    kv_meta = jnp.pad(
        rbits_with_flags, ((0, 0), (0, 0), (0, M_pad - (blocks + 2))), constant_values=0
    )
    kv_meta_flat = kv_meta.reshape(heads * q_tiles * M_pad)
    if fuse_kv:
      kv_fused = jnp.concatenate([k_hbm, v_hbm], axis=-1)
      return (kv_meta_flat, q_hbm, ck, cv, log2_len_2d, route_bf16, kv_fused)
    return (kv_meta_flat, q_hbm, ck, cv, log2_len_2d, route_bf16, k_hbm, v_hbm)
  else:
    union_mask = route_bits > 0
    n_active_raw = jnp.sum(union_mask.astype(jnp.int32), axis=-1, keepdims=True)
    # Pack n_act (bits 0..14), has_centroids (bit 15), and n_inter (bits 16..30) at [blocks].
    packed_meta = n_active_raw | (has_centroids << 15) | (n_inter << 16)
    meta_val = block_arange | (route_bits << 12)
    sort_key = jnp.where(
        is_inter,
        block_arange,
        jnp.where(union_mask, blocks + block_arange, 2 * blocks),
    )
    _, sorted_meta = lax.sort_key_val(
        sort_key, jnp.broadcast_to(meta_val, sort_key.shape), dimension=-1
    )
    sorted_valid = jnp.where(jnp.sort(sort_key, axis=-1) < 2 * blocks, sorted_meta, 0)
    sorted_with_meta = jnp.concatenate([sorted_valid, packed_meta], axis=-1)
    kv_meta = jnp.pad(sorted_with_meta, ((0, 0), (0, 0), (0, M_pad - (blocks + 1))), constant_values=0)
    kv_meta_flat = kv_meta.reshape(heads * q_tiles * M_pad)
    if fuse_kv:
      kv_fused = jnp.concatenate([k_hbm, v_hbm], axis=-1)
      return (kv_meta_flat, q_hbm, ck, cv, log2_len_2d, route_bf16, kv_fused)
    return (kv_meta_flat, q_hbm, ck, cv, log2_len_2d, route_bf16, k_hbm, v_hbm)


def _sol_pallas_kernel_impl(
    *args: Any,
    Tq: int,
    G: int,
    B_pad: int,
    M_pad: int,
    blocks: int,
    tokens: int,
    qb_per_tile: int,
    scale_log2e: float,
    coalesce_dma: bool,
    fuse_kv: bool,
    compact_in_smem: bool,
) -> None:
  """Single-stage Pallas kernel evaluating unrouted centroids + double-buffered exact KV blocks."""
  if compact_in_smem:
    if fuse_kv:
      (
          umask_smem_ref,
          q_ref,
          ck_ref,
          cv_ref,
          log2_len_ref,
          route_tile_ref,
          kv_hbm_ref,
          out_ref,
          kv_meta_smem_ref,
          m_scratch_ref,
          l_scratch_ref,
          o_scratch_ref,
          kv_vmem_ref,
          sem_kv,
      ) = args
    else:
      (
          umask_smem_ref,
          q_ref,
          ck_ref,
          cv_ref,
          log2_len_ref,
          route_tile_ref,
          k_hbm_ref,
          v_hbm_ref,
          out_ref,
          kv_meta_smem_ref,
          m_scratch_ref,
          l_scratch_ref,
          o_scratch_ref,
          k_vmem_ref,
          v_vmem_ref,
          sem_k,
          sem_v,
      ) = args
  else:
    if fuse_kv:
      (
          kv_meta_smem_ref,
          q_ref,
          ck_ref,
          cv_ref,
          log2_len_ref,
          route_tile_ref,
          kv_hbm_ref,
          out_ref,
          m_scratch_ref,
          l_scratch_ref,
          o_scratch_ref,
          kv_vmem_ref,
          sem_kv,
      ) = args
    else:
      (
          kv_meta_smem_ref,
          q_ref,
          ck_ref,
          cv_ref,
          log2_len_ref,
          route_tile_ref,
          k_hbm_ref,
          v_hbm_ref,
          out_ref,
          m_scratch_ref,
          l_scratch_ref,
          o_scratch_ref,
          k_vmem_ref,
          v_vmem_ref,
          sem_k,
          sem_v,
      ) = args

  h_idx = pl.program_id(0)
  t_idx = pl.program_id(1)
  full_bits = (1 << qb_per_tile) - 1
  full_blocks = tokens // BLOCK_SIZE
  has_tail_pad = (tokens % BLOCK_SIZE) != 0

  if compact_in_smem:
    has_cent = umask_smem_ref[blocks]
    n_inter = umask_smem_ref[blocks + 1]

    # Single-pass partition of active blocks into [0..n_inter-1] (Intersection)
    # and [n_inter..n_act-1] (Fringe) on the Scalar Core!
    def _pack_single_pass(b, ptrs):
      inter_ptr = ptrs & 0xFFFF
      fringe_ptr = ptrs >> 16
      rbits = umask_smem_ref[b]
      is_act = rbits > 0
      is_inter = (rbits == full_bits) & (b < full_blocks)
      is_fringe = is_act & (~is_inter)
      dst = jnp.where(is_inter, inter_ptr, fringe_ptr)
      kv_meta_smem_ref[dst] = b | (rbits << 12)
      next_inter = inter_ptr + jnp.where(is_inter, jnp.int32(1), jnp.int32(0))
      next_fringe = fringe_ptr + jnp.where(is_fringe, jnp.int32(1), jnp.int32(0))
      return next_inter | (next_fringe << 16)

    final_ptrs = lax.fori_loop(0, blocks, _pack_single_pass, n_inter << 16)
    n_act = final_ptrs >> 16
    for c in range(G):
      kv_meta_smem_ref[n_act + c] = 0
  else:
    packed_meta = kv_meta_smem_ref[blocks]
    n_act = packed_meta & 0x7FFF
    has_cent = (packed_meta >> 15) & 1
    n_inter = packed_meta >> 16

  n_inter_steps = n_inter // G
  n_steps = (n_act + G - 1) // G

  def _issue_kv_group(step_idx, slot):
    base = step_idx * G
    if fuse_kv:
      for c in range(G):
        b_idx = jnp.where(base + c < n_act, kv_meta_smem_ref[base + c] & 0xFFF, 0)
        pltpu.make_async_copy(
            src_ref=kv_hbm_ref.at[h_idx, pl.ds(b_idx * 64, 64), :],
            dst_ref=kv_vmem_ref.at[slot, pl.ds(c * 64, 64), :],
            sem=sem_kv.at[slot],
        ).start()
    else:
      for c in range(G):
        b_idx = jnp.where(base + c < n_act, kv_meta_smem_ref[base + c] & 0xFFF, 0)
        pltpu.make_async_copy(
            src_ref=k_hbm_ref.at[h_idx, pl.ds(b_idx * 64, 64), :],
            dst_ref=k_vmem_ref.at[slot, pl.ds(c * 64, 64), :],
            sem=sem_k.at[slot],
        ).start()
        pltpu.make_async_copy(
            src_ref=v_hbm_ref.at[h_idx, pl.ds(b_idx * 64, 64), :],
            dst_ref=v_vmem_ref.at[slot, pl.ds(c * 64, 64), :],
            sem=sem_v.at[slot],
        ).start()

  def _wait_kv_group(slot):
    if fuse_kv:
      pltpu.make_async_copy(
          src_ref=kv_vmem_ref.at[slot],
          dst_ref=kv_vmem_ref.at[slot],
          sem=sem_kv.at[slot],
      ).wait()
    else:
      pltpu.make_async_copy(
          src_ref=k_vmem_ref.at[slot],
          dst_ref=k_vmem_ref.at[slot],
          sem=sem_k.at[slot],
      ).wait()
      pltpu.make_async_copy(
          src_ref=v_vmem_ref.at[slot],
          dst_ref=v_vmem_ref.at[slot],
          sem=sem_v.at[slot],
      ).wait()

  @pl.when(n_steps > 0)
  def _prefetch_step0():
    _issue_kv_group(jnp.int32(0), 0)

  @pl.when(n_steps > 1)
  def _prefetch_step1():
    _issue_kv_group(jnp.int32(1), 1)

  q_tile = q_ref[...]  # (Tq, 128) bf16
  mask_val = jnp.float32(DEFAULT_MASK_VALUE)

  # 1. Single-shot centroid evaluation into (m_scratch_ref, l_scratch_ref, o_scratch_ref).
  @pl.when(has_cent != 0)
  def _init_from_centroids():
    ck_all = ck_ref[...]  # (B_pad, 128) bf16
    cv_all = cv_ref[...]  # (B_pad, 128) bf16
    r_all = route_tile_ref[...]  # (B_pad, 128) bf16

    col_offset = (t_idx * qb_per_tile) % 128
    q_qb = col_offset + (lax.broadcasted_iota(jnp.int32, (Tq, 128), 0) // 64)
    lane_idx = lax.broadcasted_iota(jnp.int32, (Tq, 128), 1)
    indicator = jnp.where(q_qb == lane_idx, 1.0, 0.0).astype(jnp.bfloat16)

    qk_c = (
        lax.dot_general(
            ck_all, q_tile, QK_DIM_NUMBERS, preferred_element_type=jnp.float32
        )
        * scale_log2e
    )  # (B_pad, Tq)
    bias_c = jnp.tile(log2_len_ref[...], (1, Tq // 128))
    routed = lax.dot_general(
        r_all, indicator, QK_DIM_NUMBERS, preferred_element_type=jnp.float32
    )  # (B_pad, Tq)
    qk_c = jnp.where(routed > 0.5, mask_val, qk_c + bias_c)

    m_c = qk_c.max(axis=0)[None, :]  # (1, Tq)
    m_init = jnp.broadcast_to(m_c, (8, Tq))
    s_c = jnp.exp2(qk_c - m_c)
    has_any_c = m_c > (mask_val * 0.5)
    s_c = jnp.where(jnp.broadcast_to(has_any_c, (B_pad, Tq)), s_c, 0.0)
    l_c = s_c.sum(axis=0, keepdims=True)
    l_init = jnp.broadcast_to(l_c, (8, Tq))
    o_init = lax.dot_general(
        cv_all,
        s_c.astype(jnp.bfloat16),
        SV_DIM_NUMBERS,
        preferred_element_type=jnp.float32,
    )  # (128, Tq)
    m_scratch_ref[...] = m_init
    l_scratch_ref[...] = l_init
    o_scratch_ref[...] = o_init

  @pl.when(has_cent == 0)
  def _init_empty():
    m_scratch_ref[...] = jnp.full((8, Tq), mask_val, dtype=jnp.float32)
    l_scratch_ref[...] = jnp.zeros((8, Tq), dtype=jnp.float32)
    o_scratch_ref[...] = jnp.zeros((128, Tq), dtype=jnp.float32)

  # Precompute 2D bit-position mask for slice-free Fringe masking of (G*64, Tq).
  bpw = max(1, 32 // qb_per_tile)
  row_iota = lax.broadcasted_iota(jnp.int32, (G * 64, Tq), 0)
  row_g = row_iota // 64
  col_qb = lax.broadcasted_iota(jnp.int32, (G * 64, Tq), 1) // 64
  local_bit = (row_g % bpw) * qb_per_tile + col_qb
  bit_mask_2d = jnp.uint32(1) << local_bit.astype(jnp.uint32)

  # 2. Exact KV blocks loop:
  # Phase A (0 .. n_inter_steps - 1): 100% dense across tile, zero masking of qk!
  # Phase B (n_inter_steps .. n_steps - 1): Fringe blocks with slice-free bit-packed rmask!
  def _update_kv_step(step_idx, slot, is_fringe: bool):
    if fuse_kv:
      kv_chunk = kv_vmem_ref[slot]
      k_chunk = kv_chunk[:, :128]
      v_chunk = kv_chunk[:, 128:]
    else:
      k_chunk = k_vmem_ref[slot]  # (G*64, 128) bf16
      v_chunk = v_vmem_ref[slot]  # (G*64, 128) bf16

    qk = (
        lax.dot_general(
            k_chunk, q_tile, QK_DIM_NUMBERS, preferred_element_type=jnp.float32
        )
        * scale_log2e
    )  # (G*64, Tq)

    if is_fringe:
      base = step_idx * G
      num_words = (G + bpw - 1) // bpw
      active_word = jnp.uint32(0)
      for w in range(num_words):
        word_val = jnp.uint32(0)
        for k in range(bpw):
          g = w * bpw + k
          if g < G:
            # Positions >= n_act (incl. kv_meta[blocks] = packed_meta) contribute no route bits.
            meta = jnp.where(base + g < n_act, kv_meta_smem_ref[base + g], 0)
            rbits = jnp.uint32((meta >> 12) & full_bits)
            word_val = word_val | (rbits << (k * qb_per_tile))
        if num_words == 1:
          active_word = word_val
        elif w == 0:
          active_word = jnp.broadcast_to(word_val, (G * 64, Tq))
        else:
          active_word = jnp.where(row_g >= (w * bpw), word_val, active_word)

      rmask = (bit_mask_2d & active_word) != 0
      if has_tail_pad:
        last_b = kv_meta_smem_ref[n_act - 1] & 0xFFF
        tail_row_limit = jnp.where(
            (step_idx == (n_steps - 1)) & (last_b == (blocks - 1)),
            (n_act - 1 - step_idx * G) * 64 + (tokens % 64),
            G * 64,
        )
        rmask = rmask & (row_iota < tail_row_limit)
      qk = jnp.where(rmask, qk, mask_val)

    m_prev, l_prev = m_scratch_ref[...], l_scratch_ref[...]
    m_curr = qk.max(axis=0)[None, :]
    m_next = jnp.maximum(m_prev, m_curr)
    s_curr = jnp.exp2(qk - m_next[0:1])
    s_cast = s_curr.astype(jnp.bfloat16)
    l_curr = s_curr.sum(axis=0, keepdims=True)
    alpha = jnp.exp2(m_prev - m_next)
    o_curr = lax.dot_general(
        v_chunk,
        s_cast,
        SV_DIM_NUMBERS,
        preferred_element_type=jnp.float32,
    )
    o_scratch_ref[:] = alpha[0:1, ...] * o_scratch_ref[:] + o_curr
    m_scratch_ref[...], l_scratch_ref[...] = m_next, l_curr + alpha * l_prev

  def _inter_step_body(step_idx, _):
    slot = step_idx & 1
    _wait_kv_group(slot)
    _update_kv_step(step_idx, slot, is_fringe=False)

    @pl.when(step_idx + 2 < n_steps)
    def _issue_next():
      _issue_kv_group(step_idx + 2, slot)

    return None

  def _fringe_step_body(step_idx, _):
    slot = step_idx & 1
    _wait_kv_group(slot)
    _update_kv_step(step_idx, slot, is_fringe=True)

    @pl.when(step_idx + 2 < n_steps)
    def _issue_next():
      _issue_kv_group(step_idx + 2, slot)

    return None

  lax.fori_loop(0, n_inter_steps, _inter_step_body, None)
  lax.fori_loop(n_inter_steps, n_steps, _fringe_step_body, None)

  # 3. Final normalization and transpose (128, Tq) -> (Tq, 128).
  l_final = l_scratch_ref[...]
  l_inv = 1.0 / l_final[0:1]
  o_norm = o_scratch_ref[...] * l_inv
  out_ref[...] = jnp.swapaxes(o_norm, 0, 1)


def run_sol_pallas_kernel(
    *prepared_args: jax.Array,
    tokens: int,
    Tq: int = 512,
    G: int = 16,
    Bc: int = 256,
    scale: float | None = None,
    coalesce_dma: bool = False,
    fuse_kv: bool = False,
    compact_in_smem: bool = False,
    interpret: bool = False,
) -> jax.Array:
  """Stage 2: Execute Pallas kernel on precomputed Stage 1 tuple."""
  q_hbm = prepared_args[1]
  heads, Q_pad, head_dim = q_hbm.shape
  if scale is None:
    scale = 1.0 / math.sqrt(head_dim)

  blocks, _, qb_per_tile, q_tiles, _, B_pad, M_pad = _static_dims(tokens, Tq, G, Bc)
  scale_log2e = float(scale * LOG2E)
  grid = (heads, q_tiles)

  kernel_fn = functools.partial(
      _sol_pallas_kernel_impl,
      Tq=Tq,
      G=G,
      B_pad=B_pad,
      M_pad=M_pad,
      blocks=blocks,
      tokens=tokens,
      qb_per_tile=qb_per_tile,
      scale_log2e=scale_log2e,
      coalesce_dma=coalesce_dma,
      fuse_kv=fuse_kv,
      compact_in_smem=compact_in_smem,
  )

  in_specs = [
      pl.BlockSpec((M_pad,), lambda h, t, *_: (h * q_tiles + t,), memory_space=pltpu.SMEM),
      pl.BlockSpec((None, Tq, head_dim), lambda h, t, *_: (h, t, 0)),
      pl.BlockSpec((None, B_pad, head_dim), lambda h, t, *_: (h, 0, 0)),
      pl.BlockSpec((None, B_pad, head_dim), lambda h, t, *_: (h, 0, 0)),
      pl.BlockSpec((B_pad, 128), lambda h, t, *_: (0, 0)),
      pl.BlockSpec((None, B_pad, 128), lambda h, t, *_: (h, 0, (t * qb_per_tile) // 128)),
  ]
  if fuse_kv:
    in_specs.append(pl.BlockSpec(memory_space=pltpu.HBM))
  else:
    in_specs.extend([
        pl.BlockSpec(memory_space=pltpu.HBM),
        pl.BlockSpec(memory_space=pltpu.HBM),
    ])

  scratch_shapes: list[Any] = []
  if compact_in_smem:
    scratch_shapes.append(pltpu.SMEM((M_pad,), jnp.int32))
  scratch_shapes.extend([
      pltpu.VMEM((8, Tq), jnp.float32),
      pltpu.VMEM((8, Tq), jnp.float32),
      pltpu.VMEM((head_dim, Tq), jnp.float32),
  ])
  if fuse_kv:
    scratch_shapes.extend([
        pltpu.VMEM((2, G * 64, head_dim * 2), jnp.bfloat16),
        pltpu.SemaphoreType.DMA((2,)),
    ])
  else:
    scratch_shapes.extend([
        pltpu.VMEM((2, G * 64, head_dim), jnp.bfloat16),
        pltpu.VMEM((2, G * 64, head_dim), jnp.bfloat16),
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.SemaphoreType.DMA((2,)),
    ])

  out_shape = jax.ShapeDtypeStruct((heads, Q_pad, head_dim), jnp.float32)
  out_specs = pl.BlockSpec((None, Tq, head_dim), lambda h, t, *_: (h, t, 0))

  out_padded = pl.pallas_call(
      kernel_fn,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=grid,
          in_specs=in_specs,
          out_specs=out_specs,
          scratch_shapes=scratch_shapes,
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary"),
          flags={"XLA_TPU_FORCE_LP_LLO_SCHEDULER": True},
          disable_bounds_checks=True,
          skip_device_barrier=True,
          vmem_limit_bytes=VMEM_LIMIT_BYTES,
      ),
      out_shape=out_shape,
      interpret=interpret,
  )(*prepared_args)

  return jnp.swapaxes(out_padded[:, :tokens, :], 0, 1)


def sol_attention_pallas(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    sink_tokens: int,
    tau: float = 1.0,
    Tq: int = 512,
    G: int = 16,
    Bc: int = 256,
    coalesce_dma: bool = False,
    fuse_kv: bool = False,
    compact_in_smem: bool = False,
    interpret: bool = False,
) -> jax.Array:
  """Compute Sol block-sparse + centroid attention using a Pallas TPU kernel.

  Inputs q, k, v have shape (tokens, heads, head_dim) in bfloat16; returns
  float32 output of shape (tokens, heads, head_dim).
  """
  tokens = q.shape[0]
  prepared_args = prepare_sol_pallas_inputs(
      q,
      k,
      v,
      sink_tokens=sink_tokens,
      tau=tau,
      Tq=Tq,
      G=G,
      Bc=Bc,
      fuse_kv=fuse_kv,
      compact_in_smem=compact_in_smem,
  )
  return run_sol_pallas_kernel(
      *prepared_args,
      tokens=tokens,
      Tq=Tq,
      G=G,
      Bc=Bc,
      coalesce_dma=coalesce_dma,
      fuse_kv=fuse_kv,
      compact_in_smem=compact_in_smem,
      interpret=interpret,
  )
