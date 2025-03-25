# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import math
from typing import Optional

import os
import triton
import triton.language as tl
import torch

# This code is derived from sglang and FLASHNN projects
# https://github.com/AlibabaPAI/FLASHNN/blob/main/flashnn/triton_kernels/paged_attn.py

_SEQ_PARTITION_SIZE = 1024  # HIP


def paged_attention_decode(
    output: torch.Tensor,  # [num_seqs, num_kv_heads*query_grp_sz, head_sz]
    query: torch.Tensor,  # [num_seqs, num_kv_heads*query_grp_sz, head_sz]
    key_cache: torch.Tensor,  # [num_seqs, num_kv_heads, kv_blk_sz, head_sz]
    value_cache: torch.Tensor,  # [num_blks, num_kv_heads, kv_blk_sz, head_sz]
    seq_lens: torch.Tensor,  # [num_seqs]
    block_tables: torch.Tensor,  # [num_seqs, max_num_blks_per_seq]
    attn_scale: float,
    max_seq_len: int,
    compute_type,
    num_seq_partitions: int = 0,  # TODO use this below
    alibi_slopes: torch.Tensor = None,
) -> None:
    """
    #TODO: Add Doc
    """

    # get num_seqs, num_kv_heads, kv_blk_sz, head_sz and query_grp_sz
    num_seqs = query.shape[0]
    num_q_heads = query.shape[1]
    num_kv_heads = key_cache.shape[1]
    k_scale = 1.0
    v_scale = 1.0
    paged_attn_decode_v1(
        output,
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        max_seq_len,
        compute_type,
        num_kv_heads,
        attn_scale,
        alibi_slopes,
        k_scale,
        v_scale,
    )


def paged_attn_decode_v1(
    output: torch.Tensor,  # [num_seqs, num_kv_heads*query_grp_sz, head_sz]
    query: torch.Tensor,  # [num_seqs, num_kv_heads*query_grp_sz, head_sz]
    key_cache: torch.Tensor,  # [num_seqs, num_kv_heads, kv_blk_sz, head_sz]
    value_cache: torch.Tensor,  # [num_seqs, num_kv_heads, kv_blk_sz, head_sz]
    block_tables: torch.Tensor,  # [num_seqs, max_num_blks_per_seq]
    seq_lens: torch.Tensor,  # [num_seqs]
    max_seq_len: int,
    compute_type,
    num_kv_heads: int,
    scale: float,
    alibi_slopes: Optional[torch.Tensor],
    k_scale,
    v_scale,
    tp_rank: int = 0,
    blocksparse_local_blocks: int = 0,
    blocksparse_vert_stride: int = 0,
    blocksparse_block_size: int = 64,
    blocksparse_head_sliding_step: int = 0,
):
    """
    #TODO: Add Doc
    """
    SHOULD_LOG = os.environ.get("SHOULD_LOG", None) is not None

    num_seqs = query.shape[0]
    num_q_heads = query.shape[1]
    kv_blk_sz = key_cache.shape[2]
    head_sz = key_cache.shape[3]
    query_grp_sz = query.shape[1] // num_kv_heads
    query_grp_sz_pow2 = triton.next_power_of_2(query_grp_sz)
    kv_blk_sz_pow2 = triton.next_power_of_2(kv_blk_sz)
    head_sz_pow2 = triton.next_power_of_2(head_sz)

    grid = (num_seqs, num_kv_heads, 1)
    query_grp_sz_pow2 = 16

    if SHOULD_LOG:
        print(f"grid={grid}")
        print("Arguments to kernel:")
        print(f"output: {output.shape}")
        print(f"query: {query.shape}")
        print(f"key_cache: {key_cache.shape}")
        print(f"value_cache: {value_cache.shape}")
        print(f"block_tables: {block_tables.shape}")
        print(f"seq_lens: {seq_lens.shape}")
        print(f"alibi_slopes: {alibi_slopes.shape if alibi_slopes is not None else None}")
        print(f"scale: {scale}")
        print(f"k_scale: {k_scale}")
        print(f"v_scale: {v_scale}")
        print(f"output strides: {output.stride()}")
        print(f"query strides: {query.stride()}")
        print(f"key_cache strides: {key_cache.stride()}")
        print(f"block_tables strides: {block_tables.stride()}")
        print(f"compute_type: {compute_type}")
        print(f"head_sz: {head_sz}")
        print(f"head_sz_pow2: {head_sz_pow2}")
        print(f"query_grp_sz: {query_grp_sz}")
        print(f"query_grp_sz_pow2: {query_grp_sz_pow2}")
        print(f"kv_blk_sz: {kv_blk_sz}")
        print(f"kv_blk_sz_pow2: {kv_blk_sz_pow2}")

    r: triton.compiler.CompiledKernel = _paged_attn_decode_v1_w_dot_kernel[grid](
        output,
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        alibi_slopes,
        scale,
        k_scale, # torch.tensor([k_scale], dtype=torch.float32, device="cuda"),
        v_scale, # torch.tensor([v_scale], dtype=torch.float32, device="cuda"),
        output.stride(0),
        output.stride(1),
        output.stride(2), # output.stride(2),
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        block_tables.stride(0),
        block_tables.stride(1), # torch.tensor([block_tables.stride(1),
        compute_type=compute_type,
        HEAD_SZ=head_sz,
        HEAD_SZ_POW2=head_sz_pow2,
        QUERY_GRP_SZ=query_grp_sz,
        QUERY_GRP_SZ_POW2=query_grp_sz_pow2,
        KV_BLK_SZ=kv_blk_sz,
        KV_BLK_SZ_POW2=kv_blk_sz,
    )
    if SHOULD_LOG:
        print("---")
        src: triton.compiler.compiler.ASTSource = r.src
        signature = src.signature.items()
        constants = getattr(src, "constants")
        print(dir(r.metadata), r.metadata)
        print(signature)
        print(constants)
    # print(src.signature)
    # for param_name, param in src.fn.signature.parameters.items():
        # print(dir(param))
    # print(r.asm['ttir'])

@triton.jit
def _paged_attn_decode_v1_w_dot_kernel(
    out_ptr,  # [num_seqs, num_kv_heads * query_grp_sz, head_sz]
    q_ptr,  # [num_seqs, num_kv_heads * query_grp_sz, head_sz]
    k_cache_ptr,  # [num_blocks, num_kv_heads, kv_blk_sz, head_sz]
    v_cache_ptr,  # [num_blocks, num_kv_heads, kv_blk_sz, head_sz]
    blk_tables_ptr,  # [num_seqs, max_num_blks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    alibi_slopes,  # [num_kv_heads*query_grp_sz]
    scale,
    k_scale,
    v_scale,
    stride_o_s,
    stride_o_nh,
    stride_o_hs,
    stride_q_s,
    stride_q_nh,
    stride_q_hs,
    stride_k_b,
    stride_k_nh,
    stride_k_kb,
    stride_k_hs,
    stride_bt_s,
    stride_bt_nb,
    compute_type: tl.constexpr,
    HEAD_SZ: tl.constexpr,
    HEAD_SZ_POW2: tl.constexpr,
    QUERY_GRP_SZ: tl.constexpr,
    QUERY_GRP_SZ_POW2: tl.constexpr,
    KV_BLK_SZ: tl.constexpr,
    KV_BLK_SZ_POW2: tl.constexpr,
):
    """
    #TODO: Add Doc
    """
    # q = tl.load(q_ptr)
    # tl.store(out_ptr, q)

    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    log2e: tl.constexpr = 1.4426950408889634

    seq_len = tl.load(seq_lens_ptr + seq_idx)

    num_kv_blks = tl.cdiv(seq_len, KV_BLK_SZ)

    blk_offs = tl.arange(0, KV_BLK_SZ_POW2)
    head_sz_offs = tl.arange(0, HEAD_SZ_POW2)
    q_grp_offs = tl.arange(0, QUERY_GRP_SZ_POW2)

    # load alibi slopes[QUERY_GRP_SZ_POW2]
    if alibi_slopes is None:
        alibi_slope = tl.zeros([QUERY_GRP_SZ_POW2], dtype=tl.float32)
    else:
        alibi_slope = tl.load(
            alibi_slopes + kv_head_idx * QUERY_GRP_SZ + q_grp_offs,
            mask=q_grp_offs < QUERY_GRP_SZ,
            other=0.0,
        )

    q_offs = (
        seq_idx * stride_q_s
        + (kv_head_idx * QUERY_GRP_SZ + q_grp_offs[:, None]) * stride_q_nh
        + head_sz_offs[None, :] * stride_q_hs
    )

    # load q[QUERY_GRP_SZ_POW2, HEAD_SZ_POW2]
    q_mask = (q_grp_offs[:, None] < QUERY_GRP_SZ) & (head_sz_offs[None, :] < HEAD_SZ)

    q = tl.load(q_ptr + q_offs, mask=q_mask, other=0.0)
    q = (q * scale).to(compute_type)

    acc = tl.zeros([QUERY_GRP_SZ_POW2, HEAD_SZ_POW2], dtype=tl.float32)
    max_logit = tl.zeros([QUERY_GRP_SZ_POW2], dtype=tl.float32) + float("-inf")
    exp_sum = tl.zeros([QUERY_GRP_SZ_POW2], dtype=tl.float32)

    kv_offs = (
        kv_head_idx * stride_k_nh
        + blk_offs[:, None] * stride_k_kb
        + head_sz_offs[None, :] * stride_k_hs
    )
    blk_tbl_start_ptr = blk_tables_ptr + seq_idx * stride_bt_s

    for b in range(num_kv_blks):
        kv_blk_nums = tl.load(blk_tbl_start_ptr + b)
        kv_blk_offs = kv_blk_nums * stride_k_b + kv_offs
        blk_seq_offs = b * KV_BLK_SZ + blk_offs
        kv_mask = (
            (blk_seq_offs[:, None] < seq_len)
            & (blk_offs[:, None] < KV_BLK_SZ)
            & (head_sz_offs[None, :] < HEAD_SZ)
        )

        # load k[KV_BLK_SZ_POW2, HEAD_SZ_POW2]
        k_0 = tl.load(k_cache_ptr + kv_blk_offs, mask=kv_mask, other=0.0)
        k = k_0.to(tl.float32) * k_scale if k_0.dtype.is_fp8() else k_0
        k = k.to(compute_type)

        # qk: [QUERY_GRP_SZ_POW2, KV_BLK_SZ_POW2]
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        qk = tl.where(
            (q_grp_offs[:, None] < QUERY_GRP_SZ) & (blk_seq_offs[None, :] < seq_len),
            qk,
            float("-inf"),
        )

        if alibi_slopes is not None:
            qk += (alibi_slope[:, None] * (blk_seq_offs - seq_len + 1)[None, :]).to(
                tl.float32
            )

        qk = tl.where(
            (q_grp_offs[:, None] < QUERY_GRP_SZ) & (blk_seq_offs[None, :] < seq_len),
            qk,
            float("-inf"),
        )
        max_logit_new = tl.maximum(tl.max(qk, axis=1), max_logit)

        # p: [QUERY_GRP_SZ_POW2, KV_BLK_SZ_POW2]
        p = tl.math.exp2((qk - max_logit_new[:, None]) * log2e)
        alpha = tl.math.exp2((max_logit - max_logit_new) * log2e)
        acc *= alpha[:, None]

        # v: [KV_BLK_SZ, HEAD_SZ]
        v_0 = tl.load(v_cache_ptr + kv_blk_offs, mask=kv_mask, other=0.0)
        v = v_0.to(tl.float32) * v_scale if v_0.dtype.is_fp8() else v_0
        v = v.to(compute_type)

        p = p.to(v.dtype)
        acc += tl.dot(p, v, out_dtype=tl.float32)

        exp_sum = exp_sum * alpha + tl.sum(p, axis=1)
        max_logit = max_logit_new

    acc = acc / exp_sum[:, None]

    out_offs = (
        seq_idx * stride_o_s
        + (kv_head_idx * QUERY_GRP_SZ + q_grp_offs[:, None]) * stride_o_nh
        + head_sz_offs[None, :]
    )

    out_mask = (q_grp_offs[:, None] < QUERY_GRP_SZ) & (head_sz_offs[None, :] < HEAD_SZ)
    tl.store(out_ptr + out_offs, acc.to(out_ptr.dtype.element_ty), mask=out_mask)

