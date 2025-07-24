import os
import torch
from vllm_kernels.triton_unified_attention import unified_attention

os.environ["TRITON_BACKEND_DEBUG"] = "1"
os.environ["SHOULD_LOG"] = "1"

if __name__ == '__main__':
    token_count = 8
    batch_size = 8
    num_heads = 32
    num_kv_heads = 8
    head_size = 128
    block_size = 16
    num_blocks = 4096
    max_input_len = 1
    max_seq_len = 8192
    scale = 0.08838834765
    alibi_slopes = None
    k_scale = 1.0
    v_scale = 1.0

    query = torch.randn(token_count, num_heads, head_size, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(token_count, num_kv_heads, head_size, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(token_count, num_kv_heads, head_size, dtype=torch.bfloat16, device="cuda")
    o = torch.empty_like(query)
    key_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_size , dtype=torch.bfloat16, device="cuda")
    value_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_size, dtype=torch.bfloat16, device="cuda")
    block_tables = torch.full((batch_size, max_seq_len // block_size), 65535, dtype=torch.int32, device="cuda")
    block_tables[:, 0] = 0
    seq_lens = torch.full((batch_size,), 0, dtype=torch.int32, device="cuda")
    seq_lens[0] = 8
    start_loc = torch.full((batch_size+1,), 2, dtype=torch.int32, device="cuda")
    start_loc[0] = 0
    sliding_window = 0

    print("Parameters for unified_attention:")
    print(f"query shape: {query.shape}, dtype: {query.dtype}")
    print(f"k shape: {k.shape}, dtype: {k.dtype}")
    print(f"v shape: {v.shape}, dtype: {v.dtype}")
    print(f"o shape: {o.shape}, dtype: {o.dtype}")
    print(f"key_cache shape: {key_cache.shape}, dtype: {key_cache.dtype}")
    print(f"value_cache shape: {value_cache.shape}, dtype: {value_cache.dtype}")
    print(f"block_tables shape: {block_tables.shape}, dtype: {block_tables.dtype}")
    print(f"start_loc shape: {start_loc.shape}, dtype: {start_loc.dtype}")
    print(f"seq_lens shape: {seq_lens.shape}, dtype: {seq_lens.dtype}")
    print(f"max_input_len: {max_input_len}")
    print(f"k_scale: {k_scale}")
    print(f"v_scale: {v_scale}")
    print(f"alibi_slopes: {alibi_slopes}")
    print(f"sliding_window: {sliding_window}")
    print(f"scale: {scale}")

    unified_attention(
        q=query,
        k=key_cache,
        v=value_cache,
        out=o,
        cu_seqlens_q=start_loc,
        max_seqlen_q=max_input_len,
        seqused_k=seq_lens,
        max_seqlen_k=max_seq_len,
        softmax_scale=scale,
        causal=True,
        window_size=[-1, -1],
        block_table=block_tables,
        softcap=0.0,
        q_descale=None,
        k_descale=k_scale,
        v_descale=v_scale,
    )

    print(o)
