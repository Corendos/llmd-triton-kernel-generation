from numpy import short
import torch
import random
import time
from vllm_kernels.triton_unified_attention import unified_attention

def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)

if __name__ == '__main__':
    seed_everything(0)
    torch.set_default_device("cuda")

    batch_size = 256
    num_heads = 32
    num_kv_heads = 8
    head_size = 128
    block_size = 16
    num_blocks = 32768
    max_input_len = 1
    scale = 0.08838834765
    alibi_slopes = None
    kv_cache_dtype = "auto"
    k_scale = 1.0
    v_scale = 1.0

    long_seq_len = 2047
    short_seq_len = 255
    ratio = 1.0 # Number of long sequence wrt to short sequence. 0.1 means that 10% of the sequences are long.
    long_seq_count = int(ratio * batch_size)
    short_seq_count = batch_size - long_seq_count

    query_lens = [1 for i in range(batch_size // 2)] + [1 for i in range(batch_size // 2)]
    context_lens = [long_seq_len if i < long_seq_count else short_seq_len for i in range(batch_size)]

    token_count = sum(query_lens)

    max_seq_len = 8192
    max_block_per_seq = max_seq_len // block_size

    query = torch.randn(token_count, num_heads, head_size, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(token_count, num_kv_heads, head_size, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(token_count, num_kv_heads, head_size, dtype=torch.bfloat16, device="cuda")
    o = torch.empty_like(query)
    key_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_size , dtype=torch.bfloat16, device="cuda")
    value_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_size, dtype=torch.bfloat16, device="cuda")

    values = torch.arange(0, num_blocks, dtype=torch.long, device="cuda")
    values = values[torch.randperm(num_blocks)]

    block_tables = torch.zeros((batch_size, max_block_per_seq), dtype=torch.int32, device="cuda")
    for i in range(batch_size):
        seq_len = context_lens[i] + query_lens[i]
        block_count = (seq_len + block_size - 1) // block_size
        block_tables[i][0:block_count] = torch.randint(0, num_blocks, (block_count,), dtype=torch.int32, device="cuda")


    seq_lens = torch.tensor([a + b for a, b in zip(query_lens, context_lens)], dtype=torch.long, device="cuda")
    start_loc = torch.cumsum(torch.tensor([0] + query_lens, dtype=torch.long), dim=0)

    assert seq_lens.shape[0] == batch_size
    print(f"# Short Sequences: {short_seq_count} - # Long Sequences: {long_seq_count}")
    print(block_tables)
    print(seq_lens)
    print(start_loc)

    unified_attention(
        q=query,
        k=key_cache,
        v=value_cache,
        out=o,
        cu_seqlens_q=start_loc,
        max_seqlen_q=max_input_len,
        seqused_k=seq_lens,
        max_seqlen_k=seq_len,
        softmax_scale=scale,
        causal=True,
        window_size=[-1, -1],
        block_table=block_tables,
        softcap=0.0,
        q_descale=None,
        k_descale=1.0,
        v_descale=1.0,
    )
    print(o)
    torch.cuda.synchronize()
    start_time = time.time()
    ITERATIONS = 3000
    for _ in range(ITERATIONS):
        unified_attention(
            q=query,
            k=key_cache,
            v=value_cache,
            out=o,
            cu_seqlens_q=start_loc,
            max_seqlen_q=max_input_len,
            seqused_k=seq_lens,
            max_seqlen_k=seq_len,
            softmax_scale=scale,
            causal=True,
            window_size=[-1, -1],
            block_table=block_tables,
            softcap=0.0,
            q_descale=None,
            k_descale=1.0,
            v_descale=1.0,
        )
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"triton Time: {(end_time - start_time)*1000:.2f} ms")
    print(f"triton Time per invocation: {((end_time - start_time) / ITERATIONS)*1000:.2f} ms")

