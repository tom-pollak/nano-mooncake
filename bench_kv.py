# Intra-node benchmark for NanoKVCache MVP
# - init_process_group (NCCL/Gloo) for coordination like IRIS examples
# - RPC for control plane
# - Measures t_plan, t_copy_per_MB, end-to-end p50/p99 under light contention
import argparse
import statistics
import time

import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp

from nano_kv import NanoKVCache


def kv_bytes_per_token_per_layer(n_kv_heads: int, head_dim: int, bpe: int) -> int:
    return 2 * n_kv_heads * head_dim * bpe


def print_kv_page_guidance(dtype: torch.dtype):
    bpe = torch.empty((), dtype=dtype).element_size()
    configs = [
        ("MHA-64 x 128", 64, 128),
        ("MHA-40 x 128", 40, 128),
        ("GQA-8  x 128", 8, 128),
    ]
    pages = [128 << 10, 256 << 10, 512 << 10, 1 << 20]
    print("\n[KV page sizing]")
    for name, n_kv, h in configs:
        per_tok = kv_bytes_per_token_per_layer(n_kv, h, bpe)
        msg = f"  {name}: ~{per_tok/1024:.1f} KiB / token / layer"
        print(msg)
        for pb in pages:
            print(f"    page {pb>>10:>4} KiB ≈ {pb//per_tok:>4} tokens/layer")
    print()


def run_bench(rank: int, world_size: int, args):
    # NCCL for GPU selection + barriers (IRIS does the data plane)
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method=args.init, world_size=world_size, rank=rank)
    torch.cuda.set_device(rank if torch.cuda.is_available() else 0)

    # RPC init (tiny control plane)
    rpc.init_rpc(name=f"worker{rank}", rank=rank, world_size=world_size)

    dtype = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]
    cache = NanoKVCache(heap_bytes=args.heap_gb * (1 << 30), dtype=dtype, master_rank=0)

    if rank == 0:
        print_kv_page_guidance(dtype)

    producer = 0
    consumer = 1
    sizes_mb = [1, 2, 8, 16]
    reps = args.reps

    for mb in sizes_mb:
        elem_sz = torch.empty((), dtype=dtype).element_size()
        n_elems = (mb * (1 << 20)) // elem_sz
        key_prefix = f"kv-{mb}MB"

        if rank == producer:
            x = torch.randn((n_elems,), device="cuda", dtype=dtype)

            # warmup
            cache.put(key_prefix + "-warm", x, preferred_segment=consumer)

            t_plan, t_copy, t_total = [], [], []
            for r in range(reps):
                k = f"{key_prefix}-{r}"
                # plan
                t0 = time.perf_counter()
                meta = cache.open_for_write(k, x.numel() * elem_sz, preferred_segment=consumer)
                t1 = time.perf_counter()
                # copy (persistent kernel)
                t2 = time.perf_counter()
                cache.put(k, x, preferred_segment=consumer)
                t3 = time.perf_counter()
                t_plan.append(t1 - t0)
                t_copy.append(t3 - t2)
                t_total.append(t3 - t0)

            avg_plan_ms = 1e3 * statistics.mean(t_plan)
            avg_copy_ms = 1e3 * statistics.mean(t_copy)
            copy_gibs = (mb/1024) * reps / sum(t_copy) if sum(t_copy) > 0 else float("nan")
            p50_ms = 1e3 * statistics.median(t_total)
            p99_ms = 1e3 * (sorted(t_total)[max(0, int(0.99 * (len(t_total)-1)))])
            print(f"[PUT rank{rank}] {mb}MB x{reps}: t_plan={avg_plan_ms:.2f} ms  "
                  f"copy={avg_copy_ms:.2f} ms ({copy_gibs:.2f} GiB/s)  "
                  f"end2end p50={p50_ms:.2f} ms p99={p99_ms:.2f} ms")

        if rank == consumer:
            # one GET to sample read throughput
            dist.barrier()
            y = cache.get(key_prefix + "-warm")
            t0 = time.perf_counter()
            _ = cache.get(f"{key_prefix}-0")
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            rd_gibs = (mb/1024) / (t1 - t0) if (t1 - t0) > 0 else float("nan")
            print(f"[GET rank{rank}] {mb}MB ~ {rd_gibs:.2f} GiB/s")
        dist.barrier()

    rpc.shutdown()
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--init", type=str, default="tcp://127.0.0.1:29500")
    ap.add_argument("--heap_gb", type=int, default=2)
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32", "bf16"])
    ap.add_argument("--reps", type=int, default=12)
    args = ap.parse_args()

    mp.spawn(run_bench, args=(args.world, args), nprocs=args.world, join=True)


if __name__ == "__main__":
    main()
