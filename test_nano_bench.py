#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Simple correctness + throughput smoke test for NanoKVCache MVP

import os
import time
import random
import argparse

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import iris

# Import the cache from sibling file
from nano_mooncake import NanoKVCache


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--world_size", type=int, default=2)
    p.add_argument("--heap_mb", type=int, default=256, help="per-rank payload heap in MiB")
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32", "bf16"])
    p.add_argument("--elem_mb", type=int, default=64, help="payload size per PUT in MiB")
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--page_kb", type=int, default=256)
    return p.parse_args()


def torch_dtype(name: str):
    return {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}[name]


def _worker(local_rank: int, world_size: int, init_url: str, args):
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(
        backend=backend,
        init_method=init_url,
        world_size=world_size,
        rank=local_rank,
        device_id=torch.device(f"cuda:{local_rank}"),
    )

    torch.cuda.set_device(local_rank)

    shmem = iris.iris(1 << 33)  # 8 GiB default symmetric heap (adjust as needed)

    dtype = torch_dtype(args.dtype)
    elem_bytes = torch.tensor([], dtype=dtype, device="cuda").element_size()

    heap_elems = (args.heap_mb * 1024 * 1024) // elem_bytes
    cache = NanoKVCache(
        shmem=shmem,
        heap_elems=heap_elems,
        dtype=dtype,
        page_bytes=args.page_kb * 1024,
        master_rank=0,
    )

    rank = shmem.get_rank()
    ws = shmem.get_num_ranks()

    # Sanity
    if ws != world_size:
        raise RuntimeError("IRIS world size mismatch vs torch.distributed")

    # Simple correctness once
    key = f"kv-{time.time_ns()}"
    n_elems = (args.elem_mb * 1024 * 1024) // elem_bytes

    # Source payload on producer (rank 0)
    if rank == 0:
        src = torch.arange(n_elems, dtype=dtype, device="cuda")
        cache.put(key, src, preferred_segment=1 if world_size > 1 else 0)
    else:
        # non-producer waits
        pass

    dist.barrier()

    # Consumer reads (owner is preferred_segment=1)
    if world_size > 1 and rank == 1:
        dst = torch.empty(n_elems, dtype=dtype, device="cuda")
        cache.get_into(key, dst)
        # Validate
        src_cpu = torch.arange(n_elems, dtype=dtype, device="cpu")
        assert torch.allclose(dst.cpu(), src_cpu), "Mismatch in GET data"

    dist.barrier()

    # Benchmark loop: producer pushes, consumer reads
    put_times = []
    get_times = []

    for it in range(args.iters):
        k = f"bench-{it}-{time.time_ns()}"
        if rank == 0:
            src = torch.randn(n_elems, dtype=dtype, device="cuda")
            t0 = time.time()
            cache.put(k, src, preferred_segment=1 if world_size > 1 else 0)
            torch.cuda.synchronize()
            t1 = time.time()
            put_times.append(t1 - t0)
        dist.barrier()
        if world_size > 1 and rank == 1:
            dst = torch.empty(n_elems, dtype=dtype, device="cuda")
            t0 = time.time()
            cache.get_into(k, dst)
            torch.cuda.synchronize()
            t1 = time.time()
            get_times.append(t1 - t0)
        dist.barrier()

    if rank == 0:
        bytes_per = n_elems * elem_bytes
        if put_times:
            gbps = [(bytes_per / t) / (1024**3) for t in put_times]
            print(f"PUT: p50={torch.tensor(gbps).median().item():.2f} GiB/s  p90={torch.quantile(torch.tensor(gbps), 0.9).item():.2f}  n={len(gbps)}")
        print("Stats(master):", cache.stats())
    if world_size > 1 and rank == 1:
        if get_times:
            gbps = [(bytes_per / t) / (1024**3) for t in get_times]
            print(f"GET: p50={torch.tensor(gbps).median().item():.2f} GiB/s  p90={torch.quantile(torch.tensor(gbps), 0.9).item():.2f}  n={len(gbps)}")
        print(f"Stats(rank{rank}):", cache.stats())

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    url = "tcp://127.0.0.1:29500"
    mp.spawn(_worker, args=(args.world_size, url, args), nprocs=args.world_size, join=True)
