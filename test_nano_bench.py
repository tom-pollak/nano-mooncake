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
    # ---- Robust distributed + env setup ----
    from datetime import timedelta
    # Parse master addr/port from init_url like tcp://127.0.0.1:29500
    url_body = init_url.split("//", 1)[1]
    master_addr, master_port = url_body.split(":")
    os.environ.setdefault("MASTER_ADDR", master_addr)
    os.environ.setdefault("MASTER_PORT", master_port)
    os.environ["RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    # Sometimes IRIS reads these; set them too just in case
    os.environ["IRIS_RANK"] = str(local_rank)
    os.environ["IRIS_WORLD_SIZE"] = str(world_size)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Nano Mooncake MVP (GPU-only). No CUDA device detected.")
    n_gpus = torch.cuda.device_count()
    if world_size > n_gpus:
        raise RuntimeError(f"world_size={world_size} exceeds visible GPUs={n_gpus}. Use --world_size<={n_gpus} or set CUDA_VISIBLE_DEVICES.")

    backend = "nccl"
    torch.cuda.set_device(local_rank)
    print(f"[Rank {local_rank}] init_process_group on {master_addr}:{master_port}", flush=True)
    # Torch distributed init (optionally pass device_id if supported)
    init_kwargs = dict(
        backend=backend,
        init_method=init_url,
        world_size=world_size,
        rank=local_rank,
        timeout=timedelta(seconds=120),
    )
    try:
        import inspect
        if 'device_id' in inspect.signature(dist.init_process_group).parameters:
            init_kwargs['device_id'] = torch.device(f'cuda:{local_rank}')
    except Exception:
        pass
    print(f"[Rank {local_rank}] init_process_group on {master_addr}:{master_port}", flush=True)
    # Torch distributed init (optionally pass device_id if supported)
    init_kwargs = dict(
        backend=backend,
        init_method=init_url,
        world_size=world_size,
        rank=local_rank,
        timeout=timedelta(seconds=120),
    )
    try:
        import inspect
        if 'device_id' in inspect.signature(dist.init_process_group).parameters:
            init_kwargs['device_id'] = torch.device(f'cuda:{local_rank}')
    except Exception:
        pass
    print(f"[Rank {local_rank}] init_process_group on {master_addr}:{master_port}", flush=True)
    dist.init_process_group(**init_kwargs)

    # Create a control-plane group that uses Gloo for object collectives
    try:
        ctrl_pg = dist.new_group(backend='gloo')
    except Exception:
        ctrl_pg = None

    # ---- IRIS symmetric heap (use modest default to avoid OOM) ----
    # You can bump this with --heap_mb.
    # 1 GiB symmetric heap baseline (adjust per GPU mem)
    iris_heap_bytes = 1 << 30
    try:
        shmem = iris.iris(iris_heap_bytes)
    except Exception as e:
        raise RuntimeError(f"IRIS init failed (heap={iris_heap_bytes}): {e}")

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

    # Sanity: IRIS ranks should match torch world_size
    if ws != world_size:
        raise RuntimeError(f"IRIS world size mismatch vs torch.distributed (iris={ws}, torch={world_size}). Ensure RANK/WORLD_SIZE env are set.")

    # Simple correctness once
    key = f"kv-{time.time_ns()}"
    n_elems = (args.elem_mb * 1024 * 1024) // elem_bytes

    if rank == 0:
        src = torch.arange(n_elems, dtype=dtype, device="cuda")
        cache.put(key, src, preferred_segment=1 if world_size > 1 else 0)
    dist.barrier()

    if world_size > 1 and rank == 1:
        dst = torch.empty(n_elems, dtype=dtype, device="cuda")
        cache.get_into(key, dst)
        src_cpu = torch.arange(n_elems, dtype=dtype, device="cpu")
        assert torch.allclose(dst.cpu(), src_cpu), "Mismatch in GET data"
    dist.barrier()

    # Benchmark loop
    put_times = []
    get_times = []
    for it in range(args.iters):
        k = f"bench-{it}-{time.time_ns()}"
        if rank == 0:
            src = torch.randn(n_elems, dtype=dtype, device="cuda")
            t0 = time.time()
            cache.put(k, src, preferred_segment=1 if world_size > 1 else 0)
            torch.cuda.synchronize()
            put_times.append(time.time() - t0)
        dist.barrier()
        if world_size > 1 and rank == 1:
            dst = torch.empty(n_elems, dtype=dtype, device="cuda")
            t0 = time.time()
            cache.get_into(k, dst)
            torch.cuda.synchronize()
            get_times.append(time.time() - t0)
        dist.barrier()

    if rank == 0:
        bytes_per = n_elems * elem_bytes
        if put_times:
            gbps = torch.tensor([(bytes_per / t) / (1024**3) for t in put_times])
            print(f"PUT: p50={gbps.median().item():.2f} GiB/s  p90={torch.quantile(gbps, 0.9).item():.2f}  n={len(gbps)}")
        print("Stats(master):", cache.stats())
    if world_size > 1 and rank == 1:
        bytes_per = n_elems * elem_bytes
        if get_times:
            gbps = torch.tensor([(bytes_per / t) / (1024**3) for t in get_times])
            print(f"GET: p50={gbps.median().item():.2f} GiB/s  p90={torch.quantile(gbps, 0.9).item():.2f}  n={len(gbps)}")
        print(f"Stats(rank{rank}):", cache.stats())

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    # Build init URL from env if provided (torchrun), else default
    master_addr = os.getenv('MASTER_ADDR', '127.0.0.1')
    master_port = os.getenv('MASTER_PORT', '29500')
    url = f"tcp://{master_addr}:{master_port}"

    # If launched via torchrun, LOCAL_RANK/WORLD_SIZE are set and we should NOT mp.spawn again
    local_rank_env = os.getenv('LOCAL_RANK')
    world_size_env = os.getenv('WORLD_SIZE')
    if local_rank_env is not None and world_size_env is not None:
        _worker(int(local_rank_env), int(world_size_env), url, args)
    else:
        mp.spawn(_worker, args=(args.world_size, url, args), nprocs=args.world_size, join=True)
