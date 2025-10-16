# SPDX-License-Identifier: MIT
# Minimal GPU-only distributed KV cache using IRIS + Triton
# Single-hop GPU↔GPU writes from inside Triton kernels, tiny rank-0 master.
#
# Requirements:
#   - PyTorch (CUDA)
#   - Triton
#   - IRIS (https://github.com/ROCm/iris)
#
# Notes:
#   * This is an MVP for experimentation, not production.
#   * No persistence, no replication, no HA; FIFO eviction is stubbed out.

from __future__ import annotations

import math
import os
import time
import hashlib
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import torch
import torch.distributed as dist
import triton
import triton.language as tl
import iris

# ======================
# Triton + IRIS kernels
# ======================

@triton.jit
def _contig_put_kernel(
    src_ptr,                 # local src
    n_elems,                 # number of elements
    dst_payload_ptr,         # remote dst base (same dtype as src)
    dst_rank: tl.constexpr,  # IRIS rank of owner
    src_rank: tl.constexpr,  # my IRIS rank
    heap_bases_ptr: tl.tensor,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n_elems
    vals = tl.load(src_ptr + offs, mask=m)
    iris.store(dst_payload_ptr + offs, vals, src_rank, dst_rank, heap_bases_ptr, mask=m)


@triton.jit
def _contig_get_kernel(
    dst_ptr,                 # local dst
    n_elems,
    src_payload_ptr,         # remote src base
    owner_rank: tl.constexpr,
    my_rank: tl.constexpr,
    heap_bases_ptr: tl.tensor,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n_elems
    vals = iris.load(src_payload_ptr + offs, my_rank, owner_rank, heap_bases_ptr, mask=m)
    tl.store(dst_ptr + offs, vals, mask=m)


@triton.jit
def _wait_ready_then_get_kernel(
    dst_ptr,
    n_elems,
    src_payload_ptr,
    header_state_ptr,        # u32* on owner
    owner_rank: tl.constexpr,
    my_rank: tl.constexpr,
    heap_bases_ptr: tl.tensor,
    BLOCK: tl.constexpr,
):
    # spin until state == 2 (READY)
    st = iris.atomic_cas(header_state_ptr, 2, 2, my_rank, owner_rank, heap_bases_ptr, sem="acquire", scope="sys")
    while st != 2:
        st = iris.atomic_cas(header_state_ptr, 2, 2, my_rank, owner_rank, heap_bases_ptr, sem="acquire", scope="sys")

    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n_elems
    vals = iris.load(src_payload_ptr + offs, my_rank, owner_rank, heap_bases_ptr, mask=m)
    tl.store(dst_ptr + offs, vals, mask=m)


@triton.jit
def _commit_ready_kernel(
    header_state_ptr,        # u32* on owner
    dst_rank: tl.constexpr,
    src_rank: tl.constexpr,
    heap_bases_ptr: tl.tensor,
):
    # 2 == READY
    iris.atomic_xchg(header_state_ptr, 2, src_rank, dst_rank, heap_bases_ptr, sem="release", scope="sys")


# ======================
# Host-side structures
# ======================

HEADER_U32_WORDS = 32  # 128 bytes
# Layout (u32 indices)
OFF_STATE = 10        # state: 0=RESERVED,1=WRITING,2=READY,3=DEAD
OFF_N_ELEMS = 6 * 2   # bytes_total(u64) -> place n_elems(u32) here for convenience
OFF_PAGE_ELEMS = 6 * 2 + 1

STATE_RESERVED = 0
STATE_WRITING = 1
STATE_READY = 2
STATE_DEAD = 3


def _hash_key_u64(key: str) -> int:
    # Stable 64-bit hash using BLAKE2b
    h = hashlib.blake2b(key.encode("utf-8"), digest_size=8)
    return int.from_bytes(h.digest(), byteorder="little", signed=False)


@dataclass
class ObjMeta:
    key: str
    key_hash: int
    owner_rank: int
    header_ptr: int    # device address on owner (u32*)
    payload_ptr: int   # device address on owner (dtype*)
    n_elems: int
    page_elems: int
    dtype: torch.dtype


@dataclass
class WritePlan:
    key_hash: int
    dst_rank: int
    header_ptr: int
    payload_ptr: int
    n_elems: int
    page_elems: int
    dtype: torch.dtype


# ======================
# NanoKVCache MVP
# ======================

class NanoKVCache:
    def __init__(
        self,
        shmem: iris.iris,
        heap_elems: int,
        dtype: torch.dtype = torch.float16,
        page_bytes: int = 256 * 1024,
        master_rank: int = 0,
    ) -> None:
        assert dtype in (torch.float16, torch.float32, torch.bfloat16), "MVP supports fp16/fp32/bf16"
        self.shmem = shmem
        self.dtype = dtype
        self.elem_bytes = torch.tensor([], dtype=dtype, device="cuda").element_size()
        assert page_bytes % self.elem_bytes == 0, "page_bytes must be a multiple of element size"
        self.page_elems = page_bytes // self.elem_bytes
        self.world_size = shmem.get_num_ranks()
        self.rank = shmem.get_rank()
        self.master_rank = master_rank

        # Owner-side payload pool: simple bump allocator
        self.payload_pool = shmem.empty((heap_elems,), device="cuda", dtype=dtype)
        self._payload_bump = 0  # in elements

        # Header allocations per object (one small symmetric buffer each)
        self._headers: Dict[int, torch.Tensor] = {}

        # Master-only: object index
        self._index: Dict[int, ObjMeta] = {}

        # Routing strategy: hash % world_size

    # -------------
    # RPC helpers
    # -------------
    def _all_gather_one(self, obj: Any) -> list[Any]:
        """Gather a Python object from each rank."""
        out = [None for _ in range(self.world_size)]
        dist.all_gather_object(out, obj)
        return out

    def _broadcast_one(self, obj: Any, src: int) -> Any:
        """Broadcast a Python object from src rank to all ranks."""
        buf = [obj] if self.rank == src else [None]
        dist.broadcast_object_list(buf, src=src)
        return buf[0]

    # -------------
    # Allocation on OWNER rank (invoked by protocol round 1)
    # -------------
    def _owner_alloc(self, key_hash: int, n_elems: int) -> Tuple[int, int, int]:
        # Return (header_ptr, payload_ptr, page_elems)
        # Create header buffer (symmetric, uint32)
        header = self.shmem.zeros((HEADER_U32_WORDS,), device="cuda", dtype=torch.int32)
        header_ptr = header.data_ptr()
        self._headers[key_hash] = header

        # Reserve contiguous payload range
        start = self._payload_bump
        end = start + n_elems
        if end > self.payload_pool.numel():
            raise RuntimeError("Out of payload memory on owner")
        self._payload_bump = end
        payload_ptr = self.payload_pool.data_ptr() + start * self.elem_bytes

        # Initialize header locally: set N_ELEMS, PAGE_ELEMS, STATE=WRITING
        # OFF_N_ELEMS/OFF_PAGE_ELEMS are u32 slots
        header[OFF_N_ELEMS] = n_elems
        header[OFF_PAGE_ELEMS] = self.page_elems
        header[OFF_STATE] = STATE_WRITING
        torch.cuda.synchronize()
        return header_ptr, payload_ptr, self.page_elems

    # -------------
    # Public API
    # -------------
    def open_for_write(self, key: str, tensor: torch.Tensor, *, preferred_segment: Optional[int] = None) -> WritePlan:
        assert tensor.is_cuda and tensor.dtype == self.dtype and tensor.is_contiguous()
        n_elems = tensor.numel()
        key_hash = _hash_key_u64(key)
        dst_rank = preferred_segment if preferred_segment is not None else (key_hash % self.world_size)

        # Round 1: producer announces request → master chooses owner and asks OWNER to alloc
        gathered = self._all_gather_one({
            "from_rank": self.rank,
            "key_hash": key_hash,
            "n_elems": n_elems,
            "dst_rank": dst_rank,
        })

        # Master discovers one request per call (MVP assumes single caller at a time)
        if self.rank == self.master_rank:
            producer_req = None
            for item in gathered:
                if item is not None:
                    producer_req = item
                    break
            assert producer_req is not None
            owner = int(producer_req["dst_rank"])  # chosen owner
            # Ask owner to allocate
            alloc_cmd = producer_req
        else:
            alloc_cmd = None
        # Broadcast alloc request to all; owner will act upon it
        alloc_cmd = self._broadcast_one(alloc_cmd, src=self.master_rank)

        # OWNER executes allocation and returns plan to master
        if alloc_cmd is not None:
            if self.rank == int(alloc_cmd["dst_rank"]):
                header_ptr, payload_ptr, page_elems = self._owner_alloc(alloc_cmd["key_hash"], alloc_cmd["n_elems"])
                plan = WritePlan(
                    key_hash=alloc_cmd["key_hash"],
                    dst_rank=self.rank,
                    header_ptr=header_ptr,
                    payload_ptr=payload_ptr,
                    n_elems=alloc_cmd["n_elems"],
                    page_elems=page_elems,
                    dtype=self.dtype,
                )
            else:
                plan = None
        else:
            plan = None

        # Gather plan from all to master
        plans = self._all_gather_one(plan)

        # Master selects owner's plan, stores metadata, then broadcasts final plan
        if self.rank == self.master_rank:
            owner_plan: Optional[WritePlan] = None
            for item in plans:
                if item is not None:
                    owner_plan = item
                    break
            assert owner_plan is not None
            # Persist meta
            meta = ObjMeta(
                key=key,
                key_hash=key_hash,
                owner_rank=owner_plan.dst_rank,
                header_ptr=owner_plan.header_ptr,
                payload_ptr=owner_plan.payload_ptr,
                n_elems=owner_plan.n_elems,
                page_elems=owner_plan.page_elems,
                dtype=self.dtype,
            )
            self._index[key_hash] = meta
            final = owner_plan
        else:
            final = None

        return self._broadcast_one(final, src=self.master_rank)

    def commit(self, plan: WritePlan) -> None:
        # Flip header state to READY with release semantics via IRIS atomic_xchg
        if self.rank == plan.dst_rank:
            # Owner can set state directly without IRIS, but to keep path uniform,
            # we use the same kernel (src=dst=owner)
            src_rank = self.rank
        else:
            src_rank = self.rank
        grid = (1,)
        _commit_ready_kernel[grid](
            plan.header_ptr,
            plan.dst_rank,
            src_rank,
            self.shmem.get_heap_bases(),
        )
        torch.cuda.synchronize()

    def put(self, key: str, tensor: torch.Tensor, *, preferred_segment: Optional[int] = None) -> None:
        plan = self.open_for_write(key, tensor, preferred_segment=preferred_segment)
        # Push payload
        BLOCK = 1024
        grid = (triton.cdiv(plan.n_elems, BLOCK),)
        _contig_put_kernel[grid](
            tensor,
            plan.n_elems,
            plan.payload_ptr,
            plan.dst_rank,
            self.rank,
            self.shmem.get_heap_bases(),
            BLOCK,
        )
        self.commit(plan)

    def get_location(self, key: str) -> Optional[ObjMeta]:
        key_hash = _hash_key_u64(key)
        # Master returns ObjMeta if present
        if self.rank == self.master_rank:
            meta = self._index.get(key_hash)
        else:
            meta = None
        return self._broadcast_one(meta, src=self.master_rank)

    def get_into(self, key: str, dst: torch.Tensor) -> None:
        assert dst.is_cuda and dst.dtype == self.dtype and dst.is_contiguous()
        meta = self.get_location(key)
        if meta is None:
            raise KeyError(f"Key not found: {key}")
        assert dst.numel() == meta.n_elems
        # Wait for READY + read
        BLOCK = 1024
        grid = (triton.cdiv(meta.n_elems, BLOCK),)
        header_state_ptr = meta.header_ptr + OFF_STATE * 4
        _wait_ready_then_get_kernel[grid](
            dst,
            meta.n_elems,
            meta.payload_ptr,
            header_state_ptr,
            meta.owner_rank,
            self.rank,
            self.shmem.get_heap_bases(),
            BLOCK,
        )
        torch.cuda.synchronize()

    def get(self, key: str) -> torch.Tensor:
        meta = self.get_location(key)
        if meta is None:
            raise KeyError(f"Key not found: {key}")
        out = torch.empty((meta.n_elems,), device="cuda", dtype=self.dtype)
        self.get_into(key, out)
        return out

    def remove(self, key: str) -> None:
        key_hash = _hash_key_u64(key)
        if self.rank == self.master_rank:
            meta = self._index.get(key_hash)
            if meta is None:
                resp = None
            else:
                # OWNER sets STATE=DEAD locally
                resp = meta
        else:
            resp = None
        resp = self._broadcast_one(resp, src=self.master_rank)

        if resp is not None:
            meta: ObjMeta = resp
            if self.rank == meta.owner_rank:
                header = self._headers.get(key_hash)
                if header is not None:
                    header[OFF_STATE] = STATE_DEAD
                    torch.cuda.synchronize()
            # Inform master of completion
            fin = key_hash
        else:
            fin = None
        fins = self._all_gather_one(fin)
        if self.rank == self.master_rank:
            for kh in fins:
                if kh is not None:
                    self._index.pop(kh, None)

    def stats(self) -> Dict[str, Any]:
        used = self._payload_bump * self.elem_bytes
        total = self.payload_pool.numel() * self.elem_bytes
        ready = len(self._index) if self.rank == self.master_rank else None
        return {
            "rank": self.rank,
            "bytes_used": int(used),
            "bytes_total": int(total),
            "ready_count_master": ready,
            "page_elems": int(self.page_elems),
            "dtype": str(self.dtype),
        }
