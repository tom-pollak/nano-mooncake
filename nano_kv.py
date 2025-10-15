# Minimal GPU-only KV cache over IRIS
# - Host-allocated (rank0 master), device-pushed writes via IRIS in persistent kernels
# - Control plane: tiny torch.distributed.rpc calls (no server loop in your code)
# - Data plane: IRIS one-hop RDMA
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import torch
import torch.distributed.rpc as rpc
import triton
import triton.language as tl
import iris


# =========================
# Persistent Triton kernels
# =========================

@triton.jit
def _rdma_store_persistent(
    src_ptr,            # local tensor
    n_elems,            # total elements
    dst_base_ptr,       # symmetric heap base view (remote) as 1-D tensor
    dst_off_elems,      # element offset into dst_base_ptr
    src_rank: tl.constexpr,
    dst_rank: tl.constexpr,
    heap_bases_ptr,     # iris.get_heap_bases()
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    start = pid * BLOCK
    stride = nprog * BLOCK
    for base in range(start, n_elems, stride):
        offs = base + tl.arange(0, BLOCK)
        mask = offs < n_elems
        vals = tl.load(src_ptr + offs, mask=mask, cache_modifier=".cg")   # streaming load from producer
        iris.store(
            dst_base_ptr + dst_off_elems + offs,
            vals,
            src_rank, dst_rank, heap_bases_ptr,
            mask=mask
        )


@triton.jit
def _rdma_load_persistent(
    dst_ptr,            # local output tensor
    n_elems,
    src_base_ptr,       # symmetric heap base view (remote) as 1-D tensor
    src_off_elems,
    dst_rank: tl.constexpr,
    src_rank: tl.constexpr,
    heap_bases_ptr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    start = pid * BLOCK
    stride = nprog * BLOCK
    for base in range(start, n_elems, stride):
        offs = base + tl.arange(0, BLOCK)
        mask = offs < n_elems
        vals = iris.load(
            src_base_ptr + src_off_elems + offs,
            dst_rank, src_rank, heap_bases_ptr,
            mask=mask
        )
        tl.store(dst_ptr + offs, vals, mask=mask)


# =========================
# Master (lives on rank0, hidden behind RPC)
# =========================

@dataclass
class ObjMeta:
    owner_rank: int
    offset_elems: int
    n_elems: int
    dtype: str  # "fp16"/"fp32"/"bf16" etc.


class _MasterState:
    def __init__(self, world_size: int, elem_capacity_per_rank: int, dtype: str):
        self.world_size = world_size
        self.capacity = elem_capacity_per_rank
        self.dtype = dtype
        self.bump_by_rank = [0] * world_size       # elements
        self.index: Dict[str, ObjMeta] = {}

    def open_for_write(self, key: str, n_elems: int, preferred_segment: Optional[int]) -> Tuple[int, int, int, str]:
        owner = preferred_segment if preferred_segment is not None else (hash(key) % self.world_size)
        cur = self.bump_by_rank[owner]
        new_cur = cur + n_elems
        if new_cur > self.capacity:
            raise RuntimeError(f"OOM: owner {owner} need {n_elems} elems, used {cur}, cap {self.capacity}")
        self.bump_by_rank[owner] = new_cur
        meta = ObjMeta(owner_rank=owner, offset_elems=cur, n_elems=n_elems, dtype=self.dtype)
        self.index[key] = meta
        return meta.owner_rank, meta.offset_elems, meta.n_elems, meta.dtype

    def get_location(self, key: str) -> Tuple[int, int, int, str]:
        m = self.index.get(key)
        if m is None:
            raise KeyError(key)
        return m.owner_rank, m.offset_elems, m.n_elems, m.dtype

    def remove(self, key: str) -> None:
        self.index.pop(key, None)

    def stats(self):
        return {"capacity_elems": self.capacity, "used_by_rank_elems": list(self.bump_by_rank),
                "objects": len(self.index)}


# module-local holder created on rank0 by init_master(); accessed only via RPC
_MASTER_STATE: Optional[_MasterState] = None


def init_master(world_size: int, elem_capacity_per_rank: int, dtype: str):
    """RPC-executed on rank0."""
    global _MASTER_STATE
    _MASTER_STATE = _MasterState(world_size, elem_capacity_per_rank, dtype)


def rpc_open_for_write(key: str, n_elems: int, preferred_segment: Optional[int]):
    return _MASTER_STATE.open_for_write(key, n_elems, preferred_segment)  # type: ignore


def rpc_get_location(key: str):
    return _MASTER_STATE.get_location(key)  # type: ignore


def rpc_remove(key: str):
    _MASTER_STATE.remove(key)  # type: ignore


def rpc_stats():
    return _MASTER_STATE.stats()  # type: ignore


# =========================
# Public API
# =========================

def _dtype_to_str(dt: torch.dtype) -> str:
    if dt == torch.float16: return "fp16"
    if dt == torch.float32: return "fp32"
    if dt == torch.bfloat16: return "bf16"
    raise ValueError(f"Unsupported dtype {dt}")


def _str_to_dtype(s: str) -> torch.dtype:
    return {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}[s]


class NanoKVCache:
    """Host-allocated (rank0), device-pushed, persistent-kernel IRIS KV cache."""

    def __init__(self, heap_bytes: int, dtype: torch.dtype = torch.float16, master_rank: int = 0):
        self.master_rank = master_rank
        self.dtype = dtype
        self.dtype_str = _dtype_to_str(dtype)

        # IRIS symmetric heap (per rank)
        self.shmem = iris.iris(heap_bytes)
        self.rank = self.shmem.get_rank()
        self.world_size = self.shmem.get_num_ranks()

        # 1-D heap view tensor (addressable base for remote ops)
        self.elem_size = torch.empty((), dtype=self.dtype).element_size()
        self.heap_elems = heap_bytes // self.elem_size
        self.heap_view = self.shmem.empty((self.heap_elems,), device="cuda", dtype=self.dtype)

        # IRIS heap bases tensor for device
        self.heap_bases = self.shmem.get_heap_bases()

        # Initialize the master state on rank0 (via RPC) exactly once
        if self.rank == self.master_rank:
            init_master(self.world_size, self.heap_elems, self.dtype_str)
        else:
            rpc.rpc_sync(f"worker{self.master_rank}", init_master,
                         args=(self.world_size, self.heap_elems, self.dtype_str))

        self.shmem.barrier()

    # ---- control-plane helpers ----
    def _open(self, key: str, n_elems: int, preferred_segment: Optional[int]):
        owner, off, n, dstr = rpc.rpc_sync(f"worker{self.master_rank}",
                                           rpc_open_for_write,
                                           args=(key, n_elems, preferred_segment))
        return ObjMeta(owner, off, n, dstr)

    def _loc(self, key: str):
        owner, off, n, dstr = rpc.rpc_sync(f"worker{self.master_rank}", rpc_get_location, args=(key,))
        return ObjMeta(owner, off, n, dstr)

    # ---- public API ----
    def open_for_write(self, key: str, bytes_total: int, *, preferred_segment: Optional[int] = None) -> ObjMeta:
        n_elems = (bytes_total + self.elem_size - 1) // self.elem_size
        return self._open(key, n_elems, preferred_segment)

    def put(self, key: str, tensor: torch.Tensor, *, preferred_segment: Optional[int] = None,
            block_size: int = 4096, num_programs: int = 256):
        assert tensor.is_cuda and tensor.dtype == self.dtype
        n_elems = tensor.numel()
        meta = self._open(key, n_elems, preferred_segment)

        grid = (num_programs,)
        _rdma_store_persistent[grid](
            tensor,
            n_elems,
            self.heap_view,
            meta.offset_elems,
            self.rank,
            meta.owner_rank,
            self.heap_bases,
            block_size,
        )
        torch.cuda.synchronize()

    def get(self, key: str, *, dst: Optional[torch.Tensor] = None,
            block_size: int = 4096, num_programs: int = 256) -> torch.Tensor:
        meta = self._loc(key)
        dtype = _str_to_dtype(meta.dtype)
        n_elems = meta.n_elems
        if dst is None:
            dst = torch.empty((n_elems,), device="cuda", dtype=dtype)
        else:
            assert dst.is_cuda and dst.dtype == dtype and dst.numel() >= n_elems

        grid = (num_programs,)
        _rdma_load_persistent[grid](
            dst,
            n_elems,
            self.heap_view,
            meta.offset_elems,
            self.rank,
            meta.owner_rank,
            self.heap_bases,
            block_size,
        )
        torch.cuda.synchronize()
        return dst

    def get_location(self, key: str) -> ObjMeta:
        return self._loc(key)

    def remove(self, key: str):
        rpc.rpc_sync(f"worker{self.master_rank}", rpc_remove, args=(key,))

    def stats(self):
        return rpc.rpc_sync(f"worker{self.master_rank}", rpc_stats, args=())
