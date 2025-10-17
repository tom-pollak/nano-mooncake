import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SymmetricHeap:
    """Container for the raw tensor and rendezvous handle representing a shared heap."""

    tensor: Any
    handle: Any


def create_symmetric_heap(
    heap_bytes: int,
    *,
    device: Any | None = None,
    group: Any | None = None,
) -> SymmetricHeap:
    """Allocate and rendezvous a symmetric memory buffer"""
    if not dist.is_initialized():
        raise ValueError(
            "torch.distributed must be initialised before creating symmetric heaps."
        )

    pg = group or dist.group.WORLD
    assert pg is not None
    rank = dist.get_rank(pg)
    device = device or torch.device(f"cuda:{rank}")
    heap_tensor = symm_mem.empty(
        (heap_bytes,),
        dtype=torch.uint8,
        device=device,
    )
    handle = symm_mem.rendezvous(heap_tensor, pg)
    return SymmetricHeap(tensor=heap_tensor, handle=handle)
