from .store import Manifest
from .symm import create_symmetric_heap


import torch
from dataclasses import dataclass
from enum import IntEnum


class DevState(IntEnum):
    WRITING = 1
    READY = 2
    DEAD = 3


@dataclass
class DevHeader:
    epoch: int
    state: int
    refcnt: int


class Owner:
    """Symmetric memory backed owner used in production."""

    header_stride = 256

    def __init__(self, page_bytes: int, heap_bytes: int, *, device=None, group=None):
        heap = create_symmetric_heap(heap_bytes, device=device, group=group)
        tensor = heap.tensor
        self.heap = heap
        self.rank = heap.handle.rank
        self.page_bytes = page_bytes
        self.heap_bytes = int(tensor.numel()) * tensor.element_size()
        self.next_offset = 0
        self.headers: dict[int, DevHeader] = {}
        self.allocs: dict[int, int] = {}

    def _reserve(self, bytes_total: int) -> tuple[int, int]:
        header_ptr = self.next_offset
        next_offset = header_ptr + self.header_stride
        pages = (bytes_total + self.page_bytes - 1) // self.page_bytes
        payload_ptr = next_offset
        next_offset += pages * self.page_bytes
        if next_offset > self.heap_bytes:
            raise RuntimeError(
                f"Symmetric heap exhausted on rank {self.rank}: "
                f"requested {bytes_total} bytes, only {self.heap_bytes - self.next_offset} remaining."
            )
        self.next_offset = next_offset
        return header_ptr, payload_ptr

    def alloc(self, key: str, bytes_total: int, epoch: int) -> Manifest:
        header_ptr, payload_ptr = self._reserve(bytes_total)
        self.headers[header_ptr] = DevHeader(
            epoch=epoch, state=DevState.WRITING, refcnt=0
        )
        self.allocs[payload_ptr] = bytes_total
        return Manifest(
            key=key,
            owner_rank=self.rank,
            header_ptr=header_ptr,
            payload_ptr=payload_ptr,
            bytes_total=bytes_total,
            page_bytes=self.page_bytes,
            epoch=epoch,
            last_access_ns=0,
        )

    def publish_ready(self, header_ptr: int, epoch: int):
        h = self.headers[header_ptr]
        assert h.epoch == epoch and h.state == DevState.WRITING
        h.state = DevState.READY

    def try_remove(self, header_ptr: int, epoch: int) -> bool:
        h = self.headers.get(header_ptr)
        if not h or h.epoch != epoch or h.refcnt != 0 or h.state != DevState.READY:
            return False
        h.state = DevState.DEAD
        return True

    def reader_enter(self, man: Manifest) -> bool:
        h = self.headers.get(man.header_ptr)
        if not h or h.state != DevState.READY or h.epoch != man.epoch:
            return False
        h.refcnt += 1
        return True

    def reader_exit(self, man: Manifest) -> None:
        h = self.headers.get(man.header_ptr)
        assert h and h.epoch == man.epoch and h.refcnt > 0
        h.refcnt -= 1

    def payload_view(
        self,
        man: Manifest,
        *,
        dtype=None,
        rank: int | None = None,
    ):
        """Return a tensor view over the manifest payload in symmetric memory."""
        if self.heap is None:
            raise RuntimeError("payload_view unavailable without a symmetric heap.")
        dtype = dtype or torch.uint8
        tensor = self.heap.tensor
        elem_size = torch.empty((), dtype=dtype, device=tensor.device).element_size()
        if man.bytes_total % elem_size != 0:
            raise ValueError(
                f"bytes_total ({man.bytes_total}) is not aligned with dtype {dtype} "
                f"(element size {elem_size})."
            )
        length = man.bytes_total // elem_size
        target_rank = man.owner_rank if rank is None else rank
        buf = self.heap.handle.get_buffer(
            target_rank,
            (length,),
            dtype,
            man.payload_ptr,
        )
        return buf[:length]


class FakeOwner(Owner):
    """CPU-only stub owner used in tests when symmetric memory is unavailable."""

    def __init__(self, rank: int, heap_bytes: int, page_bytes: int):
        self.heap = None
        self.rank = rank
        self.page_bytes = page_bytes
        self.heap_bytes = heap_bytes
        self.next_offset = 0
        self.headers: dict[int, DevHeader] = {}
        self.allocs: dict[int, int] = {}

    def payload_view(self, *args, **kwargs):  # type: ignore[override]
        raise NotImplementedError("FakeOwner does not expose payload views.")


__all__ = [
    "DevState",
    "DevHeader",
    "FakeOwner",
    "Owner",
]
