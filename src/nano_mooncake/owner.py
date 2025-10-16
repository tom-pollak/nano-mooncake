import etcd3
from .store import Manifest


# cli = etcd3.client()

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


class FakeOwner:
    def __init__(self, rank: int, heap_bytes: int, page_bytes: int):
        self.rank = rank
        self.heap = heap_bytes
        self.page_bytes = page_bytes
        self.next_ptr = 0x1000_0000
        self.headers: dict[int, DevHeader] = {}  # header_ptr -> DevHeader
        self.allocs: dict[int, int] = {}  # payload_ptr -> size

    def alloc(self, key: str, bytes_total: int, epoch: int) -> Manifest:
        header_ptr = self.next_ptr
        self.next_ptr += 256  # fake header stride
        payload_ptr = self.next_ptr
        self.next_ptr += ((bytes_total + 255) // 256) * 256
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
