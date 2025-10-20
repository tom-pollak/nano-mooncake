import xxhash
from etcd3 import Etcd3Client
from pydantic import BaseModel, computed_field
import time
import logging
from typing import Literal
import torch

logger = logging.getLogger(__name__)


class Manifest(BaseModel):
    key: str
    owner_rank: int
    header_ptr: int
    payload_ptr: int
    bytes_total: int
    page_bytes: int
    epoch: int
    last_access_ns: int

    @computed_field
    @property
    def last_page_bytes(self) -> int:
        r = self.bytes_total % self.page_bytes
        return r if r != 0 else self.page_bytes

    @computed_field
    @property
    def hash(self) -> str:
        return xxhash.xxh64(self.key).hexdigest()


class Client:
    def __init__(self, cli: Etcd3Client, nm_prefix="/nm/"):
        self.cli = cli
        self.nm_prefix = nm_prefix

    @staticmethod
    def key_hash(key: str) -> str:
        return xxhash.xxh64(key).hexdigest()

    def obj_k(self, h):
        return f"{self.nm_prefix}/obj/{h}"

    def intent_k(self, h):
        return f"{self.nm_prefix}/intent/{h}"

    def open_for_write(self, man: Manifest, ttl=60) -> bool:
        cli, obj_k, intent_k = self.cli, self.obj_k(man.hash), self.intent_k(man.hash)
        lease = cli.lease(ttl)
        ok, _ = cli.transaction(
            compare=[
                cli.transactions.version(obj_k) == 0,  # no READY manifest
                cli.transactions.version(intent_k) == 0,  # nobody writing
            ],
            # write data
            success=[
                cli.transactions.put(intent_k, man.model_dump_json(), lease=lease)
            ],
            failure=[],
        )
        return ok

    def commit(self, h) -> bool:
        cli, obj_k, intent_k = self.cli, self.obj_k(h), self.intent_k(h)
        # already committed?
        obj_data, _ = cli.get(obj_k)
        if obj_data:
            return True

        # no open_for_write
        raw, meta = cli.get(intent_k)
        if raw is None:
            logger.warning(f"commit before open_for_write: {h} {meta=}")
            return False
        assert meta is not None

        man = Manifest.model_validate_json(raw)
        man.last_access_ns = time.time_ns()

        ok, _ = cli.transaction(
            compare=[cli.transactions.version(intent_k) == meta.version],
            success=[
                cli.transactions.put(obj_k, man.model_dump_json()),
                cli.transactions.delete(intent_k),
            ],
            failure=[],
        )
        return ok

    def get_location(
        self, h
    ) -> tuple[Literal["READY", "WRITING", "MISSING"], Manifest | None]:
        cli, obj_k, intent_k = self.cli, self.obj_k(h), self.intent_k(h)
        obj_data, _ = cli.get(obj_k)
        if obj_data:
            return "READY", Manifest.model_validate_json(obj_data)

        intent_data, _ = cli.get(intent_k)
        if intent_data:
            return "WRITING", None

        return "MISSING", None

    def remove(self, h: str, owner) -> bool:
        cli, obj_k = self.cli, self.obj_k(h)
        raw, _ = cli.get(obj_k)
        # nothing to remove
        if not raw:
            return True

        man = Manifest.model_validate_json(raw)
        # device CAS READY->DEAD (with epoch check)
        if not owner.try_remove(man.header_ptr, man.epoch):
            return False  # busy / wrong epoch

        return cli.delete(obj_k)

    def put_tensor(
        self,
        owner,
        key: str,
        tensor: torch.Tensor,
        *,
        epoch: int,
        ttl: int = 60,
    ) -> Manifest:
        if tensor.dtype != torch.uint8:
            raise ValueError("put_tensor currently supports only torch.uint8 tensors.")
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        man = owner.alloc(key, bytes_total=tensor.numel(), epoch=epoch)
        if not self.open_for_write(man, ttl=ttl):
            raise RuntimeError(f"open_for_write conflict for key {key}")

        dest = owner.payload_view(man, dtype=torch.uint8)
        if dest.numel() != tensor.numel():
            self.remove(man.hash, owner)
            raise RuntimeError("Allocated payload size does not match tensor size.")
        dest.copy_(tensor)

        owner.publish_ready(man.header_ptr, epoch)
        if not self.commit(man.hash):
            raise RuntimeError(f"Commit failed for key {key}")
        return man

    def get_tensor(
        self,
        owner,
        key: str,
        *,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        h = self.key_hash(key)
        st, man = self.get_location(h)
        if st != "READY" or man is None:
            raise KeyError(f"Key {key} is not ready (state={st}).")

        if not owner.reader_enter(man):
            raise RuntimeError("Failed to enter reader; manifest is stale or busy.")

        try:
            view = owner.payload_view(man, dtype=torch.uint8)
            result = view.clone()
        finally:
            owner.reader_exit(man)

        if device is not None:
            result = result.to(device)
        return result
