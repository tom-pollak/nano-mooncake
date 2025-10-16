from etcd3 import Etcd3Client
from pydantic import BaseModel, computed_field
import time
import logging
from typing import Literal

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


def open_for_write(cli: Etcd3Client, h, man: Manifest, ttl=60) -> bool:
    obj_k, intent_k = f"/nm/obj/{h}", f"/nm/intent/{h}"
    lease = cli.lease(ttl)
    ok, _ = cli.transaction(
        compare=[
            cli.transactions.version(obj_k) == 0,  # no READY manifest
            cli.transactions.version(intent_k) == 0,  # nobody writing
        ],
        # write data
        success=[cli.transactions.put(intent_k, man.model_dump_json(), lease=lease)],
        failure=[],
    )
    return ok


def commit(cli: Etcd3Client, h) -> bool:
    obj_k, intent_k = f"/nm/obj/{h}", f"/nm/intent/{h}"
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
    cli: Etcd3Client, h
) -> tuple[Literal["READY", "WRITING", "MISSING"], Manifest | None]:
    obj_data, _ = cli.get(f"/nm/obj/{h}")
    if obj_data:
        return "READY", Manifest.model_validate_json(obj_data)

    intent_data, _ = cli.get(f"/nm/intent/{h}")
    if intent_data:
        return "WRITING", None

    return "MISSING", None


def remove(cli: Etcd3Client, h: str, owner) -> bool:
    obj_k = f"/nm/obj/{h}"
    raw, _ = cli.get(obj_k)
    # nothing to remove
    if not raw:
        return True

    man = Manifest.model_validate_json(raw)
    # device CAS READY->DEAD (with epoch check)
    if not owner.try_remove(man.header_ptr, man.epoch):
        return False  # busy / wrong epoch

    return cli.delete(obj_k)
