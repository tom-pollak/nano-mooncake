from etcd3 import Etcd3Client
from pydantic import BaseModel, computed_field
import time
import logging
from typing import Literal

logger = logging.getLogger(__name__)


class PageData(BaseModel):
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


def open_for_write(cli: Etcd3Client, h, data: PageData) -> bool:
    obj_k, intent_k = f"/nm/obj/{h}", f"/nm/intent/{h}"
    ok, _ = cli.transaction(
        # data must not already exist
        compare=[
            cli.transactions.version(obj_k) == 0,  # no READY manifest
            cli.transactions.version(intent_k) == 0,  # nobody writing
        ],
        # write data
        success=[cli.transactions.put(intent_k, data.model_dump_json())],
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
    intent_raw, meta = cli.get(intent_k)
    if intent_raw is None:
        logger.warning(f"commit before open_for_write: {h} {meta=}")
        return False
    assert meta is not None

    data = PageData.model_validate_json(intent_raw)
    data.last_access_ns = time.time_ns()

    ok, _ = cli.transaction(
        compare=[cli.transactions.version(intent_k) == meta.version],
        success=[
            cli.transactions.put(obj_k, data.model_dump_json()),
            cli.transactions.delete(intent_k),
        ],
        failure=[],
    )
    return ok


def get_location(
    cli: Etcd3Client, h
) -> tuple[Literal["READY", "WRITING", "MISSING"], PageData | None]:
    obj_data, _ = cli.get(f"/nm/obj/{h}")
    if obj_data:
        return "READY", PageData.model_validate_json(obj_data)

    intent_data, _ = cli.get(f"/nm/intent/{h}")
    if intent_data:
        return "WRITING", None

    return "MISSING", None
