import pytest
import xxhash
from nano_mooncake.store import Manifest, open_for_write, commit, get_location, remove
from nano_mooncake.owner import FakeOwner
import etcd3

HEAP_BYTES = 1024
PAGE_BYTES = 8


@pytest.fixture()
def cli():
    return etcd3.client()


@pytest.fixture()
def owner():
    return FakeOwner(rank=0, heap_bytes=HEAP_BYTES, page_bytes=PAGE_BYTES)


def test_happy_path(cli, owner: FakeOwner):
    key = "sess/1:t8192"
    h = xxhash.xxh64(key).hexdigest()
    epoch = 1
    man = owner.alloc(key, bytes_total=256, epoch=epoch)

    assert open_for_write(cli, h, man)
    # simulate producer data copy, then device publish
    owner.publish_ready(man.header_ptr, epoch)
    assert commit(cli, h)

    st, got = get_location(cli, h)
    assert got is not None
    assert st == "READY" and got.header_ptr == man.header_ptr

    assert remove(cli, h, owner)
    st, _ = get_location(cli, h)
    assert st == "MISSING"


def test_double_open_for_write(cli, owner):
    key = "sess/1:t8192"
    h = xxhash.xxh64(key).hexdigest()
    epoch = 1
    man = owner.alloc(key, bytes_total=256, epoch=epoch)

    assert open_for_write(cli, h, man) is True
    assert open_for_write(cli, h, man) is False
