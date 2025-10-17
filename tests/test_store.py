import uuid
import time
import etcd3
import contextlib
import pytest
from nano_mooncake.store import Client, Manifest
from nano_mooncake.owner import FakeOwner, DevHeader, DevState
import etcd3


@pytest.fixture()
def cli():
    c = etcd3.client()
    ns = f"/nm_test/{uuid.uuid4().hex}"
    yield Client(c, ns)
    # cleanup
    with contextlib.suppress(Exception):
        for k, *_ in c.get_prefix(ns):
            c.delete(k)


@pytest.fixture()
def owner():
    return FakeOwner(rank=0, heap_bytes=1024, page_bytes=8)


def test_happy_path(cli, owner: FakeOwner):
    key = "sess/1:t8192"
    epoch = 1
    man = owner.alloc(key, bytes_total=256, epoch=epoch)

    assert cli.open_for_write(man)
    # simulate producer data copy, then device publish
    owner.publish_ready(man.header_ptr, epoch)
    assert cli.commit(man.hash)

    st, got = cli.get_location(man.hash)
    assert got is not None
    assert st == "READY" and got.header_ptr == man.header_ptr

    assert cli.remove(man.hash, owner)
    st, _ = cli.get_location(man.hash)
    assert st == "MISSING"


def test_double_open_for_write(cli, owner):
    key = "sess/1:t8192"
    epoch = 1
    man = owner.alloc(key, bytes_total=256, epoch=epoch)

    assert cli.open_for_write(man)
    assert cli.open_for_write(man) is False


def test_not_ready(cli, owner):
    key = "sess/2:t4096"
    man = owner.alloc(key, bytes_total=128, epoch=1)

    assert cli.open_for_write(man)  # see wrappers below
    st, got = cli.get_location(man.hash)
    assert st == "WRITING" and got is None


def test_commit_idempotent(cli, owner):
    key = "sess/3:t4096"
    man = owner.alloc(key, 256, epoch=1)

    assert cli.open_for_write(man)
    owner.publish_ready(man.header_ptr, 1)
    assert cli.commit(man.hash)
    # second commit should be True and a no-op
    assert cli.commit(man.hash)


def test_remove_idempotent_and_refcnt(cli, owner):
    key = "sess/5:t4096"
    man = owner.alloc(key, 128, epoch=1)
    assert cli.open_for_write(man)
    owner.publish_ready(man.header_ptr, 1)
    assert cli.commit(man.hash)

    # simulate in-use
    owner.headers[man.header_ptr].refcnt = 1
    assert cli.remove(man.hash, owner) is False  # busy
    owner.headers[man.header_ptr].refcnt = 0
    assert cli.remove(man.hash, owner) is True
    assert cli.remove(man.hash, owner) is True  # second time, no-op


def test_open_for_write_lease_ttl(cli, owner):
    key = "sess/7:t2048"
    man = owner.alloc(key, 128, epoch=1)

    # Short TTL and do NOT commit
    assert cli.open_for_write(man, ttl=1)
    # Intent exists immediately
    st, _ = cli.get_location(man.hash)
    assert st == "WRITING"

    time.sleep(2.5)  # let the lease expire

    # Intent should be gone now
    st, _ = cli.get_location(man.hash)
    assert st == "MISSING"


def test_epoch_prevents_aba(cli, owner):
    key = "sess/4:t4096"

    # A1 publish
    man1 = owner.alloc(key, 256, epoch=1)
    assert cli.open_for_write(man1)
    owner.publish_ready(man1.header_ptr, 1)
    assert cli.commit(man1.hash)

    # Reader caches manifest here (simulated)
    st, cached = cli.get_location(man1.hash)
    assert st == "READY" and cached.epoch == 1

    # Remove A1, reuse data for A2
    assert cli.remove(man1.hash, owner)
    man2 = Manifest(**{**cached.model_dump(), "epoch": 2})
    # Force owner to reuse same ptrs in FakeOwner (skip alloc; we're simulating)
    owner.headers[man2.header_ptr] = DevHeader(
        epoch=2, state=DevState.WRITING, refcnt=0
    )

    assert cli.open_for_write(man2)
    owner.publish_ready(man2.header_ptr, 2)
    assert cli.commit(man2.hash)

    # A reader must compare header.epoch vs cached.epoch before touching refcnt
    assert owner.headers[man2.header_ptr].epoch == 2
    assert cached.epoch == 1
    # Your real reader should detect mismatch and refetch metadata.


def test_reader_epoch_mismatch_refetch(cli, owner):
    # Publish A1
    key = "sess/6:t4096"
    man1 = owner.alloc(key, 256, epoch=1)
    assert cli.open_for_write(man1)
    owner.publish_ready(man1.header_ptr, 1)
    assert cli.commit(man1.hash)

    # Cache manifest
    st, cached = cli.get_location(man1.hash)
    assert st == "READY" and cached.epoch == 1

    # Remove A1, reuse same ptrs for A2 (epoch 2)
    assert cli.remove(man1.hash, owner)
    owner.headers[cached.header_ptr] = DevHeader(
        epoch=2, state=DevState.WRITING, refcnt=0
    )
    man2 = Manifest(**{**cached.model_dump(), "epoch": 2})
    assert cli.open_for_write(man2)
    owner.publish_ready(man2.header_ptr, 2)
    assert cli.commit(man2.hash)

    # Reader must reject stale cached manifest
    assert owner.reader_enter(cached) is False

    # Refetch and enter with fresh manifest
    st, fresh = cli.get_location(man2.hash)
    assert st == "READY" and fresh.epoch == 2
    assert owner.reader_enter(fresh) is True
    owner.reader_exit(fresh)


def test_remove_waits_for_reader(cli, owner):
    key = "sess/8:t2048"
    man = owner.alloc(key, 128, epoch=1)
    assert cli.open_for_write(man)
    owner.publish_ready(man.header_ptr, 1)
    assert cli.commit(man.hash)

    # Enter as reader
    assert owner.reader_enter(man)
    assert cli.remove(man.hash, owner) is False  # busy
    owner.reader_exit(man)
    assert cli.remove(man.hash, owner) is True
