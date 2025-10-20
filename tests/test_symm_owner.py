import os
import sys
import uuid

import torch
import torch.distributed as dist
import etcd3
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import run_tests

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nano_mooncake.owner import Owner
from nano_mooncake.store import Client


class SymmetricOwnerSmokeTest(MultiProcessTestCase):
    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    @property
    def world_size(self) -> int:
        return 2

    @property
    def device(self) -> torch.device:
        return torch.device(f"cuda:{self.rank}")

    def _init_process(self):
        torch.cuda.set_device(self.device)
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        torch.manual_seed(1337 + self.rank)

    @skip_if_lt_x_gpu(2)
    def test_payload_view_zero_copy(self):
        self._init_process()
        try:
            owner = Owner(
                page_bytes=256,
                heap_bytes=4096,
                device=self.device,
                group=dist.group.WORLD,
            )
        except ValueError:
            self.skipTest("PyTorch symmetric memory not available")

        ns = [None]
        if self.rank == 0:
            ns[0] = f"/nm_symm/{uuid.uuid4().hex}"
        dist.broadcast_object_list(ns, src=0)
        prefix = ns[0]

        cli = etcd3.client()
        store = Client(cli, prefix)

        key = "sess/test"
        pattern = torch.arange(512, dtype=torch.uint8, device=self.device) % 251
        dist.barrier()

        if self.rank == 0:
            store.put_tensor(owner, key, pattern, epoch=1)

        dist.barrier()
        payload = store.get_tensor(owner, key).cpu()
        expected = torch.arange(512, dtype=torch.uint8) % 251
        torch.testing.assert_close(payload, expected)

        dist.barrier()
        if self.rank == 0:
            store.remove(store.key_hash(key), owner)
            for k, *_ in cli.get_prefix(prefix):
                cli.delete(k)

        dist.destroy_process_group()


if __name__ == "__main__":
    run_tests()
