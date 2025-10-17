import os
import sys

import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import run_tests

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nano_mooncake.owner import Owner
from nano_mooncake.store import Manifest


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

        manifest_json = [None]
        if self.rank == 0:
            man = owner.alloc("sess/test", bytes_total=512, epoch=1)
            pattern = torch.arange(
                man.bytes_total, dtype=torch.uint8, device=self.device
            ) % 251
            owner.payload_view(man, dtype=torch.uint8).copy_(pattern)
            manifest_json[0] = man.model_dump_json()

        dist.broadcast_object_list(manifest_json, src=0)
        assert manifest_json[0] is not None
        man = Manifest.model_validate_json(manifest_json[0])

        dist.barrier()
        payload = owner.payload_view(man, dtype=torch.uint8).cpu()
        expected = torch.arange(man.bytes_total, dtype=torch.uint8) % 251
        torch.testing.assert_close(payload, expected)
        dist.barrier()

        dist.destroy_process_group()


if __name__ == "__main__":
    run_tests()
