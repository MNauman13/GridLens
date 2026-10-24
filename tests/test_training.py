import copy
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.train_neural import load_checkpoint, save_checkpoint


class CheckpointTests(unittest.TestCase):
    def test_resume_reproduces_next_optimizer_step_and_rng(self):
        torch.manual_seed(17)
        first = torch.nn.Linear(3, 1)
        optimizer = torch.optim.Adam(first.parameters(), lr=0.001)

        def step(model, opt):
            opt.zero_grad()
            model(torch.randn(8, 3)).square().mean().backward()
            opt.step()

        step(first, optimizer)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pt"
            save_checkpoint(path, first, optimizer, 1, 5.0, copy.deepcopy(first.state_dict()), 0, [], "fixture")
            step(first, optimizer)
            second = torch.nn.Linear(3, 1)
            second_optimizer = torch.optim.Adam(second.parameters(), lr=0.001)
            load_checkpoint(path, second, second_optimizer, "fixture")
            step(second, second_optimizer)
            for a, b in zip(first.parameters(), second.parameters(), strict=True):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            with self.assertRaises(ValueError):
                load_checkpoint(path, second, second_optimizer, "different-dataset")
            with path.open("ab") as stream:
                stream.write(b"corrupt")
            with self.assertRaises(ValueError):
                load_checkpoint(path, second, second_optimizer, "fixture")
