"""Seed and isolate PyTorch random state on the selected device."""
from __future__ import annotations


from contextlib import contextmanager


import torch


def seed_device(seed, device):
    torch.random.default_generator.manual_seed(int(seed))
    device = torch.device(device)
    if device.type != "cpu":
        module = getattr(torch, device.type)
        with module.device(device):
            module.manual_seed(int(seed))


@contextmanager
def fork_device_rng(device):
    device = torch.device(device)
    if device.type == "cpu":
        with torch.random.fork_rng(devices=[]):
            yield
    else:
        module = getattr(torch, device.type)
        index = device.index if device.index is not None else module.current_device()
        with torch.random.fork_rng(devices=[index], device_type=device.type):
            yield

