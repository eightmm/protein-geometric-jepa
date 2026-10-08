"""Training samples as a pure function of (seed, rank, step, sample index).

No sampler state is carried between steps, so any worker can build any step
and resume needs only the step number. Records, crops and masks are built on
the CPU; the training loop moves them to the device.
"""
from dataclasses import replace
import hashlib
import os
import signal
import threading
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from .records import random_crop
from ..geometry.primitives import random_rotation
from ..objectives.tasks import make_observation


def sample_generator(seed: int, rank: int, step: int, index: int) -> torch.Generator:
    digest = hashlib.blake2b(f"{seed}:{rank}:{step}:{index}".encode(), digest_size=8).digest()
    return torch.Generator().manual_seed(int.from_bytes(digest, "little") & (2**63-1))


def make_microbatch(dataset, cfg, rank: int, step: int, micro: int) -> list:
    """One microbatch of (record, observation). Tasks cycle per SAMPLE so one
    step mixes objectives and every target encoder receives updates."""
    batch = []
    for b in range(cfg.batch_size):
        index = micro*cfg.batch_size+b
        task = cfg.tasks[(step*cfg.accumulation_steps*cfg.batch_size+index) % len(cfg.tasks)]
        generator = sample_generator(cfg.seed, rank, step, index)
        record = dataset[int(torch.randint(len(dataset), (), generator=generator))]
        record = random_crop(record, cfg.crop_lengths, generator, cfg.crop_min_observed)
        if cfg.rigid_augmentation:
            # One rigid transform of the source record: teacher and student
            # observations share the frame.
            rotation = random_rotation(generator)
            record = record.rigid_transform(rotation, torch.randn(3, generator=generator)*cfg.translation_std)
        batch.append((record, make_observation(record, task, cfg.mask_fraction, generator,
                                               cfg.mask_blocks, cfg.mask_mode, cfg.mask_min_span)))
    return batch


class StepStream(IterableDataset):
    """Steps [first, end) as (step, [microbatch, ...]); worker w builds steps
    first+w, first+w+W, ..., which the DataLoader returns in step order."""
    def __init__(self, dataset, cfg, rank: int, first: int, end: int):
        self.dataset, self.cfg, self.rank, self.first, self.end = dataset, cfg, rank, first, end

    def __iter__(self):
        info = get_worker_info()
        worker, workers = (info.id, info.num_workers) if info else (0, 1)
        for step in range(self.first+worker, self.end, workers):
            yield step, [make_microbatch(self.dataset, self.cfg, self.rank, step, m)
                         for m in range(self.cfg.accumulation_steps)]


def _identity(item):
    return item


def _worker_init(_):
    torch.set_num_threads(1)  # many workers x many threads oversubscribes the host
    # Launchers (torchrun, Slurm) signal the trainer's whole process group.
    # Workers leave that group, so the trainer can finish its step and save,
    # while the DataLoader can still stop them (its SIGTERM comes from the
    # parent, which PyTorch's worker handler honours). Ignoring SIGTERM instead
    # would hang the loader's shutdown of a worker blocked on a full queue.
    if hasattr(os, "setpgrp"):
        os.setpgrp()


def step_loader(dataset, cfg, rank: int, first: int, end: int):
    """Iterator of (step, microbatches); `loader_workers` > 0 prefetches in
    background processes, 0 builds each step in-process."""
    stream = StepStream(dataset, cfg, rank, first, end)
    if cfg.loader_workers == 0:
        return iter(stream)
    # spawn: fork after CUDA/NCCL initialization can deadlock (DistributedDataParallel
    # docs), and spawn workers are direct children of the trainer (a forkserver
    # process would die on a launcher's group-wide SIGTERM and take the workers'
    # liveness pipes with it). A script that calls train() with workers needs an
    # `if __name__ == "__main__":` guard (the CLI has one). Stop signals are
    # ignored while workers start, so they inherit SIG_IGN before worker_init.
    # The private generator keeps the global (dropout) RNG untouched.
    stops = [getattr(signal, n) for n in ("SIGTERM", "SIGUSR1") if hasattr(signal, n)]
    main = threading.current_thread() is threading.main_thread()
    previous = {s: signal.signal(s, signal.SIG_IGN) for s in stops} if main else {}
    try:
        return iter(DataLoader(stream, batch_size=None, num_workers=cfg.loader_workers,
                               collate_fn=_identity, worker_init_fn=_worker_init, prefetch_factor=2,
                               multiprocessing_context="spawn",
                               generator=torch.Generator().manual_seed(cfg.seed+rank)))
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)


def to_device(record, observation, device):
    return record.to(device), replace(observation, atom_visible=observation.atom_visible.to(device),
                                      seq_visible=observation.seq_visible.to(device),
                                      target_residues=observation.target_residues.to(device))
