"""Atomic, weights_only-loadable checkpoints including EMA/optimizer/RNG state."""
from pathlib import Path
from dataclasses import asdict
import os
import torch

# 3 = v0.4 typed latent heads (EMA with encoders) read by the predictor.
FORMAT_VERSION = 3


def rng_state(generator):
    return {"cpu": torch.get_rng_state(), "sampler": generator.get_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state, generator):
    torch.set_rng_state(state['cpu'].cpu())
    generator.set_state(state['sampler'].cpu())
    if state['cuda']:
        if not torch.cuda.is_available():
            raise RuntimeError("Checkpoint has CUDA RNG state but this run has no CUDA device.")
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def save_checkpoint(path, model, optimizer, scheduler, step, model_cfg, train_cfg,
                    rank_rng, fingerprint, world_size):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(format_version=FORMAT_VERSION, model=model.state_dict(), optimizer=optimizer.state_dict(),
                   scheduler=scheduler.state_dict(), step=int(step), model_config=asdict(model_cfg),
                   train_config=asdict(train_cfg), rng=rank_rng, fingerprint=fingerprint,
                   world_size=world_size, torch_version=str(torch.__version__))
    tmp = path.with_suffix(path.suffix+f".tmp-{os.getpid()}")
    try:
        with tmp.open('wb') as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def load_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    version = payload.get('format_version')
    if version in (1, 2):
        raise ValueError(f"Checkpoint format {version} predates the v0.4 typed latent heads "
                         "and predictor interface; retrain instead.")
    if version != FORMAT_VERSION:
        raise ValueError("Unsupported checkpoint format.")
    return payload
