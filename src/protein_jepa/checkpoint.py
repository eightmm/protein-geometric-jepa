"""Atomic, weights_only-loadable checkpoints including EMA/optimizer/RNG state."""
from pathlib import Path
from dataclasses import asdict
import os
import torch

# 3 = typed latent heads (EMA with encoders) read by the predictor.
# 4 = same weights; samples are a pure function of the step (no sampler state).
FORMAT_VERSION = 4
READABLE = (3, 4)


def rng_state():
    """Dropout RNG only; training samples are derived from the step number."""
    return {"cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    torch.set_rng_state(state['cpu'].cpu())
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
        raise ValueError(f"Checkpoint format {version} predates the typed latent heads "
                         "and predictor interface; retrain instead.")
    if version not in READABLE:
        raise ValueError("Unsupported checkpoint format.")
    return payload
