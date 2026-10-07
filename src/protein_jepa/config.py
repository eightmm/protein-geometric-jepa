from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
import yaml
from .models.fibers import FiberDims


@dataclass
class ModelConfig:
    scalar: int = 64
    vector: int = 16
    tensor: int = 8
    sequence_width: int = 128
    sequence_layers: int = 3
    heads: int = 4
    atom_layers: int = 1
    backbone_layers: int = 3
    aa_layers: int = 2
    internal_layers: int = 2
    latent_scalar: int = 64
    latent_vector: int = 8
    latent_tensor: int = 4
    circular_channels: int = 8
    include_chi5: bool = False
    radius_atom: float = 5.0
    radius_residue: float = 12.0
    max_neighbors: int = 24
    backend: str = "reference"
    dropout: float = 0.0

    def __post_init__(self):
        if self.heads < 1 or self.sequence_width < 1:
            raise ValueError("heads and sequence_width must be positive.")
        if self.sequence_width % self.heads:
            raise ValueError("sequence_width must be divisible by heads.")
        if self.scalar % self.heads:
            raise ValueError("scalar must be divisible by heads for the predictor.")
        for key in ("scalar", "vector", "tensor", "latent_scalar", "latent_vector", "latent_tensor",
                    "circular_channels", "sequence_layers", "internal_layers"):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive.")
        if min(self.atom_layers, self.backbone_layers, self.aa_layers) < 1:
            raise ValueError("Geometry layer counts must be positive.")
        if min(self.radius_atom, self.radius_residue) <= 0 or self.max_neighbors < 1:
            raise ValueError("Graph radii and max_neighbors must be positive.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0,1).")
        if self.backend not in {"reference", "cueq-naive", "cueq-cuda"}:
            raise ValueError("Invalid backend.")

    @property
    def dims(self):
        return FiberDims(self.scalar, self.vector, self.tensor)

    @property
    def latent_dims(self):
        return FiberDims(self.latent_scalar, self.latent_vector, self.latent_tensor)


@dataclass
class TrainConfig:
    steps: int = 1000
    batch_size: int = 2
    learning_rate: float = 0.0003
    weight_decay: float = 0.01
    ema: float = 0.99
    grad_clip: float = 1.0
    crop_lengths: list[int] = field(default_factory=lambda: [128, 256])
    mask_fraction: float = 0.35
    rigid_augmentation: bool = True
    translation_std: float = 1.0
    global_weight: float = 0.1
    atom_weight: float = 0.2
    variance_weight: float = 0.05
    covariance_weight: float = 0.005
    circular_weight: float = 0.02
    seed: int = 17
    save_every: int = 100
    log_every: int = 10
    device: str = "cpu"
    threads: int = 2
    allow_observed_order: bool = False
    tasks: list[str] = field(default_factory=lambda: [
        "seq_to_bb", "bb_to_seq", "cart_to_internal", "internal_to_bb",
        "sc_to_chi", "chi_to_sc", "sc_infill", "bb_infill", "aa_infill"])

    def __post_init__(self):
        if not (0 <= self.ema < 1) or not (0 < self.mask_fraction < 1):
            raise ValueError("Invalid EMA or mask_fraction.")
        if self.batch_size < 1 or self.steps < 1 or not self.crop_lengths or min(self.crop_lengths) < 1:
            raise ValueError("Invalid train size.")
        if self.save_every < 1 or self.log_every < 1 or self.learning_rate <= 0:
            raise ValueError("Invalid schedule.")
        if self.translation_std < 0:
            raise ValueError("translation_std must be nonnegative.")
        if self.threads < 1 or self.grad_clip <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid threads/gradient clipping/weight decay.")
        if min(self.global_weight, self.atom_weight, self.variance_weight, self.covariance_weight, self.circular_weight) < 0:
            raise ValueError("Loss weights must be nonnegative.")


def _checked(cls, data):
    unknown = set(data)-{f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} options: {sorted(unknown)}")
    return cls(**data)


def load_config(path):
    data = yaml.safe_load(Path(path).read_text())
    if set(data)-{"model", "training"}:
        raise ValueError("Only model/training config sections are supported.")
    return _checked(ModelConfig, data.get("model", {})), _checked(TrainConfig, data.get("training", {}))
