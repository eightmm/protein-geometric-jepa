from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
import warnings
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
    # Typed latent heads (EMA-tracked with the encoders): semantic scalars,
    # l=1/l=2 irreps, learned circles; S^2 directions and SO(3) frames are
    # the opt-in next ablation. latent_typing=euclidean: same heads, no
    # manifold projection (the all-Euclidean baseline).
    latent_scalar: int = 64
    latent_vector: int = 16
    latent_tensor: int = 8
    circular_channels: int = 8
    direction_channels: int = 0
    frame_channels: int = 0
    gram_channels: int = 4
    latent_typing: str = "typed"
    # Predictor: joint self-attention over context and mask tokens (I-JEPA style).
    predictor_layers: int = 4
    atom_decoder_layers: int = 2
    predictor_expansion: int = 2
    relative_positions: int = 32
    include_chi5: bool = False
    radius_atom: float = 5.0
    radius_residue: float = 12.0
    max_neighbors: int = 24
    # 'local': per-residue SC view (v0.2 contract). 'spatial' (opt-in experiment):
    # SC atoms also see other residues' SC atoms.
    sc_context: str = "local"
    backend: str = "reference"
    dropout: float = 0.0
    # Explicit architecture selection: old checkpoints/configs remain baseline.
    interaction: str = "baseline"
    effdock_radial_hidden: int = 96
    effdock_expansion: int = 2
    effdock_residual_scale: float = 0.1
    effdock_aggregation: str = "soft"
    effdock_conditioning: bool = True
    effdock_dual_radial: bool = True
    effdock_distance_decay: bool = True
    effdock_norm_rescale: bool = True
    effdock_smooth_cutoff: bool = True
    # Opt-in encoder experiments (v0.3 defaults keep the v0.2 operator family):
    # axis-projection gate invariants, bilinear cross-degree FFN, and a
    # per-destination envelope radius that vanishes where top-k truncates.
    effdock_directional: bool = False
    effdock_ffn: str = "gate"
    effdock_adaptive_cutoff: bool = False

    def __post_init__(self):
        if self.heads < 1 or self.sequence_width < 1:
            raise ValueError("heads and sequence_width must be positive.")
        if self.sequence_width % self.heads:
            raise ValueError("sequence_width must be divisible by heads.")
        if self.scalar % self.heads:
            raise ValueError("scalar must be divisible by heads for the predictor.")
        for key in ("circular_channels", "direction_channels", "frame_channels", "gram_channels",
                    "latent_vector", "latent_tensor"):
            if getattr(self, key) < 0:
                raise ValueError(f"{key} must be nonnegative.")
        if self.latent_typing not in {"typed", "euclidean"}:
            raise ValueError("latent_typing must be typed or euclidean.")
        if self.frame_channels and self.latent_typing != "typed":
            raise ValueError("frame_channels require latent_typing: typed (SO(3) projection).")
        for key in ("scalar", "vector", "tensor", "sequence_layers", "internal_layers", "latent_scalar",
                    "predictor_layers", "atom_decoder_layers", "predictor_expansion",
                    "relative_positions"):
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
        if self.sc_context not in {"spatial", "local"}:
            raise ValueError("sc_context must be spatial or local.")
        if self.effdock_ffn not in {"bilinear", "gate"}:
            raise ValueError("effdock_ffn must be bilinear or gate.")
        if self.effdock_adaptive_cutoff is True and self.effdock_aggregation == "degree":
            # A discrete degree denominator jumps when a bonded neighbour leaves
            # the top-k set, so the adaptive envelope cannot make it continuous.
            raise ValueError("effdock_adaptive_cutoff requires soft or gate aggregation.")

        if self.interaction not in {"baseline", "effdock"}:
            raise ValueError("interaction must be baseline or effdock.")
        if self.effdock_radial_hidden < 1 or self.effdock_expansion < 1:
            raise ValueError("EffDock widths and expansion must be positive.")
        if not 0 < self.effdock_residual_scale <= 1:
            raise ValueError("effdock_residual_scale must be in (0,1].")
        if self.effdock_aggregation not in {"soft", "gate", "degree"}:
            raise ValueError("Invalid effdock_aggregation.")
        for key in ("effdock_conditioning", "effdock_dual_radial", "effdock_distance_decay",
                    "effdock_norm_rescale", "effdock_smooth_cutoff", "effdock_directional",
                    "effdock_adaptive_cutoff"):
            if not isinstance(getattr(self, key), bool):
                raise ValueError(f"{key} must be boolean.")

    @property
    def dims(self):
        return FiberDims(self.scalar, self.vector, self.tensor)



@dataclass
class TrainConfig:
    steps: int = 1000
    batch_size: int = 2
    learning_rate: float = 0.0003
    weight_decay: float = 0.01
    # Teacher momentum rises linearly from ema to ema_end over the planned steps.
    ema: float = 0.996
    ema_end: float = 1.0
    grad_clip: float = 1.0
    crop_lengths: list[int] = field(default_factory=lambda: [128, 256])
    # Crops stay inside one chain; redraw windows with < this CA-observed fraction.
    crop_min_observed: float = 0.5
    mask_fraction: float = 0.35
    # Union of mask_blocks contiguous spans. 'spatial' (opt-in) picks a 3D
    # neighbourhood; its query positions then reveal hidden contacts.
    mask_blocks: int = 4
    mask_min_span: int = 8
    mask_mode: str = "span"
    # Off by default: the structure path, typed heads and losses are exactly
    # SO(3)-equivariant/invariant and translation-free, so a shared rigid
    # transform changes loss/gradients only at float precision (~1e-7/1e-5).
    # Keep for testing non-equivariant variants.
    rigid_augmentation: bool = False
    translation_std: float = 1.0
    global_weight: float = 0.1
    atom_weight: float = 0.2
    variance_weight: float = 0.05
    # On raw (un-normalized) online context semantic latents. Covariance is ON:
    # without it a 22-protein stochastic overfit collapsed to effective rank ~3
    # with lower loss but worse retrieval (docs/TYPED_LATENT_V040_KO.md).
    covariance_weight: float = 0.04
    # Learned circles: per-channel floor relu(min - (1-|E z|^2)).
    circular_weight: float = 0.05
    circular_floor: float = 0.1
    # Geometry-aware heat-kernel MMD (arXiv:2609.21656) as ablation options.
    semantic_regularizer: str = "variance"
    circular_regularizer: str = "floor"
    # Soft RMS floor for l>0 targets, relative to the sample's mean token RMS.
    target_floor: float = 0.1
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
        if not (0 <= self.ema <= self.ema_end <= 1) or self.ema >= 1:
            raise ValueError("EMA must satisfy 0 <= ema < 1 and ema <= ema_end <= 1.")
        if not (0 < self.mask_fraction < 1) or min(self.mask_blocks, self.mask_min_span) < 1:
            raise ValueError("Invalid mask_fraction or mask_blocks.")
        if self.mask_mode not in {"span", "spatial", "mixed"}:
            raise ValueError("mask_mode must be span, spatial or mixed.")
        if self.mask_mode != "span":
            warnings.warn("Spatial masks choose query positions from hidden 3D proximity; "
                          "query sets then leak contact information (SPEC 15).", stacklevel=2)
        if self.batch_size < 1 or self.steps < 1 or not self.crop_lengths or min(self.crop_lengths) < 1:
            raise ValueError("Invalid train size.")
        if self.save_every < 1 or self.log_every < 1 or self.learning_rate <= 0:
            raise ValueError("Invalid schedule.")
        if not 0 <= self.crop_min_observed <= 1:
            raise ValueError("crop_min_observed must be in [0, 1].")
        if self.translation_std < 0:
            raise ValueError("translation_std must be nonnegative.")
        if self.threads < 1 or self.grad_clip <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid threads/gradient clipping/weight decay.")
        if self.semantic_regularizer not in {"variance", "sphere_mmd"}:
            raise ValueError("semantic_regularizer must be variance or sphere_mmd.")
        if self.circular_regularizer not in {"floor", "torus_mmd"}:
            raise ValueError("circular_regularizer must be floor or torus_mmd.")
        if min(self.global_weight, self.atom_weight, self.variance_weight, self.covariance_weight,
               self.target_floor, self.circular_weight, self.circular_floor) < 0:
            raise ValueError("Loss weights must be nonnegative.")


def _checked(cls, data):
    unknown = set(data)-{f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} options: {sorted(unknown)}")
    return cls(**data)


def model_config(data: dict) -> ModelConfig:
    """Checked ModelConfig from a stored dict (removed/unknown keys explained)."""
    return _checked(ModelConfig, data)


def train_config(data: dict) -> TrainConfig:
    return _checked(TrainConfig, data)


def load_config(path):
    data = yaml.safe_load(Path(path).read_text())
    if set(data)-{"model", "training"}:
        raise ValueError("Only model/training config sections are supported.")
    return _checked(ModelConfig, data.get("model", {})), _checked(TrainConfig, data.get("training", {}))
