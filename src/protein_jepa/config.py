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
    # 'local': per-residue SC view (the default contract). 'spatial' (opt-in experiment):
    # SC atoms also see other residues' SC atoms.
    sc_context: str = "local"
    # Residue graphs carry invariant pair geometry R_i^T(x_j-x_i), R_i^T R_j (spec 6.3).
    pair_frame_features: bool = True
    # Opt-in: SC atoms also see their position in the residue's backbone frame
    # (spec 7.1 lists it as optional). On 22 proteins it did not change retrieval
    # and inflated the SC latent scale (pooled effective rank 49 -> 3).
    sc_local_frame: bool = False
    # Global readout: learned invariant attention, or uniform mean pooling (ablation).
    global_readout: str = "attention"
    # Predictor heads for the raw-geometry reconstruction baseline.
    raw_reconstruction_heads: bool = True
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
    # Opt-in encoder experiments (defaults keep the original operator family):
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
        if self.global_readout not in {"attention", "mean"}:
            raise ValueError("global_readout must be attention or mean.")
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
        for key in ("pair_frame_features", "sc_local_frame", "raw_reconstruction_heads"):
            if not isinstance(getattr(self, key), bool):
                raise ValueError(f"{key} must be boolean.")
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
    # adamw, or muon (Muon on hidden weight matrices + AdamW for the rest;
    # PyTorch >= 2.9). One learning_rate serves both (match_rms_adamw scaling).
    optimizer: str = "adamw"
    muon_momentum: float = 0.95
    # No weight decay on 1D parameters (biases, norm gains, residual scales),
    # embeddings and the relative-position bias table. False decays everything.
    decay_exclusions: bool = True
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
    node_weight: float = 1.0
    global_weight: float = 0.1
    atom_weight: float = 0.2
    # Raw-geometry reconstruction baseline (spec 19.6, default off): query
    # residues' torsions (phi/psi/omega/CA-dihedral, chi1-4) and their CA
    # displacement from the visible-CA centroid (equivariant contexts only).
    raw_angle_weight: float = 0.0
    raw_coordinate_weight: float = 0.0
    # ema: stop-gradient EMA teacher. online: teacher-free baseline (spec 18.3),
    # targets come from the online stack WITH gradient and are regularized too.
    target_encoder: str = "ema"
    # Semantic latent distance: mse (default) or cosine (spec 19.2 ablation).
    semantic_distance: str = "mse"
    variance_weight: float = 0.05
    # On raw (un-normalized) online context semantic latents. Covariance is ON:
    # without it a 22-protein stochastic overfit collapsed to effective rank ~3
    # with lower loss but worse retrieval (docs/TYPED_LATENT_KO.md).
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
    # Optimizer step = accumulation_steps microbatches of batch_size per rank;
    # per-task loss balancing happens within each microbatch.
    accumulation_steps: int = 1
    # Background processes that build crops/masks ahead of the GPU (0 = in-process).
    loader_workers: int = 0
    # Max samples per packed group (0 = all same-task samples of a microbatch);
    # bounds peak memory without changing the per-sample losses.
    pack_size: int = 0
    # Also keep step_<N>.pt for the last N saves (0: only last.pt).
    keep_checkpoints: int = 0
    # auto: nccl on CUDA, gloo on CPU. gloo on CUDA lets several ranks share one GPU (testing).
    dist_backend: str = "auto"
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
        if self.accumulation_steps < 1 or self.loader_workers < 0 or self.pack_size < 0:
            raise ValueError("accumulation_steps must be positive; loader_workers/pack_size nonnegative.")
        if self.optimizer not in {"adamw", "muon"}:
            raise ValueError("optimizer must be adamw or muon.")
        if not 0 <= self.muon_momentum < 1 or self.keep_checkpoints < 0:
            raise ValueError("muon_momentum must be in [0, 1); keep_checkpoints nonnegative.")
        if self.dist_backend not in {"auto", "nccl", "gloo"}:
            raise ValueError("dist_backend must be auto, nccl or gloo.")
        if self.semantic_regularizer not in {"variance", "sphere_mmd"}:
            raise ValueError("semantic_regularizer must be variance or sphere_mmd.")
        if self.circular_regularizer not in {"floor", "torus_mmd"}:
            raise ValueError("circular_regularizer must be floor or torus_mmd.")
        if self.target_encoder not in {"ema", "online"}:
            raise ValueError("target_encoder must be ema or online.")
        if self.semantic_distance not in {"mse", "cosine"}:
            raise ValueError("semantic_distance must be mse or cosine.")
        if min(self.node_weight, self.global_weight, self.atom_weight, self.variance_weight,
               self.covariance_weight, self.target_floor, self.circular_weight,
               self.circular_floor, self.raw_angle_weight, self.raw_coordinate_weight) < 0:
            raise ValueError("Loss weights must be nonnegative.")


def _checked(cls, data):
    unknown = set(data)-{f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} options: {sorted(unknown)}")
    return cls(**data)


# Architecture options added after checkpoints already existed: a stored
# config that omits one was built without it, so it loads with this value
# (fresh YAML configs get the dataclass defaults instead).
LEGACY_MODEL_DEFAULTS = {"pair_frame_features": False, "sc_local_frame": False,
                         "global_readout": "attention", "raw_reconstruction_heads": False}


def model_config(data: dict) -> ModelConfig:
    """Checked ModelConfig from a stored dict (removed/unknown keys explained)."""
    return _checked(ModelConfig, {**LEGACY_MODEL_DEFAULTS, **data})


def train_config(data: dict) -> TrainConfig:
    return _checked(TrainConfig, data)


def load_config(path):
    data = yaml.safe_load(Path(path).read_text())
    if set(data)-{"model", "training"}:
        raise ValueError("Only model/training config sections are supported.")
    return _checked(ModelConfig, data.get("model", {})), _checked(TrainConfig, data.get("training", {}))
