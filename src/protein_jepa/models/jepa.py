"""Online/EMA encoder sets plus predictors. No raw-coordinate reconstruction loss."""
from copy import deepcopy
import torch
from torch import nn
from ..config import ModelConfig, TrainConfig
from ..objectives.losses import latent_distance, regularize_latents
from ..objectives.tasks import Observation
from .encoders import MultiViewEncoder
from .predictors import (CrossViewPredictor, LatentProjector, VIEW_NAMES, PERIODIC_VIEWS)


class TargetStack(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = MultiViewEncoder(cfg)
        self.projectors = nn.ModuleDict({name: LatentProjector(cfg) for name in VIEW_NAMES})


class ProteinJEPA(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.online = TargetStack(cfg)
        self.teacher = deepcopy(self.online).eval()
        self.teacher.requires_grad_(False)
        self.predictor = CrossViewPredictor(cfg)

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    @torch.no_grad()
    def update_teacher(self, momentum: float):
        online_params = dict(self.online.named_parameters())
        for name, parameter in self.teacher.named_parameters():
            parameter.lerp_(online_params[name], 1-momentum)
        online_buffers = dict(self.online.named_buffers())
        for name, buffer in self.teacher.named_buffers():
            buffer.copy_(online_buffers[name])

    def task_loss(self, record, observation: Observation, train_cfg: TrainConfig):
        spec = observation.spec
        context = self.online.encoder(record, spec.context, observation.atom_visible,
                                      observation.seq_visible)
        with torch.no_grad():
            target = self.teacher.encoder(record, (spec.target,))[spec.target]
            target_nodes = self.teacher.projectors[spec.target](target.nodes)
            target_global = self.teacher.projectors[spec.target](target.global_state)
        prediction = self.predictor(context, record.seq_pos, spec.target, record.seq_pos)
        mask = observation.target_residues & target.node_valid
        node_loss, info = latent_distance(prediction, target_nodes, mask, spec.equivariant,
                                          spec.target in PERIODIC_VIEWS)
        global_pred = self.predictor(context, record.seq_pos, spec.target,
                                     record.seq_pos.new_zeros(1), level="global")
        global_mask = torch.tensor([target.global_valid and any(v.global_valid for v in context.values())],
                                    device=record.xyz.device)
        global_loss, _ = latent_distance(global_pred, target_global, global_mask, spec.equivariant, False)
        atom_loss = node_loss*0
        if spec.atom_loss and target.atoms is not None:
            select = observation.target_residues[target.atom_residue]
            if observation.task_name == "sc_infill":
                select &= target.atom_slot >= 4
            ri, ai = target.atom_residue[select], target.atom_slot[select]
            if len(ri):
                with torch.no_grad():
                    atom_target = self.teacher.projectors[spec.target](target.atoms.index(select))
                atom_pred = self.predictor(context, record.seq_pos, spec.target,
                                           record.seq_pos[ri], "atom", ai)
                atom_loss, _ = latent_distance(atom_pred, atom_target,
                                               torch.ones(len(ri), device=ri.device, dtype=torch.bool), True, False)
        # Return online latents to regularize at microbatch level, grouped by view.
        regularizer_inputs = {}
        for name, encoded in context.items():
            if bool(encoded.node_valid.any()):
                regularizer_inputs[name] = self.online.projectors[name](encoded.nodes.index(encoded.node_valid))
        loss = node_loss + train_cfg.global_weight*global_loss + train_cfg.atom_weight*atom_loss
        info.update(task=observation.task_name, node_loss=float(node_loss.detach()),
                    global_loss=float(global_loss.detach()), atom_loss=float(atom_loss.detach()))
        return loss, info, regularizer_inputs

    def forward(self, records_and_observations, train_cfg: TrainConfig):
        losses, logs, groups = [], [], {}
        for record, observation in records_and_observations:
            loss, info, inputs = self.task_loss(record, observation, train_cfg)
            losses.append(loss)
            logs.append(info)
            for name, latent in inputs.items():
                groups.setdefault(name, []).append(latent)
        if not losses:
            raise ValueError("Empty microbatch.")
        loss = torch.stack(losses).mean()
        reglogs = {}
        regs = []
        for name, latents in groups.items():
            var, cov, circ, info = regularize_latents(latents)
            regs.append(train_cfg.variance_weight*var + train_cfg.covariance_weight*cov
                        + (train_cfg.circular_weight*circ if name in PERIODIC_VIEWS else 0))
            reglogs[name] = info
        if regs:
            loss = loss + torch.stack(regs).mean()
        return loss, {"samples": logs, "regularization": reglogs}

    @torch.no_grad()
    def encode(self, record, mode="sequence"):
        mapping = {"sequence": ("seq",), "backbone": ("bb",),
                   "all_atom": ("aa",), "multimodal": ("seq", "bb", "aa", "bb_internal", "chi")}
        if mode not in mapping:
            raise ValueError(f"Unknown inference mode: {mode}")
        was_training = self.training
        self.eval()
        result = self.online.encoder(record, mapping[mode])
        self.train(was_training)
        return result
