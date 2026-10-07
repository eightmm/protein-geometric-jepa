"""Online encoder, EMA teacher encoder and JEPA predictor. No coordinate reconstruction."""
from copy import deepcopy
import torch
from torch import nn
from ..config import ModelConfig, TrainConfig
from ..objectives.losses import latent_distance, regularize_latents
from ..objectives.tasks import Observation
from .encoders import MultiViewEncoder
from .predictors import (CrossViewPredictor, make_target, reference_rms, topology_atoms,
                         low_rms_fraction)


class TargetStack(nn.Module):
    """The EMA-tracked part. v0.3 targets need no projector: every teacher
    weight that shapes a target is a weight the online encoder trains."""
    def __init__(self, cfg):
        super().__init__()
        self.encoder = MultiViewEncoder(cfg)


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
        # Mask tokens sit at the OBSERVATION's target residues and atom tokens
        # follow the visible-sequence topology; teacher validity/presence only
        # filters the loss, so hidden atom presence never shapes any query.
        query = torch.where(observation.target_residues)[0]
        atom_residues = atom_slots = None
        if spec.atom_loss and target.atoms is not None:
            atom_residues, atom_slots = topology_atoms(
                record.seq, observation.seq_visible, query, observation.task_name == "sc_infill")
        prediction = self.predictor(context, record.seq_pos, spec.target, query,
                                    atom_residues, atom_slots)
        floor = train_cfg.target_floor
        with torch.no_grad():
            target_nodes = make_target(target.nodes, target.node_valid, floor).index(query)
            target_global = make_target(target.global_state, None, floor,
                                        reference=reference_rms(target.nodes, target.node_valid))
        node_loss, info = latent_distance(prediction.nodes, target_nodes,
                                          target.node_valid[query], spec.equivariant)
        observed = any(v.global_valid or bool(v.node_valid.any()) for v in context.values())
        global_mask = torch.tensor([target.global_valid and observed], device=record.xyz.device)
        global_loss, _ = latent_distance(prediction.global_state, target_global, global_mask,
                                         spec.equivariant)
        atom_loss = node_loss*0
        if atom_residues is not None and len(atom_residues) and len(target.atom_residue):
            # Align topology queries with observed teacher atoms (key = residue*37+slot);
            # queries without an observed atom are masked out of the loss.
            keys = target.atom_residue*37+target.atom_slot
            order = keys.argsort()
            wanted = atom_residues*37+atom_slots
            where = torch.searchsorted(keys[order], wanted).clamp_max(len(keys)-1)
            observed_atom = keys[order][where] == wanted
            with torch.no_grad():
                atom_target = make_target(target.atoms, None, floor).index(order[where])
            atom_loss, _ = latent_distance(prediction.atoms, atom_target, observed_atom, True)
        # Online context scalars BEFORE normalization: layer-normed scalars sum
        # to zero, which would make the covariance penalty fight the norm itself.
        regularizer_inputs = {name: encoded.nodes.s[encoded.node_valid]
                              for name, encoded in context.items() if bool(encoded.node_valid.any())}
        loss = node_loss + train_cfg.global_weight*global_loss + train_cfg.atom_weight*atom_loss
        info.update(target_low_rms=low_rms_fraction(target.nodes, target.node_valid, floor))
        info.update(task=observation.task_name, node_loss=float(node_loss.detach()),
                    global_loss=float(global_loss.detach()), atom_loss=float(atom_loss.detach()))
        return loss, info, regularizer_inputs

    def forward(self, records_and_observations, train_cfg: TrainConfig):
        by_task, logs, groups = {}, [], {}
        for record, observation in records_and_observations:
            loss, info, inputs = self.task_loss(record, observation, train_cfg)
            by_task.setdefault(observation.task_name, []).append(loss)
            logs.append(info)
            for name, latent in inputs.items():
                groups.setdefault(name, []).append(latent)
        if not by_task:
            raise ValueError("Empty microbatch.")
        # Mean within each task, then across tasks: a mixed microbatch is not
        # dominated by whichever task happens to have more samples.
        loss = torch.stack([torch.stack(v).mean() for v in by_task.values()]).mean()
        reglogs = {}
        regs = []
        for name, latents in groups.items():
            var, cov, info = regularize_latents(latents)
            regs.append(train_cfg.variance_weight*var + train_cfg.covariance_weight*cov)
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
