"""Online encoder, EMA teacher encoder and JEPA predictor. No coordinate reconstruction."""
from copy import deepcopy
import torch
from torch import nn
from ..config import ModelConfig, TrainConfig
from ..objectives.losses import latent_distance, regularize_latents
from ..objectives.tasks import Observation
from .encoders import MultiViewEncoder
from .predictors import (CrossViewPredictor, ContextLatent, VIEW_NAMES, make_target,
                         topology_atoms, low_rms_fraction)
from .latents import (TypedLatentHead, latent_spec, make_typed_target, eq_reference,
                      typed_distance, circular_floor, torus_mmd, sphere_mmd)


@torch.no_grad()
def node_retrieval(pred, target, valid) -> dict:
    """Share of masked residues whose prediction is closest (centred cosine,
    invariant scalars) to its OWN target among all masked targets; chance 1/n.

    Loss alone cannot separate learning from collapse: predicting the common
    mean lowers MSE but scores chance here. Centring over the masked set
    removes the shared component that dominates untrained encoder states.
    """
    n = int(valid.sum())
    if n < 2:
        return {}
    p, t = pred.sem[valid], target.sem[valid]
    p = torch.nn.functional.normalize(p-p.mean(0), dim=-1)
    t = torch.nn.functional.normalize(t-t.mean(0), dim=-1)
    top1 = (p @ t.T).argmax(-1) == torch.arange(n, device=p.device)
    return {"node_top1": float(top1.float().mean()), "node_chance": 1/n}


class TargetStack(nn.Module):
    """The EMA-tracked part: view encoders AND their typed latent heads.

    Context latents come from the online heads and feed the predictor, so
    every head weight that shapes a teacher target is trained by the
    prediction loss (design B; the v0.2 projector only saw a regularizer).
    """
    def __init__(self, cfg):
        super().__init__()
        self.encoder = MultiViewEncoder(cfg)
        typed = cfg.latent_typing == "typed"
        self.heads = nn.ModuleDict({v: TypedLatentHead(cfg.dims, latent_spec(cfg, v), typed,
                                                       cfg.gram_channels) for v in VIEW_NAMES})

    def context(self, encoded: dict) -> dict[str, ContextLatent]:
        out = {}
        for name, view in encoded.items():
            index = torch.where(view.node_valid)[0]
            nodes = self.heads[name](view.nodes.index(index))
            glob = self.heads[name](view.global_state) if view.global_valid else None
            out[name] = ContextLatent(nodes, index, glob)
        return out


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

    def context_latents(self, record, views, atom_visible=None, seq_visible=None):
        """Online typed context latents for the given visible observation."""
        encoded = self.online.encoder(record, views, atom_visible, seq_visible)
        return self.online.context(encoded)

    def task_loss(self, record, observation: Observation, train_cfg: TrainConfig):
        spec = observation.spec
        typed = self.cfg.latent_typing == "typed"
        context = self.context_latents(record, spec.context, observation.atom_visible,
                                       observation.seq_visible)
        with torch.no_grad():
            target = self.teacher.encoder(record, (spec.target,))[spec.target]
            head = self.teacher.heads[spec.target]
            target_nodes_raw = head(target.nodes)
            target_global_raw = head(target.global_state)
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
            node_target, kind_masks = make_typed_target(target_nodes_raw, target.node_valid, floor)
            node_target = node_target.index(query)
            kind_masks = {k: m[query] for k, m in kind_masks.items()}
            global_target, _ = make_typed_target(
                target_global_raw, None, floor, instance=False,
                reference=eq_reference(target_nodes_raw, target.node_valid))
        node_loss, info = typed_distance(prediction.nodes, node_target, target.node_valid[query],
                                         spec.equivariant, typed, kind_masks)
        observed = any(v.global_state is not None or len(v.index) for v in context.values())
        global_mask = torch.tensor([target.global_valid and observed], device=record.xyz.device)
        # Global latents stay invariant semantic + irreps; never circles/frames.
        global_loss, _ = typed_distance(prediction.global_state, global_target, global_mask,
                                        spec.equivariant, typed, kinds=('sem', 'eq'))
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
                atom_target = make_target(target.atoms, None, floor,
                                          instance=True).index(order[where])
            atom_loss, _ = latent_distance(prediction.atoms, atom_target, observed_atom, True)
        # Online context latents BEFORE target normalization: layer-normed
        # scalars sum to zero, which would make a covariance penalty fight the norm.
        regularizer_inputs = {name: (z.nodes.sem, z.nodes.circ) for name, z in context.items()
                              if len(z.index)}
        loss = node_loss + train_cfg.global_weight*global_loss + train_cfg.atom_weight*atom_loss
        info.update(target_low_rms=low_rms_fraction(target.nodes, target.node_valid, floor),
                    **node_retrieval(prediction.nodes, node_target, target.node_valid[query]))
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
        reglogs, regs = {}, []
        typed = self.cfg.latent_typing == "typed"
        for name, latents in groups.items():
            sems = [z[0] for z in latents]
            circs = [z[1] for z in latents]
            var, cov, info = regularize_latents(sems)
            reg = train_cfg.variance_weight*var + train_cfg.covariance_weight*cov
            if train_cfg.semantic_regularizer == "sphere_mmd":
                # Directional uniformity on top of (not instead of) the scale floor.
                reg = reg + train_cfg.variance_weight*sphere_mmd(torch.cat(sems))
            circ = torch.cat(circs)
            if circ.shape[1]:
                if not typed:
                    # Raw (Euclidean) circles have no circle floor: use a variance floor.
                    circ_reg, _, _ = regularize_latents([circ.flatten(1)])
                elif train_cfg.circular_regularizer == "torus_mmd":
                    circ_reg = torus_mmd(circ)
                else:
                    circ_reg = circular_floor(circ, train_cfg.circular_floor)
                reg = reg + train_cfg.circular_weight*circ_reg
                info["circular_variance"] = float((1-circ.mean(0).square().sum(-1)).mean().detach())
            regs.append(reg)
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

    @torch.no_grad()
    def latents(self, record, mode="sequence") -> dict[str, ContextLatent]:
        """Typed latents (semantic, irreps, circles, directions, frames) of the
        online encoders for downstream use; hidden states come from encode()."""
        was_training = self.training
        self.eval()
        encoded = self.encode(record, mode)
        result = self.online.context(encoded)
        self.train(was_training)
        return result
