"""Online encoder, EMA teacher encoder and JEPA predictor. No coordinate reconstruction."""
from contextlib import nullcontext
from copy import deepcopy
import torch
from torch import nn, Tensor
from ..config import ModelConfig, TrainConfig
from ..objectives.losses import fiber_distance, regularize_latents
from ..data.batch import segment_mean
from ..data.constants import AA3, ATOM_ID, N_ATOMS, SYMMETRIC_RENAMES
from ..objectives.tasks import Observation
from .encoders import MultiViewEncoder
from ..data.batch import pack, padded_layout, to_padded
from .predictors import (CrossViewPredictor, ContextLatent, VIEW_NAMES, make_target,
                         topology_atoms, low_rms_fraction)
from ..geometry.features import backbone_features, chi_features
from ..objectives.losses import effective_rank
from .latents import _masked_term
from .latents import (TypedLatentHead, latent_spec, make_typed_target, eq_reference,
                      typed_distance, circular_floor, torus_mmd, sphere_mmd)


def _rename_table() -> Tensor:
    """[21, 37]: each slot's partner under the residue's symmetric flip (else itself)."""
    table = torch.arange(N_ATOMS).repeat(len(AA3)+1, 1)
    for name, pairs in SYMMETRIC_RENAMES.items():
        for a, b in pairs.items():
            table[AA3.index(name), ATOM_ID[a]] = ATOM_ID[b]
            table[AA3.index(name), ATOM_ID[b]] = ATOM_ID[a]
    return table


RENAMES = _rename_table()


def symmetric_atom_loss(pred, target, target_residue, target_slot, residues, slots, seq,
                        segment, size):
    """Per-record atom loss, minimized per residue over {names as given,
    symmetric flip}: equivalent naming (ASP OD1/OD2, a PHE ring flip, ...) is
    not a learnable difference. Queries without an observed atom are masked."""
    keys = target_residue*N_ATOMS+target_slot
    order = keys.argsort()
    ranked = keys[order]

    def assignment(query_slots):
        wanted = residues*N_ATOMS+query_slots
        where = torch.searchsorted(ranked, wanted).clamp_max(len(keys)-1)
        found = ranked[where] == wanted
        return fiber_distance(pred, target.index(order[where]), True)*found, found
    same, found = assignment(slots)
    flip, found_flip = assignment(RENAMES.to(slots.device)[seq[residues], slots])
    with torch.no_grad():
        n = int(residues.max())+1
        flipped = (torch.zeros(n, device=same.device).index_add_(0, residues, flip.detach())
                   < torch.zeros(n, device=same.device).index_add_(0, residues, same.detach()))[residues]
    loss = torch.where(flipped, flip, same)
    valid = torch.where(flipped, found_flip, found)
    return segment_mean(loss[valid], segment[valid], size)[0]+pred.s.sum()*0


@torch.no_grad()
def equivariant_diagnostics(vectors: list, tensors: list, gram_channels: int = 4) -> dict:
    """Collapse watch for l>0 latents (spec 20.3), no loss term: per-degree
    channel RMS, share of near-dead channels (RMS < 0.1 x mean channel RMS)
    and the effective rank of rotation-invariant Gram contractions across
    tokens of many proteins (rotation alone cannot inflate it)."""
    out = {}
    for name, xs, dof in (("l1", vectors, 3), ("l2", tensors, 5)):
        x = torch.cat(xs)
        if not x.shape[1] or not len(x):
            continue
        rms = (x.flatten(2).square().sum(-1)/dof).mean(0).sqrt()
        out[f"{name}_rms"] = float(rms.mean())
        out[f"{name}_dead_channels"] = float((rms < 0.1*rms.mean()).float().mean())
    v = torch.cat(vectors)[:, :gram_channels]
    if v.shape[1] >= 2 and len(v) >= 2:
        upper = torch.triu_indices(v.shape[1], v.shape[1], device=v.device)
        gram = torch.einsum('nci,ndi->ncd', v, v)[:, upper[0], upper[1]]
        out["l1_gram_rank"] = effective_rank(gram)
    return out


@torch.no_grad()
def node_retrieval(pred: Tensor, target: Tensor, valid: Tensor, segment: Tensor,
                   size: int) -> tuple[Tensor, Tensor]:
    """Per record: share of masked residues whose prediction is closest
    (centred cosine, invariant scalars) to its OWN target among that record's
    masked targets; chance 1/n. Returns (top1 [B], n [B]).

    Loss alone cannot separate learning from collapse: predicting the common
    mean lowers MSE but scores chance here. Centring over the masked set
    removes the shared component that dominates untrained encoder states.
    """
    seg = segment[valid]
    layout = padded_layout(seg, size)
    real = layout >= 0
    n = real.sum(-1)
    if not layout.shape[1]:
        return n.to(pred.dtype), n
    p, t = to_padded(pred[valid], layout), to_padded(target[valid], layout)
    w = real[..., None].to(p.dtype)

    def centred(x):
        mean = (x*w).sum(1, keepdim=True)/n.clamp_min(1)[:, None, None]
        x = (x-mean)*w   # padding stays zero, so it never sets the scale
        # A collapsed record leaves only rounding noise after centring; cosine
        # would amplify it into an arbitrary score, so it scores chance (1/n).
        scale = mean.norm(dim=-1, keepdim=True)+x.norm(dim=-1, keepdim=True).amax(1, keepdim=True)
        x = x.masked_fill(x.norm(dim=-1, keepdim=True) <= 1e-4*scale, 0)
        return torch.nn.functional.normalize(x, dim=-1)
    sim = torch.einsum('bid,bjd->bij', centred(p), centred(t)).masked_fill(~real[:, None, :], -torch.inf)
    hit = (sim.argmax(-1) == torch.arange(layout.shape[1], device=p.device)) & real
    return hit.sum(-1)/n.clamp_min(1), n


class TargetStack(nn.Module):
    """The EMA-tracked part: view encoders AND their typed latent heads.

    Context latents come from the online heads and feed the predictor, so
    every head weight that shapes a teacher target is trained by the
    prediction loss (design B; an earlier projector only saw a regularizer).
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
            out[name] = ContextLatent(nodes, index, self.heads[name](view.global_state),
                                      view.global_valid)
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
        teacher = list(self.teacher.named_parameters())
        torch._foreach_lerp_([p for _, p in teacher], [online_params[n] for n, _ in teacher], 1-momentum)
        online_buffers = dict(self.online.named_buffers())
        for name, buffer in self.teacher.named_buffers():
            buffer.copy_(online_buffers[name])

    def context_latents(self, record, views, atom_visible=None, seq_visible=None):
        """Online typed context latents for the given visible observation."""
        encoded = self.online.encoder(record, views, atom_visible, seq_visible)
        return self.online.context(encoded)

    def task_loss(self, record, observation: Observation, train_cfg: TrainConfig):
        """One sample: (loss, info, regularizer inputs)."""
        return self.sample_losses([(record, observation)], train_cfg)[0]

    def sample_losses(self, records_and_observations, train_cfg: TrainConfig,
                      group_size: int | None = None) -> list[tuple]:
        """Per-sample (loss, info, regularizer inputs) in input order.

        Samples of the same task share context/target views, so each task's
        samples run as one packed batch (at most `group_size` per batch).
        """
        groups = {}
        for i, (_, observation) in enumerate(records_and_observations):
            groups.setdefault(observation.task_name, []).append(i)
        out = [None]*len(records_and_observations)
        for members in groups.values():
            step = group_size or len(members)
            for start in range(0, len(members), step):
                chunk = members[start:start+step]
                results = self.group_loss([records_and_observations[i][0] for i in chunk],
                                          [records_and_observations[i][1] for i in chunk], train_cfg)
                for i, result in zip(chunk, results):
                    out[i] = result
        return out

    def group_loss(self, records, observations, train_cfg: TrainConfig) -> list[tuple]:
        """Samples of ONE task as a packed batch; per-record statistics,
        losses and diagnostics equal the separate per-sample computation."""
        spec = observations[0].spec
        if any(o.task_name != observations[0].task_name for o in observations):
            raise ValueError("A group holds one task.")
        typed = self.cfg.latent_typing == "typed"
        batch = pack(records)
        size, owner = batch.size, batch.batch
        atom_visible = torch.cat([o.atom_visible for o in observations])
        seq_visible = torch.cat([o.seq_visible for o in observations])
        target_residues = torch.cat([o.target_residues for o in observations])
        context = self.context_latents(batch, spec.context, atom_visible, seq_visible)
        # EMA teacher under stop-gradient, or (teacher-free baseline) the online
        # stack itself with gradient flowing into the targets.
        teacher_free = train_cfg.target_encoder == "online"
        stack = self.online if teacher_free else self.teacher
        targets_grad = nullcontext() if teacher_free else torch.no_grad()
        with targets_grad:
            target = stack.encoder(batch, (spec.target,))[spec.target]
            head = stack.heads[spec.target]
            target_nodes_raw = head(target.nodes)
            target_global_raw = head(target.global_state)
        # Mask tokens sit at the OBSERVATION's target residues and atom tokens
        # follow the visible-sequence topology; teacher validity/presence only
        # filters the loss, so hidden atom presence never shapes any query.
        query = torch.where(target_residues)[0]
        atom_residues = atom_slots = None
        if spec.atom_loss and target.atoms is not None:
            atom_residues, atom_slots = topology_atoms(
                batch.seq, seq_visible, query, observations[0].task_name == "sc_infill")
        raw = train_cfg.raw_angle_weight > 0 or train_cfg.raw_coordinate_weight > 0
        prediction = self.predictor(context, batch.seq_pos, spec.target, query,
                                    atom_residues, atom_slots, owner, size, raw)
        floor = train_cfg.target_floor
        every = torch.arange(size, device=owner.device)
        with targets_grad:
            node_target, kind_masks = make_typed_target(target_nodes_raw, target.node_valid, floor,
                                                        batch=owner, size=size)
            node_target = node_target.index(query)
            kind_masks = {k: m[query] for k, m in kind_masks.items()}
            global_target, _ = make_typed_target(
                target_global_raw, None, floor, instance=False,
                reference=eq_reference(target_nodes_raw, target.node_valid, owner, size),
                batch=every, size=size)
        node_valid = target.node_valid[query]
        semantic = train_cfg.semantic_distance
        node_loss, terms = typed_distance(prediction.nodes, node_target, node_valid,
                                          spec.equivariant, typed, kind_masks, segment=owner[query],
                                          size=size, semantic=semantic)
        observed = torch.zeros(size, dtype=torch.bool, device=owner.device)
        for latent in context.values():
            observed |= latent.global_valid
            observed[owner[latent.index]] = True
        # Global latents stay invariant semantic + irreps; never circles/frames.
        global_loss, _ = typed_distance(prediction.global_state, global_target,
                                        target.global_valid & observed, spec.equivariant, typed,
                                        kinds=('sem', 'eq'), segment=every, size=size,
                                        semantic=semantic)
        atom_loss = node_loss*0
        if atom_residues is not None and len(atom_residues) and len(target.atom_residue):
            # Topology queries meet observed teacher atoms by (residue, slot), up
            # to the residue's symmetric renaming.
            with torch.no_grad():
                atom_target = make_target(target.atoms, None, floor, instance=True,
                                          batch=owner[target.atom_residue], size=size)
            atom_loss = symmetric_atom_loss(prediction.atoms, atom_target, target.atom_residue,
                                            target.atom_slot, atom_residues, atom_slots, batch.seq,
                                            owner[atom_residues], size)
        raw_angle = raw_coordinate = node_loss.detach()*0
        if raw:
            raw_angle, raw_coordinate = self._raw_losses(batch, prediction, query, atom_visible,
                                                         spec.equivariant)
        loss = (train_cfg.node_weight*node_loss + train_cfg.global_weight*global_loss
                + train_cfg.atom_weight*atom_loss + train_cfg.raw_angle_weight*raw_angle
                + train_cfg.raw_coordinate_weight*raw_coordinate)
        top1, retrieved = node_retrieval(prediction.nodes.sem, node_target.sem, node_valid,
                                         owner[query], size)
        low = low_rms_fraction(target.nodes, target.node_valid, floor, owner, size)
        # Online context latents BEFORE target normalization: layer-normed
        # scalars sum to zero, which would make a covariance penalty fight the norm.
        regularizer_inputs = [{} for _ in range(size)]
        latents = [(name, z.nodes, owner[z.index]) for name, z in context.items()]
        if teacher_free:
            # Targets carry gradient here, so their latents are regularized too.
            valid = torch.where(target.node_valid)[0]
            latents.append((f"{spec.target}_target", target_nodes_raw.index(valid), owner[valid]))
        for name, z, records_of in latents:
            counts = torch.bincount(records_of, minlength=size).tolist()
            for b, parts in enumerate(zip(*(x.split(counts) for x in (z.sem, z.circ, z.v, z.t)))):
                if len(parts[0]):
                    regularizer_inputs[b][name] = parts
        # One device->host transfer for every per-record diagnostic.
        counts = terms.pop('counts')
        terms.pop('valid_targets')
        names = list(terms)
        table = torch.stack([terms[k].detach() for k in names]+[counts[k] for k in names]+[
            node_loss.detach(), global_loss.detach(), atom_loss.detach(), top1, retrieved.float(),
            low, raw_angle.detach(), raw_coordinate.detach()]).T.tolist()
        results = []
        k = len(names)
        for b, row in enumerate(table):
            info = {"valid_targets": int(row[k+names.index('sem')])}
            if info["valid_targets"]:
                info.update({name: row[i] for i, name in enumerate(names) if row[k+i] > 0})
            node, glob, atom, hit, n, low_b, angle_b, coordinate_b = row[2*k:]
            info.update(target_low_rms=low_b, task=observations[b].task_name, node_loss=node,
                        global_loss=glob, atom_loss=atom)
            if raw:
                info.update(raw_angle_loss=angle_b, raw_coordinate_loss=coordinate_b)
            if n >= 2:
                info.update(node_top1=hit, node_chance=1/n)
            results.append((loss[b], info, regularizer_inputs[b]))
        return results

    def _raw_losses(self, batch, prediction, query, atom_visible, equivariant):
        """Raw-geometry baseline per record: 1 - cos over valid torsions of the
        query residues, and the CA offset from the visible-CA centroid (Angstrom/10;
        equivariant contexts only). Targets come from the full crop."""
        owner, size = batch.batch, batch.size
        with torch.no_grad():
            bb, chi = backbone_features(batch), chi_features(batch)
            angles = torch.cat((bb.periodic, chi.encoded), 1)[query]
            valid = torch.cat((bb.periodic_valid, chi.valid), 1)[query]
        per = 1-(prediction.raw_angles*angles).sum(-1)
        angle = _masked_term(per, valid, owner[query], size)[0]
        coordinate = angle.detach()*0
        if equivariant:
            ca = batch.xyz[:, 1].float()
            seen = atom_visible[:, 1] & batch.present[:, 1]
            centroid, count = segment_mean(ca, owner, size, seen)
            target = (ca[query]-centroid[owner[query]])/10
            ok = batch.present[query, 1] & (count[owner[query]] > 0)
            error = (prediction.raw_coordinate-target).square().sum(-1)/3
            coordinate = segment_mean(error[ok], owner[query][ok], size)[0]
        return angle, coordinate+prediction.raw_coordinate.sum()*0

    def forward(self, records_and_observations, train_cfg: TrainConfig):
        by_task, logs, groups = {}, [], {}
        if not records_and_observations:
            raise ValueError("Empty microbatch.")
        results = self.sample_losses(records_and_observations, train_cfg, train_cfg.pack_size or None)
        for (_, observation), (loss, info, inputs) in zip(records_and_observations, results):
            by_task.setdefault(observation.task_name, []).append(loss)
            logs.append(info)
            for name, latent in inputs.items():
                groups.setdefault(name, []).append(latent)
        # Mean within each task, then across tasks: a mixed microbatch is not
        # dominated by whichever task happens to have more samples.
        loss = torch.stack([torch.stack(v).mean() for v in by_task.values()]).mean()
        reglogs, regs = {}, []
        typed = self.cfg.latent_typing == "typed"
        for name, latents in groups.items():
            sems = [z[0] for z in latents]
            circs = [z[1] for z in latents]
            var, cov, info = regularize_latents(sems)
            info.update(equivariant_diagnostics([z[2] for z in latents], [z[3] for z in latents]))
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
