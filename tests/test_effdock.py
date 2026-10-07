"""Behavioral tests shared by the analytic and ACTUAL CuEq TP implementations."""
from dataclasses import replace
import importlib.util

import pytest
import torch

from conftest import assert_fiber_close
from protein_jepa.config import ModelConfig, TrainConfig
from protein_jepa.data.graphs import Graph, make_graph, NUM_EDGE_TYPES
from protein_jepa.geometry.primitives import random_rotation, symmetric_traceless
from protein_jepa.models.fibers import Fiber, FiberDims
from protein_jepa.models.effdock_blocks import (
    FiberRMSNorm, FiberDropout, ConditionalFiberNorm, StableNormRescale,
    EffDockInteractionBlock, gated_aggregate,
)
from protein_jepa.models.jepa import ProteinJEPA
from protein_jepa.objectives.tasks import TASKS, make_observation

CUEQ = (importlib.util.find_spec('cuequivariance') is not None
        and importlib.util.find_spec('cuequivariance_torch') is not None)
BACKENDS = ['reference', pytest.param('cueq-naive', marks=[
    pytest.mark.cueq, pytest.mark.skipif(not CUEQ, reason='Actual CuEq package required')])]


def fiber(n=7, dims=FiberDims(8, 4, 2)):
    return Fiber(torch.randn(n, dims.scalar), torch.randn(n, dims.vector, 3),
                 symmetric_traceless(torch.randn(n, dims.tensor, 3, 3)))


def graph(n=7):
    return make_graph(torch.randn(n, 3), torch.arange(n), torch.arange(n),
                      torch.empty(2, 0, dtype=torch.long), radius=5., max_neighbors=6,
                      node_kind=torch.arange(n) % 3)


@pytest.fixture(params=BACKENDS)
def eff_cfg(request, tiny_cfg):
    return replace(tiny_cfg, geometry_block='effdock', backend=request.param,
                   eff_radial_hidden=24, eff_edge_dim=8)


@pytest.fixture
def eff_model(eff_cfg):
    return ProteinJEPA(eff_cfg).eval()


@pytest.mark.parametrize('kind', ['rms', 'drop', 'conditional', 'rescale'])
def test_typed_helpers_equivariant(kind):
    dims = FiberDims(8, 4, 2)
    h, r = fiber(), random_rotation()
    if kind == 'rms':
        op = FiberRMSNorm(dims)
    elif kind == 'drop':
        op = FiberDropout(dims, .3).train()
    elif kind == 'rescale':
        op = StableNormRescale(dims)
    else:
        mod = ConditionalFiberNorm(dims, 5)
        # Nonzero modulation verifies the trained case, not only identity init.
        torch.nn.init.normal_(mod.modulate.weight, std=.2)
        c = torch.randn(7, 5)
        op = lambda x: mod(x, c)
    torch.manual_seed(19); a = op(h).rotate(r)
    torch.manual_seed(19); b = op(h.rotate(r))
    assert_fiber_close(a, b)


def test_zero_rescale_finite_derivative():
    dims = FiberDims(8, 4, 2)
    h = Fiber.zeros(3, dims, torch.zeros(1))
    for x in (h.s, h.v, h.t):
        x.requires_grad_(True)
    layer = StableNormRescale(dims)
    torch.nn.init.normal_(layer.v_map.bias)
    out = layer(h)
    assert out.v.count_nonzero() == out.t.count_nonzero() == 0
    (out.s.sum()+out.v.sum()+out.t.sum()).backward()
    assert torch.isfinite(h.v.grad).all() and torch.isfinite(h.t.grad).all()


def test_soft_aggregation_does_not_cancel_single_edge_decay():
    dims = FiberDims(8, 4, 2)
    h = fiber(1)
    dst = torch.tensor([0])
    high = torch.ones(1, dims.invariant)
    low = high*.001
    a = gated_aggregate(h, high, dst, 1, dims, 'soft')
    b = gated_aggregate(h, low, dst, 1, dims, 'soft')
    assert b.v.norm() < a.v.norm()*.01
    # A normalized attention mean largely cancels uniform attenuation.
    a = gated_aggregate(h, high, dst, 1, dims, 'gate')
    b = gated_aggregate(h, low, dst, 1, dims, 'gate')
    assert b.v.norm() > a.v.norm()*.99


def test_typed_graph_relations_and_device_bonds():
    x = torch.tensor([[0., 0, 0], [1., 0, 0], [0., 2, 0]])
    bonds = torch.tensor([[0, 1], [1, 0]])
    kinds = torch.tensor([0, 1, 2])
    g = make_graph(x, torch.arange(3), torch.arange(3), bonds, node_kind=kinds)
    for e, (a, b) in enumerate(g.edge_index.T.tolist()):
        bonded = (a, b) in [(0, 1), (1, 0)]
        assert int(g.relation[e, 2]) == bonded
        assert int(g.edge_type[e]) == 2*(3*int(kinds[a])+int(kinds[b]))+bonded
    assert int(g.edge_type.max()) < NUM_EDGE_TYPES


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('mode', ['soft', 'gate', 'degree'])
def test_block_rotation_and_permutation(backend, mode):
    dims = FiberDims(8, 4, 2)
    layer = EffDockInteractionBlock(dims, backend, radial_hidden=24, cutoff=5., aggregation=mode).eval()
    h, g, r = fiber(), graph(), random_rotation()
    rotated = replace(g, direction=g.direction @ r.T)
    assert_fiber_close(layer(h, g).rotate(r), layer(h.rotate(r), rotated), atol=3e-4, rtol=3e-4)
    p = torch.randperm(len(h.s)); inverse = torch.argsort(p)
    permuted = replace(g, edge_index=inverse[g.edge_index])
    assert_fiber_close(layer(h, g).index(p), layer(h.index(p), permuted), atol=5e-5, rtol=5e-5)


@pytest.mark.parametrize('backend', BACKENDS)
def test_block_empty_isolated_and_cancellation_safe(backend):
    dims = FiberDims(8, 4, 2)
    layer = EffDockInteractionBlock(dims, backend, radial_hidden=24)
    empty = Graph(torch.empty(2, 0, dtype=torch.long), torch.zeros(0), torch.zeros(0, 3), torch.zeros(0, 3))
    assert len(layer(Fiber.zeros(0, dims, torch.zeros(1)), empty).s) == 0
    h = Fiber.zeros(2, dims, torch.zeros(1)); h.s = torch.randn_like(h.s)
    out = layer(h, empty)
    assert out.v.count_nonzero() == out.t.count_nonzero() == 0
    assert torch.isfinite(out.s).all()


@pytest.mark.parametrize('view', ['bb', 'sc', 'aa'])
def test_eff_encoder_rotation(eff_model, protein, view):
    r = random_rotation()
    with torch.no_grad():
        a = eff_model.online.encoder(protein, (view,))[view]
        b = eff_model.online.encoder(protein.rigid_transform(r, torch.tensor([3., -1, 7])), (view,))[view]
    for key in ('nodes', 'atoms', 'global_state'):
        assert_fiber_close(getattr(a, key).rotate(r), getattr(b, key), atol=5e-4, rtol=5e-4)
    torch.testing.assert_close(b.nodes.t, b.nodes.t.transpose(-1, -2), atol=3e-6, rtol=3e-6)
    assert b.nodes.t.diagonal(dim1=-2, dim2=-1).sum(-1).abs().max() < 5e-5


def test_eff_bb_isolation_and_no_inplace_fusion(eff_model, protein):
    xyz = protein.xyz.clone(); xyz[:, 4:] += 300*torch.randn_like(xyz[:, 4:])
    changed = replace(protein, xyz=xyz, seq=(protein.seq+7) % 20)
    with torch.no_grad():
        a = eff_model.online.encoder(protein, ('bb',))['bb']
        b = eff_model.online.encoder(changed, ('bb',))['bb']
        c = eff_model.online.encoder(protein, ('bb', 'aa'))['bb']
    assert_fiber_close(a.nodes, b.nodes, atol=0, rtol=0)
    assert_fiber_close(a.nodes, c.nodes, atol=0, rtol=0)


@pytest.mark.parametrize('task', ['bb_infill', 'sc_infill', 'aa_infill'])
def test_eff_hidden_geometry_cannot_change_predictions(eff_model, protein, task):
    obs = make_observation(protein, task, .3, torch.Generator().manual_seed(6))
    xyz = protein.xyz.clone(); hidden = protein.present & ~obs.atom_visible
    xyz[hidden] = torch.randn_like(xyz[hidden])*1000
    changed = replace(protein, xyz=xyz)
    with torch.no_grad():
        a = eff_model.online.encoder(protein, obs.spec.context, obs.atom_visible, obs.seq_visible)
        b = eff_model.online.encoder(changed, obs.spec.context, obs.atom_visible, obs.seq_visible)
        for name in a:
            assert_fiber_close(a[name].nodes, b[name].nodes, atol=0, rtol=0)
        pa = eff_model.predictor(a, protein.seq_pos, obs.spec.target, protein.seq_pos)
        pb = eff_model.predictor(b, protein.seq_pos, obs.spec.target, protein.seq_pos)
    assert_fiber_close(pa.fiber, pb.fiber, atol=0, rtol=0)


@pytest.mark.parametrize('task', list(TASKS))
def test_eff_all_tasks_backward(eff_model, protein, task):
    eff_model.train()
    obs = make_observation(protein, task, .3, torch.Generator().manual_seed(8))
    loss, _ = eff_model([(protein, obs)], TrainConfig())
    loss.backward()
    assert torch.isfinite(loss)
    grads = [p.grad for p in eff_model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert all(p.grad is None for p in eff_model.teacher.parameters())
    if eff_model.cfg.backend == 'cueq-naive' and set(obs.spec.context) & {'bb', 'sc', 'aa'}:
        tp_grads = [p.grad for n, p in eff_model.named_parameters()
                    if '.message.tp.' in n and p.grad is not None]
        assert tp_grads and any(g.abs().sum() > 0 for g in tp_grads)


def test_eff_checkpoint_exact_resume(tiny_cfg, tmp_path):
    from protein_jepa.data.dataset import SyntheticDataset
    from protein_jepa.train import train
    from protein_jepa.checkpoint import load_checkpoint
    cfg = replace(tiny_cfg, geometry_block='effdock', eff_radial_hidden=24, dropout=.1)
    tc = TrainConfig(steps=3, batch_size=1, crop_lengths=[10], threads=1)
    data = SyntheticDataset(3, 12, 5)
    train(cfg, tc, data, tmp_path/'full')
    train(cfg, tc, data, tmp_path/'resume', stop_after=1)
    train(cfg, tc, data, tmp_path/'resume', resume=tmp_path/'resume'/'last.pt')
    a = load_checkpoint(tmp_path/'full'/'last.pt')['model']
    b = load_checkpoint(tmp_path/'resume'/'last.pt')['model']
    for name in a:
        torch.testing.assert_close(a[name], b[name], atol=0, rtol=0)


def test_legacy_configuration_default_and_checkpoint_guard(tiny_cfg):
    assert ModelConfig().geometry_block == 'legacy'
    old = ProteinJEPA(tiny_cfg)
    new = ProteinJEPA(replace(tiny_cfg, geometry_block='effdock', eff_radial_hidden=24))
    with pytest.raises(RuntimeError):
        new.load_state_dict(old.state_dict())


@pytest.mark.parametrize('settings', [dict(geometry_block='bogus'), dict(eff_aggregation='bad'),
    dict(eff_expansion=0), dict(eff_edge_dim=0), dict(eff_residual_scale=0.)])
def test_eff_invalid_config(settings):
    with pytest.raises(ValueError):
        ModelConfig(**settings)


@pytest.mark.cuda
@pytest.mark.skipif(not CUEQ or not torch.cuda.is_available(), reason='Actual CuEq CUDA ops/GPU required')
def test_eff_cueq_cuda_forward_backward_rotation(tiny_cfg, protein):
    cfg = replace(tiny_cfg, backend='cueq-cuda', geometry_block='effdock', eff_radial_hidden=24)
    model = ProteinJEPA(cfg).cuda().eval()
    rec = protein.to('cuda')
    rotation = random_rotation().cuda()
    with torch.no_grad():
        a = model.online.encoder(rec, ('aa',))['aa']
        b = model.online.encoder(rec.rigid_transform(rotation, torch.zeros(3, device='cuda')), ('aa',))['aa']
    assert_fiber_close(a.nodes.rotate(rotation), b.nodes, atol=2e-3, rtol=2e-3)
    obs = make_observation(rec, 'aa_infill', .3, torch.Generator().manual_seed(1))
    loss, _ = model([(rec, obs)], TrainConfig()); loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_pre_upgrade_legacy_config_can_resume(tiny_cfg, tmp_path):
    from protein_jepa.data.dataset import SyntheticDataset
    from protein_jepa.train import train
    from protein_jepa.checkpoint import load_checkpoint
    tc = TrainConfig(steps=2, batch_size=1, crop_lengths=[8], threads=1)
    ds = SyntheticDataset(2, 10, 7)
    train(tiny_cfg, tc, ds, tmp_path/'old', stop_after=1)
    path = tmp_path/'old'/'last.pt'
    payload = load_checkpoint(path)
    payload['model_config'] = {k: v for k, v in payload['model_config'].items()
                               if k != 'geometry_block' and not k.startswith('eff_')}
    torch.save(payload, path)
    train(tiny_cfg, tc, ds, tmp_path/'old', resume=path)
    assert load_checkpoint(path)['step'] == 2
