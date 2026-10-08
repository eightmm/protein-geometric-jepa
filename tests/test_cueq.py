"""Optional tests must SKIP, not PASS, without the actual CuEq packages/GPU."""
import importlib.util
import pytest
import torch
from conftest import assert_fiber_close
from protein_jepa.models.fibers import Fiber, FiberDims
from protein_jepa.geometry.primitives import random_rotation, normalize, symmetric_traceless

available=bool(importlib.util.find_spec('cuequivariance')) and bool(importlib.util.find_spec('cuequivariance_torch'))


@pytest.mark.cueq
@pytest.mark.skipif(not available,reason='CuEq packages are not installed')
def test_cueq_bridge_and_cpu_equivariance():
    from protein_jepa.models.cueq_backend import CuEqMessage
    d=FiberDims(8,4,2); layer=CuEqMessage(d,'cueq-naive')
    h=Fiber(torch.randn(6,8),torch.randn(6,4,3),symmetric_traceless(torch.randn(6,2,3,3)))
    assert_fiber_close(h,layer.unpack(layer.pack(h)))
    u,_=normalize(torch.randn(6,3)); r=random_rotation()
    assert_fiber_close(layer(h,u).rotate(r),layer(h.rotate(r),u@r.T),atol=2e-4,rtol=2e-4)
    loss=layer(h,u).s.square().mean(); loss.backward()
    assert any(p.grad is not None for p in layer.parameters())


@pytest.mark.cueq
@pytest.mark.skipif(not available,reason='CuEq packages are not installed')
@pytest.mark.parametrize('interaction', ['baseline', 'effdock', 'effdock-full'])
def test_cueq_naive_end_to_end(tiny_cfg,protein,interaction):
    from dataclasses import replace
    from protein_jepa.models.jepa import ProteinJEPA
    from protein_jepa.config import TrainConfig
    from protein_jepa.objectives.tasks import make_observation
    extra=dict(effdock_ffn='bilinear',effdock_directional=True,effdock_adaptive_cutoff=True,
               sc_context='spatial') if interaction=='effdock-full' else {}
    model=ProteinJEPA(replace(tiny_cfg,backend='cueq-naive',interaction=interaction.split('-')[0],
                              effdock_radial_hidden=24,**extra))
    obs=make_observation(protein,'aa_infill',.3,torch.Generator().manual_seed(1))
    loss,_=model([(protein,obs)],TrainConfig()); loss.backward()
    assert torch.isfinite(loss)


@pytest.mark.cuda
@pytest.mark.skipif(not available or not torch.cuda.is_available(),reason='Requires CuEq packages and a CUDA GPU')
@pytest.mark.parametrize('interaction', ['baseline', 'effdock'])
def test_cueq_cuda_end_to_end(tiny_cfg,protein,interaction):
    from dataclasses import replace
    from protein_jepa.models.jepa import ProteinJEPA
    from protein_jepa.config import TrainConfig
    from protein_jepa.objectives.tasks import make_observation
    model=ProteinJEPA(replace(tiny_cfg,backend='cueq-cuda',interaction=interaction,effdock_radial_hidden=24)).cuda()
    rec=protein.to('cuda')
    obs=make_observation(rec,'aa_infill',.3,torch.Generator().manual_seed(1))
    loss,_=model([(rec,obs)],TrainConfig()); loss.backward()
    assert torch.isfinite(loss)


@pytest.mark.cueq
@pytest.mark.skipif(not available, reason='CuEq packages are not installed')
@pytest.mark.parametrize('view', ['bb', 'sc', 'aa'])
def test_effdock_cueq_encoder_and_global_equivariance(tiny_cfg, protein, view):
    from dataclasses import replace
    from protein_jepa.models.jepa import ProteinJEPA
    model = ProteinJEPA(replace(tiny_cfg, backend='cueq-naive', interaction='effdock',
                                effdock_radial_hidden=24)).eval()
    r = random_rotation()
    with torch.no_grad():
        a = model.online.encoder(protein, (view,))[view]
        b = model.online.encoder(protein.rigid_transform(r, torch.ones(3)), (view,))[view]
    for x, y in [(a.nodes, b.nodes), (a.atoms, b.atoms), (a.global_state, b.global_state)]:
        assert_fiber_close(x.rotate(r), y, atol=3e-4, rtol=3e-4)


@pytest.mark.cueq
@pytest.mark.skipif(not available, reason='CuEq packages are not installed')
@pytest.mark.parametrize('transport',['none','mean','learned'])
def test_effdock_cueq_all_tasks_gradients_and_mask_isolation(tiny_cfg, protein, transport):
    from dataclasses import replace
    from protein_jepa.models.jepa import ProteinJEPA
    from protein_jepa.config import TrainConfig
    from protein_jepa.objectives.tasks import make_observation, TASKS
    from protein_jepa.models.predictors import topology_atoms
    model = ProteinJEPA(replace(tiny_cfg, backend='cueq-naive', interaction='effdock',
                                effdock_radial_hidden=24,encoder_global_transport=transport))
    for task in TASKS:
        model.zero_grad(set_to_none=True)
        obs = make_observation(protein, task, .3, torch.Generator().manual_seed(2))
        loss, _ = model([(protein, obs)], TrainConfig())
        loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.eval()
    for task in ('bb_infill', 'sc_infill', 'aa_infill'):
        obs = make_observation(protein, task, .3, torch.Generator().manual_seed(2))
        x = protein.xyz.clone()
        hidden = ~obs.atom_visible & protein.present
        x[hidden] = torch.randn_like(x[hidden])*1000
        with torch.no_grad():
            a = model.online.encoder(protein, obs.spec.context, obs.atom_visible, obs.seq_visible)
            b = model.online.encoder(replace(protein, xyz=x), obs.spec.context,
                                     obs.atom_visible, obs.seq_visible)
            query = torch.where(obs.target_residues)[0]
            atoms = (topology_atoms(protein.seq, obs.seq_visible, query, task == 'sc_infill')
                     if obs.spec.atom_loss else (None, None))
            pa = model.predictor(model.online.context(a), protein.seq_pos, obs.spec.target, query, *atoms)
            pb = model.predictor(model.online.context(b), protein.seq_pos, obs.spec.target, query, *atoms)
        pairs = [(pa.nodes, pb.nodes), (pa.global_state, pb.global_state)]
        if obs.spec.atom_loss:
            pairs.append((pa.atoms, pb.atoms))
        for x, y in pairs:
            assert_fiber_close(x, y, atol=0, rtol=0)
