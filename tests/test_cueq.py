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
def test_cueq_naive_end_to_end(tiny_cfg,protein):
    from dataclasses import replace
    from protein_jepa.models.jepa import ProteinJEPA
    from protein_jepa.config import TrainConfig
    from protein_jepa.objectives.tasks import make_observation
    model=ProteinJEPA(replace(tiny_cfg,backend='cueq-naive'))
    obs=make_observation(protein,'aa_infill',.3,torch.Generator().manual_seed(1))
    loss,_=model([(protein,obs)],TrainConfig()); loss.backward()
    assert torch.isfinite(loss)


@pytest.mark.cuda
@pytest.mark.skipif(not available or not torch.cuda.is_available(),reason='Requires CuEq packages and a CUDA GPU')
def test_cueq_cuda_end_to_end(tiny_cfg,protein):
    from dataclasses import replace
    from protein_jepa.models.jepa import ProteinJEPA
    from protein_jepa.config import TrainConfig
    from protein_jepa.objectives.tasks import make_observation
    model=ProteinJEPA(replace(tiny_cfg,backend='cueq-cuda')).cuda()
    rec=protein.to('cuda')
    obs=make_observation(rec,'aa_infill',.3,torch.Generator().manual_seed(1))
    loss,_=model([(rec,obs)],TrainConfig()); loss.backward()
    assert torch.isfinite(loss)
