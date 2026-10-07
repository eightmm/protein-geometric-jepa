from dataclasses import replace
import pytest
import torch
from conftest import assert_fiber_close
from protein_jepa.geometry.primitives import random_rotation, symmetric_traceless, normalize
from protein_jepa.models.fibers import Fiber, FiberDims, GlobalReadout
from protein_jepa.models.equivariant import CartesianMessage, EquivariantBlock
from protein_jepa.models.predictors import LatentProjector
from protein_jepa.data.synthetic import sequence_record
from protein_jepa.objectives.tasks import make_observation


def test_reference_message_equivariance():
    d=FiberDims(8,4,2)
    h=Fiber(torch.randn(9,8),torch.randn(9,4,3),symmetric_traceless(torch.randn(9,2,3,3)))
    u,_=normalize(torch.randn(9,3)); r=random_rotation()
    layer=CartesianMessage(d)
    assert_fiber_close(layer(h,u).rotate(r),layer(h.rotate(r),u@r.T))


@pytest.mark.parametrize('view',['bb','sc','aa'])
def test_structure_encoder_equivariance(model,protein,view):
    r=random_rotation(); t=torch.tensor([2.,-1.,4.])
    with torch.no_grad():
        a=model.online.encoder(protein,(view,))[view]
        b=model.online.encoder(protein.rigid_transform(r,t),(view,))[view]
    assert_fiber_close(a.nodes.rotate(r),b.nodes,atol=5e-5,rtol=5e-5)
    assert_fiber_close(a.global_state.rotate(r),b.global_state,atol=5e-5,rtol=5e-5)
    assert_fiber_close(a.atoms.rotate(r),b.atoms,atol=5e-5,rtol=5e-5)
    assert torch.equal(a.node_valid,b.node_valid)


def test_backbone_isolation_from_sequence_and_sidechain(model,protein):
    x=protein.xyz.clone(); x[:,4:]+=torch.randn_like(x[:,4:])*100
    changed=replace(protein,xyz=x,seq=(protein.seq+7)%20)
    with torch.no_grad():
        a=model.online.encoder(protein,('bb',))['bb']
        b=model.online.encoder(changed,('bb',))['bb']
    assert_fiber_close(a.nodes,b.nodes,atol=0,rtol=0)
    assert_fiber_close(a.global_state,b.global_state,atol=0,rtol=0)


def test_aa_fusion_does_not_mutate_bb(model,protein):
    with torch.no_grad():
        a=model.online.encoder(protein,('bb',))['bb']
        b=model.online.encoder(protein,('bb','aa'))['bb']
    assert_fiber_close(a.nodes,b.nodes,atol=0,rtol=0)


@pytest.mark.parametrize('task',['bb_infill','sc_infill','aa_infill'])
def test_hidden_target_mutation_cannot_change_context_or_prediction(model,protein,task):
    obs=make_observation(protein,task,.3,torch.Generator().manual_seed(2))
    x=protein.xyz.clone(); hidden=~obs.atom_visible & protein.present
    x[hidden]=torch.randn_like(x[hidden])*1000
    changed=replace(protein,xyz=x)
    with torch.no_grad():
        a=model.online.encoder(protein,obs.spec.context,obs.atom_visible,obs.seq_visible)
        b=model.online.encoder(changed,obs.spec.context,obs.atom_visible,obs.seq_visible)
        for v in a:
            assert_fiber_close(a[v].nodes,b[v].nodes,atol=0,rtol=0)
            assert_fiber_close(a[v].global_state,b[v].global_state,atol=0,rtol=0)
        pa=model.predictor(a,protein.seq_pos,obs.spec.target,protein.seq_pos)
        pb=model.predictor(b,protein.seq_pos,obs.spec.target,protein.seq_pos)
    assert_fiber_close(pa.fiber,pb.fiber,atol=0,rtol=0)


def test_equivariant_predictor_without_target_coordinates(model,protein):
    r=random_rotation(); moved=protein.rigid_transform(r,torch.tensor([5.,0.,-1.]))
    obs=make_observation(protein,'aa_infill',.3,torch.Generator().manual_seed(2))
    with torch.no_grad():
        a=model.online.encoder(protein,obs.spec.context,obs.atom_visible,obs.seq_visible)
        b=model.online.encoder(moved,obs.spec.context,obs.atom_visible,obs.seq_visible)
        pa=model.predictor(a,protein.seq_pos,'aa',protein.seq_pos)
        pb=model.predictor(b,protein.seq_pos,'aa',protein.seq_pos)
    assert_fiber_close(pa.fiber.rotate(r),pb.fiber,atol=5e-5,rtol=5e-5)


def test_sequence_only_predictor_has_no_fixed_world_vector(model,protein):
    with torch.no_grad():
        a=model.online.encoder(protein,('seq',))
        z=model.predictor(a,protein.seq_pos,'bb',protein.seq_pos)
    assert torch.count_nonzero(z.fiber.v)==0
    assert torch.count_nonzero(z.fiber.t)==0


def test_empty_structure_view_is_finite(model):
    rec=sequence_record('AGKST')
    with torch.no_grad():
        out=model.online.encoder(rec,('bb','aa','chi'))
    for h in out.values():
        assert not h.node_valid.any()
        for x in (h.nodes.s,h.nodes.v,h.nodes.t,h.global_state.s):
            assert torch.isfinite(x).all()


def test_sequence_only_does_not_invoke_geometry(model,monkeypatch):
    def fail(*args,**kwargs):
        raise AssertionError('Geometry tower should not run.')
    monkeypatch.setattr(model.online.encoder.bb,'forward',fail)
    out=model.encode(sequence_record('MATK'),'sequence')
    assert out['seq'].nodes.s.shape[0]==4


def test_internal_view_has_no_cartesian_graph(model,protein,monkeypatch):
    import protein_jepa.models.encoders as enc
    def fail(*args,**kwargs):
        raise AssertionError('Internal-coordinate encoder attempted a Cartesian graph.')
    monkeypatch.setattr(enc,'make_graph',fail)
    monkeypatch.setattr(enc,'residue_graph',fail)
    out=model.online.encoder(protein,('bb_internal','chi'))
    assert set(out)=={'bb_internal','chi'}


def test_global_readout_zero_geometry_stays_zero(tiny_cfg):
    d=tiny_cfg.dims
    h=Fiber.zeros(4,d,torch.zeros(1)); h.s=torch.randn(4,d.scalar)
    out=GlobalReadout(d)(h,torch.ones(4,dtype=torch.bool))
    assert out.v.count_nonzero()==0 and out.t.count_nonzero()==0


def test_learned_circles_unit_norm(tiny_cfg):
    h=Fiber.zeros(5,tiny_cfg.dims,torch.zeros(1)); h.s=torch.randn(5,tiny_cfg.scalar)
    z=LatentProjector(tiny_cfg)(h)
    torch.testing.assert_close(z.circular.norm(dim=-1),torch.ones(5,tiny_cfg.circular_channels))


def test_tensor_channels_symmetric_traceless(model,protein):
    out=model.online.encoder(protein,('bb','aa'))
    for view in out.values():
        for h in (view.nodes,view.global_state,view.atoms):
            torch.testing.assert_close(h.t,h.t.transpose(-1,-2),atol=1e-6,rtol=1e-6)
            trace=h.t.diagonal(dim1=-2,dim2=-1).sum(-1)
            torch.testing.assert_close(trace,torch.zeros_like(trace),atol=2e-6,rtol=0)


def test_unknown_backend_rejected(tiny_cfg):
    with pytest.raises(ValueError):
        EquivariantBlock(tiny_cfg.dims,'pretend-cueq')
