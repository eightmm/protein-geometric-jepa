from dataclasses import replace
import pytest
import torch
from conftest import assert_fiber_close
from protein_jepa.geometry.primitives import random_rotation, symmetric_traceless, normalize
from protein_jepa.models.fibers import Fiber, FiberDims, GlobalReadout
from protein_jepa.models.equivariant import CartesianMessage, EquivariantBlock
from protein_jepa.models.predictors import make_target, topology_atoms
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
        query=torch.where(obs.target_residues)[0]
        atoms=topology_atoms(protein.seq,obs.seq_visible,query,task=='sc_infill') if obs.spec.atom_loss else (None,None)
        pa=model.predictor(a,protein.seq_pos,obs.spec.target,query,*atoms)
        pb=model.predictor(b,protein.seq_pos,obs.spec.target,query,*atoms)
    assert_fiber_close(pa.nodes,pb.nodes,atol=0,rtol=0)
    assert_fiber_close(pa.global_state,pb.global_state,atol=0,rtol=0)
    if obs.spec.atom_loss:
        assert_fiber_close(pa.atoms,pb.atoms,atol=0,rtol=0)


@pytest.mark.parametrize('task',['sc_infill','aa_infill'])
def test_hidden_atom_presence_never_shapes_queries_or_predictions(model,protein,task):
    """Deleting a hidden atom from the record must not change any prediction."""
    obs=make_observation(protein,task,.3,torch.Generator().manual_seed(2))
    hidden=~obs.atom_visible & protein.present
    hidden[:,:4]&=task=='aa_infill'
    r,a=torch.where(hidden)
    present=protein.present.clone(); present[r[0],a[0]]=False
    changed=replace(protein,present=present)
    with torch.no_grad():
        outs=[]
        for rec in (protein,changed):
            o=make_observation(rec,task,.3,torch.Generator().manual_seed(2))
            ctx=model.online.encoder(rec,o.spec.context,o.atom_visible,o.seq_visible)
            query=torch.where(o.target_residues)[0]
            atoms=topology_atoms(rec.seq,o.seq_visible,query,task=='sc_infill')
            outs.append((atoms,model.predictor(ctx,rec.seq_pos,o.spec.target,query,*atoms)))
    (qa,pa),(qb,pb)=outs
    assert all(torch.equal(x,y) for x,y in zip(qa,qb))
    for x,y in ((pa.nodes,pb.nodes),(pa.global_state,pb.global_state),(pa.atoms,pb.atoms)):
        assert_fiber_close(x,y,atol=0,rtol=0)


def test_glycine_only_sidechain_infill_has_no_atom_queries(model,protein):
    from protein_jepa.config import TrainConfig
    gly=replace(protein,seq=torch.full_like(protein.seq,7))
    present=gly.present.clone(); present[:,4:]=False; gly=replace(gly,present=present)
    obs=make_observation(gly,'sc_infill',.3,torch.Generator().manual_seed(1))
    model.train()
    loss,info=model([(gly,obs)],TrainConfig())
    loss.backward()
    assert torch.isfinite(loss)


def test_global_target_uses_crop_reference_scale(tiny_cfg):
    from protein_jepa.models.predictors import reference_rms
    d=tiny_cfg.dims
    nodes=Fiber(torch.randn(5,d.scalar),torch.randn(5,d.vector,3),symmetric_traceless(torch.randn(5,d.tensor,3,3)))
    glob=Fiber(torch.randn(1,d.scalar),torch.randn(1,d.vector,3)*1e-5,torch.zeros(1,d.tensor,3,3))
    z=make_target(glob,None,reference=reference_rms(nodes,torch.ones(5,dtype=torch.bool)))
    assert z.v.norm()<1e-2 and z.s[0,-2]<-1


def test_topology_atoms_follow_visible_sequence_only(protein):
    query=torch.tensor([0,1])
    visible=torch.ones(len(protein),dtype=torch.bool); visible[1]=False
    r,a=topology_atoms(protein.seq,visible,query,False)
    assert set(a[r==1].tolist())=={0,1,2,3}
    r,a=topology_atoms(protein.seq,visible,query,True)
    assert not bool((r==1).any()) and bool((a>=4).all())


def test_equivariant_predictor_without_target_coordinates(model,protein):
    r=random_rotation(); moved=protein.rigid_transform(r,torch.tensor([5.,0.,-1.]))
    obs=make_observation(protein,'aa_infill',.3,torch.Generator().manual_seed(2))
    with torch.no_grad():
        a=model.online.encoder(protein,obs.spec.context,obs.atom_visible,obs.seq_visible)
        b=model.online.encoder(moved,obs.spec.context,obs.atom_visible,obs.seq_visible)
        query=torch.arange(len(protein))
        atoms=topology_atoms(protein.seq,obs.seq_visible,query[obs.target_residues],False)
        pa=model.predictor(a,protein.seq_pos,'aa',query,*atoms)
        pb=model.predictor(b,protein.seq_pos,'aa',query,*atoms)
    for x,y in ((pa.nodes,pb.nodes),(pa.global_state,pb.global_state),(pa.atoms,pb.atoms)):
        assert_fiber_close(x.rotate(r),y,atol=5e-5,rtol=5e-5)
    assert pa.nodes.v.abs().sum()>0 and pa.atoms.t.abs().sum()>0


def test_sequence_only_predictor_has_no_fixed_world_vector(model,protein):
    with torch.no_grad():
        a=model.online.encoder(protein,('seq',))
        z=model.predictor(a,protein.seq_pos,'bb',torch.arange(len(protein)))
    for h in (z.nodes,z.global_state):
        assert torch.count_nonzero(h.v)==0
        assert torch.count_nonzero(h.t)==0


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


def test_target_normalization_contract(tiny_cfg):
    d=tiny_cfg.dims
    h=Fiber(torch.randn(6,d.scalar),torch.randn(6,d.vector,3),symmetric_traceless(torch.randn(6,d.tensor,3,3)))
    h.v[0]*=1e-5; h.t[0]*=1e-5
    r=random_rotation()
    z=make_target(h)
    assert_fiber_close(z.rotate(r),make_target(h.rotate(r)),atol=1e-5,rtol=1e-5)
    # Near-zero tokens are damped by the soft floor, not inflated to unit RMS.
    assert z.v[0].norm()<1e-2 and z.v[1:].square().sum(-1).mean()>1
    # Two invariant log-magnitude scalars follow the layer-normed scalars.
    assert z.s.shape[-1]==d.scalar+2 and z.s[0,-2]<z.s[1:,-2].min()
    torch.testing.assert_close(z.s[:,:d.scalar].mean(-1),torch.zeros(6),atol=1e-5,rtol=0)
    zero=Fiber.zeros(3,d,torch.zeros(1)); zero.s=torch.randn(3,d.scalar)
    out=make_target(zero)
    assert out.v.count_nonzero()==0 and out.t.count_nonzero()==0 and torch.isfinite(out.s).all()


def test_sidechain_context_option(tiny_cfg,protein):
    from protein_jepa.models.jepa import ProteinJEPA
    x=protein.xyz.clone(); x[7,4:]+=torch.tensor([.4,-.3,.2])
    changed=replace(protein,xyz=x)
    for context,expect in (('local',False),('spatial',True)):
        m=ProteinJEPA(replace(tiny_cfg,sc_context=context)).eval()
        with torch.no_grad():
            a=m.online.encoder(protein,('sc',))['sc'].nodes.s
            b=m.online.encoder(changed,('sc',))['sc'].nodes.s
        other=torch.cat(((a-b)[:7],(a-b)[8:])).abs().max()
        assert bool(other>0)==expect, context


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
