from dataclasses import replace
import pytest
import torch
from conftest import assert_fiber_close
from protein_jepa.config import TrainConfig
from protein_jepa.objectives.tasks import TASKS, make_observation
from protein_jepa.objectives.losses import latent_distance
from protein_jepa.checkpoint import load_checkpoint
from protein_jepa.data.dataset import SyntheticDataset
from protein_jepa.train import train


@pytest.mark.parametrize('task',list(TASKS))
def test_every_task_finite_backward(model,protein,task):
    model.train()
    obs=make_observation(protein,task,.3,torch.Generator().manual_seed(4))
    loss,info=model([(protein,obs)],TrainConfig())
    assert torch.isfinite(loss)
    loss.backward()
    grads=[p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert all(p.grad is None for p in model.teacher.parameters())
    assert any(p.grad is not None and bool(p.grad.abs().sum()) for p in model.predictor.parameters())


def test_all_encoder_paths_receive_gradients_across_tasks(model,protein):
    model.train()
    for task in TASKS:
        obs=make_observation(protein,task,.3,torch.Generator().manual_seed(6))
        loss,_=model([(protein,obs)],TrainConfig()); loss.backward()
    for view in ('seq','bb','sc','aa','bb_internal','chi'):
        module=getattr(model.online.encoder,view)
        assert any(p.grad is not None and bool(p.grad.abs().sum()) for p in module.parameters()),view


@pytest.mark.parametrize('transport',['none','mean','learned'])
def test_every_teacher_target_parameter_is_trained_online(model,protein,transport):
    """JEPA invariant: the EMA teacher may only use weights the student trains.

    One optimizer step first wakes zero-initialized gates, so only permanently
    dead target-path weights (e.g. a never-consumed online branch) fail.
    """
    if transport != 'none':
        from protein_jepa.models.jepa import ProteinJEPA
        model=ProteinJEPA(replace(model.cfg,encoder_global_transport=transport))
    model.train()
    # Adam moves every touched weight by ~lr, so zero-initialized gates wake up.
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=1e-2)
    def touched():
        return {n for n,p in model.online.named_parameters()
                if p.grad is not None and float(p.grad.abs().max())>1e-7}
    # Regularizers OFF: heads must be trained by the prediction loss itself
    # (an earlier projector got gradient only from a regularizer).
    cfg=TrainConfig(variance_weight=0.,covariance_weight=0.,circular_weight=0.)
    def cycle():
        optimizer.zero_grad(set_to_none=True)
        for task in TASKS:
            obs=make_observation(protein,task,.3,torch.Generator().manual_seed(6))
            loss,_=model([(protein,obs)],cfg); loss.backward()
        return touched()
    cycle(); optimizer.step()
    trained=cycle()
    optimizer.zero_grad(set_to_none=True)
    for view in ('seq','bb','sc','aa','bb_internal','chi'):
        out=model.online.encoder(protein,(view,))[view]
        z=model.online.context({view:out})[view]
        tensors=[x for lat in (z.nodes,z.global_state) for x in lat.tensors() if x.is_floating_point()]
        if view=='aa':
            tensors+=[out.atoms.s,out.atoms.v,out.atoms.t]
        # Random linear probe: a norm-based probe could miss sign-symmetric paths.
        g=torch.Generator().manual_seed(len(view))
        sum((x*torch.randn(x.shape,generator=g)).sum() for x in tensors).backward()
    used=touched()
    assert used, 'target probe produced no gradient'
    assert not used-trained, sorted(used-trained)


def test_teacher_ema_exact_and_eval(model):
    online=dict(model.online.named_parameters()); teacher=dict(model.teacher.named_parameters())
    name=next(iter(online)); before=teacher[name].detach().clone()
    with torch.no_grad():
        online[name].add_(1)
    model.update_teacher(.8)
    torch.testing.assert_close(teacher[name],before+.2)
    model.train(); assert not model.teacher.training


def test_zero_valid_loss_is_differentiable_zero(model,protein):
    z=model.online.encoder(protein,('seq',))['seq'].nodes
    loss,info=latent_distance(z,z.detach(),torch.zeros(len(protein),dtype=torch.bool))
    assert float(loss.detach())==0 and info['valid_targets']==0
    loss.backward()


def test_resume_is_exact(tiny_cfg,tmp_path):
    cfg=replace(tiny_cfg,dropout=.1)
    training=TrainConfig(steps=4,batch_size=1,crop_lengths=[10,12],log_every=10,save_every=2,threads=1)
    dataset=SyntheticDataset(4,14,33)
    train(cfg,training,dataset,tmp_path/'full')
    train(cfg,training,dataset,tmp_path/'resumed',stop_after=2)
    train(cfg,training,dataset,tmp_path/'resumed',resume=tmp_path/'resumed'/'last.pt')
    a,b=load_checkpoint(tmp_path/'full'/'last.pt'),load_checkpoint(tmp_path/'resumed'/'last.pt')
    assert a['step']==b['step']==4
    for key in a['model']:
        torch.testing.assert_close(a['model'][key],b['model'][key],atol=0,rtol=0)


def test_resume_rejects_dataset_change(tiny_cfg,tmp_path):
    training=TrainConfig(steps=2,batch_size=1,crop_lengths=[8],threads=1)
    train(tiny_cfg,training,SyntheticDataset(2,10,1),tmp_path/'r',stop_after=1)
    with pytest.raises(ValueError,match='same dataset'):
        train(tiny_cfg,training,SyntheticDataset(2,10,2),tmp_path/'r',resume=tmp_path/'r'/'last.pt')


def test_task_loss_normalizes_global_target_against_crop_nodes(model,protein,monkeypatch):
    """The single-token global target must receive the crop's node RMS reference."""
    import protein_jepa.models.jepa as jepa
    calls=[]
    original=jepa.make_typed_target
    def spy(z,valid,floor=.1,instance=True,reference=None,**kw):
        calls.append((len(z.sem),reference))
        return original(z,valid,floor,instance,reference,**kw)
    monkeypatch.setattr(jepa,'make_typed_target',spy)
    obs=make_observation(protein,'bb_infill',.3,torch.Generator().manual_seed(3))
    model([(protein,obs)],TrainConfig())
    single=[ref for n,ref in calls if n==1]
    assert single and all(ref is not None for ref in single)


def test_overfit_fits_residue_specific_targets(tiny_cfg):
    """Loss must fall AND predictions must retrieve their own residue's target;
    a collapsed (mean) predictor lowers loss but stays at chance."""
    from protein_jepa.data.synthetic import synthetic_record
    from protein_jepa.overfit import overfit
    cfg=replace(tiny_cfg,interaction='effdock',effdock_radial_hidden=24)
    tc=TrainConfig(tasks=['seq_to_bb','bb_infill'],ema=.999,ema_end=1.,mask_min_span=2,
                   variance_weight=0.,covariance_weight=0.,circular_weight=0.)
    # 200 steps: at 120 the margin depended on the exact initialization draw.
    result=overfit(cfg,tc,[synthetic_record(20,1)],200,5e-3,200,'cpu',0,log=lambda *_:None)
    last=result['history'][-1]
    assert result['loss_ratio']<.5
    assert last['node_top1']>2*last['chance']


def test_stochastic_overfit_uses_training_sampling(tiny_cfg):
    """Stochastic mode crops, masks and mixes tasks like `train`; it must run
    and evaluate on fixed crops of the same records."""
    from protein_jepa.data.synthetic import synthetic_record
    from protein_jepa.overfit import overfit, sample_batch
    tc=TrainConfig(crop_lengths=[10,14],mask_min_span=2)
    records=[synthetic_record(18,i) for i in range(3)]
    batch=sample_batch(records,tc,0,4,torch.Generator().manual_seed(0))
    assert len({obs.task_name for _,obs in batch})==4 and all(len(r)<=14 for r,_ in batch)
    result=overfit(tiny_cfg,tc,records,2,1e-3,2,'cpu',0,log=lambda *_:None,stochastic=True,batch_size=2)
    assert result['mode']=='stochastic' and result['pairs']==27 and len(result['history'])==2


def test_overfit_patience_stops_a_plateaued_run(tiny_cfg):
    """With lr 0 nothing improves: the run must stop after `patience` evals."""
    from protein_jepa.data.synthetic import synthetic_record
    from protein_jepa.overfit import overfit
    tc=TrainConfig(tasks=['seq_to_bb'],mask_min_span=2)
    result=overfit(tiny_cfg,tc,[synthetic_record(16,1)],50,0.0,2,'cpu',0,log=lambda *_:None,
                   patience=2,lr_schedule='cosine')
    assert result['converged_at_step']==4 and result['history'][-1]['step']==4


@pytest.mark.parametrize('transport',['none','mean','learned'])
@pytest.mark.parametrize('task',list(TASKS))
def test_packed_group_equals_separate_samples(model,task,transport):
    """Packing records of different lengths must not mix them: per-record
    losses and diagnostics equal the one-record computation."""
    if transport != 'none':
        from protein_jepa.models.jepa import ProteinJEPA
        model=ProteinJEPA(replace(model.cfg,encoder_global_transport=transport)).eval()
    from protein_jepa.data.synthetic import synthetic_record, sequence_record
    # sequence_record has no coordinates: structural contexts are empty and
    # its predictions collapse, which retrieval must score identically.
    records=[synthetic_record(n,s) for n,s in ((14,1),(19,2),(9,3))]+[sequence_record('GGAG')]
    obs=[make_observation(r,task,.3,torch.Generator().manual_seed(i),2,min_span=2)
         for i,r in enumerate(records)]
    cfg=TrainConfig()
    with torch.no_grad():
        packed=model.group_loss(records,obs,cfg)
        refs=[model.task_loss(r,o,cfg) for r,o in zip(records,obs)]
    for (loss,info,inputs),(ref_loss,ref_info,ref_inputs) in zip(packed,refs):
        torch.testing.assert_close(loss,ref_loss,atol=2e-5,rtol=2e-5)
        assert info.keys()==ref_info.keys()
        for key,value in ref_info.items():
            if isinstance(value,float):
                assert abs(info[key]-value)<=2e-5*(1+abs(value)),key
            else:
                assert info[key]==value,key
        assert inputs.keys()==ref_inputs.keys()
        for name,ref in ref_inputs.items():
            for got,want in zip(inputs[name],ref):
                torch.testing.assert_close(got,want,atol=2e-5,rtol=2e-5)


def test_group_without_valid_targets_matches_single(model,protein):
    """A group whose targets are all unobserved must not crash retrieval."""
    present=torch.zeros_like(protein.present)
    empty=replace(protein,present=present)
    obs=make_observation(empty,'bb_infill',.3,torch.Generator().manual_seed(2),min_span=2)
    (loss,info,_),=model.group_loss([empty],[obs],TrainConfig())
    ref_loss,ref_info,_=model.task_loss(empty,obs,TrainConfig())
    assert float(loss.detach())==float(ref_loss.detach())==0 and info['valid_targets']==0 and 'node_top1' not in info


def test_retrieval_of_a_record_ignores_other_records():
    """Collapse detection must use only the record's own tokens (review repro)."""
    from protein_jepa.models.jepa import node_retrieval
    pred=torch.tensor([4,3,2,1.0002,.9998])[:,None]
    target=torch.tensor([1,0,-1,1,-1.])[:,None]
    valid=torch.ones(5,dtype=torch.bool)
    packed,_=node_retrieval(pred,target,valid,torch.tensor([0,0,0,1,1]),2)
    alone,_=node_retrieval(pred[3:],target[3:],valid[3:],torch.zeros(2,dtype=torch.long),1)
    assert float(packed[1])==float(alone[0])


@pytest.mark.parametrize('optimizer',['adamw',pytest.param('muon',marks=pytest.mark.skipif(
    not hasattr(torch.optim,'Muon'),reason='needs torch.optim.Muon'))])
def test_resume_is_exact_with_workers_and_accumulation(tiny_cfg,tmp_path,optimizer):
    """Samples depend only on the step: a run split by stop/resume and built
    by background workers equals one in-process run, with accumulation on
    (and with the Muon + AdamW optimizer set)."""
    import json
    cfg=replace(tiny_cfg,dropout=.1)   # dropout RNG must survive worker start-up
    base=TrainConfig(steps=4,batch_size=2,accumulation_steps=2,crop_lengths=[10,12],
                     log_every=1,save_every=2,threads=1,mask_min_span=2,optimizer=optimizer)
    dataset=SyntheticDataset(4,14,33)
    tiny_cfg=cfg
    train(tiny_cfg,base,dataset,tmp_path/'full')
    rows=[json.loads(x) for x in (tmp_path/'full'/'metrics.jsonl').read_text().splitlines()]
    assert all(len(r['samples'])==len(r['tasks'])==4 and len(r['regularization'])==2 for r in rows)
    workers=replace(base,loader_workers=2)
    train(tiny_cfg,workers,dataset,tmp_path/'resumed',stop_after=2)
    train(tiny_cfg,workers,dataset,tmp_path/'resumed',resume=tmp_path/'resumed'/'last.pt')
    a,b=load_checkpoint(tmp_path/'full'/'last.pt'),load_checkpoint(tmp_path/'resumed'/'last.pt')
    for key in a['model']:
        torch.testing.assert_close(a['model'][key],b['model'][key],atol=0,rtol=0)


def test_step_loader_workers_match_in_process():
    from protein_jepa.data.sampling import step_loader
    cfg=TrainConfig(batch_size=3,crop_lengths=[10],mask_min_span=2)
    dataset=SyntheticDataset(5,16,3)
    inline=list(step_loader(dataset,cfg,1,2,7))
    workers=list(step_loader(dataset,replace(cfg,loader_workers=3),1,2,7))
    assert [s for s,_ in inline]==[s for s,_ in workers]==list(range(2,7))
    for (_,a),(_,b) in zip(inline,workers):
        for (ra,oa),(rb,ob) in zip(a[0],b[0]):
            assert oa.task_name==ob.task_name and torch.equal(ra.xyz,rb.xyz)
            assert torch.equal(oa.target_residues,ob.target_residues)


def test_pack_size_bounds_groups_without_changing_losses(model):
    from protein_jepa.data.sampling import make_microbatch
    cfg=TrainConfig(batch_size=18,crop_lengths=[12],mask_min_span=2)
    batch=make_microbatch(SyntheticDataset(4,14,5),cfg,0,0,0)
    with torch.no_grad():
        whole,_=model(batch,cfg)
        single,_=model(batch,replace(cfg,pack_size=1))
    torch.testing.assert_close(whole,single,atol=2e-5,rtol=2e-5)


@pytest.mark.parametrize('task',['aa_infill','sc_infill'])
def test_symmetric_atom_naming_is_not_a_learnable_difference(model,task):
    """Swapping the coordinates of equivalent atoms (ASP OD1/OD2, a PHE/TYR
    ring flip, ...) leaves residue states and the atom loss unchanged."""
    from protein_jepa.data.synthetic import synthetic_record
    from protein_jepa.data.constants import AA3, ATOM_ID, SYMMETRIC_RENAMES
    record=next(r for r in (synthetic_record(24,s) for s in range(50))
                if {'ASP','PHE'}<={AA3[int(a)] for a in r.seq})
    xyz=record.xyz.clone()
    for i,aa in enumerate(record.seq.tolist()):
        for a,b in SYMMETRIC_RENAMES.get(AA3[aa],{}).items():
            xyz[i,[ATOM_ID[a],ATOM_ID[b]]]=record.xyz[i,[ATOM_ID[b],ATOM_ID[a]]]
    renamed=replace(record,xyz=xyz)
    with torch.no_grad():
        for view in ('sc','aa'):
            a=model.online.encoder(record,(view,))[view].nodes
            b=model.online.encoder(renamed,(view,))[view].nodes
            torch.testing.assert_close(a.s,b.s,atol=2e-5,rtol=2e-5)
        obs=make_observation(record,task,.6,torch.Generator().manual_seed(1),2,min_span=2)
        cfg=TrainConfig()
        (loss,info,_),=model.group_loss([record],[obs],cfg)
        (other,other_info,_),=model.group_loss([renamed],[obs],cfg)
    assert info['atom_loss']>0
    assert abs(info['atom_loss']-other_info['atom_loss'])<=2e-5*(1+info['atom_loss'])
    torch.testing.assert_close(loss,other,atol=2e-5,rtol=2e-5)


def test_gradient_diagnostics_preserve_training_state_and_gradients(model,protein):
    from protein_jepa.evaluate import task_gradient_diagnostics
    model.train()
    parameter=next(p for p in model.parameters() if p.requires_grad)
    parameter.grad=torch.ones_like(parameter)
    before=parameter.grad.clone()
    pairs=[(protein,make_observation(protein,t,.3,torch.Generator().manual_seed(2)))
           for t in ('bb_infill','aa_infill')]
    report=task_gradient_diagnostics(model,pairs,TrainConfig())
    assert model.training and not model.teacher.training
    torch.testing.assert_close(parameter.grad,before,atol=0,rtol=0)
    assert all(p.grad is None for p in model.teacher.parameters())
    assert all(v>0 for v in report['gradient_norm'].values())
    cosine=torch.tensor(report['gradient_cosine'])
    torch.testing.assert_close(cosine.diag(),torch.ones(2),atol=1e-5,rtol=0)


def test_context_removal_control_ignores_sequence_content_but_retains_observation(model,protein):
    changed=replace(protein,seq=(protein.seq+7)%20)
    obs=make_observation(protein,'seq_to_bb',.3,torch.Generator().manual_seed(2))
    predictions=[]
    handle=model.predictor.register_forward_hook(lambda module,args,out: predictions.append(out))
    try:
        with torch.no_grad():
            for control in ('none','position_mask_only'):
                for record in (protein,changed):
                    model.sample_losses([(record,obs)],TrainConfig(),context_control=control)
    finally:
        handle.remove()
    assert not torch.equal(predictions[0].nodes.sem,predictions[1].nodes.sem)
    assert_fiber_close(predictions[2].nodes,predictions[3].nodes,atol=0,rtol=0)
    assert_fiber_close(predictions[2].global_state,predictions[3].global_state,atol=0,rtol=0)
    model.train()
    with pytest.raises(ValueError,match='evaluation-only'):
        model.sample_losses([(protein,obs)],TrainConfig(),context_control='position_mask_only')


def test_teacher_free_routes_gradient_through_targets(model,protein):
    """target_encoder=online: targets come from the online stack with gradient
    (a target-only view still trains), the EMA teacher is untouched, and the
    target latents are regularized."""
    model.train()
    cfg=TrainConfig(target_encoder='online')
    obs=make_observation(protein,'sc_to_chi',.3,torch.Generator().manual_seed(2))
    loss,details=model([(protein,obs)],cfg)
    assert torch.isfinite(loss)
    loss.backward()
    chi=model.online.encoder.chi
    assert any(p.grad is not None and bool(p.grad.abs().sum()) for p in chi.parameters())
    assert all(p.grad is None for p in model.teacher.parameters())
    assert 'chi_target' in details['regularization']
    # Geometric and atom targets: zero-RMS rows must not make gradients NaN,
    # and atom targets carry gradient like the node targets.
    for task in ('chi_to_sc','aa_infill'):
        model.zero_grad(set_to_none=True)
        obs=make_observation(protein,task,.3,torch.Generator().manual_seed(2),min_span=2)
        loss,_,_=model.task_loss(protein,obs,replace(cfg,node_weight=float(task!='aa_infill'),
                                                     global_weight=0.))
        loss.backward()
        grads=[p.grad for p in model.online.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads),task
    import protein_jepa.models.jepa as jepa
    seen=[]
    original=jepa.make_target
    def spy(*a,**k):
        out=original(*a,**k); seen.append(out.s.requires_grad); return out
    jepa.make_target=spy
    try:
        obs=make_observation(protein,'aa_infill',.3,torch.Generator().manual_seed(2),min_span=2)
        model.task_loss(protein,obs,cfg)
    finally:
        jepa.make_target=original
    assert seen==[True]


def test_semantic_cosine_and_mean_readout_options(tiny_cfg,protein):
    from protein_jepa.models.jepa import ProteinJEPA
    from protein_jepa.models.fibers import GlobalReadout, Fiber
    obs=make_observation(protein,'bb_infill',.3,torch.Generator().manual_seed(2),min_span=2)
    model=ProteinJEPA(replace(tiny_cfg,global_readout='mean')).eval()
    with torch.no_grad():
        mse,_,_=model.task_loss(protein,obs,TrainConfig())
        cos,_,_=model.task_loss(protein,obs,TrainConfig(semantic_distance='cosine'))
    assert torch.isfinite(cos) and float(cos)!=float(mse)
    readout=GlobalReadout(tiny_cfg.dims,learned=False)
    h=Fiber(torch.randn(5,16),torch.randn(5,4,3),torch.randn(5,2,3,3))
    valid=torch.ones(5,dtype=torch.bool)
    out=readout(h,valid)
    mean=Fiber(h.s.mean(0,keepdim=True)+readout.scalar_query,h.v.mean(0,keepdim=True),
               h.t.mean(0,keepdim=True))
    torch.testing.assert_close(out.s,readout.mix(mean).s)   # uniform weights
    assert readout.score is None


def test_raw_reconstruction_baseline_is_equivariant_and_trains(model,protein):
    """Raw torsion/CA-offset losses: finite, train their heads, and the loss is
    invariant to a rigid motion of the record (the CA offset rotates with it)."""
    from protein_jepa.geometry.primitives import random_rotation
    cfg=TrainConfig(node_weight=0.,global_weight=0.,atom_weight=0.,raw_angle_weight=1.,
                    raw_coordinate_weight=1.)
    obs=make_observation(protein,'bb_infill',.3,torch.Generator().manual_seed(2),min_span=2)
    loss,info,_=model.task_loss(protein,obs,cfg)
    assert info['raw_angle_loss']>0 and info['raw_coordinate_loss']>0
    loss.backward()
    assert bool(model.predictor.raw_angle_head.weight.grad.abs().sum())
    assert bool(model.predictor.raw_coordinate_head.linear.weight.grad.abs().sum())
    r=random_rotation(torch.Generator().manual_seed(9))
    moved=protein.rigid_transform(r,torch.tensor([2.,-4.,1.]))
    with torch.no_grad():
        a,_,_=model.task_loss(protein,obs,cfg)
        b,_,_=model.task_loss(moved,obs,cfg)
    torch.testing.assert_close(a,b,atol=2e-5,rtol=2e-5)


def test_equivariant_collapse_diagnostics_are_logged(model,protein):
    obs=make_observation(protein,'bb_infill',.3,torch.Generator().manual_seed(2),min_span=2)
    _,details=model([(protein,obs)],TrainConfig())
    reg=details['regularization']['bb']
    assert {'l1_rms','l2_rms','l1_dead_channels','l1_gram_rank'}<=reg.keys()


def test_window_inference_rejects_gaps_and_diagnostics_flag_full_collapse(model):
    from protein_jepa.data.synthetic import synthetic_record
    from protein_jepa.models.jepa import equivariant_diagnostics
    with pytest.raises(ValueError,match='covered'):
        model.encode_windows(synthetic_record(37,5),'backbone',window=8,stride=12)
    zero=equivariant_diagnostics([torch.zeros(6,4,3)],[torch.zeros(6,2,3,3)])
    assert zero['l1_dead_channels']==1.0 and zero['l2_dead_channels']==1.0


def test_raw_reconstruction_ignores_unobserved_coordinates(model,protein):
    """Unobserved coordinates may hold NaN; raw losses (and an angle-only
    setup) must stay finite."""
    xyz=protein.xyz.clone(); present=protein.present.clone()
    present[3,1]=False; xyz[3,1]=float('nan')
    broken=replace(protein,xyz=xyz,present=present)
    obs=make_observation(broken,'bb_infill',.3,torch.Generator().manual_seed(2),min_span=2)
    for weights in ((1.,1.),(1.,0.)):
        cfg=TrainConfig(node_weight=0.,global_weight=0.,atom_weight=0.,raw_angle_weight=weights[0],
                        raw_coordinate_weight=weights[1])
        model.zero_grad(set_to_none=True)
        loss,_,_=model.task_loss(broken,obs,cfg)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_muon_takes_hidden_matrices_and_adamw_the_rest(tiny_cfg):
    from protein_jepa.models.jepa import ProteinJEPA
    from protein_jepa.optim import parameter_groups
    model=ProteinJEPA(replace(tiny_cfg,interaction='effdock',effdock_radial_hidden=24))
    muon,decay,no_decay=parameter_groups(model,True,True)
    names={id(p):n for n,p in model.named_parameters()}
    embeddings={id(m.weight) for m in model.modules() if isinstance(m,torch.nn.Embedding)}
    assert muon and all(p.ndim==2 and min(p.shape)>=2 for p in muon)
    assert not any('.heads.' in names[id(p)] or '_head' in names[id(p)] or id(p) in embeddings
                   or names[id(p)].endswith(('attention.bias','tp.weight','seed.v','seed.t'))
                   for p in muon)
    assert all(p.ndim<2 or id(p) in embeddings or names[id(p)].endswith('attention.bias')
               for p in no_decay)
    trainable={id(p) for p in model.parameters() if p.requires_grad}
    assert {id(p) for p in muon+decay+no_decay}==trainable and len(muon+decay+no_decay)==len(trainable)
    _,all_decay,none=parameter_groups(model,False,False)
    assert not none and len(all_decay)==len(trainable)


def test_auto_resume_keep_checkpoints_and_init_from(tiny_cfg,tmp_path):
    from protein_jepa.checkpoint import load_checkpoint as load
    training=TrainConfig(steps=4,batch_size=1,crop_lengths=[10],threads=1,save_every=1,
                         keep_checkpoints=2,mask_min_span=2)
    dataset=SyntheticDataset(3,12,5)
    assert train(tiny_cfg,training,dataset,tmp_path/'r',resume='auto',stop_after=2)['steps']==2
    assert train(tiny_cfg,training,dataset,tmp_path/'r',resume='auto')['steps']==4
    assert sorted(p.name for p in (tmp_path/'r').glob('step_*.pt'))==['step_00000003.pt','step_00000004.pt']
    train(tiny_cfg,training,dataset,tmp_path/'full')
    a,b=load(tmp_path/'full'/'last.pt'),load(tmp_path/'r'/'last.pt')
    assert all(torch.equal(a['model'][k],b['model'][k]) for k in a['model'])
    fresh=train(tiny_cfg,replace(training,steps=1),dataset,tmp_path/'init',init_from=tmp_path/'r'/'last.pt')
    assert fresh['steps']==1 and load(tmp_path/'init'/'last.pt')['step']==1
    with pytest.raises(ValueError,match='different model'):
        train(replace(tiny_cfg,scalar=8,heads=4),training,dataset,tmp_path/'bad',
              init_from=tmp_path/'r'/'last.pt')


def test_stop_signal_saves_and_resumes_exactly(tiny_cfg,tmp_path):
    """SIGTERM mid-run: the step finishes, a checkpoint is written, the run
    reports interrupted, and resuming reproduces the uninterrupted run."""
    import os, signal
    from protein_jepa.checkpoint import load_checkpoint as load
    training=TrainConfig(steps=5,batch_size=1,crop_lengths=[10],threads=1,save_every=10,
                         mask_min_span=2)
    class SignalOnce(SyntheticDataset):
        calls=0
        def __getitem__(self,i):
            SignalOnce.calls+=1
            if SignalOnce.calls==2:      # during step 1
                os.kill(os.getpid(),signal.SIGTERM)
            return super().__getitem__(i)
    stopped=train(tiny_cfg,training,SignalOnce(3,12,5),tmp_path/'r')
    assert stopped['interrupted'] and stopped['steps']==2 and load(tmp_path/'r'/'last.pt')['step']==2
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL or callable(signal.getsignal(signal.SIGTERM))
    dataset=SyntheticDataset(3,12,5)
    train(tiny_cfg,training,dataset,tmp_path/'r',resume=tmp_path/'r'/'last.pt')
    train(tiny_cfg,training,dataset,tmp_path/'full')
    a,b=load(tmp_path/'full'/'last.pt'),load(tmp_path/'r'/'last.pt')
    assert all(torch.equal(a['model'][k],b['model'][k]) for k in a['model'])


def test_group_stop_signal_with_loader_workers(tmp_path):
    """A launcher signals the whole process group (trainer + loader workers):
    workers must survive so the trainer saves, then resume is exact."""
    import subprocess, sys, textwrap
    from pathlib import Path
    script=tmp_path/'run.py'
    script.write_text(textwrap.dedent('''
        import os, signal, sys, json, torch
        from dataclasses import replace
        from protein_jepa.config import ModelConfig, TrainConfig
        from protein_jepa.data.dataset import SyntheticDataset
        from protein_jepa.train import train
        class GroupSignal(SyntheticDataset):
            def __getitem__(self, i):
                marker = sys.argv[2]+'.sent'
                if sys.argv[3] == 'signal' and not os.path.exists(marker):
                    open(marker, 'w').close()
                    os.killpg(os.getpgid(0), signal.SIGTERM)   # like torchrun/Slurm
                return super().__getitem__(i)
        if __name__ == '__main__':
            cfg = ModelConfig(scalar=16, vector=4, tensor=2, sequence_width=32, sequence_layers=1,
                              atom_layers=1, backbone_layers=1, aa_layers=1, internal_layers=1,
                              predictor_layers=2, latent_scalar=16, latent_vector=4,
                              latent_tensor=2, circular_channels=4)
            tc = TrainConfig(steps=6, batch_size=1, crop_lengths=[10], threads=1, save_every=10,
                             mask_min_span=2, loader_workers=1)
            data = GroupSignal(3, 12, 5)
            result = train(cfg, tc, data, sys.argv[1], resume='auto')
            print(json.dumps({'interrupted': result['interrupted'], 'steps': result['steps']}))
    '''))
    env={**__import__('os').environ,'PYTHONPATH':str(Path(__file__).resolve().parents[1]/'src')}
    def run(out,mode):
        done=subprocess.run([sys.executable,str(script),str(out),str(tmp_path/out.name),mode],
                            capture_output=True,text=True,env=env,start_new_session=True,timeout=600)
        assert done.returncode==0,done.stderr[-2000:]
        return __import__('json').loads(done.stdout.strip().splitlines()[-1])
    first=run(tmp_path/'r','signal')
    assert first['interrupted'] and first['steps']<6
    assert run(tmp_path/'r','signal')['steps']==6        # auto-resume; marker prevents a second signal
    run(tmp_path/'full','none')
    a,b=load_checkpoint(tmp_path/'full'/'last.pt'),load_checkpoint(tmp_path/'r'/'last.pt')
    assert all(torch.equal(a['model'][k],b['model'][k]) for k in a['model'])


def test_mean_readout_loads_states_with_an_unused_score_network(tiny_cfg):
    from protein_jepa.models.fibers import GlobalReadout
    learned=GlobalReadout(tiny_cfg.dims)
    mean=GlobalReadout(tiny_cfg.dims,learned=False)
    mean.load_state_dict(learned.state_dict())        # strict: score.* keys are dropped
    torch.testing.assert_close(mean.scalar_query,learned.scalar_query)


def test_evaluation_counts_raw_only_supervision(model,protein):
    """A record whose latent targets are all missing but whose raw targets
    exist still contributes to the raw-baseline summary."""
    from protein_jepa.evaluate import summarize_pairs
    present=protein.present.clone(); present[:,4:]=False          # no sidechains
    bare=replace(protein,present=present)
    obs=make_observation(bare,'chi_to_sc',.3,torch.Generator().manual_seed(2))
    cfg=TrainConfig(node_weight=0.,global_weight=0.,atom_weight=0.,raw_angle_weight=1.)
    pairs=[(bare,obs)]
    with torch.no_grad():
        results=model.sample_losses(pairs,cfg)
    info=results[0][1]
    assert info['valid_targets']==0 and info['raw_angle_targets']>0
    summary=summarize_pairs(pairs,results)['chi_to_sc']
    assert summary['records']==1 and summary['raw_angle_records']==1
    assert summary['mean_raw_angle_loss'] is not None and summary['mean_pretext_loss'] is not None
