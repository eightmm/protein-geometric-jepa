from dataclasses import replace
import pytest
import torch
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


def test_every_teacher_target_parameter_is_trained_online(model,protein):
    """JEPA invariant: the EMA teacher may only use weights the student trains.

    One optimizer step first wakes zero-initialized gates, so only permanently
    dead target-path weights (e.g. a never-consumed online branch) fail.
    """
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
    result=overfit(cfg,tc,[synthetic_record(20,1)],120,5e-3,120,'cpu',0,log=lambda *_:None)
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


@pytest.mark.parametrize('task',list(TASKS))
def test_packed_group_equals_separate_samples(model,task):
    """Packing records of different lengths must not mix them: per-record
    losses and diagnostics equal the one-record computation."""
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
        for name,(sem,circ) in ref_inputs.items():
            torch.testing.assert_close(inputs[name][0],sem,atol=2e-5,rtol=2e-5)


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


def test_resume_is_exact_with_workers_and_accumulation(tiny_cfg,tmp_path):
    """Samples depend only on the step: a run split by stop/resume and built
    by background workers equals one in-process run, with accumulation on."""
    import json
    cfg=replace(tiny_cfg,dropout=.1)   # dropout RNG must survive worker start-up
    base=TrainConfig(steps=4,batch_size=2,accumulation_steps=2,crop_lengths=[10,12],
                     log_every=1,save_every=2,threads=1,mask_min_span=2)
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
