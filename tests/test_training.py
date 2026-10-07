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
    def cycle():
        optimizer.zero_grad(set_to_none=True)
        for task in TASKS:
            obs=make_observation(protein,task,.3,torch.Generator().manual_seed(6))
            loss,_=model([(protein,obs)],TrainConfig()); loss.backward()
        return touched()
    cycle(); optimizer.step()
    trained=cycle()
    optimizer.zero_grad(set_to_none=True)
    for view in ('seq','bb','sc','aa','bb_internal','chi'):
        out=model.online.encoder(protein,(view,))[view]
        parts=[out.nodes.index(out.node_valid),out.global_state]
        if view=='aa':
            parts.append(out.atoms)
        # Random linear probe: a norm-based probe could miss sign-symmetric paths.
        g=torch.Generator().manual_seed(len(view))
        sum((x*torch.randn(x.shape,generator=g)).sum() for h in parts for x in (h.s,h.v,h.t)).backward()
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
    torch.testing.assert_close(a['rng'][0]['sampler'],b['rng'][0]['sampler'])


def test_resume_rejects_dataset_change(tiny_cfg,tmp_path):
    training=TrainConfig(steps=2,batch_size=1,crop_lengths=[8],threads=1)
    train(tiny_cfg,training,SyntheticDataset(2,10,1),tmp_path/'r',stop_after=1)
    with pytest.raises(ValueError,match='same dataset'):
        train(tiny_cfg,training,SyntheticDataset(2,10,2),tmp_path/'r',resume=tmp_path/'r'/'last.pt')
