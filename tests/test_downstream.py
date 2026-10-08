import pytest
import torch
from protein_jepa.downstream import InvariantNodeHead,InvariantProteinHead
from protein_jepa.geometry.primitives import random_rotation


def test_downstream_head_rotation_invariance(model,protein,tiny_cfg):
    with torch.no_grad():
        state=model.encode(protein,'all_atom')['aa']
    r=random_rotation()
    node=InvariantNodeHead(tiny_cfg.dims,3)
    glob=InvariantProteinHead(tiny_cfg.dims,2)
    torch.testing.assert_close(node(state.nodes),node(state.nodes.rotate(r)),atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(glob(state.nodes,state.node_valid),glob(state.nodes.rotate(r),state.node_valid),atol=1e-6,rtol=1e-6)


def test_protein_head_rejects_empty_input(model,protein,tiny_cfg):
    state=model.encode(protein,'backbone')['bb']
    with pytest.raises(ValueError):
        InvariantProteinHead(tiny_cfg.dims,1)(state.nodes,torch.zeros_like(state.node_valid))


@pytest.mark.parametrize('task',['regression','classification'])
def test_frozen_probe_uses_train_statistics_and_disjoint_clusters(task):
    from protein_jepa.downstream import frozen_linear_probe
    x=torch.tensor([[-2.],[-1.],[1.],[2.]],requires_grad=True)
    z=torch.tensor([[10.],[-10.]])
    y=3*x.detach()[:,0]+2 if task=='regression' else (x.detach()[:,0]>0).long()
    truth=3*z[:,0]+2 if task=='regression' else (z[:,0]>0).long()
    kw=dict(train_clusters=['a','a','b','b'],eval_clusters=['c','d'],task=task,ridge=1e-6)
    result=frozen_linear_probe(x,y,z,truth,**kw)
    assert not result['weight'].requires_grad and x.grad is None
    assert float(result['feature_mean'])==0
    if task=='regression':
        assert result['metrics']['mse']<1e-6 and result['metrics']['mean_baseline_mse']>100
    else:
        assert result['metrics']['accuracy']==1 and result['metrics']['majority_accuracy']==.5
    with pytest.raises(ValueError,match='overlap'):
        frozen_linear_probe(x,y,z,truth,**{**kw,'eval_clusters':['a','d']})
