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
