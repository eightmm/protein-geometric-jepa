"""Task-head building blocks, not trained downstream models."""
import torch
from torch import nn
from .models.fibers import Fiber, FiberDims, GlobalReadout


class InvariantNodeHead(nn.Module):
    """Residue or atom regression/classification logits; caller chooses task loss."""
    def __init__(self, dims: FiberDims, outputs: int, hidden: int = 128):
        super().__init__()
        if outputs < 1:
            raise ValueError('outputs must be positive')
        self.mlp = nn.Sequential(nn.Linear(dims.invariant, hidden), nn.SiLU(), nn.Linear(hidden, outputs))

    def forward(self, states: Fiber):
        return self.mlp(states.invariant())


class InvariantProteinHead(nn.Module):
    def __init__(self, dims: FiberDims, outputs: int, hidden: int = 128):
        super().__init__()
        self.pool = GlobalReadout(dims)
        self.head = InvariantNodeHead(dims, outputs, hidden)

    def forward(self, nodes: Fiber, valid: torch.Tensor):
        if not bool(valid.any()):
            raise ValueError('Protein head needs at least one valid node.')
        return self.head(self.pool(nodes, valid))
