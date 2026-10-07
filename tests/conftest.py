import pytest
import torch
from protein_jepa.config import ModelConfig
from protein_jepa.data.synthetic import synthetic_record
from protein_jepa.models.jepa import ProteinJEPA


@pytest.fixture(autouse=True)
def threads_and_seed():
    torch.set_num_threads(1)
    torch.manual_seed(23)


@pytest.fixture
def tiny_cfg():
    return ModelConfig(scalar=16, vector=4, tensor=2, sequence_width=32, sequence_layers=1,
                       atom_layers=1, backbone_layers=1, aa_layers=1, internal_layers=1,
                       latent_scalar=16, latent_vector=4, latent_tensor=2, circular_channels=4)


@pytest.fixture
def protein():
    return synthetic_record(14, 11)


@pytest.fixture
def model(tiny_cfg):
    return ProteinJEPA(tiny_cfg).eval()


def assert_fiber_close(a, b, atol=2e-5, rtol=2e-5):
    for key in ('s','v','t'):
        torch.testing.assert_close(getattr(a, key), getattr(b, key), atol=atol, rtol=rtol)
