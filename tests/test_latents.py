"""Typed latent contracts: each kind on its own space, correct symmetry."""
from dataclasses import replace
import pytest
import torch
from conftest import assert_fiber_close
from protein_jepa.config import ModelConfig, TrainConfig
from protein_jepa.models.fibers import Fiber, FiberDims
from protein_jepa.models.latents import (TypedLatentHead, LatentSpec, gram_schmidt, circular_floor,
                                         torus_mmd, sphere_mmd, make_typed_target)
from protein_jepa.geometry.primitives import random_rotation, symmetric_traceless


def fiber(n=6, d=FiberDims(8, 4, 2)):
    return Fiber(torch.randn(n, d.scalar), torch.randn(n, d.vector, 3),
                 symmetric_traceless(torch.randn(n, d.tensor, 3, 3)))


SPEC = LatentSpec(sem=5, vector=3, tensor=2, circ=4, dir=2, frame=2)


def test_each_kind_has_its_symmetry():
    head = TypedLatentHead(FiberDims(8, 4, 2), SPEC)
    h, r = fiber(), random_rotation()
    a, b = head(h), head(h.rotate(r))
    # sem and circles are invariant; irreps, directions and frames rotate.
    assert_fiber_close(a.rotate(r), b, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(a.sem, b.sem, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(a.circ, b.circ, atol=2e-5, rtol=2e-5)


def test_kinds_lie_on_their_manifolds():
    z = TypedLatentHead(FiberDims(8, 4, 2), SPEC)(fiber())
    torch.testing.assert_close(z.circ.norm(dim=-1), torch.ones(6, 4))
    torch.testing.assert_close(z.dir.norm(dim=-1), torch.ones(6, 2))
    eye = torch.eye(3).expand(6, 2, 3, 3)
    torch.testing.assert_close(z.frame.transpose(-1, -2) @ z.frame, eye, atol=1e-5, rtol=0)
    torch.testing.assert_close(torch.linalg.det(z.frame), torch.ones(6, 2), atol=1e-5, rtol=0)


def test_degenerate_frames_and_directions_are_masked():
    a = torch.tensor([[1., 0, 0], [0, 0, 0], [1, 0, 0]])
    b = torch.tensor([[0., 1, 0], [0, 1, 0], [2, 0, 0]])
    _, valid = gram_schmidt(a, b)
    assert valid.tolist() == [True, False, False]
    head = TypedLatentHead(FiberDims(8, 4, 2), SPEC)
    h = fiber()
    h.v[0] = 0  # no geometry at token 0
    _, masks = make_typed_target(head(h), None)
    assert not bool(masks['dir'][0].any()) and not bool(masks['frame'][0].any())
    assert bool(masks['dir'][1:].any())


def test_tiny_raw_vectors_are_never_valid_targets():
    """Relative strength alone let 1e-8 vectors (not unit after projection) through."""
    head = TypedLatentHead(FiberDims(8, 4, 2), SPEC)
    h = fiber()
    h.v *= 1e-8
    z = head(h)
    _, masks = make_typed_target(z, None)
    assert not bool(masks['dir'].any()) and not bool(masks['frame'].any())


def test_missing_rows_do_not_change_dir_frame_validity():
    head = TypedLatentHead(FiberDims(8, 4, 2), SPEC)
    h = fiber(4)
    h.v[1] *= 0.05  # weak token
    z = head(h)
    _, observed = make_typed_target(z, None)
    zeros = Fiber(torch.zeros(98, 8), torch.zeros(98, 4, 3), torch.zeros(98, 2, 3, 3))
    from protein_jepa.models.fibers import cat_fibers
    z2 = head(cat_fibers([h, zeros]))
    valid = torch.cat((torch.ones(4, dtype=torch.bool), torch.zeros(98, dtype=torch.bool)))
    _, with_missing = make_typed_target(z2, valid)
    for kind in ('dir', 'frame'):
        assert torch.equal(observed[kind], with_missing[kind][:4])


def test_euclidean_ablation_skips_manifold_projection():
    head = TypedLatentHead(FiberDims(8, 4, 2), replace(SPEC, frame=0), typed=False)
    z = head(fiber())
    assert not torch.allclose(z.circ.norm(dim=-1), torch.ones(6, 4))
    with pytest.raises(ValueError):
        TypedLatentHead(FiberDims(8, 4, 2), SPEC, typed=False)
    with pytest.raises(ValueError):
        ModelConfig(latent_typing='euclidean', frame_channels=1)


def test_circular_floor_and_torus_mmd_detect_collapse():
    spread = torch.nn.functional.normalize(torch.randn(256, 3, 2), dim=-1)
    collapsed = torch.tensor([1., 0.]).expand(256, 3, 2).clone()
    assert float(circular_floor(spread)) == 0 and float(circular_floor(collapsed)) > 0
    assert float(torus_mmd(spread)) < 0.1 < float(torus_mmd(collapsed))
    torch.testing.assert_close(torus_mmd(collapsed), torch.tensor(1.), atol=1e-4, rtol=0)
    # Near collapse the floor still pushes the raw (pre-normalization) circles apart.
    raw = (collapsed+0.01*torch.randn_like(collapsed)).requires_grad_()
    circular_floor(torch.nn.functional.normalize(raw, dim=-1)).backward()
    assert raw.grad.abs().sum() > 0


def test_sphere_mmd_detects_collapse():
    spread, collapsed = torch.randn(256, 16), torch.ones(256, 16)
    assert float(sphere_mmd(spread)) < 0.2 < float(sphere_mmd(collapsed))
    # An all-zero collapse is maximal, not "spread out" (it used to score below 0).
    torch.testing.assert_close(sphere_mmd(torch.zeros(64, 16)), torch.tensor(1.), atol=1e-6, rtol=0)
    # High dimension stays finite and ordered; too-small dimensions are rejected.
    assert 0 <= float(sphere_mmd(torch.randn(128, 512))) < float(sphere_mmd(torch.ones(128, 512)))
    with pytest.raises(ValueError):
        sphere_mmd(torch.randn(8, 2))


@pytest.mark.parametrize('options', [dict(semantic_regularizer='sphere_mmd'),
                                     dict(circular_regularizer='torus_mmd')])
def test_geometry_aware_regularizers_train(model, protein, options):
    from protein_jepa.objectives.tasks import make_observation
    model.train()
    obs = make_observation(protein, 'cart_to_internal', .3, torch.Generator().manual_seed(1))
    loss, info = model([(protein, obs)], TrainConfig(**options))
    loss.backward()
    assert torch.isfinite(loss)


def test_typed_latent_api_for_downstream(model, protein):
    out = model.latents(protein, 'multimodal')
    assert set(out) == {'seq', 'bb', 'aa', 'bb_internal', 'chi'}
    assert out['seq'].nodes.circ.shape[1] == 0 and out['chi'].nodes.circ.shape[1] > 0
    assert out['bb'].nodes.v.shape[1] > 0 and out['bb_internal'].nodes.v.shape[1] == 0
