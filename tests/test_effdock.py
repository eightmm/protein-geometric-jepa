"""Regression contracts for the EFF-Dock-inspired interaction architecture."""
from dataclasses import replace
import pytest
import torch
from conftest import assert_fiber_close
from protein_jepa.config import ModelConfig, TrainConfig
from protein_jepa.models.fibers import Fiber, FiberDims
from protein_jepa.models.effdock_blocks import (
    EffDockInteractionBlock, FiberDropout, FiberRMSNorm, ConditionalFiberNorm,
    StableNormRescale, NormGate, gated_aggregate, EquivariantFFN, GatedFFN,
)
from protein_jepa.models.jepa import ProteinJEPA
from protein_jepa.geometry.primitives import random_rotation, symmetric_traceless
from protein_jepa.data.graphs import Graph, make_graph, NUM_EDGE_TYPES
from protein_jepa.data.dataset import SyntheticDataset
from protein_jepa.checkpoint import load_checkpoint
from protein_jepa.train import train


def random_fiber(n=5, dims=FiberDims(8, 4, 2)):
    return Fiber(torch.randn(n, dims.scalar), torch.randn(n, dims.vector, 3),
                 symmetric_traceless(torch.randn(n, dims.tensor, 3, 3)))


def empty_graph():
    return Graph(torch.empty(2, 0, dtype=torch.long), torch.empty(0),
                 torch.empty(0, 3), torch.empty(0, 3), torch.empty(0, dtype=torch.long))


@pytest.mark.parametrize('factory', [FiberRMSNorm, NormGate, StableNormRescale,
                                     EquivariantFFN, GatedFFN])
def test_degree_operations_equivariant(factory):
    d = FiberDims(8, 4, 2)
    h, r = random_fiber(dims=d), random_rotation()
    layer = factory(d)
    assert_fiber_close(layer(h).rotate(r), layer(h.rotate(r)))


def test_conditioning_and_dropout_equivariant_in_training():
    d = FiberDims(8, 4, 2)
    h, r = random_fiber(dims=d), random_rotation()
    norm = ConditionalFiberNorm(d, 7)
    torch.nn.init.normal_(norm.modulate.weight, std=.1)
    context = torch.randn(5, 7)
    assert_fiber_close(norm(h, context).rotate(r), norm(h.rotate(r), context))
    dropout = FiberDropout(d, .4).train()
    torch.manual_seed(123); a = dropout(h).rotate(r)
    torch.manual_seed(123); b = dropout(h.rotate(r))
    assert_fiber_close(a, b)


def test_zero_norm_rescale_and_gradient():
    d = FiberDims(8, 4, 2)
    h = Fiber.zeros(3, d, torch.zeros(1))
    h.v.requires_grad_(); h.t.requires_grad_()
    layer = StableNormRescale(d)
    with torch.no_grad():
        layer.v_map.bias.fill_(7); layer.t_map.bias.fill_(7)
    out = layer(h)
    assert out.v.count_nonzero() == out.t.count_nonzero() == 0
    (out.v.sum()+out.t.sum()).backward()
    assert torch.isfinite(h.v.grad).all() and torch.isfinite(h.t.grad).all()


@pytest.mark.parametrize('mode', ['soft', 'gate', 'degree'])
def test_empty_aggregation_and_zero_gate(mode):
    d = FiberDims(8, 4, 2)
    out = gated_aggregate(Fiber.zeros(0, d, torch.zeros(1)), torch.empty(0, d.invariant),
                          torch.empty(0, dtype=torch.long), 3, d, mode)
    assert out.s.count_nonzero() == out.v.count_nonzero() == out.t.count_nonzero() == 0
    h = random_fiber(2, d)
    out = gated_aggregate(h, torch.zeros(2, d.invariant), torch.tensor([0, 0]), 3, d, mode)
    assert out.s.count_nonzero() == out.v.count_nonzero() == out.t.count_nonzero() == 0


def test_soft_aggregation_retains_weak_edge_attenuation():
    d = FiberDims(1, 1, 1)
    h = Fiber(torch.ones(1, 1), torch.ones(1, 1, 3), torch.zeros(1, 1, 3, 3))
    w, dst = torch.full((1, 3), 1e-3), torch.tensor([0])
    soft = gated_aggregate(h, w, dst, 1, d, 'soft').s.item()
    gate = gated_aggregate(h, w, dst, 1, d, 'gate').s.item()
    assert soft < .002 and gate > .99


def test_directed_typed_graph_and_legacy_constructor():
    x = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.]])
    ids = torch.arange(3)
    kinds = torch.tensor([0, 1, 2])
    bonds = torch.tensor([[0, 1], [1, 0]])
    g = make_graph(x, ids, ids, bonds, node_kind=kinds)
    s, t = g.edge_index
    expected = 1+2*(kinds[s]*3+kinds[t])+g.relation[:, 2].long()
    assert torch.equal(g.edge_type, expected)
    assert int(g.edge_type.max()) < NUM_EDGE_TYPES
    assert g.relation[:, 2].sum() == 2
    assert Graph(g.edge_index, g.distance, g.direction, g.relation).edge_type is None
    with pytest.raises(ValueError, match='node_kind'):
        make_graph(x, ids, ids, bonds, node_kind=torch.tensor([0, 1, 3]))


def test_cutoff_zero_edge_equals_absent_edge():
    d = FiberDims(8, 4, 2)
    block = EffDockInteractionBlock(d, cutoff=5).eval()
    # Activate conditioning so the test catches discrete-degree leakage as well.
    torch.nn.init.normal_(block.ffn_norm.modulate.weight, std=.1)
    h = random_fiber(2, d)
    at_cutoff = Graph(torch.tensor([[0], [1]]), torch.tensor([5.]),
                      torch.tensor([[1., 0., 0.]]), torch.tensor([[0., 1/32, 0.]]))
    assert_fiber_close(block(h, at_cutoff), block(h, empty_graph()), atol=1e-7, rtol=1e-7)
    inside = replace(at_cutoff, distance=torch.tensor([4.999]))
    assert_fiber_close(block(h, inside), block(h, empty_graph()), atol=2e-5, rtol=2e-5)
    bond = replace(at_cutoff, relation=torch.tensor([[0., 1/32, 1.]]))
    assert not torch.allclose(block(h, bond).s, block(h, empty_graph()).s)


@pytest.mark.parametrize('aggregation', ['soft', 'gate', 'degree'])
@pytest.mark.parametrize('extensions', [{}, dict(directional=True, ffn='bilinear',
                                                adaptive_cutoff=True)])
def test_block_deep_backward_equivariance(aggregation, extensions):
    d = FiberDims(8, 4, 2)
    h, r = random_fiber(6, d), random_rotation()
    x = torch.randn(6, 3)
    ids = torch.arange(6)
    bonds = torch.empty(2, 0, dtype=torch.long)
    g = make_graph(x, ids, ids, bonds, node_kind=torch.full_like(ids, 2))
    if aggregation == 'degree' and extensions:
        extensions = {**extensions, 'adaptive_cutoff': False}  # unsupported pair
    layers = torch.nn.ModuleList(EffDockInteractionBlock(d, aggregation=aggregation,
                                                        radial_hidden=24, **extensions)
                                 for _ in range(8))
    a, b = h, h.rotate(r)
    for layer in layers:
        a = layer(a, g)
        b = layer(b, replace(g, direction=g.direction@r.T))
    assert_fiber_close(a.rotate(r), b, atol=2e-4, rtol=2e-4)
    sum(z.square().mean() for z in (a.s, a.v, a.t)).backward()
    assert all(torch.isfinite(p.grad).all() for p in layers.parameters() if p.grad is not None)


def test_stage_wiring_and_all_ablation_switches(tiny_cfg, protein):
    cfg = replace(tiny_cfg, interaction='effdock', effdock_radial_hidden=24)
    model = ProteinJEPA(cfg)
    enc = model.online.encoder
    stages = [enc.bb.atom_stem.layers[0], enc.sc.stem.layers[0], enc.bb.layers[0],
              enc.aa.atom_layers[0], enc.aa.res_layer]
    assert [x.stage for x in stages] == list(range(5))
    assert all(isinstance(x, EffDockInteractionBlock) for x in stages)
    assert not any(hasattr(x, 'stage_embedding') for x in stages)
    for change in (dict(effdock_conditioning=False, effdock_dual_radial=False,
                        effdock_distance_decay=False, effdock_norm_rescale=False,
                        effdock_smooth_cutoff=False, effdock_expansion=1),
                   dict(effdock_directional=True, effdock_ffn='bilinear',
                        effdock_adaptive_cutoff=True, sc_context='spatial')):
        model = ProteinJEPA(replace(cfg, **change))
        out = model.online.encoder(protein, ('bb', 'aa'))
        for view in out.values():
            assert torch.isfinite(view.nodes.s).all()
    assert isinstance(model.online.encoder.bb.layers[0].ffn, EquivariantFFN)
    assert not hasattr(ProteinJEPA(replace(cfg, effdock_dual_radial=False)).online.encoder.bb.layers[0],
                       'radial_in')


@pytest.mark.parametrize('options', [dict(interaction='typo'), dict(effdock_aggregation='typo'),
    dict(effdock_expansion=0), dict(effdock_radial_hidden=0), dict(effdock_residual_scale=0),
    dict(effdock_conditioning='false'), dict(effdock_ffn='typo'), dict(sc_context='typo'),
    dict(effdock_adaptive_cutoff=1), dict(predictor_layers=0),
    dict(global_latent_types='typo'),dict(encoder_global_transport='typo'),
    dict(sc_shape_features='false'),
    dict(effdock_adaptive_cutoff=True, effdock_aggregation='degree')])
def test_invalid_config(options):
    with pytest.raises(ValueError):
        ModelConfig(**options)


def test_effdock_checkpoint_exact_resume_and_architecture_rejection(tiny_cfg, tmp_path):
    cfg = replace(tiny_cfg, interaction='effdock', effdock_radial_hidden=24, dropout=.1)
    training = TrainConfig(steps=3, batch_size=1, crop_lengths=[10, 12], threads=1,
                           tasks=['aa_infill', 'bb_infill', 'sc_infill'])
    dataset = SyntheticDataset(3, 14, 17)
    train(cfg, training, dataset, tmp_path/'full')
    train(cfg, training, dataset, tmp_path/'resume', stop_after=1)
    train(cfg, training, dataset, tmp_path/'resume', resume=tmp_path/'resume/last.pt')
    a, b = [load_checkpoint(tmp_path/x/'last.pt') for x in ('full', 'resume')]
    for key in a['model']:
        torch.testing.assert_close(a['model'][key], b['model'][key], atol=0, rtol=0)
    with pytest.raises(ValueError, match='model configuration'):
        train(tiny_cfg, training, dataset, tmp_path/'bad', resume=tmp_path/'resume/last.pt')


def test_degree_aggregation_rejects_adaptive_cutoff():
    with pytest.raises(ValueError, match='adaptive_cutoff'):
        EffDockInteractionBlock(FiberDims(8, 4, 2), aggregation='degree', adaptive_cutoff=True)


def test_resume_rejects_unknown_training_keys(tiny_cfg, tmp_path):
    training = TrainConfig(steps=2, batch_size=1, crop_lengths=[10], threads=1)
    dataset = SyntheticDataset(2, 12, 17)
    train(tiny_cfg, training, dataset, tmp_path/'r', stop_after=1)
    path = tmp_path/'r/last.pt'
    payload = load_checkpoint(path)
    payload['train_config']['not_an_option'] = 0.02
    torch.save(payload, path)
    with pytest.raises(ValueError, match='Unknown TrainConfig'):
        train(tiny_cfg, training, dataset, tmp_path/'r', resume=path)


def test_spatial_mask_mode_warns():
    with pytest.warns(UserWarning, match='leak'):
        TrainConfig(mask_mode='spatial')


def test_ablation_writer_accepts_every_shipped_gpu_preset(tmp_path):
    import subprocess, sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    for preset in ('effdock_cueq_gpu.yaml', 'jepa_full_cueq_gpu.yaml'):
        out = tmp_path/preset
        subprocess.run([sys.executable, str(root/'scripts/make_effdock_ablations.py'),
                        '--base', str(root/'configs'/preset), '--output', str(out), '--seeds', '1'],
                       check=True, capture_output=True)
        assert len(list(out.glob('*.yaml'))) == 27


def test_format1_checkpoint_rejected(tiny_cfg, tmp_path):
    training = TrainConfig(steps=1, batch_size=1, crop_lengths=[10], threads=1)
    train(tiny_cfg, training, SyntheticDataset(2, 12, 17), tmp_path/'r')
    payload = load_checkpoint(tmp_path/'r/last.pt')
    payload['format_version'] = 1
    torch.save(payload, tmp_path/'old.pt')
    with pytest.raises(ValueError, match='retrain'):
        load_checkpoint(tmp_path/'old.pt')


def test_resume_with_omitted_effdock_config_keys(tiny_cfg, tmp_path):
    training = TrainConfig(steps=2, batch_size=1, crop_lengths=[10], threads=1)
    dataset = SyntheticDataset(2, 12, 17)
    train(tiny_cfg, training, dataset, tmp_path/'r', stop_after=1)
    path = tmp_path/'r/last.pt'
    payload = load_checkpoint(path)
    payload['model_config'] = {k: v for k, v in payload['model_config'].items()
                               if k != 'interaction' and not k.startswith('effdock_')}
    torch.save(payload, path)
    assert train(tiny_cfg, training, dataset, tmp_path/'r', resume=path)['steps'] == 2


def test_bilinear_ffn_synthesizes_out_of_span_direction():
    """Two orthogonal input directions can produce their cross product."""
    d = FiberDims(4, 2, 1)
    ffn = EquivariantFFN(d)
    h = Fiber.zeros(1, d, torch.zeros(1))
    h.v[0, 0] = torch.tensor([1., 0., 0.]); h.v[0, 1] = torch.tensor([0., 1., 0.])
    out = ffn(h)
    assert out.v[..., 2].abs().max() > 1e-4  # z is outside span{x, y}
    gated = GatedFFN(d)(h)
    assert gated.v[..., 2].abs().max() == 0


@pytest.mark.parametrize('bonded', [False, True])
def test_adaptive_cutoff_makes_topk_swap_continuous(bonded):
    """Two neighbours exchanging the k-th rank change the output by O(delta),
    including when one of them is an explicit bond that survives truncation."""
    d = FiberDims(8, 4, 2)
    torch.manual_seed(5)
    h = random_fiber(4, d)
    bonds = (torch.tensor([[2, 0], [0, 2]]) if bonded else torch.empty(2, 0, dtype=torch.long))
    def run(delta, adaptive):
        x = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 2.-delta, 0.], [0., 0., 2.+delta]])
        ids = torch.arange(4)
        g = make_graph(x, ids, ids, bonds, radius=6.,
                       max_neighbors=2, node_kind=torch.full_like(ids, 2))
        block = EffDockInteractionBlock(d, cutoff=6., radial_hidden=24, adaptive_cutoff=adaptive)
        block.load_state_dict(reference.state_dict())
        return block(h, g)
    reference = EffDockInteractionBlock(d, cutoff=6., radial_hidden=24, adaptive_cutoff=True)
    for name, p in reference.named_parameters():
        if 'scale' in name:
            torch.nn.init.constant_(p, 1.)
    gap = {a: (run(1e-4, a).s-run(-1e-4, a).s).abs().max().item() for a in (True, False)}
    assert gap[True] < 1e-3
    if not bonded:
        assert gap[False] > 1e-3


def test_checkpoint_from_before_new_architecture_options_still_loads(tiny_cfg, tmp_path):
    """A stored config without later-added architecture keys rebuilds the
    architecture it was trained with; resuming across the format change is refused."""
    from protein_jepa.config import model_config
    from protein_jepa.models.jepa import ProteinJEPA
    old_cfg = replace(tiny_cfg, pair_frame_features=False, sc_local_frame=False, sc_shape_features=False,
                      global_latent_types='legacy')
    training = TrainConfig(steps=2, batch_size=1, crop_lengths=[10], threads=1)
    dataset = SyntheticDataset(2, 12, 17)
    train(old_cfg, training, dataset, tmp_path/'r', stop_after=1)
    path = tmp_path/'r/last.pt'
    payload = load_checkpoint(path)
    reference=ProteinJEPA(old_cfg).eval()
    reference.load_state_dict(payload['model'])
    payload['model_config'] = {k: v for k, v in payload['model_config'].items()
                               if k not in ('pair_frame_features', 'sc_local_frame', 'sc_shape_features',
                                            'global_latent_types', 'encoder_global_transport')}
    payload['format_version'] = 5
    torch.save(payload, path)
    stored = load_checkpoint(path)
    rebuilt=ProteinJEPA(model_config(stored['model_config']))
    rebuilt.load_state_dict(stored['model'])
    rebuilt.eval()
    assert rebuilt.cfg.global_latent_types=='legacy' and rebuilt.cfg.encoder_global_transport=='none'
    from protein_jepa.objectives.tasks import make_observation
    protein=dataset[0]
    obs=make_observation(protein,'bb_infill',.3,torch.Generator().manual_seed(2))
    with torch.no_grad():
        original=reference.context_latents(protein,obs.spec.context,obs.atom_visible,obs.seq_visible)
        restored=rebuilt.context_latents(protein,obs.spec.context,obs.atom_visible,obs.seq_visible)
        assert original['bb'].global_state.circ.shape[1]==old_cfg.circular_channels
        assert_fiber_close(original['bb'].global_state,restored['bb'].global_state,atol=0,rtol=0)
        query=torch.where(obs.target_residues)[0]
        a=reference.predictor(original,protein.seq_pos,'bb',query)
        b=rebuilt.predictor(restored,protein.seq_pos,'bb',query)
    assert_fiber_close(a.nodes,b.nodes,atol=0,rtol=0)
    assert_fiber_close(a.global_state,b.global_state,atol=0,rtol=0)
    with pytest.raises(ValueError, match='cannot resume'):
        train(old_cfg, training, dataset, tmp_path/'r', resume=path)


@pytest.mark.parametrize('mode',['mean','learned'])
def test_global_transport_equivariance_isolation_and_zero_gate(mode):
    from protein_jepa.models.fibers import GlobalTransport
    d=FiberDims(8,4,2)
    h=random_fiber(7,d);r=random_rotation();batch=torch.tensor([0,0,0,1,1,1,1])
    layer=GlobalTransport(d,mode)
    out=layer(h,batch,2)
    assert_fiber_close(out.rotate(r),layer(h.rotate(r),batch,2))
    changed=Fiber(h.s.clone(),h.v.clone(),h.t.clone())
    changed.s[3:]+=10;changed.v[3:]*=20;changed.t[3:]*=20
    assert_fiber_close(out.index(slice(0,3)),layer(changed,batch,2).index(slice(0,3)),atol=0,rtol=0)
    assert not torch.equal(out.s,h.s)
    zero=Fiber.zeros(7,d,h.s);zero.s=h.s
    no_geometry=layer(zero,batch,2)
    assert no_geometry.v.count_nonzero()==no_geometry.t.count_nonzero()==0
    sum(x.square().mean() for x in (out.s,out.v,out.t)).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters())
    with torch.no_grad():
        layer.scale.zero_()
    assert_fiber_close(layer(h,batch,2),h,atol=0,rtol=0)


@pytest.mark.parametrize('mode',['mean','learned'])
def test_encoder_global_zero_gate_recovers_existing_model(tiny_cfg,protein,mode):
    from protein_jepa.models.jepa import ProteinJEPA
    from protein_jepa.models.fibers import GlobalTransport
    reference=ProteinJEPA(tiny_cfg).eval()
    extended=ProteinJEPA(replace(tiny_cfg,encoder_global_transport=mode)).eval()
    extended.load_state_dict(reference.state_dict(),strict=False)
    with torch.no_grad():
        for block in extended.modules():
            if isinstance(block,GlobalTransport):
                block.scale.zero_()
        a=reference.online.encoder(protein,('bb','sc','aa'))
        b=extended.online.encoder(protein,('bb','sc','aa'))
    for view in a:
        assert_fiber_close(a[view].nodes,b[view].nodes,atol=0,rtol=0)
        assert_fiber_close(a[view].global_state,b[view].global_state,atol=0,rtol=0)


def test_mean_transport_matches_uniform_attention_with_shared_projection():
    from protein_jepa.models.fibers import GlobalTransport
    d=FiberDims(8,4,2)
    h=random_fiber(7,d);batch=torch.tensor([0,0,0,1,1,1,1])
    mean=GlobalTransport(d,'mean')
    learned=GlobalTransport(d,'learned')
    learned.load_state_dict(mean.state_dict(),strict=False)
    with torch.no_grad():
        for parameter in learned.readout.score.parameters():
            parameter.zero_()
    assert_fiber_close(mean(h,batch,2),learned(h,batch,2))


def test_checkpoint_before_shape_option_restores_sc_aa_and_resumes_without_shape(tiny_cfg,tmp_path):
    from protein_jepa.config import model_config
    cfg=replace(tiny_cfg,sc_shape_features=False)
    training=TrainConfig(steps=2,batch_size=1,crop_lengths=[10],threads=1)
    dataset=SyntheticDataset(2,12,17)
    train(cfg,training,dataset,tmp_path/'r',stop_after=1)
    path=tmp_path/'r/last.pt';payload=load_checkpoint(path)
    reference=ProteinJEPA(cfg).eval();reference.load_state_dict(payload['model'])
    payload['model_config'].pop('sc_shape_features')
    torch.save(payload,path)
    stored=load_checkpoint(path)
    rebuilt=ProteinJEPA(model_config(stored['model_config'])).eval()
    rebuilt.load_state_dict(stored['model'])
    assert not rebuilt.cfg.sc_shape_features
    with torch.no_grad():
        a=reference.online.encoder(dataset[0],('sc','aa'))
        b=rebuilt.online.encoder(dataset[0],('sc','aa'))
    for view in a:
        assert_fiber_close(a[view].nodes,b[view].nodes,atol=0,rtol=0)
        assert_fiber_close(a[view].global_state,b[view].global_state,atol=0,rtol=0)
        assert_fiber_close(a[view].atoms,b[view].atoms,atol=0,rtol=0)
    with pytest.raises(ValueError,match='model configuration'):
        train(tiny_cfg,training,dataset,tmp_path/'bad',resume=path)
    assert train(cfg,training,dataset,tmp_path/'r',resume=path)['steps']==2
