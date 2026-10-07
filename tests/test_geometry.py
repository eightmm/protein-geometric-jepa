from dataclasses import replace
import pytest
import torch
from protein_jepa.geometry.primitives import normalize, angle, dihedral, local_frame, random_rotation
from protein_jepa.geometry.features import backbone_features, chi_features
from protein_jepa.data.constants import AA_TO_ID, ATOM_ID, CHI
from protein_jepa.data.synthetic import synthetic_record


def test_origin_is_present(protein):
    assert torch.equal(protein.xyz[0,0], torch.zeros(3))
    assert protein.present[0,0]
    assert backbone_features(protein).frame_valid[0]


def test_frame_columns_are_orthonormal(protein):
    feat = backbone_features(protein)
    r = feat.frames[feat.frame_valid]
    torch.testing.assert_close(r.transpose(-1,-2)@r, torch.eye(3).expand(len(r),3,3), atol=2e-6,rtol=2e-6)
    torch.testing.assert_close(torch.linalg.det(r), torch.ones(len(r)),atol=2e-6,rtol=2e-6)
    torch.testing.assert_close(feat.directions[:,6:], feat.frames.transpose(-1,-2))


def test_geometry_rotation_translation(protein):
    r = random_rotation()
    moved = protein.rigid_transform(r, torch.tensor([7.,-3.,5.]))
    a,b = backbone_features(protein), backbone_features(moved)
    torch.testing.assert_close(a.internal,b.internal,atol=2e-5,rtol=2e-5)
    torch.testing.assert_close(a.directions@r.T,b.directions,atol=2e-5,rtol=2e-5)
    torch.testing.assert_close(r@a.frames,b.frames,atol=2e-5,rtol=2e-5)
    ca,cb = chi_features(protein),chi_features(moved)
    torch.testing.assert_close(ca.encoded,cb.encoded,atol=2e-5,rtol=2e-5)


@pytest.mark.parametrize('n',[1,2,3,4])
def test_short_chains(n):
    rec=synthetic_record(n)
    feat=backbone_features(rec)
    assert feat.internal.shape==(n,18)
    assert feat.directions.shape==(n,9,3)
    assert torch.isfinite(feat.internal).all()
    if n<4:
        assert not feat.periodic_valid[:,3].any()


def test_break_invalidates_every_dependent_window(protein):
    broken=protein.peptide.clone(); broken[6]=False
    f=backbone_features(replace(protein,peptide=broken))
    assert not f.periodic_valid[7,0]  # phi after break
    assert not f.periodic_valid[6,1]  # psi before break
    assert not f.periodic_valid[7,2]  # omega after break
    assert not f.ca_angle_valid[6:8].any()
    assert not f.periodic_valid[5:8,3].any()  # ALL three four-CA windows
    assert not f.direction_valid[6,1] and not f.direction_valid[7,0]


def test_missing_atom_masks_torsion_and_frame(protein):
    present=protein.present.clone(); present[5,0]=False
    f=backbone_features(replace(protein,present=present))
    assert not f.frame_valid[5]
    assert not f.periodic_valid[5,:3].any()
    assert not f.periodic_valid[4,1]


def test_hidden_atom_dependency_closure(protein):
    visible=protein.present.clone(); visible[5:8]=False
    feat=backbone_features(protein,visible)
    assert not feat.frame_valid[5:8].any()
    assert not feat.periodic_valid[4:9,3].any()
    x=protein.xyz.clone(); x[~visible]=torch.randn_like(x[~visible])*500
    altered=replace(protein,xyz=x)
    other=backbone_features(altered,visible)
    torch.testing.assert_close(feat.internal,other.internal,rtol=0,atol=0)
    torch.testing.assert_close(feat.directions,other.directions,rtol=0,atol=0)


def test_crop_recomputes_boundary(protein):
    full=backbone_features(protein)
    crop=protein.crop(4,7)
    small=backbone_features(crop)
    assert full.periodic_valid[4,0] and not small.periodic_valid[0,0]
    assert full.periodic_valid[9,3] and not small.periodic_valid[-2:,3].any()
    assert not small.direction_valid[0,0] and not small.direction_valid[-1,1]


def test_degenerate_geometry_has_explicit_invalidity():
    z=torch.zeros(3)
    for result,valid in (normalize(z), dihedral(z,z,z,z), angle(z,z,z),local_frame(z,z,z)):
        assert not bool(valid)
        assert torch.isfinite(result).all()


def test_chi_defined_is_not_observed(protein):
    seq=protein.seq.clone(); seq[0]=AA_TO_ID['ARG']
    present=protein.present.clone(); present[0,4:]=False
    feat=chi_features(replace(protein,seq=seq,present=present),include_chi5=True)
    assert feat.defined[0].all()
    assert not feat.valid[0].any()
    assert torch.equal(feat.encoded[0],torch.zeros(5,2))


def test_asp_pi_periodicity():
    rec=synthetic_record(1)
    seq=torch.tensor([AA_TO_ID['ASP']])
    x=torch.zeros_like(rec.xyz); present=torch.zeros_like(rec.present)
    values={'N':[-1,1,1],'CA':[0,1,0],'CB':[0,0,0],'CG':[1,0,0],
            'OD1':[1,1,0],'OD2':[1,-1,0]}
    for name,value in values.items():
        x[0,ATOM_ID[name]]=torch.tensor(value); present[0,ATOM_ID[name]]=True
    a=replace(rec,xyz=x,present=present,seq=seq)
    y=x.clone(); y[0,ATOM_ID['OD1']]=x[0,ATOM_ID['OD2']]; y[0,ATOM_ID['OD2']]=x[0,ATOM_ID['OD1']]
    b=replace(a,xyz=y)
    fa,fb=chi_features(a),chi_features(b)
    assert fa.valid[0,1] and fb.valid[0,1]
    torch.testing.assert_close(fa.encoded,fb.encoded,atol=1e-6,rtol=1e-6)
    assert fa.periodicity[0,1]==2


def test_ile_uses_cg1_cd1():
    assert CHI['ILE'][1]==('CA','CB','CG1','CD1')
