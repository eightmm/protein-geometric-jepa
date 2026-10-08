from dataclasses import replace
from types import SimpleNamespace
import json
import pytest
import torch
from protein_jepa.data.records import ProteinRecord
from protein_jepa.data.io import from_atoms, from_plmol_parser, read_structure
from protein_jepa.data.dataset import ManifestDataset
from protein_jepa.data.constants import AA3, ATOMS
from protein_jepa.data.graphs import residue_graph


def atoms_of(record, count=2):
    result=[]
    for i in range(min(count,len(record))):
        for a in torch.where(record.present[i])[0].tolist():
            result.append(SimpleNamespace(chain_id='A',res_num=100+i,insertion_code='',
                res_name=AA3[int(record.seq[i])],atom_name=ATOMS[a],coords=record.xyz[i,a].tolist(),
                occupancy=1.0,alt_loc='',element=ATOMS[a][0]))
    return result


def test_npz_safe_roundtrip(protein,tmp_path):
    path=tmp_path/'protein.npz'; protein.save(path); loaded=ProteinRecord.load(path)
    for k in ('xyz','present','seq','seq_pos','peptide'):
        torch.testing.assert_close(getattr(protein,k),getattr(loaded,k))
    assert protein.residue_ids==loaded.residue_ids


def test_canonical_map_inserts_missing_residue(protein):
    atoms=atoms_of(protein)
    from protein_jepa.data.constants import AA1
    mapping=[{'position':0,'residue_id':'A:100:','aa':AA1[int(protein.seq[0])]},
             {'position':1,'residue_id':None,'aa':'G'},
             {'position':2,'residue_id':'A:101:','aa':AA1[int(protein.seq[1])]}]
    rec=from_atoms(atoms,sequence_map=mapping)
    assert len(rec)==3 and not rec.present[1].any()
    assert not rec.peptide.any() and rec.position_source=='canonical'


def test_unmapped_structure_marked_unverified(protein):
    rec=from_atoms(atoms_of(protein))
    assert rec.position_source=='observed_order_unverified'


def test_insertion_code_does_not_merge_residues(protein):
    atoms=atoms_of(protein)
    for atom in atoms:
        if atom.res_num==101:
            atom.res_num=100; atom.insertion_code='A'
    rec=from_atoms(atoms)
    assert len(rec)==2 and rec.residue_ids==('A:100:','A:100:A')


def test_plmol_adapter_contract(protein):
    parser=SimpleNamespace(protein_atoms=atoms_of(protein))
    assert len(from_plmol_parser(parser))==2
    with pytest.raises(TypeError):
        from_plmol_parser(object())


def test_altloc_residue_consistency(protein):
    atoms=atoms_of(protein,1)
    a=atoms[0]
    a.alt_loc='A'; a.occupancy=.4
    alternate=SimpleNamespace(**vars(a)); alternate.alt_loc='B'; alternate.occupancy=.6
    alternate.coords=[10.,20.,30.]
    rec=from_atoms(atoms+[alternate])
    torch.testing.assert_close(rec.xyz[0,0],torch.tensor([10.,20.,30.]))


def test_manifest_cluster_leakage_rejected(tmp_path):
    path=tmp_path/'manifest.jsonl'
    path.write_text('\n'.join(json.dumps(dict(path=f'{s}.npz',split=s,cluster_id='same'))
                              for s in ('train','val')))
    with pytest.raises(ValueError,match='multiple splits'):
        ManifestDataset(path)


def test_manifest_duplicate_path_rejected(tmp_path):
    path=tmp_path/'manifest.jsonl'
    row=dict(path='a.npz',split='train',cluster_id='one')
    path.write_text(json.dumps(row)+'\n'+json.dumps(row))
    with pytest.raises(ValueError,match='Repeated'):
        ManifestDataset(path)


def test_manifest_refuses_unverified_sequence_positions(protein,tmp_path):
    rec=replace(protein,position_source='observed_order_unverified'); rec.save(tmp_path/'p.npz')
    path=tmp_path/'manifest.jsonl'; path.write_text(json.dumps(dict(path='p.npz',split='train',cluster_id='x')))
    data=ManifestDataset(path)
    with pytest.raises(ValueError,match='Canonical'):
        data[0]
    assert len(ManifestDataset(path,allow_observed_order=True)[0])==len(rec)


def test_graph_signed_polymer_edges_and_visibility(protein):
    visible=torch.ones(len(protein),dtype=torch.bool); visible[5]=False
    ids,graph=residue_graph(protein,visible,max_neighbors=1)
    assert 5 not in ids.tolist()
    edges=set(map(tuple,graph.edge_index.T.tolist()))
    assert (0,1) in edges and (1,0) in edges
    i=((graph.edge_index[0]==0)&(graph.edge_index[1]==1)).nonzero()[0,0]
    assert graph.relation[i,1]>0 and graph.relation[i,2]==1


def test_invalid_shape_or_duplicate_ids_rejected(protein):
    with pytest.raises(ValueError):
        replace(protein,xyz=protein.xyz[:,:4])
    with pytest.raises(ValueError):
        replace(protein,residue_ids=tuple('duplicate' for _ in range(len(protein))))


def test_graph_excludes_self_edges_after_rotation(protein):
    from protein_jepa.data.graphs import make_graph, atom_bonds
    from protein_jepa.geometry.primitives import random_rotation
    for rec in [protein,protein.rigid_transform(random_rotation(),torch.tensor([12.,10.,-20.]))]:
        ri,ai=torch.where(rec.present)
        graph=make_graph(rec.xyz[ri,ai],ri,rec.seq_pos,atom_bonds(rec,ri,ai),5.0,24)
        assert not (graph.edge_index[0]==graph.edge_index[1]).any()


def test_canonical_positions_cannot_skip_missing_rows(protein):
    pos=protein.seq_pos.clone();pos[4:]+=2
    with pytest.raises(ValueError,match='enumerate missing'):
        replace(protein,seq_pos=pos)


def test_record_rejects_multiple_chains(protein):
    ids=tuple(f"{'A' if i<5 else 'B'}:{i+1}:" for i in range(len(protein)))
    with pytest.raises(ValueError,match='exactly one chain'):
        replace(protein,residue_ids=ids)


def test_crop_stays_in_observed_part_of_the_chain():
    """Mostly-unobserved windows (missing loops/termini) are redrawn."""
    import torch
    from protein_jepa.data.records import random_crop
    from protein_jepa.data.synthetic import synthetic_record
    rec=synthetic_record(40,3)
    present=rec.present.clone(); present[:28]=False      # only the last 12 rows observed
    rec=replace(rec,present=present,peptide=rec.peptide&present[1:,0]&present[:-1,2])
    def fractions(minimum):
        g=torch.Generator().manual_seed(0)
        return [float(random_crop(rec,[8],g,minimum).present[:,1].float().mean()) for _ in range(200)]
    assert min(fractions(.5))>=.5                        # every crop has >= half observed
    assert min(fractions(0.))<.5                         # the old uniform draw did not
    whole=random_crop(rec,[64],torch.Generator().manual_seed(0),.5)
    assert len(whole)==len(rec)                          # short chains are used whole


def test_packed_graph_ties_do_not_depend_on_other_records():
    """Equidistant neighbours (a straight CA line) must resolve the same way
    alone and packed next to a longer record (different padded width)."""
    from protein_jepa.data.graphs import make_graph
    line=lambda n:torch.stack((torch.arange(n,dtype=torch.float32)*3.0,torch.zeros(n),torch.zeros(n)),-1)
    empty=torch.empty(2,0,dtype=torch.long)
    a,b=line(4),line(9)+100
    alone=make_graph(a,torch.arange(4),torch.arange(4),empty,3.1,1)
    x=torch.cat((a,b)); ids=torch.arange(13)
    packed=make_graph(x,ids,ids,empty,3.1,1,batch=torch.tensor([0]*4+[1]*9))
    mine=packed.edge_index[:,packed.edge_index[1]<4]
    assert torch.equal(mine,alone.edge_index)
    assert bool((packed.edge_index[0]<4).eq(packed.edge_index[1]<4).all())
