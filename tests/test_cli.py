"""File-format/CLI integration tests using explicit synthetic fixtures, not a benchmark."""
from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import pytest
import torch
import yaml
from protein_jepa.cli import main
from protein_jepa.config import TrainConfig, ModelConfig
from protein_jepa.data.constants import AA3, AA1, ATOMS
from protein_jepa.data.records import ProteinRecord
from protein_jepa.data.io import read_structure, from_atoms
from protein_jepa.data.synthetic import synthetic_record
from protein_jepa.evaluate import evaluate


def write_synthetic_pdb(record, path):
    lines=[]; serial=0
    for i in range(len(record)):
        for j,name in enumerate(ATOMS):
            if not record.present[i,j]:
                continue
            serial+=1
            x,y,z=record.xyz[i,j].tolist()
            element=name[0]
            lines.append(f'ATOM  {serial:5d} {name:^4s} {AA3[int(record.seq[i])]:>3s} A{i+1:4d}    '
                         f'{x:8.3f}{y:8.3f}{z:8.3f}{1.:6.2f}{20.:6.2f}          {element:>2s}  ')
    Path(path).write_text('\n'.join(lines)+'\nTER\nEND\n')


@pytest.mark.parametrize('extension',['.pdb','.cif'])
def test_native_parsing_with_canonical_mapping(tmp_path,extension):
    record=synthetic_record(8,42)
    pdb=tmp_path/'fixture.pdb'; write_synthetic_pdb(record,pdb)
    path=pdb
    if extension=='.cif':
        from Bio.PDB import PDBParser, MMCIFIO
        parsed=PDBParser(QUIET=True).get_structure('fixture',str(pdb))
        writer=MMCIFIO(); writer.set_structure(parsed)
        path=tmp_path/'fixture.cif'; writer.save(str(path))
    sequence_map=[{'position':i,'residue_id':f'A:{i+1}:','aa':
                  AA1[int(record.seq[i])]} for i in range(len(record))]
    recovered=read_structure(path,'A',sequence_map)
    assert recovered.position_source=='canonical'
    assert torch.equal(recovered.present,record.present)
    torch.testing.assert_close(recovered.xyz[record.present],record.xyz[record.present],atol=.00051,rtol=0)
    assert torch.equal(recovered.seq,record.seq)


def test_auto_chain_ignores_water_only_chains():
    atoms=[SimpleNamespace(chain_id='A',res_name='ALA',res_num=1,insertion_code='',
                           atom_name='CA',coords=(0.,0.,0.),occupancy=1.,alt_loc=''),
           SimpleNamespace(chain_id='W',res_name='HOH')]
    rec=from_atoms(atoms)
    assert rec.present[0,1] and len(rec)==1


def test_record_rejects_implicit_dtype_and_wrong_suffix(protein,tmp_path):
    with pytest.raises(TypeError,match='float32'):
        replace(protein,xyz=protein.xyz.double())
    with pytest.raises(TypeError,match='int64'):
        replace(protein,seq=protein.seq.float())
    with pytest.raises(ValueError,match='suffix'):
        protein.save(tmp_path/'wrong.out')


def test_cli_prepare_train_encode_and_evaluate(tmp_path,tiny_cfg):
    record=synthetic_record(10,32)
    pdb=tmp_path/'input.pdb'; write_synthetic_pdb(record,pdb)
    output=tmp_path/'prepared.npz'
    main(['prepare',str(pdb),'--chain','A','--output',str(output)])
    assert ProteinRecord.load(output).position_source=='observed_order_unverified'
    with pytest.raises(FileExistsError):
        main(['prepare',str(pdb),'--chain','A','--output',str(output)])
    training=TrainConfig(steps=9,batch_size=1,crop_lengths=[8],threads=1,save_every=9,log_every=9)
    config=tmp_path/'config.yaml'; config.write_text(yaml.safe_dump({'model':asdict(tiny_cfg),'training':asdict(training)}))
    rows=[]
    for i,split in enumerate(('train','train','val')):
        sample=synthetic_record(10,100+i)
        path=tmp_path/f'record-{i}.npz'; sample.save(path)
        rows.append({'path':path.name,'cluster_id':f'SYNTHETIC-{i}','split':split})
    manifest=tmp_path/'manifest.jsonl'; manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    main(['audit-manifest',str(manifest)])
    main(['train','--config',str(config),'--manifest',str(manifest),'--output',str(tmp_path/'run')])
    checkpoint=tmp_path/'run'/'last.pt'
    exported=tmp_path/'features.pt'
    main(['encode','--checkpoint',str(checkpoint),'--record',str(tmp_path/'record-2.npz'),
          '--mode','all_atom','--output',str(exported)])
    features=torch.load(exported,weights_only=True)
    assert set(features['views'])=={'aa'}
    assert len(features['views']['aa']['atom_slot'])>0
    main(['encode','--checkpoint',str(checkpoint),'--sequence','ACDEFGHIK',
          '--mode','sequence','--output',str(tmp_path/'sequence.pt')])
    report=tmp_path/'evaluation.json'
    main(['evaluate','--checkpoint',str(checkpoint),'--manifest',str(manifest),
          '--max-records','1','--output',str(report)])
    result=json.loads(report.read_text())
    assert result['split']=='val' and len(result['metrics'])==9
    with pytest.raises(ValueError,match='positive'):
        evaluate(checkpoint,manifest,tmp_path/'bad.json',max_records=0)


@pytest.mark.parametrize('kwargs',[{'heads':0},{'radius_atom':0},{'dropout':1},{'atom_layers':0}])
def test_invalid_model_config(kwargs):
    with pytest.raises(ValueError):
        ModelConfig(**kwargs)


def test_publisher_dry_run_and_name_validation(capsys):
    script=Path(__file__).resolve().parents[1]/'scripts/publish_github.py'
    spec=importlib.util.spec_from_file_location('publisher',script)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.publish('eightmm/protein-geometric-jepa',script.parents[1],dry_run=True)
    assert 'No commands executed' in capsys.readouterr().out
    with pytest.raises(ValueError):
        module.publish('https://github.com/eightmm/repo',script.parents[1],dry_run=True)
