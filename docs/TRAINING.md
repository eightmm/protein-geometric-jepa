# 학습·평가·운영

## 1. Smoke → real corpus 순서

```bash
python -m pip install -e '.[dev]'
pytest -q
protein-jepa demo --config configs/smoke.yaml --output runs/smoke
```

Demo는 9개 task를 순환하며 online/EMA/predictor를 업데이트한다. Synthetic sidechain은 물리적 ring closure/force-field 검증을 받은 구조가 아니다. Demo loss의 감소를 단백질 표현의 품질로 해석하지 않는다.

실제 데이터 준비 이후:

```bash
protein-jepa audit-manifest data/manifest.jsonl
protein-jepa train --config configs/reference.yaml \
  --manifest data/manifest.jsonl --output runs/reference_real
```

Unknown configuration keys는 조용히 무시하지 않고 오류를 낸다. Crop lengths, task schedule, loss weight, backend는 실행 폴더의 `resolved_config.json`에 기록된다.

## 2. 한 step의 처리

Single-chain record를 uniform하게 sample하고, 그 chain 안에서 parent crop을 선택한다. crop 위치는 CA 관측 비율이 `crop_min_observed` 이상인 위치 중에서 고른다. Sample마다 task를 round-robin으로 배정하고(한 step 안에 task가 섞인다), multi-block context observation mask를 만든다. Mask 이후 기하량과 graph를 구성한다. Teacher는 동일 parent crop의 target view 전체를 읽는다. Predictor는 visible representation으로 node/atom/global latent를 예측한다. Valid targets로 loss를 task 안에서 평균한 뒤 task 사이에서 평균하고, 정규화 전 온라인 context scalar에 variance floor를 건다. Optimizer와 scheduler 업데이트 후 `ema`→`ema_end` schedule의 momentum으로 teacher를 갱신한다.

Task schedule은 기본적으로 순환형이다. 모든 view가 context에도 등장하여 학습 gradient를 받도록 만든다. `tasks`를 줄이면 특정 tower가 학습되지 않을 수 있으므로 configuration과 ablation을 함께 기록한다.

## 3. Loss 기록

`metrics.jsonl`에는 step/tasks/loss/gradient norm/LR/EMA momentum 및 sample별 node/global/atom loss, valid target count, `target_low_rms`, view별 scalar std와 `effective_rank`(RankMe)가 저장된다. 서로 다른 task의 loss scale은 다르므로 task가 바뀐 전체 loss 숫자만 보고 수렴을 판단하지 않는다.

목표가 충분히 예측 가능한지도 확인한다. 특히 learned latent가 상수에 가까워지면 loss가 작아도 유용한 representation이 아닐 수 있다. Frozen probes, retrieval, residue/atom downstream을 함께 평가해야 한다.

`node_top1`은 masked residue의 예측이 자기 target을 다른 residue의 target보다 가깝게 맞힌 비율입니다(centred cosine). `node_chance`는 우연 수준입니다. `protein-jepa overfit`은 작은 고정 세트에서 이 두 지표와 loss를 함께 보여 줍니다.

## 4. Resume

```bash
protein-jepa demo --config configs/smoke.yaml --output runs/partial --stop-after 9
protein-jepa demo --config configs/smoke.yaml --output runs/partial \
  --resume runs/partial/last.pt
```

`stop-after`는 총 planned steps를 바꾸지 않고 일찍 종료하므로 scheduler trajectory를 보존한다. Resume 시 planned steps, task schedule, seed, dataset, world size와 backend를 바꾸지 않는다. Log/save frequency와 device/threads 정도만 허용된다. Device를 바꾼 실행의 수치적 bitwise equality는 보장하지 않는다.

Checkpoint는 `weights_only=True`로 읽을 수 있는 tensor/basic-type dictionary다. Custom Python object를 pickle해서 학습 record를 저장하지 않는다. `last.pt`는 임시 파일에 fsync 후 replace한다.

## 5. 실행 환경: 단일 GPU, 다중 GPU, 다중 node

```bash
# 단일 GPU
protein-jepa train --config configs/effdock_cueq_gpu.yaml --manifest data/manifest.jsonl --output runs/one
# 한 node 4 GPU
torchrun --standalone --nproc_per_node=4 -m protein_jepa.cli train \
  --config configs/effdock_cueq_gpu.yaml --manifest data/manifest.jsonl --output runs/ddp
# Slurm (N node × NPROC GPU): node마다 torchrun 하나, 첫 node에서 rendezvous.
# 사이트의 account/partition 옵션은 sbatch에 직접 덧붙인다.
MANIFEST=/abs/manifest.jsonl OUTDIR=/abs/runs/jepa CONFIG=configs/effdock_cueq_gpu.yaml \
  sbatch --nodes=2 scripts/train_slurm.sh
```

| 옵션 (`training:`) | 기본값 | 의미 |
|---|---|---|
| `batch_size` | 2 | rank당 microbatch의 protein 수 |
| `accumulation_steps` | 1 | optimizer step 하나 = microbatch K개. DDP gradient all-reduce는 마지막 microbatch에서 한 번만 한다(`no_sync`) |
| `pack_size` | 0 | packed group당 최대 sample 수. 0은 같은 task 전부. 작은 GPU에서 peak memory를 제한하며 sample별 loss는 바뀌지 않는다 |
| `loader_workers` | 0 | crop·mask를 미리 만드는 background process 수(spawn). 0은 main process에서 만든다. `train()`을 직접 부르는 script는 `if __name__ == "__main__":` guard가 필요하다 |
| `dist_backend` | auto | auto는 CUDA면 nccl, CPU면 gloo. gloo + CUDA는 GPU 하나를 여러 rank가 나눠 쓰는 시험용 |

최적화와 checkpoint 옵션:

| 옵션 (`training:`) | 기본값 | 의미 |
|---|---|---|
| `optimizer` | muon | `adamw`는 전부 AdamW. `muon`: hidden layer의 `Linear` weight와 Transformer QKV 행렬은 Muon, 나머지(embedding, direction seed, CuEq tensor product weight, 상대 위치 bias, 1D 파라미터, latent head)는 AdamW. `match_rms_adamw` 보정으로 learning rate 하나를 같이 쓴다. 22개 단백질 비교에서 AdamW 대비 정답 찾기 정확도 2.6배([기록](../reports/optimizer/README.md)). PyTorch 2.9 이상 필요 |
| `muon_momentum` | 0.95 | Muon momentum (Nesterov) |
| `decay_exclusions` | true | 1D 파라미터(bias, norm gain, residual scale), embedding, 상대 위치 bias table에는 weight decay를 주지 않는다. false면 전부 decay(이전 동작) |
| `keep_checkpoints` | 0 | `last.pt` 외에 최근 N개 저장본을 `step_<N>.pt`로 남긴다 |

- `--resume auto`: 출력 폴더에 `last.pt`가 있으면 이어서, 없으면 처음부터 학습한다. 재제출(requeue)된 job에 그대로 쓴다.
- `--init-from ckpt`: weight만 불러와 step 0부터 새 optimizer로 학습한다. 모델 설정이 같아야 한다.
- **중단 신호:** SIGTERM이나 SIGUSR1을 받으면 현재 optimizer step을 마치고 `last.pt`를 저장한 뒤 `interrupted: true`로 끝낸다. CLI는 exit code 3을 돌려준다. 여러 rank 중 하나만 신호를 받아도 모든 rank가 같은 step에서 멈춘다. DataLoader worker는 이 신호를 무시하므로, launcher가 process group 전체에 신호를 보내도 학습 프로세스가 step을 마치고 저장할 수 있다. CPU에서는 이어 학습한 결과가 끊김 없이 돌린 것과 bit 단위로 같다(`test_stop_signal_saves_and_resumes_exactly`, `test_group_stop_signal_with_loader_workers`). CUDA에서는 atomic 연산의 비결정성 때문에 1e-5 수준의 차이가 날 수 있다.
- Slurm 스크립트는 종료 5분 전 SIGTERM을 받도록 설정되어 있고, 중단된 경우 스스로 requeue한 뒤 `--resume auto`로 이어 간다. torchrun은 SIGTERM을 worker에 전달하고 일정 시간 뒤 강제 종료하므로, step 하나가 그 시간 안에 끝나야 한다. **실제 Slurm 환경에서는 실행해 보지 않았다.**
- Resume에서는 `optimizer`, `muon_momentum`, `decay_exclusions`를 바꿀 수 없다. checkpoint format 7부터 optimizer와 scheduler 상태를 목록으로 저장한다.

목적함수 비교용 옵션(설계 명세 18.3, 19.2, 19.6, 27절)은 기본값이 현재 방식이다.

| 옵션 (`training:`) | 기본값 | 의미 |
|---|---|---|
| `target_encoder` | ema | `online`은 teacher-free baseline: target을 online stack이 gradient와 함께 만들고, target latent에도 regularizer를 건다 |
| `semantic_distance` | mse | `cosine`은 semantic latent의 cosine 거리 ablation |
| `node_weight` / `global_weight` / `atom_weight` | 1 / 0.1 / 0.2 | latent 예측 항의 가중치 |
| `raw_angle_weight` / `raw_coordinate_weight` | 0 / 0 | raw 기하 재구성 baseline: 질의 residue의 torsion(1−cos)과 visible Cα 중심 기준 Cα 변위(Å/10, 좌표가 있는 context만) |
| `tasks` | 9개 | 선택형 `seq_infill`(서열만의 JEPA)을 더해 sequence-only baseline을 만든다 |

- **Sample은 (seed, rank, step, sample 번호)의 순수 함수다.** sampler 상태를 checkpoint에 두지 않으므로 resume은 step 번호만으로 정확히 이어지고, worker 수가 달라도 같은 sample이 나온다(`test_resume_is_exact_with_workers_and_accumulation`, `test_step_loader_workers_match_in_process`). 현재 checkpoint는 format 7이다. format 3·4·5·6 weight는 당시 설정으로 추론용으로 읽지만 그 학습을 resume하지는 않는다.
- **Batch:** microbatch 안에서 같은 task의 sample들은 하나의 disjoint-union batch로 계산한다. graph edge는 record 안에만 생기고, Transformer와 predictor attention은 record별 padding으로 서로를 보지 않으며, loss와 진단값은 record별로 계산한다. 결과는 sample을 하나씩 계산한 것과 같다(`test_packed_group_equals_separate_samples`). task가 9개이므로 task당 여러 sample이 모이도록 batch를 수십 단위로 잡아야 처리량이 오른다([측정](../reports/batching/README.md)).
- **Accumulation:** metrics의 `samples`에는 모든 microbatch의 sample이 들어가고, `regularization`은 microbatch별 목록이 된다. task별 loss 평균은 microbatch 안에서 하므로, `batch_size=B, accumulation_steps=K`는 `batch_size=B×K` 한 번과 수치가 정확히 같지는 않다. 유효 batch는 `world × K × B`이고 learning rate는 자동으로 조정하지 않는다.
- **정밀도:** float32 전용이다. TF32는 속도 이득이 없었고 loss의 회전 불변 오차를 3e-7에서 6e-4로 키웠다. bf16 autocast는 지원하지 않는다([측정](../reports/training_env/README.md)).
- **Resume에서 바꿔도 되는 옵션:** device, threads, log/save 주기, `loader_workers`, `pack_size`, `dist_backend`, `keep_checkpoints`. `accumulation_steps`는 최적화를 바꾸므로 바꿀 수 없다. world size도 같아야 한다.
- **Regularizer 통계는 rank-local이다.** 전역 batch의 variance/covariance를 계산하는 distributed regularizer와 같지 않다.

검증 범위([기록](../reports/training_env/README.md)):
- CPU gloo 2 rank: accumulation과 loader worker를 켠 4 step 학습을 중간에 멈췄다가 resume한 결과가 연속 학습과 bit 단위로 같다.
- GPU 1장을 gloo 2 rank가 나눠 쓰는 CuEq CUDA 학습: 동작하고 resume된다. 연속 학습과의 weight 차이는 최대 1.4e-5이며, CUDA atomic 연산의 비결정성 때문이다.
- **NCCL 다중 GPU와 다중 node Slurm은 GPU가 한 장인 환경이라 실행하지 못했다.**

## 6. Held-out pretext evaluation

```bash
protein-jepa evaluate --checkpoint runs/reference_real/last.pt \
  --manifest data/manifest.jsonl --split val --max-records 32 \
  --output runs/reference_real/validation.json --controls --audit-content
```

이는 동일 frozen online/teacher로 계산한 held-out latent prediction loss다. Downstream function/affinity/interface accuracy가 아니다. 모델을 검증 split에 fine-tune하지 않는다.

평가에는 task별 node/global/atom loss, centred retrieval·chance·lift, kind별 유효 target 수, 유효 atom target 수와 full observed crop의 view별 global 진단이 포함된다. Global sem의 std/rank와 irreps의 invariant Gram std/rank를 보고하므로 전역 회전만으로 다양성을 부풀리지 않는다. 수치 잡음을 rank로 세지 않도록 평균 std가 feature RMS의 1e-4 이하이면 rank는 0, 표본이 한 개면 null이다. `training_manifest_matches`가 false면 checkpoint의 학습 데이터와의 분리를 이 manifest만으로 검증할 수 없다.

`--controls`는 동일 mask·position·관측 유효성 metadata를 남기고 context latent 내용을 0으로 만드는 frozen intervention이다. 별도로 학습한 position-only baseline이 아니며, 낮은 control 점수만으로 shortcut을 배제하지 않는다. `--gradients`는 task당 첫 pair 하나의 eval-mode gradient norm/cosine을 추가한다. 기존 optimizer·gradient는 변경하지 않으며 실제 DDP/accumulation step의 기여도를 측정한 값은 아니다. Gradient probe는 forward-only 평가보다 비용이 크므로 선택 실행한다.

`--audit-content`는 전체 manifest의 NPZ를 읽어 split 간 동일 서열·동일 content를 검사한다. 큰 corpus에서는 별도 audit을 먼저 실행한다. 선택적 identity screen은 아래처럼 pair 수를 제한하며, cap 때문에 미검사 pair가 남으면 실패/불완전 상태로 보고한다. Global alignment(2/-1, gap open -10, extension -0.5)의 동일한 known residue 수를 두 서열 중 긴 길이로 나눈다. 이는 sequence clustering이나 remote homology 검증을 대신하지 않는다.

```bash
protein-jepa audit-manifest data/manifest.jsonl --content
protein-jepa audit-manifest data/manifest.jsonl --min-identity 0.8 --max-pairs 10000
```

새 학습은 checkpoint format 7과 `global_latent_types: sem_eq`를 쓴다. Format 3/4/5/6은 legacy inference로 읽고 학습 resume은 거부한다. `encoder_global_transport: mean|learned`는 opt-in이며 기본은 `none`이다. `sc_shape_features`는 새 설정에서 기본 true이며 이전 저장 설정에서 키가 없으면 false로 복원한다. 같은 format의 checkpoint도 shape 옵션이 다른 모델로 resume하면 거부한다. Ablation writer는 global legacy/mean/learned 및 `no_sc_shape` 대조군을 포함한 41개 variant를 만들며 학습은 시작하지 않는다.

Feature 계산의 CPU 측정은 `PYTHONPATH=src python scripts/benchmark_geometry.py --output /tmp/geometry-benchmark.json`으로 실행할 수 있다. `--no-sc-shape`로 shape 입력 비용을 분리한다. Synthetic 128/256 residue, 작은 reference 모델, CPU 한 thread의 timing이며 GPU throughput이나 학습 품질 측정이 아니다. 구현·검증 범위는 [SC shape 보고서](../reports/sc_shape/README.md)에 있다.

## 7. 운영상 현재 한계

CPU reference는 correctness baseline이며 high-throughput GPU implementation을 대체하지 않는다. Neighbor discovery는 record별 padded O(N²) 거리 계산이다. Gradient accumulation(`no_sync`), background sample loader, packed batch는 있다(5절). AMP(측정상 정밀도 손실), torch.compile/CUDA graphs는 없다.

Memory/throughput 측정을 한 뒤 fixed-token packing, accelerated radius graph, activation checkpointing을 확장한다. 검증 없이 128/256 config가 어떤 GPU 메모리에도 들어간다고 가정하지 않는다.

## Dataset immutability

The checkpoint fingerprint hashes the manifest text, not all NPZ bytes. Do not mutate preprocessed files in place between runs. Store them immutably or manage file checksums externally. The split checker validates declared cluster IDs; it does not run sequence clustering.

## Rigid augmentation

`rigid_augmentation` defaults to **false**: the structure path, typed latent heads and every loss/regularizer are exactly SO(3)-equivariant or invariant and use only relative geometry, so a shared rigid transform changes loss and gradients only at float precision (measured on 1UBQ over all nine tasks: relative loss difference ≤1.8e-7, gradient ≤3.5e-5, reference CPU and CuEq CUDA). When enabled, it applies a shared proper rotation and translation to the parent crop before teacher/student view construction. `translation_std: 1.0` is in Angstrom. One per-sample generator, derived from (seed, rank, step, sample index), drives crop, rigid augmentation and masks, so resume and loader workers reproduce them exactly. No reflection augmentation or independent teacher/student rotation is used.
