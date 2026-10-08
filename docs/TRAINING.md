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

## 5. DDP

```bash
torchrun --standalone --nproc_per_node=4 -m protein_jepa.cli train \
  --config configs/cueq_gpu.yaml --manifest data/manifest.jsonl --output runs/ddp
```

현재 batch_size는 **rank당 protein 수**다. Variable-length records를 list microbatch로 처리하므로 길이에 맞는 padding batch optimization은 아직 없다. 각 rank는 서로 다른 crop/mask를 sample하며 DDP가 gradient를 평균한다.

Regularizer 통계는 rank-local이다. 전역 batch의 variance/covariance를 계산하는 distributed regularizer와 같지 않다. 두 rank의 CPU/Gloo smoke 실행은 검증했다. CUDA/NCCL 및 실제 Slurm submission은 검증하지 않았다.

Slurm 예시:

```bash
MANIFEST=/absolute/path/manifest.jsonl OUTDIR=/absolute/path/runs/jepa \
  sbatch --partition=YOUR_PARTITION --account=YOUR_ACCOUNT scripts/train_slurm.sh
```

기본 스크립트는 1 node, 4 GPU다. 실제 cluster policy에 맞춰 GPU/CPU/memory/time를 수정한다. 자동으로 사용자 cluster에 job을 제출하지 않는다.

## 6. Held-out pretext evaluation

```bash
protein-jepa evaluate --checkpoint runs/reference_real/last.pt \
  --manifest data/manifest.jsonl --split val --max-records 32 \
  --output runs/reference_real/validation.json
```

이는 동일 frozen online/teacher로 계산한 held-out latent prediction loss다. Downstream function/affinity/interface accuracy가 아니다. 모델을 검증 split에 fine-tune하지 않는다.

## 7. 운영상 현재 한계

CPU reference는 correctness baseline이며 high-throughput GPU implementation을 대체하지 않는다. Neighbor discovery는 chunked O(N²) 거리 계산, sequence/internal Transformer는 record별 실행이다. AMP, gradient accumulation/no_sync scheduling, asynchronous sharded input pipeline, torch.compile/CUDA graphs는 이번 release에 없다.

Memory/throughput 측정을 한 뒤 fixed-token packing, accelerated radius graph, activation checkpointing을 확장한다. 검증 없이 128/256 config가 어떤 GPU 메모리에도 들어간다고 가정하지 않는다.

## Dataset immutability

The checkpoint fingerprint hashes the manifest text, not all NPZ bytes. Do not mutate preprocessed files in place between runs. Store them immutably or manage file checksums externally. The split checker validates declared cluster IDs; it does not run sequence clustering.

## Rigid augmentation

`rigid_augmentation` defaults to **false** since v0.4: the structure path, typed latent heads and every loss/regularizer are exactly SO(3)-equivariant or invariant and use only relative geometry, so a shared rigid transform changes loss and gradients only at float precision (measured on 1UBQ over all nine tasks: relative loss difference ≤1.8e-7, gradient ≤3.5e-5, reference CPU and CuEq CUDA). When enabled, it applies a shared proper rotation and translation to the parent crop before teacher/student view construction. `translation_std: 1.0` is in Angstrom. The same saved sampler RNG drives crop, rigid augmentation, and masks, so the exact-resume test covers their random state. No reflection augmentation or independent teacher/student rotation is used.
