

# JELLY Trainer

Training framework for **JELLY** (Joint Efficient Learning with Layer-Yielding), a custom PEFT adapter with sequential-to-parallel mode switching.

## Features

* **JELLY Adapter**: Novel adapter architecture with mode switching
* Sequential mode: `adapter_input = F.linear(x, W, bias=None)` → adapter sees base layer's output
* Parallel mode: `adapter_input = x` → adapter sees raw input (same as LoRA)
* Smooth transition via **Task-Isomorphic Projection** or **Merge-and-Reinit**

* **Multi-task Support**: NLU (GLUE), ViT image classification, Commonsense reasoning
* **Experiment Tracking**: Automatic Git Hash logging for both Trainer and JELLY library to ensure 100% reproducibility.
* **Comparison Methods**: LoRA, DoRA, PiSSA, AdaLoRA, BitFit

## Recent Updates (2025-02-19)

### 1. Task-Isomorphic Projection (Double SVD) — `layer.py`

`switch_to_parallel_with_correction()` 메서드를 Double SVD 기반으로 재구현:

- **Step 1**: Sequential adapter A의 태스크 순수 스케일(`S_A`) 추출 — `U_A, S_A, Vh_A = SVD(A)`
- **Step 2**: `Vh_A` 방향을 W_base 공간으로 Pull-back — `M_dir = Vh_A @ W_base`
- **Step 3**: W_base의 스케일 왜곡 제거 — `U_M, _, Vh_M = SVD(M_dir)` → `O_clean = U_M @ Vh_M`
- **Step 4**: 재조립 — `A_new = U_A @ diag(S_A) @ O_clean`, `B = 0`

스위치 시점에서 B를 0으로 초기화하고 stale gradient를 클리어합니다.

### 2. Dynamic Mode — Gradient Coherence 기반 자동 스위칭

고정된 `switch_epoch` 대신, **Gradient Coherence**(연속 그래디언트 코사인 유사도)로 최적의 스위치 타이밍을 자동 결정:

```
C_t = cos_sim(∇A_t, ∇A_{t-1})
```

- Sequential 학습이 효율적일 때: `C_t ≈ high` (그래디언트 방향 일관)
- 지식 포화/오버피팅 시작: `C_t → 0` (그래디언트 방향 무작위화)
- EMA 스무딩 + Peak 대비 drop ratio로 트리거

**핵심 파일들:**
- `trainer/dynamic_switch.py` — `DynamicSwitchCallback` (Gradient Coherence 모니터링)
- `trainer/jelly_base.py` — `training_step()` 오버라이드로 backward 후 gradient 캡처

**실험 결과** (arc_challenge, Llama-2-7B):
- Dynamic switch at epoch 0.42 → final train loss **0.014** (LoRA: 0.11)
- arc_easy eval accuracy: **79.05%**

### 3. Merge-and-Reinit + Optimizer Reset — `jelly_base.py`

`_execute_switch()` 메서드:
1. **Weight Switch**: 각 JellyLayer에서 `switch_to_parallel_with_correction()` 호출
2. **Optimizer State Reset**: Adapter 파라미터의 momentum(exp_avg, exp_avg_sq) 초기화
3. **LR Schedule 유지**: LoRA와 동일한 단일 LR 스케줄 (restart 없음)

### 4. Data Slicing — 대규모 데이터셋 학습 시간 단축

`--max_train_samples` 옵션 추가 (기존 `--train_data_ratio`도 수정):

- `train_CS.py`: 기존에 `--train_data_ratio`가 정의만 되고 미사용 → 실제 슬라이싱 구현
- `train_nlu.py`, `train_vit.py`: `--max_train_samples` 추가
- WandB 로깅: `validation/original_train_data_size`, `validation/sliced_train_data_size`

**적용 순서**: `train_data_ratio` (비율) → `max_train_samples` (절대값 상한)

```bash
# 예시: 1000개 샘플로 제한 (~30분 내 학습 완료)
python train_CS.py --task winogrande --max_train_samples 1000
```

| Dataset | Original | MAX=1000 Est. |
|---------|----------|---------------|
| winogrande (40k) | ~15h | ~22 min |
| siqa (38k) | ~32h | ~50 min |
| piqa (16k) | ~15h | ~56 min |
| openbookqa (5k) | ~3h | ~36 min |
| arc_challenge (1.2k) | ~22 min | ~18 min |

## Installation

### 1. Create Virtual Environment

```bash
conda create -n jelly python=3.10 -y
conda activate jelly

```

### 2. Install Dependencies

```bash
# PyTorch with CUDA (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Core dependencies
pip install transformers==4.45.1 datasets evaluate wandb accelerate deepspeed scipy sentencepiece protobuf scikit-learn

```

### 3. Clone Repository with Submodule

**주의**: 반드시 빈 디렉토리나 새 위치에 클론하세요. 기존 프로젝트 폴더 안에서 실행하면 중복 폴더가 생깁니다.

```bash
# 홈 디렉토리나 원하는 위치로 이동
cd ~

# 새 폴더로 클론 (기존 jelly_trainer 폴더가 없어야 함)
git clone --recursive https://github.com/merrybabyxmas/jelly_trainer.git
cd jelly_trainer
```

이미 레포지토리가 있다면 서브모듈만 초기화:
```bash
cd /path/to/existing/jelly_trainer
git submodule update --init --recursive
```

### 4. Install JELLY PEFT Extension (Editable Mode)

기존의 복잡한 파일 복사 과정 없이, 서브모듈을 직접 패키지로 연결합니다. 이 방식을 사용하면 `peft_jelly` 폴더 내의 코드를 수정하는 즉시 실험에 반영됩니다.

```bash
# 공식 peft가 설치되어 있다면 먼저 삭제
pip uninstall peft -y

# 서브모듈을 수정한 즉시 반영되는 Editable 모드로 설치
pip install -e ./peft_jelly

```

## Verify Installation

설치가 완료되면 아래 명령어를 통해 `peft`가 서브모듈 경로에서 정상적으로 로드되는지 확인합니다.

```bash
python -c "
import peft
print(f'JELLY Path: {peft.__file__}')
from peft.tuners.jelly import JellyConfig, JellyModel
from trainer import register_jelly
register_jelly()
print('JELLY installation verified!')
"

```

## Usage

모든 실행 스크립트(`.sh`)는 실행 시 자동으로 서브모듈을 최신 상태로 동기화하도록 구성되어 있습니다.

### Commonsense Reasoning (Llama-2-7B)

```bash
# Dynamic mode (recommended) — auto-switch via Gradient Coherence
python train_CS.py --task piqa --adapter jelly --r 8 --jelly_mode dynamic

# With data slicing for fast iteration
python train_CS.py --task winogrande --adapter jelly --r 8 --jelly_mode dynamic --max_train_samples 1000

# Fixed switch epoch
python train_CS.py --task piqa --adapter jelly --r 8 --jelly_mode seq2par --switch_epoch 1
```

### NLU Training (GLUE Tasks)

```bash
python train_nlu.py --task sst2 --adapter jelly --r 8 --jelly_mode dynamic
```

### ViT Image Classification

```bash
python train_vit.py --task eurosat --adapter jelly --r 8 --jelly_mode dynamic
```

## Research Reproducibility

본 프레임워크는 연구의 재현성을 위해 실험마다 Git Hash를 기록합니다.

* **WandB Logging**: `trainer_hash`와 `peft_hash`가 자동으로 WandB 결과에 포함됩니다.
* **Submodule Auto-Sync**: 훈련 스크립트 실행 시 `git submodule update --init --recursive`가 수행되어 코드 버전 불일치를 방지합니다.

## JELLY Modes

| Mode | Description |
| --- | --- |
| `parallel` | Standard parallel adapter (like LoRA) |
| `sequential` | Sequential adapter mode |
| `seq2par` | Start with sequential, switch to parallel at `--switch_epoch` |
| `dynamic` | Start sequential, auto-switch via Gradient Coherence (recommended) |

## Project Structure

```
jelly_trainer/
├── peft_jelly/              # JELLY PEFT extension (submodule)
│   └── tuners/jelly/
│       ├── layer.py         # JellyLayer with switch_to_parallel_with_correction()
│       └── config.py        # JellyConfig
├── trainer/
│   ├── jelly_base.py        # JellyBaseTrainer (mode switching, optimizer reset, LR control)
│   ├── dynamic_switch.py    # DynamicSwitchCallback (Gradient Coherence)
│   └── utils.py             # Reproducibility utils (Git Hash)
├── experiments/
│   ├── base_runner.py       # BaseExperimentRunner (TrainingConfig, LoRAConfig, JELLYConfig)
│   ├── cs_comparison.py     # Commonsense Reasoning comparison runner
│   ├── glue_comparison.py   # GLUE comparison runner
│   └── img_comparison.py    # Image classification comparison runner
├── scripts/
│   ├── llama2_7B_base/      # Llama-2-7B run scripts
│   ├── deberta_v3_base/     # DeBERTa-v3 run scripts
│   └── vit_base/            # ViT-B/16 run scripts
├── train_CS.py              # Commonsense training script
├── train_nlu.py             # NLU training script
└── train_vit.py             # ViT training script
```

## RTX 4000 Series Note

For RTX 4000 series GPUs (4080, 4090), set these environment variables:

```bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
```

## License

MIT License
