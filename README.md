

# JELLY Trainer

Training framework for **JELLY** (Joint Efficient Learning with Layer-Yielding), a custom PEFT adapter with sequential-to-parallel mode switching.

## Features

* **JELLY Adapter**: Novel adapter architecture with mode switching
* Sequential mode: `output = W_base @ (B @ (A @ x))`
* Parallel mode: `output = (W_base + B @ A) @ x`
* Smooth transition via weight projection: `W_A_new = W_A @ W_base`


* **Multi-task Support**: NLU (GLUE), ViT image classification, Commonsense reasoning
* **Experiment Tracking**: Automatic Git Hash logging for both Trainer and JELLY library to ensure 100% reproducibility.
* **Comparison Methods**: LoRA, DoRA, PiSSA, AdaLoRA, BitFit

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
pip install transformers datasets evaluate wandb accelerate deepspeed

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

### NLU Training (GLUE Tasks)

```bash
python train_nlu.py --task sst2 --adapter jelly --r 8 --jelly_mode seq2par --switch_epoch 3

```

### ViT Image Classification

```bash
python train_vit.py --task eurosat --adapter jelly --r 8 --jelly_mode seq2par --switch_epoch 3

```

### Commonsense Reasoning

```bash
python train_CS.py --task piqa --adapter jelly --r 8 --jelly_mode seq2par --switch_epoch 3

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

## Project Structure

```
jelly_trainer/
├── peft_jelly/          # JELLY PEFT extension (submodule)
│   └── tuners/jelly/    # JELLY tuner implementation
├── trainer/             # Training utilities
│   ├── jelly_base.py    # Base trainer with mode switching
│   ├── utils.py         # Reproducibility utils (Git Hash)
├── scripts/             # Optimized run scripts (.sh)
├── train_nlu.py         # NLU training script
├── train_vit.py         # ViT training script
└── train_CS.py          # Commonsense training script

```

## RTX 4000 Series Note

For RTX 4000 series GPUs (4080, 4090), set these environment variables:

```bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

```

## License

MIT License
