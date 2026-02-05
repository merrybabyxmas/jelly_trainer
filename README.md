# JELLY Trainer

Training framework for **JELLY** (Joint Efficient Learning with Layer-Yielding), a custom PEFT adapter with sequential-to-parallel mode switching.

## Features

- **JELLY Adapter**: Novel adapter architecture with mode switching
  - Sequential mode: `output = W_base @ (B @ (A @ x))`
  - Parallel mode: `output = (W_base + B @ A) @ x`
  - Smooth transition via weight projection: `W_A_new = W_A @ W_base`
- **Multi-task Support**: NLU (GLUE), ViT image classification, Commonsense reasoning
- **Comparison Methods**: LoRA, DoRA, PiSSA, AdaLoRA, BitFit

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

# Install base peft package
pip install peft
```

### 3. Clone Repository with Submodule

```bash
git clone --recursive https://github.com/merrybabyxmas/jelly_trainer.git
cd jelly_trainer
```

If you already cloned without `--recursive`:
```bash
git submodule update --init --recursive
```

### 4. Install JELLY PEFT Extension

The JELLY adapter needs to be installed in your peft package location:

```bash
# Find peft installation path
PEFT_PATH=$(python -c "import peft; print(peft.__path__[0])")

# Copy JELLY tuner to peft
cp -r peft_jelly/tuners/jelly ${PEFT_PATH}/tuners/

# Append JELLY enum to peft_types.py (if not already present)
if ! grep -q "JELLY" ${PEFT_PATH}/utils/peft_types.py; then
    sed -i '/^class PeftType/,/^class / { /^class /!{ /^$/d; }; /^[[:space:]]*[A-Z]/{ H; $!d; }; ${x; s/\n//g; s/$/\n    JELLY = "JELLY"\n/; p; }; }' ${PEFT_PATH}/utils/peft_types.py 2>/dev/null || echo "Manual edit may be required for peft_types.py"
fi

# Append JELLY mapping to mapping.py (if not already present)
if ! grep -q "JELLY" ${PEFT_PATH}/mapping.py; then
    cat >> ${PEFT_PATH}/mapping.py << 'JELLY_MAPPING'

# JELLY Registration
PEFT_TYPE_TO_PREFIX_MAPPING[PeftType.JELLY] = "adapter_model"

try:
    from .tuners.jelly import JellyConfig, JellyModel
    PEFT_TYPE_TO_CONFIG_MAPPING[PeftType.JELLY] = JellyConfig
    PEFT_TYPE_TO_TUNER_MAPPING[PeftType.JELLY] = JellyModel
except ImportError:
    pass
JELLY_MAPPING
fi

echo "JELLY PEFT extension installed successfully!"
```

### 5. Verify Installation

```bash
python -c "
from peft.tuners.jelly import JellyConfig, JellyModel
from trainer import register_jelly, JellyBaseTrainer
register_jelly()
print('JELLY installation verified!')
"
```

## Usage

### NLU Training (GLUE Tasks)

```bash
python train_nlu.py \
    --task sst2 \
    --adapter jelly \
    --r 8 \
    --jelly_mode seq2par \
    --switch_epoch 3 \
    --epochs 10 \
    --batch 32 \
    --seed 42
```

### ViT Image Classification

```bash
python train_vit.py \
    --task eurosat \
    --adapter jelly \
    --r 8 \
    --jelly_mode seq2par \
    --switch_epoch 3 \
    --epochs 10 \
    --batch 32 \
    --seed 42
```

### Commonsense Reasoning

```bash
python train_CS.py \
    --task piqa \
    --adapter jelly \
    --model microsoft/deberta-v3-base \
    --r 8 \
    --jelly_mode seq2par \
    --switch_epoch 3 \
    --epochs 5 \
    --batch 16 \
    --seed 42
```

## JELLY Modes

| Mode | Description |
|------|-------------|
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
│   ├── jelly_nlu.py     # NLU trainer
│   ├── jelly_vit.py     # ViT trainer
│   └── utils.py         # Utilities
├── configs/             # Task configurations
├── scripts/             # Run scripts
├── train_nlu.py         # NLU training script
├── train_vit.py         # ViT training script
└── train_CS.py          # Commonsense training script
```

## Supported Tasks

### NLU (GLUE Benchmark)
- sst2, cola, mrpc, sts-b, qqp, mnli, qnli, rte

### Image Classification
- eurosat, dtd, gtsrb, resisc45, sun397, svhn

### Commonsense Reasoning
- piqa, siqa, arc_easy, arc_challenge, hellaswag, winogrande, openbookqa

## RTX 4000 Series Note

For RTX 4000 series GPUs (4080, 4090), set these environment variables:

```bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
```

## License

MIT License
