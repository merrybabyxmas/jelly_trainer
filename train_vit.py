#!/usr/bin/env python
"""
ViT Image Classification Training
==================================
ViT-B/16을 사용한 이미지 분류 학습 (DTD, EuroSAT, GTSRB, RESISC45, SUN397, SVHN)
"""
import argparse
import torch
import random
import numpy as np
import os
import json
import tempfile
import shutil
import ssl

# SSL 인증서 검증 우회 (데이터셋 다운로드용)
ssl._create_default_https_context = ssl._create_unverified_context
from datasets import load_dataset
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
from torchvision import transforms
from torchvision import datasets as tv_datasets
from datasets import Dataset
import wandb

from peft import get_peft_model, LoraConfig, AdaLoraConfig
from peft.tuners.jelly.config import JellyConfig

from trainer import JellyViTTrainer, setup_seed, register_jelly, BestMetricCallback, verify_param_equality, log_adapter_params_to_wandb

# JELLY 등록
register_jelly()


def get_worker_init_fn(seed):
    def worker_init_fn(worker_id):
        import numpy as np
        import random
        worker_seed = (seed + worker_id) % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    return worker_init_fn


# ============================================================
# 데이터셋 설정
# ============================================================
IMG_TASK_META = {
    "dtd": dict(
        source="torchvision",
        tv_class=tv_datasets.DTD,
        num_labels=47,
    ),
    "eurosat": dict(
        source="torchvision",
        tv_class=tv_datasets.EuroSAT,
        num_labels=10,
    ),
    "gtsrb": dict(
        source="torchvision",
        tv_class=tv_datasets.GTSRB,
        num_labels=43,
    ),
    "resisc45": dict(
        source="huggingface",
        dataset_name="timm/resisc45",
        num_labels=45,
        split_train="train",
        split_val="test"
    ),
    "sun397": dict(
        source="huggingface",
        dataset_name="tanganke/sun397",
        num_labels=397,
        split_train="train",
        split_val="test"
    ),
    "svhn": dict(
        source="torchvision",
        tv_class=tv_datasets.SVHN,
        num_labels=10,
    ),
}

# 태스크별 하이퍼파라미터
IMG_TASK_CONFIG = {
    "dtd": dict(epochs=50, batch=32, lr=1e-4),
    "eurosat": dict(epochs=20, batch=32, lr=1e-4),
    "gtsrb": dict(epochs=20, batch=32, lr=1e-4),
    "resisc45": dict(epochs=20, batch=32, lr=1e-4),
    "sun397": dict(epochs=30, batch=32, lr=1e-4),
    "svhn": dict(epochs=10, batch=32, lr=1e-4),
}


# ============================================================
# Torchvision Dataset Loader
# ============================================================
def load_torchvision_dataset(task: str, meta: dict, data_root: str = "./data", seed: int = 42):
    """
    Torchvision 데이터셋을 HuggingFace Dataset 형식으로 변환 (캐시 지원)

    Args:
        task: 태스크 이름
        meta: 태스크 메타 정보
        data_root: 데이터 저장 경로
        seed: random_split에 사용할 시드 (재현성 보장)
    """
    # HF Dataset 캐시 확인 (DTD 등 변환이 오래 걸리는 데이터셋용)
    cache_base = os.path.join(os.path.dirname(__file__), ".cache", task, "hf_dataset")
    train_cache_path = os.path.join(cache_base, "train")
    val_cache_path = os.path.join(cache_base, "test")

    if os.path.exists(train_cache_path) and os.path.exists(val_cache_path):
        print(f"[*] Loading cached HF dataset from {cache_base}")
        train_hf = Dataset.load_from_disk(train_cache_path)
        val_hf = Dataset.load_from_disk(val_cache_path)
        return {"train": train_hf, "test": val_hf}

    print(f"[*] Converting torchvision dataset to HF format (this may take a while for {task})...")
    tv_class = meta["tv_class"]

    # DTD는 split 파라미터 사용
    if task == "dtd":
        train_ds = tv_class(root=data_root, split="train", download=True)
        val_ds = tv_class(root=data_root, split="test", download=True)
    # GTSRB는 split 파라미터 사용
    elif task == "gtsrb":
        train_ds = tv_class(root=data_root, split="train", download=True)
        val_ds = tv_class(root=data_root, split="test", download=True)
    # SVHN은 split 파라미터 사용
    elif task == "svhn":
        train_ds = tv_class(root=data_root, split="train", download=True)
        val_ds = tv_class(root=data_root, split="test", download=True)
    # EuroSAT, SUN397는 train 파라미터 사용
    elif task in ["eurosat", "sun397"]:
        # 전체 데이터셋 로드 후 분할
        full_ds = tv_class(root=data_root, download=True)
        # 80/20 split
        total_len = len(full_ds)
        train_len = int(0.8 * total_len)
        val_len = total_len - train_len
        train_ds, val_ds = torch.utils.data.random_split(
            full_ds, [train_len, val_len],
            generator=torch.Generator().manual_seed(seed)  
        )
    else:
        raise ValueError(f"Unknown task: {task}")

    def convert_to_hf_dataset(tv_dataset):
        """Torchvision dataset을 HuggingFace Dataset으로 변환"""
        images = []
        labels = []

        # random_split 결과인 경우 Subset 처리
        if hasattr(tv_dataset, 'dataset'):
            # Subset인 경우
            for idx in tv_dataset.indices:
                img, label = tv_dataset.dataset[idx]
                images.append(img)
                labels.append(label)
        else:
            for i in range(len(tv_dataset)):
                img, label = tv_dataset[i]
                images.append(img)
                labels.append(label)

        return Dataset.from_dict({"image": images, "label": labels})

    train_hf = convert_to_hf_dataset(train_ds)
    val_hf = convert_to_hf_dataset(val_ds)

    # 캐시 저장 (다음 실행 시 빠르게 로드)
    os.makedirs(cache_base, exist_ok=True)
    train_hf.save_to_disk(train_cache_path)
    val_hf.save_to_disk(val_cache_path)
    print(f"[*] Saved HF dataset cache to {cache_base}")

    return {"train": train_hf, "test": val_hf}


def build_adapter(adapter_type, r=8, alpha=8, total_step=None, lora_dropout=0.0,
                  target_modules=None, init_weights="jelly"):
    at = adapter_type.lower()
    if target_modules is None:
        target_modules = ["query", "key", "value"]

    # NOTE: task_type="SEQ_CLS" ensures classifier head is trainable via modules_to_save
    # This is critical for fair comparison - all adapters should train the classifier
    if at in ["lora", "dora", "pissa"]:
        kwargs = dict(
            r=r,
            lora_alpha=alpha,
            target_modules=target_modules,
            task_type="SEQ_CLS",
            lora_dropout=lora_dropout,
        )
        if at == "pissa":
            kwargs["init_lora_weights"] = "pissa"
        if at == "dora":
            kwargs["use_dora"] = True
        return LoraConfig(**kwargs)

    if at == "adalora":
        return AdaLoraConfig(
            init_r=r,
            target_r=r // 2,
            lora_alpha=alpha,
            target_modules=target_modules,
            total_step=total_step if total_step else 1000,
            task_type="SEQ_CLS",
            lora_dropout=lora_dropout,
        )

    if at == "jelly":
        return JellyConfig(r=r, alpha=alpha, target_modules=target_modules,
                           task_type="SEQ_CLS", lora_dropout=lora_dropout,
                           init_weights=init_weights)

    if at == "bitfit":
        return "bitfit"

    raise ValueError(f"Unknown adapter: {adapter_type}")


def main(args):
    task = args.task
    # --- CUDA 확인 로그 추가 ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print(f"[DEVICE INFO] Using Device: {device.upper()}")
    if torch.cuda.is_available():
        print(f"[DEVICE INFO] GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"[DEVICE INFO] Available GPUs: {torch.cuda.device_count()}")
    else:
        print("[WARNING] CUDA is not available. Training will be slow on CPU.")
    print("=" * 60)
    # -----------------------
    
    adapter_type = args.adapter.lower()

    meta = IMG_TASK_META[task]
    cfg = IMG_TASK_CONFIG.get(task, {"epochs": 20, "batch": 32, "lr": 1e-4})

    num_labels = meta["num_labels"]
    epochs = args.epochs if args.epochs is not None else cfg["epochs"]
    batch = args.batch if args.batch else cfg["batch"]
    lr = args.learning_rate if args.learning_rate else cfg["lr"]

    # 데이터셋 로드
    if meta.get("source") == "torchvision":
        raw = load_torchvision_dataset(task, meta, seed=args.seed)
        split_train, split_val = "train", "test"
    else:
        # HuggingFace datasets
        if "subset" in meta:
            raw = load_dataset(meta["dataset_name"], meta["subset"])
        else:
            raw = load_dataset(meta["dataset_name"])
        split_train = meta["split_train"]
        split_val = meta["split_val"]

    # 모델 및 프로세서 로드
    model_name = "google/vit-base-patch16-224"
    processor = ViTImageProcessor.from_pretrained(model_name)
    base = ViTForImageClassification.from_pretrained(model_name, num_labels=num_labels, ignore_mismatched_sizes=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base.to(device)
    # 이미지 전처리
    def preprocess(examples):
        images = examples["image"]
        images = [img.convert("RGB") for img in images]
        inputs = processor(images, return_tensors="pt")
        # ViT에 필요한 pixel_values와 labels만 반환
        return {
            "pixel_values": inputs["pixel_values"],
            "labels": examples["label"]
        }

    # 캐시 디렉토리 설정 (태스크별로 캐시 저장)
    cache_dir = os.path.join(os.path.dirname(__file__), ".cache", task)
    os.makedirs(cache_dir, exist_ok=True)

    train_cache = os.path.join(cache_dir, "train_preprocessed.arrow")
    val_cache = os.path.join(cache_dir, "val_preprocessed.arrow")

    # 데이터 전처리 (캐시 파일 명시적 지정)
    train_ds = raw[split_train].map(
        preprocess,
        batched=True,
        remove_columns=raw[split_train].column_names,
        keep_in_memory=False,
        load_from_cache_file=True,
        cache_file_name=train_cache,
        batch_size=100,
    )
    val_ds = raw[split_val].map(
        preprocess,
        batched=True,
        remove_columns=raw[split_val].column_names,
        keep_in_memory=False,
        load_from_cache_file=True,
        cache_file_name=val_cache,
        batch_size=100,
    )
    train_ds.set_format("torch")
    val_ds.set_format("torch")

    # Apply train_data_ratio (use first N% for reproducibility)
    original_train_size = len(train_ds)
    if args.train_data_ratio < 100:
        subset_size = int(original_train_size * args.train_data_ratio / 100)
        subset_size = max(1, subset_size)  # At least 1 sample
        train_ds = train_ds.select(range(subset_size))
        print(f"[*] Using {args.train_data_ratio}% of training data: {subset_size}/{original_train_size} samples")
    if args.max_train_samples > 0 and len(train_ds) > args.max_train_samples:
        train_ds = train_ds.select(range(args.max_train_samples))
        print(f"[*] max_train_samples={args.max_train_samples}: {len(train_ds)}/{original_train_size} samples")
    sliced_train_size = len(train_ds)
    if sliced_train_size < original_train_size:
        print(f"[*] Training data sliced: {original_train_size} → {sliced_train_size}")

    total_train_samples = len(train_ds)

    # AdaLoRA를 위한 total_step 계산
    total_step = (len(train_ds) // batch) * epochs

    # Target modules 파싱
    target_modules = [m.strip() for m in args.target_modules.split(",")]

    # Verify LoRA == Jelly trainable param count (before adapter application)
    if adapter_type in ["lora", "pissa", "jelly"]:
        verify_param_equality(base, target_modules, r=args.r, alpha=args.alpha, lora_dropout=args.lora_dropout)

    # Adapter 적용
    peft_cfg = None
    if adapter_type == "bitfit":
        model = base
        for name, param in model.named_parameters():
            if "bias" in name or "classifier" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif adapter_type.lower() == "pissa":
        # PiSSA precompute 로직: SVD 계산 결과를 캐시하여 재사용
        peft_cfg = build_adapter(adapter_type, r=args.r, alpha=args.alpha, total_step=total_step, lora_dropout=args.lora_dropout, target_modules=target_modules)

        cache_dir = ".precomputed"
        os.makedirs(cache_dir, exist_ok=True)

        # ViT 모델명과 Rank를 조합해 고유 파일명 생성
        model_name_safe = "vit-base-patch16-224"
        cache_path = os.path.join(cache_dir, f"{model_name_safe}_r{args.r}.pt")

        if os.path.exists(cache_path):
            print(f"[*] Found precomputed PiSSA weights at {cache_path}. Loading...")
            # 캐시가 있으면 SVD 연산을 건너뜀
            peft_cfg.init_lora_weights = False
            model = get_peft_model(base, peft_cfg)

            # 저장된 PiSSA 가중치 로드
            checkpoint = torch.load(cache_path, map_location="cpu")
            model.load_state_dict(checkpoint, strict=False)
            print(f"[*] PiSSA initialization loaded from cache.")
        else:
            print(f"[*] No precomputed weights found. Computing PiSSA SVD (this may take a while)...")
            peft_cfg.init_lora_weights = "pissa"
            model = get_peft_model(base, peft_cfg)

            # 초기화된 가중치 저장 (lora_A, lora_B 및 수정된 base_layer)
            to_save = {}
            for name, param in model.named_parameters():
                if "lora_" in name or any(tm in name for tm in peft_cfg.target_modules):
                    if param.requires_grad or "base_layer" in name:
                        to_save[name] = param.cpu().detach()

            # base_layer 가중치도 저장 (PiSSA에서 수정됨)
            for name, module in model.named_modules():
                if hasattr(module, 'base_layer') and hasattr(module.base_layer, 'weight'):
                    to_save[f"{name}.base_layer.weight"] = module.base_layer.weight.cpu().detach()

            torch.save(to_save, cache_path)
            print(f"[*] PiSSA SVD computation finished and saved to {cache_path}")
    else:
        jelly_init = "lora" if (adapter_type.lower() == "jelly" and args.switch_epoch <= 0) else "jelly"
        if jelly_init == "lora":
            print(f"[JELLY] switch_epoch={args.switch_epoch} <= 0 → LoRA-equivalent mode (init_weights='lora')")
        peft_cfg = build_adapter(adapter_type, r=args.r, alpha=args.alpha, total_step=total_step,
                                 lora_dropout=args.lora_dropout, target_modules=target_modules,
                                 init_weights=jelly_init)
        model = get_peft_model(base, peft_cfg)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    frozen = total - trainable

    # Count adapter modules and calculate params per adapter
    num_adapter_modules = 0
    adapter_only_params = 0
    for name, module in model.named_modules():
        # PEFT adapter layers (LoRA, LAVA, etc.)
        if hasattr(module, 'lora_A') or hasattr(module, 'W_mu'):
            num_adapter_modules += 1
            # Count params in this adapter module
            for p in module.parameters():
                if p.requires_grad:
                    adapter_only_params += p.numel()

    params_per_adapter = adapter_only_params / num_adapter_modules if num_adapter_modules > 0 else 0

    print("=" * 60)
    print(f"[CONFIG] Task: {task} | Adapter: {adapter_type}")
    print(f"[CONFIG] Seed: {args.seed} | Epochs: {epochs} | Batch: {batch} | LR: {lr}")
    print(f"[CONFIG] Rank: {args.r} | Alpha: {args.alpha}")
    if adapter_type == "jelly":
        print(f"[CONFIG] JELLY Mode: {args.jelly_mode} | Switch Epoch: {args.switch_epoch}")
    print(f"[MODEL] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")
    print(f"[MODEL] Adapter modules: {num_adapter_modules} | Params per adapter: {params_per_adapter:,.0f}")
    print(f"[DATA] Train: {len(train_ds)} | Val: {len(val_ds)}")

    print("=" * 60)

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = preds.argmax(-1)
        acc = (preds == labels).mean()
        return {"accuracy": acc}

    run_name = f"{adapter_type}_{task}_r{args.r}_s{args.seed}"

    # Wandb 설정
    if hasattr(args, 'no_wandb') and args.no_wandb:
        wandb_mode = "disabled"
        report_to = "none"
    else:
        wandb_mode = "online"
        report_to = "wandb"

    wandb_project = getattr(args, 'wandb_project', "ViT-ImageClassification")
    wandb.init(project=wandb_project, name=run_name, config=vars(args), mode=wandb_mode)

    # Log training data info to wandb
    if wandb_mode != "disabled":
        wandb.run.summary["total_train_samples"] = total_train_samples
        wandb.run.summary["original_train_size"] = original_train_size
        wandb.run.summary["train_data_ratio"] = args.train_data_ratio
        wandb.run.summary["validation/original_train_data_size"] = original_train_size
        wandb.run.summary["validation/sliced_train_data_size"] = sliced_train_size
        # Parameter metrics
        wandb.run.summary["trainable_params"] = trainable
        wandb.run.summary["all_params"] = total
        wandb.run.summary["frozen_params"] = frozen
        wandb.run.summary["trainable_percentage"] = 100 * trainable / total
        wandb.run.summary["num_adapter_modules"] = num_adapter_modules
        wandb.run.summary["adapter_only_params"] = adapter_only_params
        wandb.run.summary["params_per_adapter"] = params_per_adapter

    # Parameter validation & wandb logging (all methods)
    log_adapter_params_to_wandb(model, adapter_type, peft_config=peft_cfg, target_modules=target_modules)

    tmp_dir = tempfile.mkdtemp()

    training_args = TrainingArguments(
        output_dir=tmp_dir,
        evaluation_strategy="epoch",
        save_strategy="no",
        learning_rate=lr,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler,
        max_grad_norm=args.max_grad_norm,
        report_to=report_to,
        seed=args.seed,
        logging_steps=10,
        logging_first_step=True,
        disable_tqdm=False,
        log_level="info",
        label_names=["labels"],  # compute_metrics 호출을 위해 명시적으로 설정
        dataloader_num_workers=4,  # 시드 재현성을 위해 메인 프로세스에서 로드
        dataloader_pin_memory=True,  
        use_cpu=False,  # GPU 강제 사용
        no_cuda=False,  # CUDA 비활성화 안함
    )

    callback = BestMetricCallback("accuracy")

    # JELLY adapter 사용 시 JellyViTTrainer 사용 (adapter_mode 로깅용)
    if adapter_type == "jelly":
        trainer = JellyViTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
            callbacks=[callback],
            jelly_mode=args.jelly_mode,
            switch_epoch=args.switch_epoch,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
            callbacks=[callback],
        )

    trainer.train()

    best_acc = None
    for log in trainer.state.log_history:
        if "eval_accuracy" in log:
            val = log["eval_accuracy"]
            best_acc = val if best_acc is None else max(best_acc, val)

    if best_acc is not None:
        wandb.run.summary["best_accuracy"] = best_acc

    wandb.finish()

    # 결과 저장
    result_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(result_dir, exist_ok=True)

    result_file = os.path.join(
        result_dir,
        f"img_result_{adapter_type}_{task}_r{args.r}_s{args.seed}.json"
    )

    result_data = {
        "task": task,
        "seed": args.seed,
        "adapter": adapter_type,
        "best_accuracy": best_acc if best_acc else 0.0,
    }
    if adapter_type == "jelly":
        result_data["jelly_mode"] = args.jelly_mode
        result_data["switch_epoch"] = args.switch_epoch

    with open(result_file, "w") as f:
        json.dump(result_data, f, indent=2)

    print("=" * 60)
    if best_acc is not None:
        print(f"[RESULT] Task: {task} | Adapter: {adapter_type}")
        print(f"[RESULT] Best Accuracy: {best_acc:.4f}")
    else:
        print(f"[RESULT] No valid metric")
    print(f"[RESULT] Saved to: {result_file}")
    print("=" * 60)

    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    return best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Task & Model
    parser.add_argument("--task", type=str, required=True, choices=list(IMG_TASK_META.keys()))
    parser.add_argument("--adapter", type=str, required=True)

    # General Training Parameters
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs from config")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--lr_scheduler", type=str, default="linear",
                        choices=["linear", "cosine", "constant"])
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # LoRA Parameters
    parser.add_argument("--r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--target_modules", type=str, default="query,key,value,dense",
                        help="Comma-separated target modules (e.g., 'query,key,value,dense')")

    # JELLY Specific Parameters
    parser.add_argument("--jelly_mode", type=str, default="seq2par",
                        choices=["parallel", "sequential", "seq2par"],
                        help="JELLY mode: parallel, sequential, or seq2par (sequential->parallel)")
    parser.add_argument("--switch_epoch", type=int, default=3,
                        help="Epoch to switch from sequential to parallel (only for seq2par mode)")

    # Wandb Settings
    parser.add_argument("--wandb_project", type=str, default="ViT-ImageClassification", help="Wandb project name")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")

    # Data Ratio Parameter
    parser.add_argument("--train_data_ratio", type=int, default=100,
                        help="Percentage of training data to use (1-100). Uses first N%% for reproducibility.")
    parser.add_argument("--max_train_samples", type=int, default=0,
                        help="Max number of training samples (0=no limit). Applied after train_data_ratio.")

    # Gradient Accumulation
    parser.add_argument("--grad_accum", type=int, default=1)

    args = parser.parse_args()
    setup_seed(args.seed)
    main(args)
