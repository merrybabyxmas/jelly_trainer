import argparse
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import math
from evaluate import load as load_metric
import wandb
import os
import json
import tempfile
import shutil
import numpy as np
import random

from peft import get_peft_model, LoraConfig, AdaLoraConfig
from peft.tuners.jelly.config import JellyConfig

from trainer import (
    JellyNLUTrainer,
    setup_seed,
    register_jelly,
    BestMetricCallback,
    verify_param_equality,
    log_adapter_params_to_wandb,
)
from trainer.utils import get_git_hash

# JELLY 등록
register_jelly()

# ============================================================
# GLUE 메타데이터 & 기본 설정
# ============================================================
GLUE_META = {
    "cola": {"keys": ("sentence", None), "num_labels": 2, "metric": "matthews_correlation"},
    "mnli": {"keys": ("premise", "hypothesis"), "num_labels": 3, "metric": "accuracy"},
    "mrpc": {"keys": ("sentence1", "sentence2"), "num_labels": 2, "metric": "f1"}, # accuracy/f1
    "qnli": {"keys": ("question", "sentence"), "num_labels": 2, "metric": "accuracy"},
    "qqp": {"keys": ("question1", "question2"), "num_labels": 2, "metric": "f1"}, # accuracy/f1
    "rte": {"keys": ("sentence1", "sentence2"), "num_labels": 2, "metric": "accuracy"},
    "sst2": {"keys": ("sentence", None), "num_labels": 2, "metric": "accuracy"},
    "stsb": {"keys": ("sentence1", "sentence2"), "num_labels": 1, "metric": "pearson"}, # pearson/spearmanr
    "wnli": {"keys": ("sentence1", "sentence2"), "num_labels": 2, "metric": "accuracy"},
}

# 태스크별 기본 하이퍼파라미터 (train_vit.py 스타일)
GLUE_TASK_CONFIG = {
    "cola": dict(epochs=20, batch=32, lr=2e-4),
    "mnli": dict(epochs=3, batch=32, lr=2e-4),
    "mrpc": dict(epochs=20, batch=32, lr=2e-4),
    "qnli": dict(epochs=3, batch=32, lr=2e-4),
    "qqp": dict(epochs=3, batch=32, lr=2e-4),
    "rte": dict(epochs=20, batch=32, lr=2e-4),
    "sst2": dict(epochs=10, batch=32, lr=2e-4),
    "stsb": dict(epochs=20, batch=32, lr=2e-4),
    "wnli": dict(epochs=20, batch=32, lr=2e-4),
}

def build_adapter(adapter_type, r=8, alpha=8, total_step=None, lora_dropout=0.0,
                  target_modules=None, init_weights="jelly"):
    at = adapter_type.lower()
    if target_modules is None:
        target_modules = ["query_proj", "key_proj", "value_proj"]

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

    if at in ["jelly", "lava"]:
        return JellyConfig(
            r=r,
            alpha=alpha,
            target_modules=target_modules,
            task_type="SEQ_CLS",
            lora_dropout=lora_dropout,
            init_weights=init_weights,
        )

    if at == "bitfit":
        return "bitfit"

    raise ValueError(f"Unknown adapter type: {adapter_type}")


def main(args):
    task = args.task
    adapter_type = args.adapter.lower()
    
    # --- CUDA 확인 로그 ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print(f"[DEVICE INFO] Using Device: {device.upper()}")
    if torch.cuda.is_available():
        print(f"[DEVICE INFO] GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"[DEVICE INFO] Available GPUs: {torch.cuda.device_count()}")
    else:
        print("[WARNING] CUDA is not available. Training will be slow.")
    print("=" * 60)
    # --------------------

    if task not in GLUE_META:
        raise ValueError(f"Unknown task: {task}")
    
    meta = GLUE_META[task]
    cfg = GLUE_TASK_CONFIG.get(task, {"epochs": 3, "batch": 32, "lr": 2e-4})

    num_labels = meta["num_labels"]
    metric_name = meta["metric"]
    
    # 인자 우선순위 적용
    epochs = args.epochs if args.epochs is not None else cfg["epochs"]
    batch = args.batch if args.batch else cfg["batch"]
    lr = args.learning_rate if args.learning_rate else cfg["lr"]

    # 데이터셋 로드
    raw_dataset = load_dataset("glue", task)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # 전처리 함수
    key1, key2 = meta["keys"]
    def preprocess_function(examples):
        args_tokenizer = (examples[key1],) if key2 is None else (examples[key1], examples[key2])
        return tokenizer(*args_tokenizer, truncation=True, max_length=128)

    encoded_dataset = raw_dataset.map(preprocess_function, batched=True)
    
    # Data Subset (재현성용)
    original_train_size = len(encoded_dataset["train"])
    if args.train_data_ratio < 100:
        subset_size = int(original_train_size * args.train_data_ratio / 100)
        subset_size = max(1, subset_size)
        encoded_dataset["train"] = encoded_dataset["train"].select(range(subset_size))
        print(f"[*] Using {args.train_data_ratio}% of training data: {subset_size}/{original_train_size}")
    
    total_train_samples = len(encoded_dataset["train"])

    # 모델 로드
    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.model, 
        num_labels=num_labels
    )
    base_model.to(device)

    # AdaLoRA용 total_step
    total_step = (len(encoded_dataset["train"]) // batch) * epochs
    
    # Target Modules
    target_modules = [m.strip() for m in args.target_modules.split(",")]

    # Verify LoRA == Jelly trainable param count (before adapter application)
    if adapter_type in ["lora", "pissa", "jelly", "lava"]:
        verify_param_equality(base_model, target_modules, r=args.r, alpha=args.alpha, lora_dropout=args.lora_dropout)

    # Adapter 적용
    peft_cfg = None
    if adapter_type == "bitfit":
        model = base_model
        for name, param in model.named_parameters():
            if "bias" in name or "classifier" in name or "pooler" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif adapter_type == "pissa":
        # PiSSA Precompute & Cache (train_vit.py 로직과 동일)
        peft_cfg = build_adapter(adapter_type, r=args.r, alpha=args.alpha, total_step=total_step, 
                                 lora_dropout=args.lora_dropout, target_modules=target_modules)
        
        cache_dir = ".precomputed_nlu"
        os.makedirs(cache_dir, exist_ok=True)
        model_name_safe = args.model.replace("/", "_")
        cache_path = os.path.join(cache_dir, f"{model_name_safe}_{task}_r{args.r}.pt")

        if os.path.exists(cache_path):
            print(f"[*] Found precomputed PiSSA weights at {cache_path}. Loading...")
            peft_cfg.init_lora_weights = False
            model = get_peft_model(base_model, peft_cfg)
            checkpoint = torch.load(cache_path, map_location="cpu")
            model.load_state_dict(checkpoint, strict=False)
            print(f"[*] PiSSA initialization loaded from cache.")
        else:
            print(f"[*] No precomputed weights. Computing PiSSA SVD...")
            peft_cfg.init_lora_weights = "pissa"
            model = get_peft_model(base_model, peft_cfg)
            
            to_save = {}
            for name, param in model.named_parameters():
                if "lora_" in name or any(tm in name for tm in peft_cfg.target_modules):
                    if param.requires_grad or "base_layer" in name:
                        to_save[name] = param.cpu().detach()
            # base_layer 저장
            for name, module in model.named_modules():
                if hasattr(module, 'base_layer') and hasattr(module.base_layer, 'weight'):
                    to_save[f"{name}.base_layer.weight"] = module.base_layer.weight.cpu().detach()
            
            torch.save(to_save, cache_path)
            print(f"[*] PiSSA SVD saved to {cache_path}")
    else:
        jelly_init = "lora" if (adapter_type.lower() in ["jelly", "lava"] and args.switch_epoch <= 0) else "jelly"
        if jelly_init == "lora":
            print(f"[JELLY] switch_epoch={args.switch_epoch} <= 0 → LoRA-equivalent mode (init_weights='lora')")
        peft_cfg = build_adapter(adapter_type, r=args.r, alpha=args.alpha, total_step=total_step,
                                 lora_dropout=args.lora_dropout, target_modules=target_modules,
                                 init_weights=jelly_init)
        model = get_peft_model(base_model, peft_cfg)

    # 파라미터 수 계산
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    print("=" * 60)
    print(f"[CONFIG] Task: {task} | Adapter: {adapter_type}")
    print(f"[CONFIG] Model: {args.model}")
    print(f"[CONFIG] Epochs: {epochs} | Batch: {batch} | LR: {lr}")
    print(f"[CONFIG] Rank: {args.r} | Alpha: {args.alpha}")
    if adapter_type in ["jelly", "lava"]:
        print(f"[CONFIG] JELLY Mode: {args.jelly_mode} | Switch Epoch: {args.switch_epoch}")
    print(f"[MODEL] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")

    print("=" * 60)

    # Metric 함수
    metric = load_metric("glue", task)
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if num_labels > 1:
            preds = np.argmax(preds, axis=1)
        elif num_labels == 1:
            preds = preds.squeeze()
        
        result = metric.compute(predictions=preds, references=labels)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values()))
        return result

    # Wandb 설정
    run_name = f"{adapter_type}_{task}_r{args.r}_s{args.seed}"
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                **vars(args),
                "trainer_hash": get_git_hash("."),
            }
        )
        wandb.run.summary["trainable_params"] = trainable
        wandb.run.summary["total_train_samples"] = total_train_samples

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
        num_train_epochs=epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler,
        max_grad_norm=args.max_grad_norm,
        report_to="wandb" if not args.no_wandb else "none",
        seed=args.seed,
        logging_steps=10,
        disable_tqdm=False,
    )

    # Callback
    target_metric_key = "combined_score" if task in ["mrpc", "stsb"] else metric_name
    callback = BestMetricCallback(target_metric_key)
    
    validation_key = "validation_mismatched" if task == "mnli" else "validation"

    # Trainer 선택
    if adapter_type in ["jelly", "lava"]:
        trainer = JellyNLUTrainer(
            model=model,
            args=training_args,
            train_dataset=encoded_dataset["train"],
            eval_dataset=encoded_dataset[validation_key],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[callback],
            jelly_mode=args.jelly_mode,
            switch_epoch=args.switch_epoch,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=encoded_dataset["train"],
            eval_dataset=encoded_dataset[validation_key],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[callback],
        )

    trainer.train()

    # Best Metric 추출
    best_score = None
    for log in trainer.state.log_history:
        if f"eval_{target_metric_key}" in log:
            val = log[f"eval_{target_metric_key}"]
            best_score = val if best_score is None else max(best_score, val)
    
    if best_score is not None and not args.no_wandb:
        wandb.run.summary["best_score"] = best_score
        wandb.finish()

    # 결과 저장
    result_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(result_dir, exist_ok=True)
    result_file = os.path.join(result_dir, f"glue_result_{adapter_type}_{task}_r{args.r}_s{args.seed}.json")

    result_data = {
        "task": task,
        "seed": args.seed,
        "adapter": adapter_type,
        "best_score": best_score if best_score else 0.0,
        "metric": target_metric_key
    }
    if adapter_type in ["jelly", "lava"]:
        result_data["jelly_mode"] = args.jelly_mode
        result_data["switch_epoch"] = args.switch_epoch

    with open(result_file, "w") as f:
        json.dump(result_data, f, indent=2)

    print("=" * 60)
    print(f"[RESULT] Task: {task} | Adapter: {adapter_type}")
    print(f"[RESULT] Best Score ({target_metric_key}): {best_score:.4f}")
    print(f"[RESULT] Saved to: {result_file}")
    print("=" * 60)

    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Task & Model
    parser.add_argument("--task", type=str, required=True, choices=list(GLUE_META.keys()))
    parser.add_argument("--adapter", type=str, required=True)
    parser.add_argument("--model", type=str, default="microsoft/deberta-v3-base")

    # Hyperparameters
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--lr_scheduler", type=str, default="linear")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    
    # LoRA Params
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=8)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--target_modules", type=str, default="query_proj,key_proj,value_proj,dense")

    # JELLY Params
    parser.add_argument("--jelly_mode", type=str, default="seq2par", choices=["parallel", "sequential", "seq2par"])
    parser.add_argument("--switch_epoch", type=int, default=1)

    # Wandb
    parser.add_argument("--wandb_project", type=str, default="GLUE-Comparison")
    parser.add_argument("--no_wandb", action="store_true")
    
    # Data Ratio
    parser.add_argument("--train_data_ratio", type=int, default=100)

    # Gradient Accumulation
    parser.add_argument("--grad_accum", type=int, default=1)

    args = parser.parse_args()
    setup_seed(args.seed)
    main(args)