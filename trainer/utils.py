"""
Trainer Utilities
=================
Common utility functions for train_vit.py and train_nlu.py
"""

import random
import numpy as np
import torch
import wandb
from transformers import TrainerCallback

import peft.utils.save_and_load
import peft.mapping
from peft.utils.peft_types import PeftType
from peft.tuners.jelly.config import JellyConfig
from peft.tuners.jelly.model import JellyModel

import subprocess

def setup_seed(seed: int):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)



def reset_jelly_generators(model, seed: int = None):
    """No-op: JellyAdapter no longer exists. Kept for API compatibility."""
    pass


def register_jelly():
    """Register JELLY to PEFT mappings"""
    if not hasattr(PeftType, "JELLY"):
        PeftType.JELLY = "JELLY"

    for jelly_key in ["JELLY", PeftType.JELLY]:
        peft.mapping.PEFT_TYPE_TO_CONFIG_MAPPING[jelly_key] = JellyConfig
        peft.mapping.PEFT_TYPE_TO_TUNER_MAPPING[jelly_key] = JellyModel
        peft.utils.save_and_load.PEFT_TYPE_TO_PREFIX_MAPPING[jelly_key] = "adapter_model"
        peft.mapping.PEFT_TYPE_TO_PREFIX_MAPPING[jelly_key] = "adapter_model"


class BestMetricCallback(TrainerCallback):
    """
    Callback to track best metric during training (higher is better: accuracy, f1, etc.)

    Args:
        main_metric: Metric name to track (e.g., "accuracy", "f1", "pearson")
                    Uses "accuracy" if None
    """
    def __init__(self, main_metric: str = None):
        if main_metric:
            self.main_metric = f"eval_{main_metric}"
        else:
            self.main_metric = "eval_accuracy"
        self.best_score = -float("inf")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        epoch = int(state.epoch) if state.epoch else 0
        if metrics:
            current = metrics.get(self.main_metric)
            if current is None:
                current = metrics.get("eval_accuracy", metrics.get("accuracy"))

            if current is not None:
                is_best = current > self.best_score
                if is_best:
                    self.best_score = current
                print(f"[EVAL] Epoch {epoch}: {self.main_metric} = {current:.4f} | Best = {self.best_score:.4f}" + (" *" if is_best else ""))
                if wandb.run is not None:
                    wandb.log({"eval/best_main": self.best_score}, step=state.global_step)
            else:
                loss = metrics.get("eval_loss", 0)
                print(f"[EVAL] Epoch {epoch}: Loss = {loss:.4f}")

    def on_train_begin(self, args, state, control, **kwargs):
        print("[TRAIN] Training started...")

    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch = int(state.epoch) if state.epoch else 0
        print(f"[TRAIN] Epoch {epoch + 1}/{int(args.num_train_epochs)} starting...")

    def on_train_end(self, args, state, control, **kwargs):
        print(f"[TRAIN] Training completed. Best score: {self.best_score:.4f}")


class BestLossCallback(TrainerCallback):
    """
    Callback to track best (minimum) eval_loss during training (lower is better).
    Logs eval/best_loss, eval/best_perplexity, eval/perplexity to wandb each eval step.
    """
    def __init__(self):
        self.best_loss = float("inf")
        self.best_perplexity = float("inf")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        import math
        epoch = state.epoch or 0
        if metrics:
            loss = metrics.get("eval_loss")
            if loss is not None:
                perplexity = math.exp(min(loss, 20))  # cap to avoid overflow
                is_best = loss < self.best_loss
                if is_best:
                    self.best_loss = loss
                    self.best_perplexity = perplexity
                print(
                    f"[EVAL] Epoch {epoch:.2f}: eval_loss={loss:.4f} | "
                    f"perplexity={perplexity:.2f} | "
                    f"best_loss={self.best_loss:.4f}"
                    + (" *" if is_best else "")
                )
                if wandb.run is not None:
                    wandb.log({
                        "eval/best_loss": self.best_loss,
                        "eval/best_perplexity": self.best_perplexity,
                        "eval/perplexity": perplexity,
                        "eval/loss": loss,
                    }, step=state.global_step)

    def on_train_begin(self, args, state, control, **kwargs):
        print("[TRAIN] Training started...")

    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch = int(state.epoch) if state.epoch else 0
        print(f"[TRAIN] Epoch {epoch + 1}/{int(args.num_train_epochs)} starting...")

    def on_train_end(self, args, state, control, **kwargs):
        print(
            f"[TRAIN] Training completed. "
            f"Best eval_loss={self.best_loss:.4f} | "
            f"Best perplexity={self.best_perplexity:.2f}"
        )


def print_trainable_parameters(model):
    """Print trainable parameter count and percentage"""
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        all_params += num_params
        if param.requires_grad:
            trainable_params += num_params

    percentage = 100 * trainable_params / all_params
    print(f"trainable params: {trainable_params:,} || all params: {all_params:,} || trainable%: {percentage:.4f}")
    return trainable_params, all_params, percentage



def verify_param_equality(base_model, target_modules, r, alpha, lora_dropout=0.0):
    """
    Analytically verify that LoRA and Jelly would have the same trainable parameters.
    Scans base model for matching target layers and computes expected adapter param counts.
    """
    import torch.nn as nn

    # 1. Find all target layers
    matched_layers = []
    for name, module in base_model.named_modules():
        if isinstance(module, nn.Linear):
            if any(t in name for t in target_modules):
                matched_layers.append((name, module.in_features, module.out_features))

    # 2. Compute adapter params per layer
    # LoRA:  A = Linear(in, r) + B = Linear(r, out) → in*r + r*out
    # Jelly: W_A = Linear(adapter_in, r) + W_B = Linear(r, out)
    #   square (in==out): adapter_in = out = in → same as LoRA
    #   non-square:       adapter_in = in       → same as LoRA
    total_adapter_params = 0
    print(f"[PARAM CHECK] Target modules: {target_modules} | Rank: {r}")
    print(f"[PARAM CHECK] {'Layer':<60} {'in':>6} {'out':>6} {'A(in×r)':>10} {'B(r×out)':>10} {'subtotal':>10}")
    print(f"[PARAM CHECK] {'-'*104}")
    for name, in_f, out_f in matched_layers:
        a_params = in_f * r
        b_params = r * out_f
        subtotal = a_params + b_params
        total_adapter_params += subtotal
        print(f"[PARAM CHECK] {name:<60} {in_f:>6} {out_f:>6} {a_params:>10,} {b_params:>10,} {subtotal:>10,}")

    # 3. Find classifier/score head params (modules_to_save)
    classifier_params = 0
    classifier_names = ["classifier", "score"]
    for name, module in base_model.named_modules():
        if any(c in name for c in classifier_names) and hasattr(module, 'weight'):
            p_count = sum(p.numel() for p in module.parameters())
            classifier_params += p_count
            print(f"[PARAM CHECK] {name + ' (head)':<60} {'':<6} {'':<6} {'':>10} {'':>10} {p_count:>10,}")

    total_trainable = total_adapter_params + classifier_params
    print(f"[PARAM CHECK] {'-'*104}")
    print(f"[PARAM CHECK] {'Adapter layers:':<60} {len(matched_layers):>6} layers {'':<6} {'':>10} {'':>10} {total_adapter_params:>10,}")
    print(f"[PARAM CHECK] {'Classifier head:':<60} {'':<6} {'':<6} {'':>10} {'':>10} {classifier_params:>10,}")
    print(f"[PARAM CHECK] {'TOTAL trainable (LoRA == Jelly):':<60} {'':<6} {'':<6} {'':>10} {'':>10} {total_trainable:>10,}")

    return total_trainable, len(matched_layers)


def _classify_module_type(param_name, target_modules=None):
    """Classify a parameter into a module type based on its name path."""
    # Classifier/score head (highest priority)
    if any(kw in param_name for kw in ("classifier", "score")):
        return "classifier_head"

    # Pooler
    if "pooler" in param_name:
        return "pooler"

    # Match against user-specified target modules (case-sensitive, longest first)
    if target_modules:
        sorted_tms = sorted(target_modules, key=len, reverse=True)
        for tm in sorted_tms:
            if tm in param_name:
                return tm

    # Fallback: common module keywords (longest first to avoid partial matches)
    common = [
        "query_proj", "key_proj", "value_proj",
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "query", "key", "value", "dense",
    ]
    for cm in common:
        if cm in param_name:
            return cm

    return "other"


def log_adapter_params_to_wandb(model, adapter_type, peft_config=None, target_modules=None):
    """
    Log detailed per-module trainable parameter counts, alpha, and rank to wandb.

    Works for ALL adapter types: LoRA, DoRA, PiSSA, AdaLoRA, JELLY, BitFit.
    Call this AFTER wandb.init() and after model (with adapter) is created.

    Args:
        model: The model (with adapter applied)
        adapter_type: str, e.g. "lora", "jelly", "bitfit"
        peft_config: The PEFT config object (LoraConfig, JellyConfig, etc.) or None
        target_modules: List of target module names (e.g. ["query_proj", "key_proj"])
    """
    at = adapter_type.lower()

    # ===== 1. Per-module trainable param counts =====
    module_params = {}
    total_trainable = 0
    total_all = 0

    for name, param in model.named_parameters():
        total_all += param.numel()
        if param.requires_grad:
            total_trainable += param.numel()
            module_type = _classify_module_type(name, target_modules)
            module_params[module_type] = module_params.get(module_type, 0) + param.numel()

    # ===== 2. Extract alpha/rank from config =====
    config_alpha = None
    config_rank = None

    if peft_config is not None and peft_config != "bitfit":
        if hasattr(peft_config, 'lora_alpha'):
            config_alpha = peft_config.lora_alpha
        elif hasattr(peft_config, 'alpha'):
            config_alpha = peft_config.alpha

        if hasattr(peft_config, 'r'):
            config_rank = peft_config.r
        elif hasattr(peft_config, 'init_r'):
            config_rank = peft_config.init_r

    # ===== 3. Verify alpha/rank from actual model layers =====
    model_alphas = set()
    model_ranks = set()

    for _, module in model.named_modules():
        # LoRA / DoRA / PiSSA / AdaLoRA layers
        if hasattr(module, 'lora_alpha') and isinstance(getattr(module, 'lora_alpha', None), dict):
            for _, alpha_val in module.lora_alpha.items():
                model_alphas.add(alpha_val)
            if hasattr(module, 'r') and isinstance(getattr(module, 'r', None), dict):
                for _, r_val in module.r.items():
                    model_ranks.add(r_val)

    # ===== 4. Console output =====
    print(f"\n{'='*70}")
    print(f"[PARAM VALIDATION] Adapter: {at.upper()}")
    print(f"[PARAM VALIDATION] Total: {total_trainable:,} trainable / {total_all:,} ({100*total_trainable/total_all:.4f}%)")
    print(f"[PARAM VALIDATION] Per-module breakdown:")
    for module_type, count in sorted(module_params.items()):
        pct = 100 * count / total_trainable if total_trainable > 0 else 0
        print(f"[PARAM VALIDATION]   {module_type:<30} {count:>12,} ({pct:>5.1f}%)")
    if config_alpha is not None:
        print(f"[PARAM VALIDATION] Config Alpha: {config_alpha} | Config Rank: {config_rank}")
    if model_alphas:
        print(f"[PARAM VALIDATION] Model Alpha (verified): {sorted(model_alphas)}")
    if model_ranks:
        print(f"[PARAM VALIDATION] Model Rank  (verified): {sorted(model_ranks)}")
    print(f"{'='*70}\n")

    # ===== 5. Log to wandb =====
    if wandb.run is None:
        return

    summary = {
        "validation/adapter_type": at,
        "validation/total_params": total_all,
        "validation/trainable_params": total_trainable,
        "validation/trainable_pct": round(100 * total_trainable / total_all, 4) if total_all > 0 else 0,
    }

    for module_type, count in sorted(module_params.items()):
        summary[f"validation/params/{module_type}"] = count

    if config_alpha is not None:
        summary["validation/config_alpha"] = config_alpha
    if config_rank is not None:
        summary["validation/config_rank"] = config_rank

    if model_alphas:
        alpha_list = sorted(model_alphas)
        summary["validation/model_alpha"] = alpha_list[0] if len(alpha_list) == 1 else str(alpha_list)
    if model_ranks:
        rank_list = sorted(model_ranks)
        summary["validation/model_rank"] = rank_list[0] if len(rank_list) == 1 else str(rank_list)

    wandb.run.summary.update(summary)


def get_git_hash(path="."):
    """Get git commit hash for a given directory"""
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=path,
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def get_git_info() -> dict:
    """Get git hashes for trainer and peft_jelly for experiment tracking"""
    import os

    # Trainer (jelly_trainer) hash
    trainer_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    trainer_hash = get_git_hash(trainer_path)

    # PEFT/JELLY hash (from peft_jelly submodule)
    peft_jelly_path = os.path.join(trainer_path, "peft_jelly")
    if os.path.exists(peft_jelly_path):
        peft_hash = get_git_hash(peft_jelly_path)
    else:
        # Fallback: try to get from installed peft location
        import peft
        peft_path = os.path.dirname(peft.__file__)
        peft_hash = get_git_hash(os.path.dirname(peft_path))

    return {
        "trainer_hash": trainer_hash,
        "peft_hash": peft_hash,
    }


def log_git_info_to_wandb():
    """Log git hashes to wandb for reproducibility"""
    if wandb.run is not None:
        git_info = get_git_info()
        wandb.config.update(git_info, allow_val_change=True)
        wandb.run.summary["trainer_hash"] = git_info["trainer_hash"]
        wandb.run.summary["peft_hash"] = git_info["peft_hash"]
        print(f"[GIT] Trainer: {git_info['trainer_hash']} | PEFT: {git_info['peft_hash']}")