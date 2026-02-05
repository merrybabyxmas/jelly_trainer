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

    # Set JellyAdapter global seed
    try:
        from peft.tuners.jelly.layer import JellyAdapter
        JellyAdapter.set_global_seed(seed)
    except ImportError:
        pass


def reset_jelly_generators(model, seed: int = None):
    """Reset all JellyAdapter generators in model (can be called at epoch start)"""
    try:
        from peft.tuners.jelly.layer import JellyAdapter
        for module in model.modules():
            if isinstance(module, JellyAdapter):
                if hasattr(module, 'reset_generator'):
                    module.reset_generator(seed)
    except ImportError:
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
    Callback to track best metric during training

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