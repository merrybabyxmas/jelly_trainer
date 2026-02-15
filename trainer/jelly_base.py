from transformers import Trainer
from typing import Dict

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class JellyBaseTrainer(Trainer):
    """
    JELLY Trainer with configurable adapter mode

    Modes:
    - "parallel": Start with Parallel mode (same as LoRA)
    - "sequential": Use Sequential mode throughout
    - "seq2par": Start Sequential -> Switch to Parallel at switch_epoch
    """

    def __init__(self, *args, jelly_mode="seq2par", switch_epoch=3, **kwargs):
        super().__init__(*args, **kwargs)

        self.jelly_mode = jelly_mode.lower()
        self.switch_epoch = float(switch_epoch)
        self.has_switched = False
        self.jelly_layers = []
        self.loss_track = {}
        self.param_stats_logged = False
        self.mode_logged = False

    def _ensure_layers_cached(self):
        if self.jelly_layers:
            return
        for m in self.model.modules():
            if hasattr(m, "set_mode"):
                self.jelly_layers.append(m)

    def _apply_initial_mode(self):
        """Set initial mode at training start"""
        if self.jelly_mode == "parallel":
            for layer in self.jelly_layers:
                layer.set_mode("parallel")
            self.has_switched = True
            print(f">>> [JELLY] Mode: PARALLEL (from start)")
        elif self.jelly_mode == "sequential":
            for layer in self.jelly_layers:
                layer.set_mode("sequential")
            print(f">>> [JELLY] Mode: SEQUENTIAL (fixed)")
        elif self.switch_epoch <= 0:
            # switch_epoch <= 0: parallel from start, no correction needed
            # (init_weights="lora" in config makes this identical to LoRA)
            for layer in self.jelly_layers:
                layer.set_mode("parallel")
            self.has_switched = True
            print(f">>> [JELLY] Mode: PARALLEL (switch_epoch={self.switch_epoch}, LoRA-equivalent)")
        else:  # seq2par (default)
            for layer in self.jelly_layers:
                layer.set_mode("sequential")
            print(f">>> [JELLY] Mode: SEQ2PAR (switch at epoch {self.switch_epoch})")

    def _log_parameter_info_to_wandb(self):
        """Log parameter info directly to wandb"""
        if not HAS_WANDB or wandb.run is None:
            return

        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        stats = {
            "stats/total_params": total,
            "stats/trainable_params": trainable,
            "stats/trainable_ratio_percent": 100 * (trainable / total)
        }

        for name, module in self.model.named_modules():
            if "JellyLayer" in str(type(module)):
                layer_p = sum(p.numel() for p in module.parameters() if p.requires_grad)
                if layer_p > 0:
                    clean_name = name.replace("base_model.model.", "")
                    stats[f"params_by_layer/{clean_name}"] = layer_p

        wandb.log(stats, commit=False)

    def _log_mode_to_wandb(self):
        """Save JELLY mode config to wandb"""
        if not HAS_WANDB or wandb.run is None:
            return

        wandb.config.update({
            "jelly_mode": self.jelly_mode,
            "switch_epoch": self.switch_epoch if self.jelly_mode == "seq2par" else None,
        }, allow_val_change=True)

    def _get_current_mode_value(self):
        """Return current mode as number (for wandb logging)"""
        # 0.0 = sequential, 1.0 = parallel, 0.5 = seq2par (before switch)
        if self.jelly_mode == "parallel":
            return 1.0
        elif self.jelly_mode == "sequential":
            return 0.0
        else:  # seq2par
            return 1.0 if self.has_switched else 0.5

    def compute_loss(self, model, inputs, return_outputs=False):
        self._ensure_layers_cached()

        # Set initial mode on first call
        if not self.mode_logged:
            self._apply_initial_mode()
            self._log_mode_to_wandb()
            self.mode_logged = True

        # Check for mode switch (seq2par only)
        if self.jelly_mode == "seq2par":
            curr_epoch = self.state.epoch or 0
            if curr_epoch >= self.switch_epoch and not self.has_switched:
                for layer in self.jelly_layers:
                    if hasattr(layer, 'switch_to_parallel_with_correction'):
                        layer.switch_to_parallel_with_correction()
                    else:
                        layer.set_mode("parallel")
                print(f"\n>>> [SWITCH] Epoch {curr_epoch:.2f}: Sequential -> Parallel (with weight correction)")
                self.has_switched = True

        # Forward
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) and "loss" in outputs else outputs[0]

        # Logging
        self.loss_track = {
            "task_loss": loss.item(),
            "adapter_mode": self._get_current_mode_value()
        }

        # Log parameter stats once
        if not self.param_stats_logged:
            self._log_parameter_info_to_wandb()
            self.param_stats_logged = True

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float]) -> None:
        if self.model.training and self.loss_track:
            for k, v in self.loss_track.items():
                logs[k] = v
        super().log(logs)
