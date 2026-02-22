from transformers import Trainer
from typing import Dict

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class JellyBaseTrainer(Trainer):
    """
    JELLY Trainer with Task-Aware Subspace Initialization (TASI)

    Pipeline:
    1. Probe phase (steps 0 … probe_steps-1):
       - Square layers train in sequential mode (adapter_input = W_base · x)
       - Non-square layers train in parallel mode (same as LoRA)
    2. At step probe_steps: execute TASI
       - Pull-back + SVD purification → re-initialize A and B for every layer
       - Reset optimizer state (clear momentum built up during probe)
    3. Main training (steps probe_steps … end):
       - All layers in parallel mode (standard LoRA-style training)

    probe_steps=0 skips the probe phase entirely (starts parallel from step 0).
    """

    def __init__(self, *args, probe_steps=200, probe_init_scale=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.probe_steps = int(probe_steps)
        self.probe_init_scale = probe_init_scale  # None = auto √(1/d_in)
        self.probe_done = (probe_steps == 0)  # if 0, skip probe entirely
        self.jelly_layers = []
        self.loss_track = {}
        self.param_stats_logged = False
        self.mode_logged = False

    def _ensure_layers_cached(self):
        if self.jelly_layers:
            return
        for m in self.model.modules():
            if hasattr(m, "initialize_from_probe") and hasattr(m, "set_mode"):
                self.jelly_layers.append(m)

    def _apply_probe_mode(self):
        """Set probe-phase modes: sequential for square layers, parallel for non-square."""
        if self.probe_done:
            # probe_steps=0: all layers start in parallel
            for layer in self.jelly_layers:
                layer.set_mode("parallel")
            print(f">>> [TASI] probe_steps=0 → all {len(self.jelly_layers)} layers start parallel")
            return

        for layer in self.jelly_layers:
            if layer.is_square:
                layer.set_mode("sequential")
            else:
                layer.set_mode("parallel")

        n_seq = sum(1 for l in self.jelly_layers if l.mode == "sequential")
        n_par = sum(1 for l in self.jelly_layers if l.mode == "parallel")
        print(f">>> [TASI] Probe phase: {n_seq} sequential (square) + {n_par} parallel (non-square)"
              f" | TASI trigger at step {self.probe_steps}")

    def _execute_tasi(self, model):
        """Execute Task-Aware Subspace Initialization on all layers."""
        step = self.state.global_step
        print(f"\n{'='*60}")
        print(f">>> [TASI] Executing at step {step} (probe_steps={self.probe_steps})")

        n_initialized = 0
        for layer in self.jelly_layers:
            layer.initialize_from_probe(init_scale=self.probe_init_scale)
            n_initialized += 1

        # Reset optimizer state for all adapter params
        if self.optimizer is not None:
            reset_count = 0
            total_momentum_norm = 0.0
            for name, param in model.named_parameters():
                if param.requires_grad and ("lora" in name or "jelly" in name):
                    if param in self.optimizer.state:
                        state = self.optimizer.state[param]
                        if "exp_avg" in state:
                            total_momentum_norm += state["exp_avg"].norm().item()
                        self.optimizer.state[param] = {}
                        reset_count += 1
            print(f">>> [TASI] Optimizer reset: {reset_count} param states cleared"
                  f" (momentum_norm_before={total_momentum_norm:.4f})")

        self.probe_done = True

        if HAS_WANDB and wandb.run is not None:
            wandb.log({
                "tasi/trigger_step": step,
                "tasi/layers_initialized": n_initialized,
                "tasi/probe_steps": self.probe_steps,
            }, commit=False)

        print(f">>> [TASI] Done. {n_initialized} layers re-initialized → all parallel")
        print(f"{'='*60}\n")

    def _log_parameter_info_to_wandb(self):
        """Log parameter info to wandb."""
        if not HAS_WANDB or wandb.run is None:
            return
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        wandb.log({
            "stats/total_params": total,
            "stats/trainable_params": trainable,
            "stats/trainable_ratio_percent": 100 * (trainable / total),
        }, commit=False)

    def _log_config_to_wandb(self):
        """Save TASI config to wandb."""
        if not HAS_WANDB or wandb.run is None:
            return
        wandb.config.update({
            "tasi_probe_steps": self.probe_steps,
            "tasi_init_scale": self.probe_init_scale,
        }, allow_val_change=True)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        self._ensure_layers_cached()

        # Set probe modes on first call
        if not self.mode_logged:
            self._apply_probe_mode()
            self._log_config_to_wandb()
            self.mode_logged = True

        # Execute TASI when probe_steps reached
        if not self.probe_done:
            if self.state.global_step >= self.probe_steps:
                self._execute_tasi(model)

        # Forward
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) and "loss" in outputs else outputs[0]

        # Track mode: 0.0 = probe phase, 1.0 = main training
        self.loss_track = {
            "task_loss": loss.item(),
            "adapter_mode": 1.0 if self.probe_done else 0.0,
        }

        if not self.param_stats_logged:
            self._log_parameter_info_to_wandb()
            self.param_stats_logged = True

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float]) -> None:
        if self.model.training and self.loss_track:
            for k, v in self.loss_track.items():
                logs[k] = v
        super().log(logs)
