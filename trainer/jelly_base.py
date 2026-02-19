from transformers import Trainer, get_scheduler
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
    - "dynamic": Start Sequential -> Switch when DynamicSwitchCallback triggers
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
        elif self.jelly_mode in ("seq2par", "dynamic") and self.switch_epoch <= 0:
            for layer in self.jelly_layers:
                layer.set_mode("parallel")
            self.has_switched = True
            print(f">>> [JELLY] Mode: PARALLEL (switch_epoch={self.switch_epoch}, LoRA-equivalent)")
        elif self.jelly_mode == "dynamic":
            for layer in self.jelly_layers:
                layer.set_mode("sequential")
            print(f">>> [JELLY] Mode: DYNAMIC (auto-switch via PSV + SES indicators)")
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
        # 0.0 = sequential, 1.0 = parallel, 0.5 = seq2par/dynamic (before switch)
        if self.jelly_mode == "parallel":
            return 1.0
        elif self.jelly_mode == "sequential":
            return 0.0
        else:  # seq2par or dynamic
            return 1.0 if self.has_switched else 0.5

    def _execute_switch(self, model, curr_epoch):
        """
        Execute the seq→par switch: merge weights, reset optimizer, restart LR.
        Shared by both seq2par (epoch-based) and dynamic (indicator-based) modes.
        """
        # 1. Weight Merge + Reinit (in each JellyLayer)
        for layer in self.jelly_layers:
            if hasattr(layer, 'switch_to_parallel_with_correction'):
                layer.switch_to_parallel_with_correction()
            else:
                layer.set_mode("parallel")

        # 2. Optimizer State Reset
        if self.optimizer is not None:
            print(f"\n>>> [RESET] Clearing Optimizer State for JELLY Adapters...")

            matched_params = 0
            has_state_count = 0
            reset_count = 0
            total_momentum_norm = 0.0

            for name, param in model.named_parameters():
                if param.requires_grad and ("jelly" in name or "lora" in name):
                    matched_params += 1
                    if param in self.optimizer.state:
                        has_state_count += 1
                        state = self.optimizer.state[param]
                        if "exp_avg" in state:
                            total_momentum_norm += state["exp_avg"].norm().item()
                        self.optimizer.state[param] = {}
                        reset_count += 1

            print(f">>> [RESET] Adapter params matched: {matched_params}, "
                  f"had optimizer state: {has_state_count}, "
                  f"reset: {reset_count}")
            print(f">>> [RESET] Total exp_avg norm before reset: {total_momentum_norm:.6f}")

            if reset_count == 0 and matched_params > 0:
                print(f">>> [WARN] Params matched but NO state found! "
                      f"optimizer.state has {len(self.optimizer.state)} entries. "
                      f"Possible param identity mismatch.")

            if HAS_WANDB and wandb.run is not None:
                wandb.log({
                    "switch/epoch": curr_epoch,
                    "switch/global_step": self.state.global_step,
                    "switch/optimizer_reset_count": reset_count,
                    "switch/momentum_norm_before_reset": total_momentum_norm,
                }, commit=False)
        else:
            print(f">>> [WARN] self.optimizer is None at switch point!")

        # 3. LR Schedule: keep original schedule (same as LoRA)
        # No restart — the existing warmup→decay continues uninterrupted.
        if self.lr_scheduler is not None:
            current_lr = self.lr_scheduler.get_last_lr()[0]
            print(f">>> [LR] Keeping original schedule (current_lr={current_lr:.6f})")

            if HAS_WANDB and wandb.run is not None:
                wandb.log({
                    "switch/lr_at_switch": current_lr,
                }, commit=False)

        print(f">>> [SWITCH] Epoch {curr_epoch:.2f}: Sequential -> Parallel "
              f"(merge-reinit + optimizer reset, LR unchanged)")
        self.has_switched = True

    def _get_dynamic_callback(self):
        """Find DynamicSwitchCallback from the trainer's attached instance."""
        if hasattr(self, '_dyn_callback_cache'):
            return self._dyn_callback_cache
        # Search in callback_handler
        for cb in self.callback_handler.callbacks:
            if type(cb).__name__ == "DynamicSwitchCallback":
                self._dyn_callback_cache = cb
                return cb
        self._dyn_callback_cache = None
        return None

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Override to capture adapter gradients after backward (before zero_grad)
        for Gradient Coherence dynamic switching.
        """
        if num_items_in_batch is not None:
            loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        else:
            loss = super().training_step(model, inputs)

        # Capture gradients for dynamic switch (after backward, before zero_grad)
        if self.jelly_mode == "dynamic" and not self.has_switched:
            dyn_cb = self._get_dynamic_callback()
            if dyn_cb is not None:
                dyn_cb.capture_gradients(
                    model, self.state.global_step, self.state.epoch or 0
                )

        return loss

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        self._ensure_layers_cached()

        # Set initial mode on first call
        if not self.mode_logged:
            self._apply_initial_mode()
            self._log_mode_to_wandb()
            self.mode_logged = True

        # Check for mode switch
        if not self.has_switched:
            curr_epoch = self.state.epoch or 0

            if self.jelly_mode == "seq2par":
                # Fixed epoch-based switch
                if curr_epoch >= self.switch_epoch:
                    self._execute_switch(model, curr_epoch)

            elif self.jelly_mode == "dynamic":
                # Indicator-based switch (check callback flag)
                dyn_cb = self._get_dynamic_callback()
                if dyn_cb is not None and dyn_cb.should_switch:
                    self._execute_switch(model, curr_epoch)

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
