"""
Dynamic Switch Callback for JELLY Trainer — Gradient Coherence

Automatically determines the optimal Sequential → Parallel switch point
by tracking the cosine similarity (coherence) between consecutive gradients
of the adapter A weights.

Math:
    C_t = <∇A_t, ∇A_{t-1}> / (||∇A_t|| · ||∇A_{t-1}||)

- During efficient sequential learning: gradients are aligned → C_t ≈ 1
- When knowledge is saturated / overfitting begins: gradients become
  incoherent (random directions) → C_t drops sharply toward 0 or below
- The inflection point = optimal switch timing

Cost: Single dot product per step → effectively zero overhead.
"""

import torch
import torch.nn.functional as F

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class DynamicSwitchCallback:
    """
    Monitors gradient coherence and triggers seq→par switch when the
    adapter's learning signal degrades (coherence collapses).

    No fixed epoch threshold needed — the model tells us when it's done
    learning in sequential space.
    """

    def __init__(
        self,
        ema_alpha: float = 0.1,
        drop_ratio: float = 0.2,
        warmup_steps: int = 50,
        min_steps: int = 100,
        log_interval: int = 10,
    ):
        """
        Args:
            ema_alpha: Smoothing factor for coherence EMA. Lower = smoother.
            drop_ratio: Trigger switch when coherence_ema < peak * drop_ratio.
                0.2 = switch when coherence drops to 20% of its peak (80% decline).
            warmup_steps: Steps before starting to track peak coherence.
                Early gradients can be noisy; wait for stable signal.
            min_steps: Minimum steps before allowing switch.
            log_interval: Print/log coherence every N steps.
        """
        self.ema_alpha = ema_alpha
        self.drop_ratio = drop_ratio
        self.warmup_steps = warmup_steps
        self.min_steps = min_steps
        self.log_interval = log_interval

        # State
        self.prev_grad = None
        self.ema_coherence = None
        self.peak_coherence = None
        self.step_count = 0
        self.coherence_history = []

        # Control
        self.should_switch = False
        self.switch_step = None
        self.switch_epoch = None

    def capture_gradients(self, model, global_step, epoch):
        """
        Called from JellyBaseTrainer.training_step() after backward,
        before optimizer.step() / zero_grad().

        Captures adapter A gradients, computes coherence with previous step,
        and checks for switch condition.
        """
        if self.should_switch:
            return

        # Collect all sequential adapter A gradients into a single vector
        grad_vec = []
        for name, module in model.named_modules():
            if not hasattr(module, "mode") or module.mode != "sequential":
                continue
            if not hasattr(module, "lora_A"):
                continue
            for adapter_name in module.active_adapters:
                if adapter_name not in module.lora_A:
                    continue
                grad = module.lora_A[adapter_name].weight.grad
                if grad is not None:
                    grad_vec.append(grad.detach().float().flatten())

        if not grad_vec:
            return

        current_grad = torch.cat(grad_vec)
        self.step_count += 1

        if self.prev_grad is not None:
            # Cosine similarity between consecutive gradients
            cos_sim = F.cosine_similarity(
                current_grad.unsqueeze(0),
                self.prev_grad.unsqueeze(0),
            ).item()

            # Update EMA
            if self.ema_coherence is None:
                self.ema_coherence = cos_sim
            else:
                a = self.ema_alpha
                self.ema_coherence = a * cos_sim + (1 - a) * self.ema_coherence

            self.coherence_history.append(self.ema_coherence)

            # Track peak coherence (after warmup)
            if self.step_count >= self.warmup_steps:
                if self.peak_coherence is None:
                    self.peak_coherence = self.ema_coherence
                else:
                    self.peak_coherence = max(self.peak_coherence, self.ema_coherence)

            # Periodic logging
            if global_step % self.log_interval == 0:
                peak_str = (
                    f"{self.peak_coherence:.4f}" if self.peak_coherence is not None else "warmup"
                )
                threshold_str = (
                    f"{self.peak_coherence * self.drop_ratio:.4f}"
                    if self.peak_coherence is not None
                    else "N/A"
                )
                print(
                    f"  [GradCoherence] step={global_step} epoch={epoch:.2f} "
                    f"raw={cos_sim:.4f} ema={self.ema_coherence:.4f} "
                    f"peak={peak_str} trigger<{threshold_str}"
                )

                if HAS_WANDB and wandb.run is not None:
                    log_data = {
                        "dynamic_switch/coherence_raw": cos_sim,
                        "dynamic_switch/coherence_ema": self.ema_coherence,
                    }
                    if self.peak_coherence is not None:
                        log_data["dynamic_switch/coherence_peak"] = self.peak_coherence
                        log_data["dynamic_switch/trigger_threshold"] = (
                            self.peak_coherence * self.drop_ratio
                        )
                    wandb.log(log_data, commit=False)

            # Switch condition: coherence has collapsed relative to its peak
            if (
                self.step_count >= self.min_steps
                and self.peak_coherence is not None
                and self.peak_coherence > 0
            ):
                threshold = self.peak_coherence * self.drop_ratio
                if self.ema_coherence <= threshold:
                    self.should_switch = True
                    self.switch_step = global_step
                    self.switch_epoch = epoch
                    print(
                        f"\n{'='*60}\n"
                        f">>> [GRADIENT COHERENCE SWITCH] step={global_step} "
                        f"epoch={epoch:.3f}\n"
                        f"    coherence_ema  = {self.ema_coherence:.4f}\n"
                        f"    peak_coherence = {self.peak_coherence:.4f}\n"
                        f"    trigger_at     = {threshold:.4f} "
                        f"(peak × {self.drop_ratio})\n"
                        f"    total_checks   = {self.step_count}\n"
                        f"{'='*60}"
                    )

                    if HAS_WANDB and wandb.run is not None:
                        wandb.log(
                            {
                                "dynamic_switch/triggered_step": global_step,
                                "dynamic_switch/triggered_epoch": epoch,
                                "dynamic_switch/coherence_at_trigger": self.ema_coherence,
                                "dynamic_switch/peak_at_trigger": self.peak_coherence,
                            },
                            commit=False,
                        )

        self.prev_grad = current_grad.clone()
