from .jelly_base import JellyBaseTrainer
import torch
import torch.nn as nn


class JellyNLUTrainer(JellyBaseTrainer):
    def compute_task_loss(self, logits, labels):
        # Regression check
        if labels.dtype in [torch.float32, torch.float64]:
            loss_fct = nn.MSELoss()
            return loss_fct(logits.view(-1), labels.view(-1))
        else:
            # Classification
            return super().compute_task_loss(logits, labels)
