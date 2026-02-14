from .jelly_base import JellyBaseTrainer


class JellyCsTrainer(JellyBaseTrainer):
    """JELLY Trainer for Commonsense Reasoning tasks (Llama-based sequence classification)"""

    def _get_task_specific_inputs(self, inputs):
        cs_keys = {"input_ids", "attention_mask", "labels"}
        return {k: v for k, v in inputs.items() if k in cs_keys}

    def compute_loss(self, model, inputs, return_outputs=False):
        # Filter to CS-relevant keys only
        filtered = self._get_task_specific_inputs(inputs)
        return super().compute_loss(model, filtered, return_outputs=return_outputs)
