from .jelly_base import JellyBaseTrainer


class JellyViTTrainer(JellyBaseTrainer):
    def _get_task_specific_inputs(self, inputs):
        # Keep only keys needed for ViT training
        vit_keys = {"pixel_values", "labels"}
        return {k: v for k, v in inputs.items() if k in vit_keys}
