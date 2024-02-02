import torch

from .hookbase import HookBase


class CleanEmbedGradHook(HookBase):
    def __init__(self, save_embed_id):
        super().__init__()
        self.save_embed_id = save_embed_id

    def after_backward(self):
        for param in (
            self.trainer.model_or_module.text.get_text_encoder()
            .get_input_embeddings()
            .parameters()
        ):
            assert param.grad.shape[0] == len(
                self.trainer.model_or_module.text.tokenizer
            )
            # Keep other embeddings frozen.
            mask = torch.arange(param.grad.shape[0]) != self.save_embed_id
            param.grad[mask, :] = 0

    def after_step(self):
        with torch.no_grad():
            frozen_norm = torch.norm(
                self.trainer.model_or_module.text.get_text_encoder()
                .get_input_embeddins()
                .weight[:-1, :],
                dim=1,
            ).mean(0)
            trainable_weight = (
                self.trainer.model_or_module.text.get_text_encoder()
                .get_input_embeddins()
                .weight[-1, :]
            )
            self.trainer.model_or_module.text.get_text_encoder().get_input_embeddins().weight[
                -1, :
            ].div_(
                torch.norm(trainable_weight) / frozen_norm
            )
