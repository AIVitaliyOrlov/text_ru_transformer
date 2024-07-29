import torch
import torch.nn as nn
from models.transformer import Encoder
from models.embeddings import PatchEmbedding


class VisionTransformerOutput(nn.Module):
    """Vision Transformer output layer."""

    def __init__(self, d_model, classes_num):
        super(VisionTransformerOutput, self).__init__()
        self.output_feed_forward = nn.Linear(d_model, classes_num)

        self._init_weights()

    def _init_weights(self):
        """Weights initialization."""
        nn.init.zeros_(self.output_feed_forward.weight)
        nn.init.zeros_(self.output_feed_forward.bias)

    def forward(self, inputs):
        """Forward pass for Vision Transformer output layer."""
        return self.output_feed_forward(inputs)


class VisionTransformer(nn.Module):
    """Vision Transformer model.

    A class for implementing Vision Transformer model from
        'An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale' (https://arxiv.org/abs/2010.11929).
    """

    def __init__(self, config, input_channels: int = 3, classes_num: int = 200):
        """Layers initialization."""
        super().__init__()
        self.config = config

        num_patches = (config.image_size // config.patch_size) ** 2
        self.embeddings = PatchEmbedding(
            config.d_model, input_channels, config.patch_size, num_patches, config.dropout_rate
        )

        self.encoder = Encoder(config)
        self.mlp_head = VisionTransformerOutput(config.d_model, classes_num)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Transformer forward pass.

        Args:
            inputs: tensor of shape (batch size, n_channels, height, width)

        Returns:
            Tensor of shape (batch size, classes num)
        """
        # TODO: Implement Vision Transformer forward pass:
        #       1. Get image embeddings with self.embeddings layer
        #       2. Pass embeddings through the model encoder
        #       3. Pass CLS token representation through the classification layer (self.mlp_head)
        raise NotImplementedError
