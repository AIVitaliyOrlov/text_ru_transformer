import math

import torch
from torch import nn


class Embedding(nn.Module):
    """A class for the embedding layer.

    These embeddings are often used to represent textual data inputs.
    """

    def __init__(self, vocabulary_size: int, d_model: int):
        """Embedding layer initialization.

        Args:
            vocabulary_size: data vocabulary size (i.e. the number of embeddings to store)
            d_model: embedding dimension
        """
        super().__init__()
        self.d_model = d_model
        self.embeddings = nn.Embedding(vocabulary_size, d_model, padding_idx=0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass for the Embedding layer.

        Args:
            inputs: tensor of shape (batch size, sequence length) representing raw inputs data

        Returns:
            Tensor of shape (batch_size, sequence length, d_model) representing the inputs embeddings
        """
        embeddings = self.embeddings(inputs)
        return embeddings


class PatchEmbedding(nn.Module):
    """A class for the patch embedding layer.

    These embeddings are often used to represent image data inputs.
    """

    def __init__(self, d_model, input_channels, patch_size, num_patches, dropout_rate):
        super().__init__()

        self.d_model = d_model
        self.conv_projection = nn.Conv2d(
            in_channels=input_channels, out_channels=d_model, kernel_size=patch_size, stride=patch_size
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, d_model) * .02)

        self._init_weights()

    def _init_weights(self):
        """Weights initialization."""
        fan_in = self.conv_projection.in_channels * self.conv_projection.kernel_size[0] \
                 * self.conv_projection.kernel_size[1]
        nn.init.trunc_normal_(self.conv_projection.weight, std=math.sqrt(1 / fan_in))
        nn.init.zeros_(self.conv_projection.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass for the Embedding layer.

        Args:
            inputs: tensor of shape (batch size, n_channels, height, width) representing raw inputs data

        Returns:
            Tensor of shape (batch_size, number of patches + 1, d_model) representing the inputs embeddings
        """
        #  TODO: Implement patch embeddings forward pass:
        #        1. Pass inputs through convolution layer (self.conv_projection)
        #        2. Reshape the result to (batch size, number of patches, d_model)
        #        3. Repeat CLS token (self.cls_token) batch size times
        #        4. Concatenate (along second dim) repeated CLS token with passed through convolution and reshaped inputs
        #        5. Sum up the result and positional embeddings parameter (self.pos_embeddings)
        raise NotImplementedError
