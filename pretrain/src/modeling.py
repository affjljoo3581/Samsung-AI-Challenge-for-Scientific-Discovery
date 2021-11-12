from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from apex.normalization import FusedLayerNorm as MoTLayerNorm
except ModuleNotFoundError:
    from torch.nn import LayerNorm as MoTLayerNorm


@dataclass
class MoTConfig:
    num_embeddings: List[int]
    num_attention_types: int
    num_layers: int = 12
    hidden_dim: int = 768
    intermediate_dim: int = 3072
    num_attention_heads: int = 12
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    position_scale: float = 100.0
    initialize_range: float = 0.02


class PositionalEncoding(nn.Module):
    """A sinusoid-based positional encoding module.

    Args:
        hidden_dim: The dimensionality of hidden layers.
        position_scale: The scale of positions. It is regarded as a maximum position.
            But this argument does not exactly the same as limitation, because you can
            use longer values. Default is `10000.0`
    """

    def __init__(self, hidden_dim: int, position_scale: float = 10000.0):
        super().__init__()
        self.register_buffer(
            "inverse_freqs",
            position_scale ** (torch.arange(0, hidden_dim, 2) / hidden_dim),
        )

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        x = position_ids.unsqueeze(-1) / self.inverse_freqs
        x = torch.cat((x.sin(), x.cos()), dim=-1)
        return x.type_as(self.inverse_freqs)


class MoTEmbeddings(nn.Module):
    """Embeddings for MoT model.

    MoT supports multiple input embeddings and relative positional encodings. The
    relative position informations are encoded to sinusoidal vectors. Also you can add
    attention weights (attention types) to the relative attention informations.

    Args:
        config: A configuration dataclass object for the MoT model.
    """

    def __init__(self, config: MoTConfig):
        super().__init__()
        self.attention_type_embeddings = nn.Embedding(
            config.num_attention_types, config.hidden_dim // config.num_attention_heads
        )
        self.position_encoding = PositionalEncoding(
            config.hidden_dim // config.num_attention_heads, config.position_scale
        )

        self.input_embeddings = nn.ModuleList(
            nn.Embedding(num_embeddings, config.hidden_dim)
            for num_embeddings in config.num_embeddings
        )
        self.embedding_dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_ids: List[torch.Tensor],
        attention_type_ids: torch.Tensor,
        relative_position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_embeddings = []
        for x, embedding in zip(input_ids, self.input_embeddings):
            input_embeddings.append(embedding(x))

        input_embeddings = sum(input_embeddings)
        input_embeddings = self.embedding_dropout(input_embeddings)

        position_embeddings = self.position_encoding(relative_position_ids)
        attention_types = self.attention_type_embeddings(attention_type_ids)
        relative_attentions = position_embeddings + attention_types

        return input_embeddings, relative_attentions


class MoTAttention(nn.Module):
    """A self-attention layer for MoT model.

    This class is designed for relative self-attention. The relative attention
    representations can be positional embeddings, attention weights, or both.

    Args:
        config: A configuration dataclass object for the MoT model.
    """

    def __init__(self, config: MoTConfig):
        super().__init__()
        self.attention_dim = config.hidden_dim // config.num_attention_heads

        self.query_projection = nn.Linear(config.hidden_dim, config.hidden_dim, False)
        self.key_projection = nn.Linear(config.hidden_dim, config.hidden_dim, False)
        self.value_projection = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.output_projection = nn.Linear(config.hidden_dim, config.hidden_dim)

        self.attention_dropout = nn.Dropout(config.attention_dropout_prob)
        self.output_dropout = nn.Dropout(config.hidden_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        return x.unflatten(-1, (-1, self.attention_dim)).transpose(-3, -2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        relative_attentions: torch.Tensor,
    ) -> torch.Tensor:
        queries = self.query_projection(hidden_states)
        keys = self.key_projection(hidden_states)
        values = self.value_projection(hidden_states)

        # Instead of do matrix multiplication to high-dimensional relative attention
        # representations, we multiply to the query and key vectors. While they are
        # multiplied to the representations to calculate attention scores, this
        # association is mathematically equivalent.
        relative_queries = F.linear(queries, self.key_projection.weight.T)
        relative_keys = F.linear(keys, self.query_projection.weight.T)

        # Split and transpose the representation vectors into multi-heads. Since the
        # attention scores are calculated regarding to each attention head, the value
        # representation vectors should be splitted as well.
        (queries, keys, values, relative_queries, relative_keys) = map(
            self.transpose_for_scores,
            (queries, keys, values, relative_queries, relative_keys),
        )

        attention_scores = (
            torch.matmul(queries, keys.transpose(-1, -2))
            + torch.einsum("bhld,blrd->bhlr", relative_queries, relative_attentions)
            + torch.einsum("bhrd,blrd->bhlr", relative_keys, relative_attentions)
        )
        attention_scores = attention_scores / (3 * self.attention_dim) ** 0.5
        attention_scores = attention_scores + attention_mask

        attention_probs = self.attention_dropout(attention_scores.softmax(-1))
        attended = torch.matmul(attention_probs, values).transpose(-3, -2)
        attended = attended.contiguous().flatten(-2)

        return self.output_dropout(self.output_projection(attended))


class MoTFeedForward(nn.Sequential):
    """An intermediate feed-forward layer for MoT model.

    Since there is nothing special about this layer, it is implemented by using
    `nn.Sequential` simply.

    Args:
        config: A configuration dataclass object for the MoT model.
    """

    def __init__(self, config: MoTConfig):
        super().__init__(
            nn.Linear(config.hidden_dim, config.intermediate_dim),
            nn.GELU(),
            nn.Linear(config.intermediate_dim, config.hidden_dim),
            nn.Dropout(config.hidden_dropout_prob),
        )


class MoTLayer(nn.Module):
    """A transformer layer for MoT model.

    Note that the implementation of MoT transformer layer uses pre-layernorm. That is,
    layer normalizations are performed before the layers, not after them. And the
    residual skip-connections are not placed before the layer normalizations. The hidden
    representations (which are not passed to the normalizations yet) will be summed to
    the output of the layers (self-attention layer, feed-forward layer).

    Args:
        config: A configuration dataclass object for the MoT model.
    """

    def __init__(self, config: MoTConfig):
        super().__init__()
        self.attention_layernorm = MoTLayerNorm(config.hidden_dim)
        self.attention = MoTAttention(config)
        self.feedforward_layernorm = MoTLayerNorm(config.hidden_dim)
        self.feedforward = MoTFeedForward(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        relative_attentions: torch.Tensor,
    ) -> torch.Tensor:
        attention_output = self.attention_layernorm(hidden_states)
        attention_output = self.attention(
            attention_output, attention_mask, relative_attentions
        )
        hidden_states = hidden_states + attention_output

        feedforward_output = self.feedforward_layernorm(hidden_states)
        feedforward_output = self.feedforward(feedforward_output)
        hidden_states = hidden_states + feedforward_output

        return hidden_states


class MoTModel(nn.Module):
    """The base implementation of MoT model.

    MoT uses pre-layernorm structure, so all layer normalizations are shifted and
    performed after the below (or originally next) layers. For example, the first
    normalization (which should be in embedding layer originally) will be performed on
    the first transformer layer. So contrary to the other transformer models, this class
    performed the layer normalization to the output of entire transformer layers, not
    the embedding tensor.

    Args:
        config: A configuration dataclass object for the MoT model.
    """

    def __init__(self, config: MoTConfig):
        super().__init__()
        self.config = config
        self.embeddings = MoTEmbeddings(config)

        self.layers = nn.ModuleList(MoTLayer(config) for _ in range(config.num_layers))
        self.output_layernorm = MoTLayerNorm(config.hidden_dim)

        self.init_weights()

    def forward(
        self,
        input_ids: List[torch.Tensor],
        attention_mask: torch.Tensor,
        attention_type_ids: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        if position_ids.dim() == 2:
            position_ids = position_ids.unsqueeze(-1)

        # Calculate relative positions by using euclidean distance. Using this, you can
        # measure for multidimentional positions (e.g. 2D and 3D)
        relative_position_ids = position_ids.unsqueeze(1) - position_ids.unsqueeze(2)
        relative_position_ids = relative_position_ids.square().sum(-1).sqrt()

        hidden_states, relative_attentions = self.embeddings(
            input_ids, attention_type_ids, relative_position_ids
        )

        if attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(-3)
        elif attention_mask.dim() == 2:
            attention_mask = attention_mask.unsqueeze(-2).unsqueeze(-2)

        # Change the type of masking tensor and make the tensor to be masks of attention
        # scores (softmax-logits), not attention probabilities. Note that the minimum
        # masking value is `-10000`, because of half-precision (fp16) overflow problem.
        attention_mask = attention_mask.type_as(hidden_states)
        attention_mask = -10000.0 * (1.0 - attention_mask)

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, relative_attentions)

        # As mentioned above, MoT uses pre-layernorm so all layer normalizations are
        # shifted to the belows. Since the normalization of embeddings is moved to the
        # first transformer layer, one of the last transformer layer is shifted and
        # performed here.
        return self.output_layernorm(hidden_states)

    @torch.no_grad()
    def init_weights(self, module: Optional[nn.Module] = None):
        if module is not None:
            if isinstance(module, nn.Linear):
                module.weight.normal_(mean=0.0, std=self.config.initialize_range)
                if module.bias is not None:
                    module.bias.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.normal_(mean=0.0, std=self.config.initialize_range)
                if module.padding_idx is not None:
                    module.weight[module.padding_idx].zero_()
            elif isinstance(module, MoTLayerNorm):
                module.weight.fill_(1.0)
                module.bias.zero_()
        else:
            # If the module is not specified, then apply the initialization to all
            # layers in this model.
            self.apply(self.init_weights)
            return
