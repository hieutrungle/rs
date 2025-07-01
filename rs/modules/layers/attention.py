from torch import nn
import torch
from torch.nn import functional as F
import numpy as np
import math


# class SharedAgentEmbedding(nn.Module):
#     def __init__(self, obs_dim: int, embed_dim: int, max_agents: int):
#         super().__init__()
#         self.obs_dim = obs_dim
#         self.embed_dim = embed_dim
#         self.max_agents = max_agents

#         # Shared observation embedding
#         self.obs_embedding = nn.Sequential(
#             nn.Linear(obs_dim, embed_dim // 2), nn.ReLU(), nn.Linear(embed_dim // 2, embed_dim)
#         )

#         # Agent position encoding (optional)
#         self.pos_encoding = nn.Parameter(torch.randn(max_agents, embed_dim) * 0.1)

#     def forward(self, observations, agent_mask):
#         # observations: (batch, n_agents, obs_dim)
#         # agent_mask: (batch, n_agents) - True for real agents
#         batch_size, n_agents = observations.shape[:2]

#         # Apply shared embedding to all agents
#         embedded = self.obs_embedding(observations)  # (batch, n_agents, embed_dim)

#         # Add positional encoding (optional)
#         embedded = embedded + self.pos_encoding[:n_agents].unsqueeze(0)

#         # Zero out embeddings for padded agents
#         embedded = embedded * agent_mask.unsqueeze(-1)

#         return embedded


class MultiAgentAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, device="cpu"):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True, device=device
        )
        self.layer_norm = nn.LayerNorm(embed_dim, device=device)

    def forward(self, x: torch.Tensor, pre_mask: torch.Tensor = None, post_attn_mask=None):
        """
        x: embedding representations
            shape: batch size, n_agents, embedding dimension
        pre_mask: Which agents are not available (observability and/or padding).
                  Mask out before attention.
            shape: batch_size, n_agents, n_agents
        post_mask: Which agents are not available. Zero out their outputs to
                   prevent gradients from flowing back. Shape of 2nd dim determines
                   whether to compute queries for all agents.
            shape: batch size, n_agents

        Return shape: batch size, n_agents, embedding dimension
        """
        embed_dim = x.shape[-1]
        n_agents = x.shape[-2]
        batch_size = math.prod(x.shape[:-2])
        # batch_size, n_agents, embed_dim = x.shape
        # if the shape of x is greater than 3
        attn_output, attn_weights = self.attention(x, x, x, attn_mask=pre_mask)
        if post_attn_mask is not None:
            n_broadcast_dims = len(attn_output.shape) - 2
            attn_output = attn_output.masked_fill(
                post_attn_mask.reshape(batch_size, n_agents, *[1 for _ in range(n_broadcast_dims)]),
                0,
            )
        # Residual connection and layer norm
        attn_output = self.layer_norm(x + attn_output)
        return attn_output, attn_weights
