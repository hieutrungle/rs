import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from rs.modules.layers import efficient_attention


def initialize_shallow_transformer(model: nn.Module):
    """
    Initialize shallow transformer for stable training
    """
    for name, param in model.named_parameters():
        if "weight" in name:
            if len(param.shape) >= 2:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.zeros_(param)
        elif "bias" in name:
            nn.init.zeros_(param)

    # Scale attention weights for stability
    for module in model.modules():
        if isinstance(module, nn.MultiheadAttention):
            module.in_proj_weight.data *= 0.1
            if module.in_proj_bias is not None:
                module.in_proj_bias.data *= 0.1


class AttentionCritic(nn.Module):
    """
    Centralized critic with attention mechanism for MARL
    Based on MAAC (Multi-Agent Actor-Attention-Critic)
    """

    def __init__(
        self,
        obs_dim: int,
        n_agents: int,
        # action_dims: List[int],
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 1,  # Intentionally shallow
        dropout: float = 0.0,
        device: str = "cpu",
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Shared observation encoder for all agents
        self.obs_encoder = nn.Sequential(
            nn.Linear(self.obs_dim, hidden_dim, device=device), nn.ReLU()
        )

        # Multi-head attention layers (deliberately shallow)
        self.attention_layers = nn.ModuleList(
            [
                efficient_attention.EfficientAttention(
                    hidden_dim=hidden_dim, num_heads=num_heads, device=device
                )
                for _ in range(num_layers)
            ]
        )

        # Layer normalization
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim, device=device) for _ in range(num_layers)]
        )

        # Individual critics for each agent
        self.critics = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, 1, device=device),
                )
                for _ in range(self.n_agents)
            ]
        )

        self.apply(initialize_shallow_transformer)

    def forward(
        self,
        observations: torch.Tensor,
        # actions: torch.Tensor,
        agent_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through attention-based centralized critic

        Args:
            observations: (batch_size, n_agents, obs_dim)
            actions: (batch_size, n_agents, action_dim)
            agent_mask: (batch_size, n_agents) - True for active agents

        Returns:
            q_values: (batch_size, n_agents, 1) - Q-values for each agent
        """

        # Encode observations
        obs_encoded = self.obs_encoder(observations)  # (batch, n_agents, hidden_dim)

        # Apply attention layers
        attended_features = obs_encoded
        for attention, norm in zip(self.attention_layers, self.layer_norms):
            # Self-attention across agents
            attn_output = attention(
                attended_features,
                mask=~agent_mask if agent_mask is not None else None,
            )

            # Residual connection and layer norm
            attended_features = norm(attended_features + attn_output)

        # Generate Q-values for each agent
        q_values = []
        for i, critic in enumerate(self.critics):
            q_val = critic(attended_features[..., i, :])
            q_values.append(q_val)
        q_values = torch.stack(q_values, dim=-2)  # (..., n_agents, 1)
        return q_values
