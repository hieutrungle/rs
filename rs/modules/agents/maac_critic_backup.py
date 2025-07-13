import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional


class AttentionCritic(nn.Module):
    """
    Centralized critic with attention mechanism for MARL
    Based on MAAC (Multi-Agent Actor-Attention-Critic)
    """

    def __init__(
        self,
        obs_dims: List[int],
        action_dims: List[int],
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,  # Intentionally shallow
        dropout: float = 0.1,
    ):
        super().__init__()

        self.n_agents = len(obs_dims)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Shared observation encoder for all agents
        self.obs_encoder = nn.Sequential(
            nn.Linear(max(obs_dims), hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )

        # Shared action encoder for all agents
        self.action_encoder = nn.Sequential(
            nn.Linear(max(action_dims), hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
        )

        # Multi-head attention layers (deliberately shallow)
        self.attention_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=hidden_dim + hidden_dim // 2,  # obs + action
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )

        # Layer normalization
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim + hidden_dim // 2) for _ in range(num_layers)]
        )

        # Individual critics for each agent
        self.critics = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, 1),
                )
                for _ in range(self.n_agents)
            ]
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
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
        batch_size, n_agents = observations.shape[:2]

        # Encode observations and actions
        obs_encoded = self.obs_encoder(observations)  # (batch, n_agents, hidden_dim)
        action_encoded = self.action_encoder(actions)  # (batch, n_agents, hidden_dim//2)

        # Concatenate encoded observations and actions
        combined_features = torch.cat([obs_encoded, action_encoded], dim=-1)

        # Apply attention layers
        attended_features = combined_features
        for attention, norm in zip(self.attention_layers, self.layer_norms):
            # Self-attention across agents
            attn_output, attn_weights = attention(
                attended_features,
                attended_features,
                attended_features,
                key_padding_mask=~agent_mask if agent_mask is not None else None,
            )

            # Residual connection and layer norm
            attended_features = norm(attended_features + self.dropout(attn_output))

        # Generate Q-values for each agent
        q_values = []
        for i, critic in enumerate(self.critics):
            q_val = critic(attended_features[:, i])
            q_values.append(q_val)

        return torch.stack(q_values, dim=1)  # (batch_size, n_agents, 1)
