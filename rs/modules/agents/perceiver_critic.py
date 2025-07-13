import torch
import torch.nn as nn
from typing import Optional


class PerceiverAttentionCritic(nn.Module):
    """
    Perceiver-based centralized critic for multi-agent systems
    Handles variable numbers of agents efficiently
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_latents: int = 64,
        num_heads: int = 8,
        num_layers: int = 2,
        max_agents: int = 10,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_latents = num_latents
        self.max_agents = max_agents

        # Latent array - learned global representations
        self.latent_array = nn.Parameter(torch.randn(num_latents, hidden_dim))

        # Input projections
        self.obs_proj = nn.Linear(obs_dim, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)

        # Cross-attention: latents attend to inputs
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )

        # Self-attention: latents attend to themselves
        self.self_attention_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
                for _ in range(num_layers)
            ]
        )

        # Layer normalization
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.self_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        # Output projection back to agent-specific values
        self.output_cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )

        self.output_norm = nn.LayerNorm(hidden_dim)

        # Final critic heads
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1)
        )

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        agent_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through Perceiver-based centralized critic
        """
        batch_size, n_agents = observations.shape[:2]

        # Project inputs
        obs_features = self.obs_proj(observations)
        action_features = self.action_proj(actions)

        # Combine observation and action features
        input_features = obs_features + action_features  # (batch, n_agents, hidden_dim)

        # Expand latent array for batch
        latents = self.latent_array.unsqueeze(0).expand(batch_size, -1, -1)

        # Cross-attention: latents attend to input features
        cross_attended, _ = self.cross_attention(
            latents,
            input_features,
            input_features,
            key_padding_mask=~agent_mask if agent_mask is not None else None,
        )
        latents = self.cross_norm(latents + cross_attended)

        # Self-attention among latents
        for self_attention, norm in zip(self.self_attention_layers, self.self_norms):
            self_attended, _ = self_attention(latents, latents, latents)
            latents = norm(latents + self_attended)

        # Output: project latents back to agent-specific representations
        agent_features = obs_features + action_features
        output_attended, _ = self.output_cross_attention(agent_features, latents, latents)

        final_features = self.output_norm(agent_features + output_attended)

        # Generate Q-values
        q_values = self.critic_head(final_features)

        return q_values  # (batch_size, n_agents, 1)
