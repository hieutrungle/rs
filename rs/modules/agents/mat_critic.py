import torch
import torch.nn as nn
from typing import Optional


class MultiAgentTransformerCritic(nn.Module):
    """
    Multi-Agent Transformer Critic with intentionally shallow architecture
    Based on MAT (Multi-Agent Transformer)
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,  # Shallow by design
        max_agents: int = 10,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_agents = max_agents

        # Input embeddings
        self.obs_embedding = nn.Linear(obs_dim, hidden_dim)
        self.action_embedding = nn.Linear(action_dim, hidden_dim)

        # Positional encoding for agents
        self.agent_pos_embedding = nn.Parameter(torch.randn(max_agents, hidden_dim))

        # Transformer encoder layers (shallow)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,  # Wider instead of deeper
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Decoder for agent-specific Q-values
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )

        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output head
        self.q_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        agent_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through MAT-based centralized critic
        """
        batch_size, n_agents = observations.shape[:2]

        # Embed observations and actions
        obs_emb = self.obs_embedding(observations)
        action_emb = self.action_embedding(actions)

        # Combine embeddings
        combined_emb = obs_emb + action_emb

        # Add positional encoding
        combined_emb = combined_emb + self.agent_pos_embedding[:n_agents]

        # Create padding mask
        if agent_mask is not None:
            src_key_padding_mask = ~agent_mask
        else:
            src_key_padding_mask = None

        # Encoder: process global information
        encoded = self.transformer_encoder(combined_emb, src_key_padding_mask=src_key_padding_mask)

        # Decoder: generate agent-specific representations
        decoded = self.transformer_decoder(
            combined_emb,  # Target sequence
            encoded,  # Memory from encoder
            tgt_key_padding_mask=src_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )

        # Generate Q-values
        q_values = self.q_head(decoded)

        return q_values  # (batch_size, n_agents, 1)
