from torch import nn
import torch
from torch.nn import functional as F
from rs.modules.layers import attention


class MultiAgentAttentionCritics(nn.Module):
    """
    Multi-agent network with attention mechanism for task allocation
    """

    def __init__(
        self,
        obs_dim: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        n_agents: int = 2,
        device: str = "cpu",
    ):
        super().__init__()
        self.n_agents = n_agents
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Shared observation embedding
        self.obs_embedding = nn.Sequential(
            nn.Linear(obs_dim, embed_dim, device=device),
            nn.ReLU(),
        )

        self.position_embedding = nn.Embedding(
            num_embeddings=n_agents, embedding_dim=embed_dim, device=device
        )

        self.agent_attention = attention.MultiAgentAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            device=device,
        )

        # Value head for critic
        self.value_head = nn.Sequential(
            # nn.Linear(embed_dim, embed_dim // 2, device=device),
            # nn.ReLU(),
            nn.Linear(embed_dim, 1, device=device),
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim, device=device)

    def apply_mha_on_3d_input(self, embedded, mha_layer):
        return mha_layer(embedded)[0]  # [0] to get attn_output

    def forward(self, observations, agent_mask=None, post_attn_mask=None):
        """Forward pass with attention mechanism"""

        # Embed observations
        obs_embedded = self.obs_embedding(observations)
        # Shape: (1, n_agents)
        position_indices = torch.arange(self.n_agents, device=observations.device).unsqueeze(0)
        pos_embedded = self.position_embedding(position_indices)
        embedded = obs_embedded + pos_embedded  # Add position embedding
        if len(observations.shape) == 3:
            # If observations are 3D, apply MHA directly
            attended_features = self.apply_mha_on_3d_input(embedded, self.agent_attention)
        else:
            # use vmap to apply MHA on each env's observations
            embedded = embedded.squeeze(1)  # remove batch dimension
            # Apply MHA on each batch independently
            attended_features = torch.vmap(self.apply_mha_on_3d_input, in_dims=(0, None))(
                embedded, self.agent_attention
            )
            attended_features = attended_features.unsqueeze(1)

        # attended_features, _ = self.agent_attention(
        #     embedded,
        #     pre_mask=agent_mask,  # Mask out unavailable agents before attention
        #     post_attn_mask=post_attn_mask,  # Zero out outputs for unavailable agents after attention, preventing gradients from flowing back
        # )

        # Generate value estimates
        values = self.value_head(attended_features)

        return values

        # return {
        #     "values": values,
        #     "attention_weights": attention_weights,
        #     "embeddings": attended_features,
        # }
