from torch import nn
import torch
from torch.nn import functional as F
from rs.modules.layers import attention


class MultiAgentAttentionAllocator(nn.Module):
    """
    Multi-agent network with attention mechanism for task allocation
    """

    def __init__(
        self,
        obs_dim: int,
        embed_dim: int = 64,
        num_heads: int = 4,
        n_agents: int = 2,
        n_tasks: int = 2,
        device: str = "cpu",
    ):
        super().__init__()
        self.n_agents = n_agents
        self.n_tasks = n_tasks
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Shared observation embedding
        self.obs_embedding = nn.Sequential(
            nn.Linear(obs_dim, embed_dim // 2, device=device),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim, device=device),
        )

        self.agent_attention = attention.MultiAgentAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            max_agents=n_agents,
            device=device,
        )

        # Task allocation head
        self.allocation_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2, device=device),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, n_tasks, device=device),
            nn.Softmax(dim=-1),  # Probability distribution over tasks
        )

        # Value head for critic
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2, device=device),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1, device=device),
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim, device=device)

    def forward(self, observations, agent_mask=None, post_attn_mask=None):
        """Forward pass with attention mechanism"""
        batch_size, n_agents = observations.shape[:2]

        # Embed observations
        embedded = self.obs_embedding(observations)
        attended_features, attention_weights = self.agent_attention(
            embedded,
            pre_mask=agent_mask,  # Mask out unavailable agents before attention
            post_attn_mask=post_attn_mask,  # Zero out outputs for unavailable agents after attention, preventing gradients from flowing back
        )

        # Generate task allocation probabilities
        allocation_probs = self.allocation_head(attended_features)

        # Generate value estimates
        values = self.value_head(attended_features)

        return {
            "allocation_probs": allocation_probs,
            "values": values,
            "attention_weights": attention_weights,
            "embeddings": attended_features,
        }
