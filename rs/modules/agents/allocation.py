from torch import nn
import torch
from torch.nn import functional as F
from rs.modules.layers import attention
import math


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


class GraphAttentionTaskAllocator(nn.Module):
    """Graph attention network for multi-agent task allocation"""

    def __init__(
        self,
        agent_state_dim=6,
        target_state_dim=6,
        embed_dim=128,
        num_heads=8,
        num_layers=3,
        device="cpu",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # State embeddings
        self.agent_encoder = nn.Sequential(
            nn.Linear(agent_state_dim, embed_dim, device=device), nn.ReLU()
        )

        self.target_encoder = nn.Sequential(
            nn.Linear(target_state_dim, embed_dim, device=device), nn.ReLU()
        )

        # Multi-head attention layers
        self.attention_layer = attention.MultiAgentAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            device=device,
        )

        # # Layer normalization
        # self.layer_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])

        # Pointer network for task selection
        self.pointer_network = PointerNetwork(
            embed_dim,
            embed_dim,
            device=device,
        )

        # Value network for critic
        self.value_network = nn.Sequential(nn.Linear(embed_dim, 1, device=device))

    def apply_mha_on_3d_input(self, embedded, mha_layer):
        return mha_layer(embedded)[0]  # [0] to get attn_output

    def forward(self, agent_states, target_states, compatibility_scores=None, mask=None):
        """Forward pass with attention mechanism"""
        # batch_size = agent_states.shape[0]
        n_agents = agent_states.shape[-2]
        n_targets = target_states.shape[-2]

        # Encode states
        agent_embeddings = self.agent_encoder(agent_states)  # (batch, n_agents, embed_dim)
        target_embeddings = self.target_encoder(target_states)  # (batch, n_targets, embed_dim)

        # Create combined node embeddings
        # (batch, n_agents + n_targets, embed_dim)
        all_embeddings = torch.cat([agent_embeddings, target_embeddings], dim=-2)

        # Apply attention layers
        attended_features = all_embeddings

        if len(attended_features.shape) == 3:
            # If observations are 3D, apply MHA directly
            attended_features = self.apply_mha_on_3d_input(attended_features, self.attention_layer)
        else:
            # use vmap to apply MHA on each env's observations
            attended_features = attended_features.squeeze(1)  # remove batch dimension
            # Apply MHA on each batch independently
            attended_features = torch.vmap(self.apply_mha_on_3d_input, in_dims=(0, None))(
                attended_features, self.attention_layer
            )
            attended_features = attended_features.unsqueeze(1)

        # Split back to agents and targets
        agent_features = attended_features[..., :n_agents, :]
        target_features = attended_features[..., n_agents:, :]

        # Generate allocation probabilities using pointer network
        allocation_logits = self.pointer_network(
            agent_features, target_features, compatibility_scores
        )

        # Generate value estimates
        global_features = torch.mean(agent_features, dim=-2)  # Global state representation
        values = self.value_network(global_features)

        return allocation_logits, values


class PointerNetwork(nn.Module):
    """Pointer network for task selection with attention mechanism"""

    def __init__(self, query_dim, key_dim, hidden_dim=128, device="cpu"):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, hidden_dim, device=device)
        self.key_proj = nn.Linear(key_dim, hidden_dim, device=device)
        # self.value_proj = nn.Linear(key_dim, hidden_dim, device=device)
        self.output_proj = nn.Linear(hidden_dim, 1, device=device)
        self.scale = math.sqrt(hidden_dim)

    def apply_bmm(self, Q, K):
        """
        Apply batch matrix multiplication to compute attention scores.
        Q: (batch, n_agents, hidden_dim)
        K: (batch, n_targets, hidden_dim)
        Returns:
            scores: (batch, n_agents, n_targets) - raw logits for task assignment
        """
        return torch.bmm(Q, K.transpose(1, 2)) / self.scale

    def forward(self, queries, keys, compatibility_scores=None):
        """
        queries: (batch, n_agents, query_dim)
        keys: (batch, n_targets, key_dim)
        compatibility_scores: (batch, n_agents, n_targets) or None
        Returns:
            scores: (batch, n_agents, n_targets) - raw logits for task assignment
        """
        # batch_size, n_agents, _ = queries.size()
        # n_targets = keys.size(1)

        # Project queries and keys
        Q = self.query_proj(queries)  # (batch, n_agents, hidden_dim)
        K = self.key_proj(keys)  # (batch, n_targets, hidden_dim)
        # V = self.value_proj(keys)  # (batch, n_targets, hidden_dim)

        # Compute scaled dot-product attention scores
        # Q: (batch, n_agents, hidden_dim)
        # K: (batch, n_targets, hidden_dim)
        # We want scores of shape (batch, n_agents, n_targets)
        if len(queries.shape) == 3:
            scores = self.apply_bmm(Q, K)
        else:
            Q = Q.squeeze(1)  # remove batch dimension
            K = K.squeeze(1)  # remove batch dimension
            scores = torch.vmap(self.apply_bmm, in_dims=(0, 0))(Q, K)
            scores = scores.unsqueeze(1)

        # Add compatibility scores if provided
        if compatibility_scores is not None:
            scores = scores + compatibility_scores

        # Optionally, softmax can be applied outside this module

        return scores
