from rs.modules.agents import allocation_critics
import torch

torch.manual_seed(0)  # For reproducibility


def create_attn_mask_from_key_padding_mask(key_padding_mask, num_heads):
    batch_size, seq_length = key_padding_mask.shape
    attn_mask = torch.ones((batch_size, seq_length, seq_length), dtype=torch.bool)
    for i in range(batch_size):
        valid_len = seq_length - key_padding_mask[i].sum().item()
        # convert to int
        valid_len = valid_len.item() if isinstance(valid_len, torch.Tensor) else valid_len
        valid_len = int(valid_len)
        attn_mask[i, :valid_len, :valid_len] = False

    attn_mask = attn_mask.unsqueeze(1).expand(batch_size, num_heads, seq_length, seq_length)
    attn_mask = attn_mask.reshape(batch_size * num_heads, seq_length, seq_length)
    return attn_mask


def pad_to_attn_mask(pad_mask, num_heads):
    B, L, H = *pad_mask.shape, num_heads
    return pad_mask.view(B, 1, 1, L).expand(-1, H, L, -1).reshape(B * H, L, L)


if __name__ == "__main__":
    # Example usage of the MultiAgentAttentionAllocator
    obs_dim = 10  # Example observation dimension
    embed_dim = 6  # Embedding dimension
    n_agents = 4  # Number of agents
    n_tasks = 3  # Number of tasks
    num_heads = 2  # Number of attention heads
    device = "cpu"  # Device to run on

    allocator = allocation_critics.MultiAgentAttentionAllocator(
        obs_dim=obs_dim,
        embed_dim=embed_dim,
        num_heads=num_heads,
        n_agents=n_agents,
        n_tasks=n_tasks,
        device=device,
    )

    # Example input data
    batch_size = 1  # Number of samples in the batch
    observations = torch.randn(batch_size, n_agents, obs_dim, device=device)  # Random observations
    agent_mask = torch.zeros((batch_size, n_agents), device=device)
    agent_mask[:, 2:] = 1  # Example mask where only the first two agents are active
    agent_mask = agent_mask.bool()  # Convert to boolean mask
    attn_mask = pad_to_attn_mask(agent_mask, num_heads)

    outputs = allocator(observations, attn_mask, agent_mask)
    for key, value in outputs.items():
        print(f"{key}: {value.shape}")
        print(f"{key}: {value}")
