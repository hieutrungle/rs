import torch
import torch.nn as nn
from typing import Optional


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


class EfficientAttention(nn.Module):
    """
    Efficient attention mechanism avoiding redundancy
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, device: str = "cpu"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Use different attention patterns
        self.local_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads // 2, batch_first=True, device=device
        )

        self.global_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads // 2, batch_first=True, device=device
        )

        self.mix_projection = nn.Linear(hidden_dim * 2, hidden_dim, device=device)

        # initialize weights
        self.apply(initialize_shallow_transformer)

    def apply_mha_on_3d_input(self, mha_layer, query, key, value, mask=None):
        return mha_layer(query, key, value, attn_mask=mask)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):

        if len(x.shape) == 3:
            # Local attention (agent-to-agent)
            local_out, _ = self.local_attention(x, x, x, attn_mask=mask)
            # Global attention (to mean representation)
            global_context = x.mean(dim=-2, keepdim=True).expand_as(x)
            global_out, _ = self.global_attention(x, global_context, global_context)
        else:
            # use vmap to apply MHA on each env's observations
            x = x.squeeze(1)
            local_out, _ = torch.vmap(self.apply_mha_on_3d_input, in_dims=(None, 0, 0, 0, None))(
                self.local_attention, x, x, x, mask
            )
            global_context = x.mean(dim=-2, keepdim=True).expand_as(x)
            global_out, _ = torch.vmap(self.apply_mha_on_3d_input, in_dims=(None, 0, 0, 0))(
                self.global_attention, x, global_context, global_context
            )
            local_out = local_out.unsqueeze(1)
            global_out = global_out.unsqueeze(1)

        # Combine both attention outputs
        combined = torch.cat([local_out, global_out], dim=-1)
        return self.mix_projection(combined)
