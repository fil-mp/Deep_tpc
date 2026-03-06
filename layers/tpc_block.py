import torch
import torch.nn as nn
from typing import Optional
from transformers.models.gpt2.modeling_gpt2 import GPT2MLP
from layers.gca import GatedCrossAttention

# TPC Block component that will be added to GPT2Block
class TPCBlock(nn.Module):
    """
    TPC Block that combines Gated Cross-Attention with a Feed-Forward Network.
    This is added to GPT2Block to enable cross-modal fusion with patch embeddings.
    """
    def __init__(self, config):
        super().__init__()
        self.ln_cross_attn = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.cross_attention = GatedCrossAttention(config)
        
        # Feed Forward layer after cross-attention (initialized from GPT2MLP)
        self.ln_ff = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(4 * config.hidden_size, config)
        
        # Gating parameter for FFW
        #for etth1 we keep it 0.5, most others 0.0
        self.gate_param_ff = nn.Parameter(torch.ones(1) * 0.5)
        
        # Store the number of fusion tokens for processing  
        self.num_fusion_tokens = config.num_fusion_tokens

    def forward(
        self,
        hidden_states: torch.Tensor,
        patch_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        # Process only fusion tokens with cross-attention
        # Assuming hidden_states = [language_tokens, fusion_tokens]
        # First split the hidden states
        language_tokens = hidden_states[:, :-self.num_fusion_tokens, :]
        fusion_tokens = hidden_states[:, -self.num_fusion_tokens:, :]
        
        # Apply cross-attention only to fusion tokens
        cross_attn_norm = self.ln_cross_attn(fusion_tokens)
        cross_attn_outputs = self.cross_attention(
            cross_attn_norm,
            patch_embeddings,
            attention_mask=None,
            output_attentions=output_attentions,
        )
        cross_attn_output = cross_attn_outputs[0]
        
        # Apply residual connection
        fusion_tokens = fusion_tokens + cross_attn_output
        
        # Combine back with language tokens
        hidden_states = torch.cat([language_tokens, fusion_tokens], dim=1)
        
        # Apply FFW to entire hidden states with gating
        residual = hidden_states
        ff_output = self.mlp(self.ln_ff(hidden_states))
        
        # Apply sigmoid gating to FFW
        gate = torch.sigmoid(self.gate_param_ff)
        gated_ff_output = gate * ff_output
        
        # Residual connection
        hidden_states = residual + gated_ff_output
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs += cross_attn_outputs[1:]
            
        return outputs