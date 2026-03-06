import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Union, List
from transformers.models.gpt2.modeling_gpt2 import GPT2Model, GPT2Block, GPT2Attention, GPT2MLP
from transformers import GPT2Tokenizer, GPT2Config

# Gated Cross-Attention module for TPC blocks
class GatedCrossAttention(nn.Module):
    """
    Gated Cross-Attention module for TPC fusion as described in DeepMLF paper.
    This module allows fusion tokens to attend to patch embeddings.
    """
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        
        # Cross-attention projections
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.o_proj = nn.Linear(self.embed_dim, self.embed_dim)
        
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        
        # Gating parameter (initialized to 0.5 as per paper)
        self.gate_param = nn.Parameter(torch.ones(1) * 0.5)
        
    def _attn(self, query, key, value, attention_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        
        attn_weights = attn_weights / math.sqrt(self.head_dim)
        
        # Apply the attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
            
        attn_output = torch.matmul(attn_weights, value)
        
        return attn_output, attn_weights
        
    def forward(
        self,
        query_states: torch.Tensor,  # Fusion tokens
        key_value_states: torch.Tensor,  # Patch embeddings
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        batch_size = query_states.size(0)
        
        # Project query (from fusion tokens)
        query = self.q_proj(query_states)
        # Project key/value (from patch embeddings)
        key = self.k_proj(key_value_states)
        value = self.v_proj(key_value_states)
        
        # Reshape for multi-head attention
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention
        attn_output, attn_weights = self._attn(query, key, value, attention_mask)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        
        # Project output
        attn_output = self.o_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        
        # Apply sigmoid gating as per paper
        gate = torch.sigmoid(self.gate_param)
        gated_output = gate * attn_output
        
        outputs = (gated_output,)
        if output_attentions:
            outputs += (attn_weights,)
            
        return outputs