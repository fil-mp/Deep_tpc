import torch
from torch import nn
from typing import Optional, Tuple
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from layers.tpc_block import TPCBlock

# Modified GPT2Block to include TPC components
class GPT2BlockWithTPC(GPT2Block):
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx)
        
        # Flag to determine if this block contains TPC components
        self.has_tpc_block = False
        
        # Add TPC Block if this layer is in the tpc_layers list
        if hasattr(config, 'tpc_layers') and layer_idx in config.tpc_layers:
            self.has_tpc_block = True
            self.tpc_block = TPCBlock(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        patch_embeddings: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ):
        # Call the original GPT2Block forward pass
        outputs = super().forward(
            hidden_states=hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        
        # Extract the hidden states from outputs
        hidden_states = outputs[0]
        
        # Apply TPC block if available and patch embeddings are provided
        if self.has_tpc_block and patch_embeddings is not None:
            tpc_outputs = self.tpc_block(
                hidden_states,
                patch_embeddings,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
            )
            hidden_states = tpc_outputs[0]
            if output_attentions and len(tpc_outputs) > 1:
                # Add cross-attention outputs to the existing outputs
                outputs = outputs + (tpc_outputs[1],)
        
        # Update the hidden states in the outputs
        outputs = (hidden_states,) + outputs[1:]
        
        return outputs