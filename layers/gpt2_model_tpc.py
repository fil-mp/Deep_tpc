import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers import GPT2Model
from layers.gpt2_tpc_block import GPT2BlockWithTPC

from transformers import GPT2Config
import os
from transformers.modeling_utils import load_state_dict



# Modified GPT2Model to work with patch embeddings and TPC blocks
class GPT2ModelWithTPC(GPT2Model):
    def __init__(self, config):
        # Initialize with the parent class constructor
        super().__init__(config)
        
        # Number of learnable fusion tokens to append
        self.num_fusion_tokens = getattr(config, 'num_fusion_tokens', 20)
        config.num_fusion_tokens = self.num_fusion_tokens  # Ensure this is stored in config
        
        # Learnable fusion tokens
        self.fusion_tokens = nn.Parameter(torch.zeros(1, self.num_fusion_tokens, config.hidden_size))
        # Initialize fusion tokens
        nn.init.normal_(self.fusion_tokens, std=0.02)
        
        # Replace the standard GPT2Blocks with our TPC-enhanced blocks
        self.h = nn.ModuleList([
            GPT2BlockWithTPC(config, layer_idx=i) 
            for i in range(config.num_hidden_layers)
        ])
        
        # Re-initialize weights for the model
        # self.init_weights()
    
    @classmethod
    def from_pretrained(cls, model_path: str, config: Optional[GPT2Config] = None, **kwargs):
        """
        Custom from_pretrained loader for GPT2ModelWithTPC.

        Args:
            model_path (str): Path to the checkpoint directory (must contain config.json and pytorch_model.bin).
            config (GPT2Config, optional): If provided, overrides loaded config.
            kwargs: Additional arguments passed to the constructor.

        Returns:
            GPT2ModelWithTPC: model instance with pretrained weights.
        """
        # 1. Load config from model path if not provided
        if config is None:
            config = GPT2Config.from_pretrained(model_path)

        # 2. Instantiate model with config
        model = cls(config, **kwargs)

        # 3. Load state dict from file
        weights_path = os.path.join(model_path, "pytorch_model.bin")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"No weights found at {weights_path}")

        state_dict = load_state_dict(weights_path)

        # Load pretrained weights with strict=False because the standard checkpoint 
        # won't contain our custom TPC blocks or fusion tokens.
        model.load_state_dict(state_dict, strict=False)
        
        # Initialize the new custom TPC components (fusion tokens, cross-attention layers, etc.)
        # that are not part of the standard pretrained GPT-2 weights.
        model._initialize_custom_components()
        return model
    
    def _initialize_custom_components(self):
        """Initialize custom TPC components that weren't in pretrained weights."""
        for i, block in enumerate(self.h):
            if hasattr(block, 'tpc_block') and block.tpc_block is not None:
                # Initialize TPC block components
                for name, module in block.tpc_block.named_modules():
                    if isinstance(module, nn.Linear):
                        nn.init.normal_(module.weight, std=0.02)
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)
                    elif isinstance(module, nn.LayerNorm):
                        nn.init.ones_(module.weight)
                        nn.init.zeros_(module.bias)
                    elif isinstance(module, nn.Parameter):
                        nn.init.normal_(module, std=0.02)
        
        nn.init.normal_(self.fusion_tokens, std=0.02)

        for name, param in self.named_parameters():
            if 'fusion_tokens' in name and param.requires_grad:
                nn.init.normal_(param, std=0.02)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        patch_embeddings: Optional[torch.FloatTensor] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('Cannot pass both input_ids and inputs_embeds')
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError('Must pass input_ids or inputs_embeds')

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)

        # ---------------- device‑safe position ids ---------------------
        if position_ids is None:
            device = (inputs_embeds if inputs_embeds is not None else self.wte.weight).device
            position_ids = torch.arange(past_length, input_shape[-1] + past_length,
                                        dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        position_embeds = self.wpe(position_ids)
        token_type_embeds = self.wte(token_type_ids) if token_type_ids is not None else 0

        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        hidden_states = self.drop(hidden_states)

        # ---------------- append fusion tokens -------------------------
        fusion_tokens_expanded = self.fusion_tokens.expand(batch_size, -1, -1)
        hidden_states_with_fusion = torch.cat([hidden_states, fusion_tokens_expanded], dim=1)

        # ---------------- causal mask (no prompt assumption) -----------
        T = hidden_states.size(1)
        F = self.num_fusion_tokens
        total_len = T + F
        device = hidden_states.device

        causal = torch.triu(torch.ones((T, T), device=device), diagonal=1).masked_fill_(
            torch.triu(torch.ones((T, T), device=device), diagonal=1) == 1, float('-inf'))
        upper_right = torch.full((T, F), float('-inf'), device=device)
        fusion_rows = torch.zeros((F, total_len), device=device)
        full_causal_mask = torch.cat([torch.cat([causal, upper_right], 1), fusion_rows], 0)
        full_causal_mask = full_causal_mask.unsqueeze(0).unsqueeze(1).expand(batch_size, 1, total_len, total_len)

        if attention_mask is not None:
            attn = attention_mask.unsqueeze(1).unsqueeze(2)
            fusion_vis = torch.ones((batch_size, 1, 1, F), device=device)
            pad_mask = torch.cat([attn, fusion_vis], dim=-1)
            full_causal_mask = full_causal_mask + (1.0 - pad_mask) * -10000.0

        attention_mask = full_causal_mask

        head_mask = self.get_head_mask(head_mask, self.config.n_layer)
        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None

        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states_with_fusion,)

            outputs = block(
                hidden_states_with_fusion,
                patch_embeddings=patch_embeddings,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            hidden_states_with_fusion = outputs[0]

            if use_cache:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

        hidden_states_with_fusion = self.ln_f(hidden_states_with_fusion)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states_with_fusion,)

        language_hidden_states = hidden_states_with_fusion[:, :-self.num_fusion_tokens, :]
        fusion_hidden_states = hidden_states_with_fusion[:, -self.num_fusion_tokens:, :]

        if not return_dict:
            return tuple(
                v for v in [language_hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return {
            'last_hidden_state': language_hidden_states,
            'fusion_hidden_states': fusion_hidden_states,
            'past_key_values': presents,
            'hidden_states': all_hidden_states,
            'attentions': all_self_attentions,
            'cross_attentions': all_cross_attentions,
        }