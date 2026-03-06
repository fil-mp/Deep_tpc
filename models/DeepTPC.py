import os
import torch
import torch.nn as nn
from transformers import GPT2Config

from layers.gpt2_model_tpc import GPT2ModelWithTPC
from layers.mlp import MLP


class Model(nn.Module):
    """Time‑series ⇄ GPT‑2 (with TPC blocks)

    * Each variable is treated as an independent "row" inside the batch (B·N).
    * `x_mark_enc` (e.g. time‑stamp / positional embedding) provides the **token stream**.
    * `times_embeds` (patches of the real signal) are delivered as *keys/values* through the
      TPC blocks (`patch_embeddings`).
    """

    def __init__(self, cfg):
        super().__init__()

        self.token_len = cfg.token_len
        self.mix       = cfg.mix_embeds
        # Ensure directory exists and contains config + weights
        ckpt_dir = os.path.abspath(cfg.llm_ckp_dir)
        # Auto-download GPT-2 if not present
        config_path = os.path.join(ckpt_dir, "config.json")
        model_path = os.path.join(ckpt_dir, "pytorch_model.bin")
        if not (os.path.exists(config_path) and os.path.exists(model_path)):
            raise FileNotFoundError(
                f"GPT-2 checkpoint not found at {ckpt_dir}. "
                "Download with: transformers.GPT2Model.from_pretrained('gpt2').save_pretrained(ckpt_dir)"
            )

        # Load and update GPT2 config with our TPC-specific and shared parameters
        base_cfg = GPT2Config.from_pretrained(cfg.llm_ckp_dir)
        base_cfg.update({
            'tpc_layers': cfg.tpc_layers,
            'num_fusion_tokens': cfg.num_fusion_tokens,
            'add_cross_attention': True,
            'layer_norm_epsilon': cfg.layer_norm_epsilon,
            'attn_pdrop': cfg.attn_pdrop,
            'resid_pdrop': cfg.resid_pdrop,
            'embd_pdrop': cfg.embd_pdrop
        })

        self.gpt2 = GPT2ModelWithTPC.from_pretrained(cfg.llm_ckp_dir, config=base_cfg)
        self.add_scale = nn.Parameter(torch.ones([]))

        # Freeze base GPT-2 weights, leaving only TPC blocks and fusion tokens trainable
        for name, p in self.gpt2.named_parameters():
            if "tpc_block" not in name and "fusion_tokens" not in name:
                p.requires_grad = False

        if cfg.mlp_hidden_layers == 0:
            self.encoder = nn.Linear(self.token_len, self.gpt2.config.hidden_size)
            self.decoder = nn.Linear(self.gpt2.config.hidden_size, self.token_len)
        else:
            self.encoder = MLP(
                self.token_len,
                self.gpt2.config.hidden_size,
                cfg.mlp_hidden_dim,
                cfg.mlp_hidden_layers,
                cfg.dropout,
                cfg.mlp_activation,
            )
            self.decoder = MLP(
                self.gpt2.config.hidden_size,
                self.token_len,
                cfg.mlp_hidden_dim,
                cfg.mlp_hidden_layers,
                cfg.dropout,
                cfg.mlp_activation,
            )


    def forecast(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None, prompt_text=None):

        # 1. normalise 
        means = x_enc.mean(1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_norm = (x_enc - means) / stdev

        B, T, N = x_norm.shape                      # batch, full length, variables

        # 2. fold each variable into non‑overlapping patches 
        x_var_first = x_norm.permute(0, 2, 1)       # [B, N, T]
        x_flat      = x_var_first.reshape(B * N, -1)  # [B·N, T]

        patches = x_flat.unfold(dimension=-1, size=self.token_len, step=self.token_len)
        token_num = patches.size(1)                 # how many patches per series

        # ‑‑ patch embeddings (keys/values for TPC)
        times_embeds = self.encoder(patches)        # [B·N, token_num, D]

        # 3. mark embeddings become the **token stream** 
        # x_mark_enc is expected already in shape [B·N, token_num, D]
        #x_mark_enc = x_mark_enc / x_mark_enc.norm(dim=2, keepdim=True)
        x_mark_enc = x_mark_enc / (x_mark_enc.norm(dim=2, keepdim=True))
        mark_tokens = self.add_scale * x_mark_enc
        # mark_tokens = 3.0 * x_mark_enc

        # 4. pass through GPT‑2
        gpt_out = self.gpt2(
            inputs_embeds=times_embeds,             # queries
            patch_embeddings=mark_tokens,         # keys/values to TPC block
            use_cache=False,
            return_dict=True,
            output_hidden_states=False,
            output_attentions=False,
        )

        hid = gpt_out["last_hidden_state"]          # [B·N, token_num, D]
        # 5. decode back to time domain
        dec_out = self.decoder(hid)                     # [B·N, token_num, token_len]
        dec_out = dec_out.reshape(B, N, -1)                 # [B, N, token_num*token_len]
        dec_out = dec_out.permute(0, 2, 1)                  # [B, token_num*token_len, N]

        # 6. denormalise
        dec_out = dec_out * \
            (stdev[:, 0, :].unsqueeze(1).repeat(1, token_num * self.token_len, 1))
        dec_out = dec_out + \
            (means[:, 0, :].unsqueeze(1).repeat(1, token_num * self.token_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None, prompt_text=None):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, prompt_text)

