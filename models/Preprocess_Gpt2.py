import torch
import torch.nn as nn
from transformers import (
    GPT2Model,
    GPT2Tokenizer,
)

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.device = torch.device("cpu")

        
        self.gpt2 = GPT2Model.from_pretrained(configs.llm_ckp_dir)

 
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained(configs.llm_ckp_dir)
        self.gpt2_tokenizer.pad_token = self.gpt2_tokenizer.eos_token
        self.vocab_size = self.gpt2_tokenizer.vocab_size
        self.hidden_dim_of_gpt2 = 768
        
        for name, param in self.gpt2.named_parameters():
            param.requires_grad = False

    def tokenizer(self, x):
        output = self.gpt2_tokenizer(x, return_tensors="pt")['input_ids'].to(self.device)
        result = self.gpt2.get_input_embeddings()(output)
        return result   
    
    def forecast(self, x_mark_enc):        
        # x_mark_enc: [bs x T x hidden_dim_of_gpt2]
        # Tokenize all sequences first
        tokenized_sequences = [self.tokenizer(x_mark_enc[i]) for i in range(len(x_mark_enc))]
        
        # Find the maximum sequence length
        max_len = max(seq.shape[1] for seq in tokenized_sequences)
        
        # Pad all sequences to the same length
        padded_sequences = []
        for seq in tokenized_sequences:
            if seq.shape[1] < max_len:
                # Pad with zeros (embedding for padding token)
                pad_size = max_len - seq.shape[1]
                padding = torch.zeros(seq.shape[0], pad_size, seq.shape[2]).to(self.device)
                padded_seq = torch.cat([seq, padding], dim=1)
            else:
                padded_seq = seq
            padded_sequences.append(padded_seq)
        
        # Now concatenate the padded sequences
        x_mark_enc = torch.cat(padded_sequences, 0)
        text_outputs = self.gpt2(inputs_embeds=x_mark_enc)[0]
        text_outputs = text_outputs[:, -1, :]
        return text_outputs
    
    def forward(self, x_mark_enc):
        return self.forecast(x_mark_enc) 