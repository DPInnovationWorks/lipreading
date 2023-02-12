import torch.nn as nn
import torch
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # self.register_buffer('pe', pe)
        # fix the bug in ddp
        self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))


    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Backend(nn.Module):
    def __init__(self,d_model,vocab_size,num_heads,num_encoder_layers,num_decoder_layers,p_dropout) -> None:
        super(Backend, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.p_dropout = p_dropout
        self.vocab_size = vocab_size
        self.transformer = nn.Transformer(d_model=self.d_model,
            nhead=self.num_heads,num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,dropout=self.p_dropout,batch_first=True)
        self.pe = PositionalEncoding(d_model=self.d_model,dropout=self.p_dropout)
        self.embedding = nn.Embedding(self.vocab_size, embedding_dim=self.d_model)
        self.fc = nn.Linear(self.d_model,self.vocab_size)
        self.init_weights()
    
    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)
    
    def forward(self,src,src_padding_mask,tgt,tgt_padding_mask,tgt_mask):
        # src已经经过visual encoding
        src = self.pe(src)
        tgt = self.pe(self.embedding(tgt))
        output = self.transformer(src,tgt, 
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            tgt_mask=tgt_mask)
        output = self.fc(output)
        return output

    