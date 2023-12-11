from torch import nn
from multihead_attention import MultiHeadMaskedAttention


class Block():
    """ A simple Transformer block, inspired by gpt2
    """

    def __init__(self, embedding_size, num_heads, dropout, hiddle_layer_scale):
        
        self.ln1 = nn.LayerNorm(embedding_size)
        self.ln2 = nn.LayerNorm(embedding_size)
        self.attn = MultiHeadMaskedAttention(embedding_size, num_heads)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_size, hiddle_layer_scale * embedding_size),
            nn.GELU(),
            nn.Linear(hiddle_layer_scale * embedding_size, embedding_size),
            dropout = nn.Dropout(dropout)
        )

        
    def forward(self, x):
        
        x = self.ln1(self.attn(x) + x)
        x = self.ln2(self.mlp(x) + x)
        
        return x
