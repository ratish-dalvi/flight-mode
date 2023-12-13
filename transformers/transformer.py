import torch
from torch import nn
from multihead_attention import MultiHeadMaskedAttention
from torch.nn.functional import cross_entropy


class Block(nn.Module):
    """ A simple Transformer block, inspired by gpt2
    """

    def __init__(self, embedding_size, num_heads, dropout, mult):
        super().__init__()

        self.ln1 = nn.LayerNorm(normalized_shape=embedding_size)
        self.ln2 = nn.LayerNorm(normalized_shape=embedding_size)
        self.attn = MultiHeadMaskedAttention(embedding_size, num_heads)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_size, mult * embedding_size),
            nn.GELU(),
            nn.Linear(mult * embedding_size, embedding_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):

        x = self.ln1(self.attn(x) + x)
        x = self.ln2(self.mlp(x) + x)

        return x

    
class Transformer(nn.Module):
    """ A basic Transformer using positional embeddings instead of encodings
    """
    def __init__(self, embedding_size, vocab_size, context_length, num_layers,
                 dropout, mult, num_heads, device):
        super().__init__()
        self.vocab_size = vocab_size
        self.device = device
        
        self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size)
        self.pos_embedding = nn.Embedding(num_embeddings=context_length, embedding_dim=embedding_size)
    
        blocks = [Block(embedding_size, num_heads, dropout, mult) for i in range(num_layers)]
        self.blocks = nn.Sequential(*blocks)
        self.layer_norm = nn.LayerNorm(embedding_size)        
        self.lm_head = nn.Linear(embedding_size, vocab_size)

        
    def forward(self, x, y):
        token_embedding = self.token_embedding(x)  # batch_size, seq_length, embedding_size
        batch_size, context_length = x.size()
        pos_embedding = self.pos_embedding(
            torch.arange(0, context_length, dtype=torch.long, device=self.device).unsqueeze(0)
        )  # batch_size, seq_length, embedding_size
        x = token_embedding + pos_embedding

        x = self.blocks(x)
        x = self.layer_norm(x)
        logits = self.lm_head(x)
        
        # Get logits and y both into shape (batch_size * context_length, vocab_size)
        loss = cross_entropy(logits.view(-1, self.vocab_size), y.view(-1))
        return logits, loss

