import torch
from torch import nn
from multihead_attention import MultiHeadMaskedAttention
from torch.nn.functional import cross_entropy, softmax


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

        
    def forward(self, input_ids, labels=None):
        batch_size, context_length = input_ids.size()
        token_embedding = self.token_embedding(input_ids)  # size: (batch_size, context_length, embedding_size)

        p = torch.arange(0, context_length, dtype=torch.long, device=self.device).unsqueeze(0)
        # p has size (1, context_length)
        pos_embedding = self.pos_embedding(p)  # Size: (1, context_length, embedding_size)
        
        # Add position embedding to every sequence in batch
        x = token_embedding + pos_embedding  # # Size: (batch_size, context_length, embedding_size)

        x = self.blocks(x)
        x = self.layer_norm(x)
        logits = self.lm_head(x)
        
        # Get logits and y both into shape (batch_size * context_length, vocab_size)
        loss = 0
        if labels is not None:
            loss = cross_entropy(logits.view(-1, self.vocab_size), labels.view(-1))
            return {"loss": loss}

        return logits

    
    @torch.no_grad()
    def generate(self, x, max_tokens, temperature=1.0):
        self.eval()  # Ensure the model is in evaluation mode

        for _ in range(max_tokens):
            logits, _ = self(x)
            logits = logits[:, -1, :] / temperature
            probs = softmax(logits, dim=-1)
            _, x_next = torch.topk(probs, k=1, dim=-1)
            x = torch.cat((x, x_next), dim=1)
        return x    
