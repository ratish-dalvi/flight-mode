import math
import torch
from torch import nn


class MultiHeadMaskedAttention(nn.Module):

    def __init__(self, embedding_size, num_heads):
        """
        Initialize the MultiHeadAttention layer.

        Args:
            embedding_size (int): The size of the input embedding. Also known as d_model.
                It must be divisible by `num_heads`
            num_heads (int): The number of attention heads.
        """
        super().__init__()

        self.keys_linear = nn.Linear(embedding_size, embedding_size, bias=False)
        self.queries_linear = nn.Linear(embedding_size, embedding_size, bias=False)
        self.values_linear = nn.Linear(embedding_size, embedding_size, bias=False)
        self.output_linear = nn.Linear(embedding_size, embedding_size)

        self.num_heads = num_heads


    def forward(self, x):
        """
        Forward pass for the MultiHeadAttention layer.

        Args:
            X (torch.Tensor): Input tensor with shape (batch_size, seq_length, embedding_size)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, embedding_size)
        """
        batch_size, seq_length, embedding_size = x.size()
        num_heads = self.num_heads

        # Apply the linear layer of size embedding_size * embedding_size
        K = self.keys_linear(x)
        Q = self.queries_linear(x)
        V = self.values_linear(x)

        # For multi-head attention, split it into separate heads
        embedding_split = embedding_size // num_heads

        # Reshape: split the embedding into `num_heads` parts
        K = K.view(batch_size, seq_length, num_heads, embedding_split)
        Q = Q.view(batch_size, seq_length, num_heads, embedding_split)
        V = Q.view(batch_size, seq_length, num_heads, embedding_split)

        # Create 3D tensors containing arrays of seq_length * embedding_split arrays.
        # The easiest way is to swap 2nd and 3rd dimensions, and reshape them to flatten
        # the first two dimensions (which are batch_size, and num_heads after the swap)
        K = K.transpose(1, 2).reshape(batch_size * num_heads, seq_length, embedding_split)
        Q = Q.transpose(1, 2).reshape(batch_size * num_heads, seq_length, embedding_split)
        V = V.transpose(1, 2).reshape(batch_size * num_heads, seq_length, embedding_split)

        # Compute scaled dot product
        scaled_dot_product = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(embedding_size)

        # Mask everything above the diagonal to prevent lookahead.
        assert scaled_dot_product.size() == (batch_size * num_heads, seq_length, seq_length)
        idxs = torch.triu_indices(seq_length, seq_length, offset=1)
        scaled_dot_product[..., idxs[0], idxs[1]] = float('-inf')

        attention_probs = torch.softmax(scaled_dot_product, dim=2)
        attn_out = torch.bmm(attention_probs, V)

        # Convert back to the original size and send through a linear layer
        attn_out = attn_out.view(batch_size, num_heads, seq_length, embedding_split)
        attn_out = attn_out.transpose(1, 2)
        attn_out = attn_out.reshape(batch_size, seq_length, num_heads * embedding_split)

        return self.output_linear(attn_out)


if __name__ == "__main__":
    mha = MultiHeadMaskedAttention(embedding_size=4, num_heads=2)
    X = torch.rand(6, 4, 4)    
    out = mha.forward(X)
    print(out.shape)
