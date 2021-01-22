import torch
import logging
import math

from torch import nn
from torch.nn.functional import softmax

class SelfAttention(nn.Module):
    def __init__(self, embed_size, hidden_size, attention_head_num):
        super().__init__()
        assert hidden_size % attention_head_num == 0
        self.attention_head_num = attention_head_num
        self.attention_head_size = hidden_size // attention_head_num
        self.query = nn.Linear(embed_size, hidden_size)
        self.key = nn.Linear(embed_size, hidden_size)
        self.value = nn.Linear(embed_size, hidden_size)

    def forward(self, input, input_mask):
        '''
        Args:
            input: (batch_size, seq_len, embed_size)
            input_mask: (batch_size, seq_len)
        Returns:
            output: (batch_size, seq_len, hidden_size)
        '''
        batch_size, seq_len, _ = input.shape
        Q = self.query(input).view(batch_size, seq_len, -1, self.attention_head_size).permute(0, 2, 1, 3)
        K = self.key(input).view(batch_size, seq_len, -1, self.attention_head_size).permute(0, 2, 3, 1)  # transpose
        V = self.value(input).view(batch_size, seq_len, -1, self.attention_head_size).permute(0, 2, 1, 3)
        weight = torch.matmul(Q, K) / math.sqrt(self.attention_head_size)   # (batch_size, attention_head_num, seq_len, seq_len)
        input_mask = input_mask.repeat(self.attention_head_num, 1).view(-1, batch_size, seq_len).permute(1, 0, 2)
        input_mask *= -1e9
        weight += input_mask.unsqueeze(dim=2)
        weight = softmax(weight, dim=3)
        output = torch.matmul(weight, V).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)

        return output


class Attention(nn.Module):
    def __init__(self, input_size, attention_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_size, attention_size),
            nn.Tanh(),
            nn.Linear(attention_size, 1)
        )
    
    def forward(self, input, input_mask):
        '''
        Args:
            input: (batch_size, seq_len, input_size)
            input_mask: (batch_size, seq_len)
        Returns:
            output: (batch_size, input_size)
        '''
        weight = self.attention(input).squeeze()   # (batch_size, seq_len)
        input_mask *= -1e9
        weight += input_mask
        weight = softmax(weight, dim=1)
        output = (input * weight.unsqueeze(2)).sum(dim=1)   # (batch_size, input_size)

        return output


if __name__ == '__main__':
    import numpy as np

    batch_size = 2
    seq_len = 3
    embed_size = hidden_size = input_size = 8
    attention_head_num = 4
    attention_size = 8
    self_attention = SelfAttention(embed_size, hidden_size, attention_head_num)
    input = torch.rand(batch_size, seq_len, embed_size)
    input_mask = torch.tensor(np.random.randint(low=0, high=2, size=(batch_size, seq_len)), dtype=torch.float)
    self_attention(input, input_mask)
    attention = Attention(input_size, attention_size)
    attention(input, input_mask)