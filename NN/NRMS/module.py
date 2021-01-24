import torch
import logging
import math
import numpy as np

from torch import nn
from torch.nn.functional import softmax

class NRMS(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.news_encoder = NewsEncoder(
            vocab_size=config['model']['vocab_size'],
            embed_size=config['model']['embed_size'],
            news_seq_len=config['preprocess']['news_max_len'],
            hidden_size=config['model']['hidden_size'],
            attention_head_num=config['model']['attention_head_num'],
            attention_size=config['model']['attention_size'],
            dropout=config['model']['dropout']
            )
        self.user_encoder = UserEncoder(
            hist_seq_len=config['preprocess']['hist_max_len'],
            hidden_size=config['model']['hidden_size'],
            attention_head_num=config['model']['attention_head_num'],
            attention_size=config['model']['attention_size']
            )

    def forward(self, can_news, can_word_mask, user_hist_news, user_hist_word_mask, user_hist_news_mask):
        '''
        Args:
            can_news: (batch_size, news_seq_len)
            can_word_mask: (batch_size, news_seq_len)
            user_hist_news: (batch_size, hist_seq_len, news_seq_len)
            user_hist_word_mask: (batch_size, hist_seq_len, news_seq_len)
            user_hist_news_mask: (batch_size, hist_seq_len)
        Returns:
            scores: (batch_size)
        '''
        can_news_embedding = self.news2embed(can_news, can_word_mask)  # (batch_size, hidden_size)
        user_embedding = self.user2embed(user_hist_news, user_hist_word_mask, user_hist_news_mask)  # (batch_size, hidden_size)
        scores = (can_news_embedding * user_embedding).sum(dim=1)

        return scores

    def news2embed(self, can_news, can_word_mask):
        '''
        Args:
            can_news: (batch_size, news_seq_len)
        Returns:
            can_word_mask: (batch_size, hidden_size)
        '''
        return self.news_encoder(can_news, can_word_mask)
    
    def user2embed(self, user_hist_news, user_hist_word_mask, user_hist_news_mask):
        '''
        Args:
            user_hist_news: (batch_size, hist_seq_len, news_seq_len)
            user_hist_word_mask: (batch_size, hist_seq_len, news_seq_len)
            user_hist_news_mask: (batch_size, hist_seq_len)
        Returns:
            user_embedding: (batch_size, hidden_size)
        '''
        batch_size, hist_seq_len, news_seq_len = user_hist_news.shape
        user_hist_news = user_hist_news.view(-1, news_seq_len)  # (batch_size * hist_seq_len, news_seq_len)
        user_hist_word_mask = user_hist_word_mask.view(-1, news_seq_len)  # (batch_size * hist_seq_len, news_seq_len)
        user_hist_news_embedding = self.news_encoder(user_hist_news, user_hist_word_mask).\
            view(batch_size, hist_seq_len, -1)  # (batch_size, hist_seq_len, hidden_size)
        user_embedding = self.user_encoder(user_hist_news_embedding, user_hist_news_mask)  # (batch_size, hidden_size)

        return user_embedding

class UserEncoder(nn.Module):
    def __init__(self, hist_seq_len, hidden_size, attention_head_num, attention_size):
        super().__init__()
        self.pos_encoding = generate_pos_encoding(hidden_size, hist_seq_len)
        self.self_attention = SelfAttention(hidden_size, hidden_size, attention_head_num)
        self.attention = Attention(hidden_size, attention_size)
    
    def forward(self, input, input_mask):
        '''
        Args:
            input: (batch_size, hist_seq_len, hidden_size)
            input_mask: (batch_size, his_seq_len)
        Returns:
            output: (batch_size, hidden_size)
        '''
        input += self.pos_encoding   # (batch_size, his_seq_len, hidden_size)
        self_att_output = self.self_attention(input, input_mask)
        output = self.attention(self_att_output, input_mask)

        return output

class NewsEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, news_seq_len, hidden_size, attention_head_num, attention_size, dropout):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoding = generate_pos_encoding(embed_size, news_seq_len)
        self.dropout1 = nn.Dropout(p=dropout)
        self.self_attention = SelfAttention(embed_size, hidden_size, attention_head_num)
        self.dropout2 = nn.Dropout(p=dropout)
        self.attention = Attention(hidden_size, attention_size)
    
    def forward(self, input, input_mask):
        '''
        Args:
            input: (batch_size, seq_len)
            input_mask: (batch_size, seq_len)
        Returns:
            output: (batch_size, hidden_size)
        '''
        input = self.word_embedding(input) + self.pos_encoding
        input = self.dropout1(input)
        self_att_output = self.self_attention(input, input_mask)
        self_att_output = self.dropout2(self_att_output)
        output = self.attention(self_att_output, input_mask)

        return output

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
        # weight: (batch_size, attention_head_num, seq_len, seq_len)
        weight = torch.matmul(Q, K) / math.sqrt(self.attention_head_size)
        input_mask = input_mask.repeat(self.attention_head_num, 1).view(-1, batch_size, seq_len).permute(1, 0, 2)
        input_mask = input_mask * (-1e9)
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
        input_mask = input_mask * (-1e9)
        weight += input_mask
        weight = softmax(weight, dim=1)
        output = (input * weight.unsqueeze(2)).sum(dim=1)   # (batch_size, input_size)

        return output

def generate_pos_encoding(embed_size, seq_len):
    pos_encoding = np.array(
        [[pos / np.power(10000, 2 * (i // 2) / embed_size) for i in range(embed_size)]\
            for pos in range(seq_len)])
    pos_encoding[:, 0::2] = np.sin(pos_encoding[:, 0::2])
    pos_encoding[:, 0::2] = np.cos(pos_encoding[:, 0::2])
    pos_encoding = torch.from_numpy(pos_encoding).to(torch.float).unsqueeze(0)  # (1, seq_len, embed_size)
    pos_encoding = nn.Parameter(pos_encoding, requires_grad=False)

    return pos_encoding

if __name__ == '__main__':
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