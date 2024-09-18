import torch
from torch import nn
from typing import List
from model.Attention import SelfAttention


class CNN(nn.Module):
    """CNN-layer with multiple kernel-sizes over the word embeddings"""

    def __init__(self, in_word_embedding_dimension: int = 300, out_channels: int = 128,
                 kernel_sizes: List[int] = [3, 5, 7], stride_sizes: List[int] = None, num_classes: int = 2,
                 dropout: float = 0.15):
        nn.Module.__init__(self)
        # self.config_keys = ['in_word_embedding_dimension', 'out_channels', 'kernel_sizes']
        self.in_word_embedding_dimension = in_word_embedding_dimension
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes

        self.embeddings_dimension = out_channels * len(kernel_sizes)
        self.convs = nn.ModuleList()

        in_channels = in_word_embedding_dimension
        if stride_sizes is None:
            stride_sizes = [1] * len(kernel_sizes)

        for kernel_size, stride in zip(kernel_sizes, stride_sizes):
            padding_size = int(kernel_size / 2)
            conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                             kernel_size=kernel_size,
                             stride=stride,
                             padding=padding_size)
            self.convs.append(conv)

        self.dropout = nn.Dropout(dropout)
        self.self_attention = SelfAttention(hidden_size=out_channels * len(kernel_sizes), num_heads=12)
        self.fc = nn.Linear(self.embeddings_dimension, num_classes)

    def forward(self, features):
        token_embeddings = features['token_embeddings']

        token_embeddings = token_embeddings.transpose(1, -1)
        vectors = [conv(token_embeddings) for conv in self.convs]
        out = torch.cat(vectors, 1).transpose(1, -1)

        features.update({'token_embeddings': out})
        out = self.self_attention(out)
        out = self.dropout(out)
        out = self.fc(out.mean(dim=1))
        return out

    def get_word_embedding_dimension(self) -> int:
        return self.embeddings_dimension

    def tokenize(self, text: str) -> List[int]:
        raise NotImplementedError()
