import torch
from torch import nn
import torch.nn.functional as F
from ..model.Attention import SelfAttention


class BiLSTMAttention(nn.Module):
    """
    Bidirectional LSTM with Attention mechanism for sequence classification tasks.
    """

    def __init__(self, word_embedding_dimension: int = 300, hidden_dim: int = 128, num_layers: int = 1, dropout: float = 0.15,
                 bidirectional: bool = True, num_classes: int = 2):
        super(BiLSTMAttention, self).__init__()

        # LSTM parameters
        self.word_embedding_dimension = word_embedding_dimension
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.bidirectional = bidirectional
        self.num_classes = num_classes

        # BiLSTM layer
        self.encoder = nn.LSTM(word_embedding_dimension, hidden_dim, num_layers=num_layers, dropout=dropout,
                               bidirectional=bidirectional, batch_first=True)

        # Attention mechanism
        self.tanh1 = nn.Tanh()
        self.w = nn.Parameter(torch.Tensor(hidden_dim * 2, 1))  # hidden_dim * 2 because of bidirectional
        nn.init.uniform_(self.w, -0.1, 0.1)

        self.self_attention = SelfAttention(hidden_size=hidden_dim * 2, num_heads=8)

        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, features):
        token_embeddings = features['token_embeddings']
        sentence_lengths = torch.clamp(features['sentence_lengths'], min=1)

        # BiLSTM encoding
        packed = nn.utils.rnn.pack_padded_sequence(token_embeddings, sentence_lengths, batch_first=True,
                                                   enforce_sorted=False)
        packed, _ = self.encoder(packed)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(packed, batch_first=True)

        # Attention mechanism
        M = self.tanh1(unpacked)
        alpha = F.softmax(torch.matmul(M, self.w), dim=1)
        out = unpacked * alpha
        out = self.self_attention(out)
        out = self.dropout(out)
        out = torch.sum(out, 1)
        out = self.fc(out)

        return out

    def get_word_embedding_dimension(self) -> int:
        return self.hidden_dim * 2 if self.bidirectional else self.hidden_dim

