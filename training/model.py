import torch
import torch.nn as nn


class MusicLSTM(nn.Module):
    """LSTM-based model for music token sequence generation."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        padding_idx: int = 0
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, hidden=None):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)
        # embedded shape: (batch_size, seq_len, embed_dim)
        out, hidden = self.lstm(embedded, hidden)
        # out shape: (batch_size, seq_len, hidden_dim)
        # Take the output at the last time step for predicting the next token
        last_step = out[:, -1, :]
        last_step = self.dropout(last_step)
        logits = self.fc(last_step)
        # logits shape: (batch_size, vocab_size)
        return logits, hidden
