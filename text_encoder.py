
import torch
import torch.nn as nn
from typing import Optional

class QuestionEncoder(nn.Module):
    """Embedding + stacked BiLSTM -> 768-D (concatenated last forward+backward from top layer)"""
    def __init__(self, vocab_size: int, emb_dim: int = 300, lstm_hidden: int = 384, lstm_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=emb_dim,
                            hidden_size=lstm_hidden,
                            num_layers=lstm_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout=dropout if lstm_layers > 1 else 0.0)

    def forward(self, token_ids: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        token_ids: (B, T)
        lengths: optional (B,) with real lengths for packing
        returns: (B, 2 * lstm_hidden)
        """
        emb = self.embedding(token_ids)

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, (h_n, c_n) = self.lstm(packed)
        else:
            _, (h_n, c_n) = self.lstm(emb)

        forward_top = h_n[-2]
        backward_top = h_n[-1]
        q_feat = torch.cat([forward_top, backward_top], dim=-1)
        return q_feat
