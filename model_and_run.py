import torch
import torch.nn as nn
from image_encoder import ImageEncoder
from text_encoder import QuestionEncoder
from typing import Optional, Tuple


class FusionBlock(nn.Module):
    def __init__(self, in_dim: int = 512 + 768, fused_dim: int = 512, dropout_p: float = 0.35):
        super().__init__()
        self.fc = nn.Linear(in_dim, fused_dim)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, img_feat: torch.Tensor, q_feat: torch.Tensor) -> torch.Tensor:
        x = torch.cat([img_feat, q_feat], dim=-1)
        x = self.fc(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class DecoderLSTM(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 300, hidden_size: int = 512, num_layers: int = 2, fused_dim: int = 512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.init_hidden_proj = nn.Linear(fused_dim, num_layers * hidden_size)
        self.init_cell_proj = nn.Linear(fused_dim, num_layers * hidden_size)

    def _init_states(self, fused: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = fused.size(0)
        h = self.init_hidden_proj(fused)
        c = self.init_cell_proj(fused)
        h = h.view(batch, self.num_layers, self.hidden_size).permute(1, 0, 2).contiguous()
        c = c.view(batch, self.num_layers, self.hidden_size).permute(1, 0, 2).contiguous()
        return h, c

    def forward(self, fused: torch.Tensor, answer_tokens: Optional[torch.Tensor] = None, teacher_forcing: bool = True, max_len: int = 20, sos_idx: int = 1, eos_idx: int = 2) -> torch.Tensor:
        device = fused.device
        h0, c0 = self._init_states(fused)
        batch = fused.size(0)

        if answer_tokens is not None:
            emb = self.embedding(answer_tokens)
            outputs, _ = self.lstm(emb, (h0, c0))
            logits = self.out(outputs)
            return logits
        else:
            generated = []
            input_token = torch.full((batch, 1), sos_idx, dtype=torch.long, device=device)
            hidden = (h0, c0)
            for t in range(max_len):
                emb = self.embedding(input_token)
                out, hidden = self.lstm(emb, hidden)
                logits = self.out(out.squeeze(1))
                next_token = logits.argmax(dim=-1, keepdim=True)
                generated.append(next_token)
                input_token = next_token
            generated = torch.cat(generated, dim=1)
            return generated


class VQAModel(nn.Module):
    def __init__(self, vocab_size_q: int, vocab_size_ans: int, q_emb_dim: int = 300, a_emb_dim: int = 300, q_lstm_hidden: int = 384, q_lstm_layers: int = 2, decoder_hidden: int = 512, decoder_layers: int = 2, pretrained_resnet: bool = True):
        super().__init__()
        self.image_encoder = ImageEncoder(pretrained=pretrained_resnet)
        self.question_encoder = QuestionEncoder(vocab_size=vocab_size_q, emb_dim=q_emb_dim, lstm_hidden=q_lstm_hidden, lstm_layers=q_lstm_layers)
        self.fusion = FusionBlock(in_dim=512 + 2 * q_lstm_hidden, fused_dim=512, dropout_p=0.35)
        self.decoder = DecoderLSTM(vocab_size=vocab_size_ans, emb_dim=a_emb_dim, hidden_size=decoder_hidden, num_layers=decoder_layers, fused_dim=512)

    def forward(self, images: torch.Tensor, question_tokens: torch.Tensor, question_lengths: Optional[torch.Tensor], answer_tokens: Optional[torch.Tensor] = None, teacher_forcing: bool = True, max_gen_len: int = 20, sos_idx: int = 1, eos_idx: int = 2):
        img_feat = self.image_encoder(images)
        q_feat = self.question_encoder(question_tokens, lengths=question_lengths)
        fused = self.fusion(img_feat, q_feat)

        if answer_tokens is not None:
            logits = self.decoder(fused, answer_tokens=answer_tokens, teacher_forcing=teacher_forcing)
            return logits
        else:
            generated = self.decoder(fused, answer_tokens=None, teacher_forcing=False, max_len=max_gen_len, sos_idx=sos_idx, eos_idx=eos_idx)
            return generated



if __name__ == "__main__":
    # hyperparams (change to match your vocab & tokens)
    vocab_size_q = 10000
    vocab_size_ans = 8000
    B = 2
    img_H = img_W = 224
    T_q = 12
    T_a = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VQAModel(vocab_size_q=vocab_size_q, vocab_size_ans=vocab_size_ans).to(device)

    dummy_images = torch.randn(B, 3, img_H, img_W, device=device)
    dummy_q_tokens = torch.randint(1, vocab_size_q, (B, T_q), device=device)
    dummy_q_lens = torch.full((B,), T_q, dtype=torch.long, device=device)
    dummy_a_tokens = torch.randint(1, vocab_size_ans, (B, T_a), device=device)

    # Training forward (logits)
    logits = model(dummy_images, dummy_q_tokens, dummy_q_lens, answer_tokens=dummy_a_tokens)
    print("Logits shape (B, T_a, vocab):", logits.shape)

    # Inference (greedy)
    preds = model(dummy_images, dummy_q_tokens, dummy_q_lens, answer_tokens=None, max_gen_len=15)
    print("Generated tokens shape (B, max_len):", preds.shape)
