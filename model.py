

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from torchvision import transforms

from transformers import T5ForConditionalGeneration, T5TokenizerFast
from tqdm.auto import tqdm
from utils import compute_contrastive_loss
from dataloader import train_loader, val_loader
try:
    from torchvision.models import resnet18, ResNet18_Weights
    def make_resnet18(pretrained=True):
        if pretrained:
            try:
                return resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            except Exception:
                return resnet18(pretrained=True)
        else:
            return resnet18(weights=None)
except Exception:
    from torchvision.models import resnet18
    def make_resnet18(pretrained=True):
        return resnet18(pretrained=pretrained)

class ImageEncoderResNet18(nn.Module):
    """ResNet18 conv backbone returning pooled vector (B,C) or spatial tokens (B,V,C)."""
    def __init__(self, pretrained: bool = True, spatial: bool = True):
        super().__init__()
        self.spatial = spatial
        base = make_resnet18(pretrained=pretrained)
        modules = list(base.children())[:-2]  
        self.features = nn.Sequential(*modules)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.out_dim = 512
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)             
        B, C, H, W = feats.shape
        if not self.spatial:
            pooled = self.pool(feats).view(B, C)
            return pooled
        tokens = feats.view(B, C, H*W).permute(0, 2, 1).contiguous()  
        return tokens

class QFormer(nn.Module):
    """
    Tiny Q-former: learnable M queries cross-attend to image patch tokens.
    Uses multihead attention where query dim = q_dim and keys/vals are projected to q_dim.
    """
    def __init__(self, n_q: int = 4, q_dim: int = 256, kv_dim: int = 512,
                 n_heads: int = 8, ff_dim: int = 1024, dropout: float = 0.1, num_layers: int = 1):
        super().__init__()
        self.n_q = n_q
        self.q_dim = q_dim
        self.kv_dim = kv_dim
        self.num_layers = num_layers

        self.queries = nn.Parameter(torch.randn(n_q, q_dim) * 0.02)

        self.kv_proj = nn.Linear(kv_dim, q_dim) if kv_dim != q_dim else None

        self.cross_attn_layers = nn.ModuleList()
        self.ffns = nn.ModuleList()
        for _ in range(num_layers):
            self.cross_attn_layers.append(
                nn.MultiheadAttention(embed_dim=q_dim, kdim=q_dim, vdim=q_dim,
                                      num_heads=n_heads, dropout=dropout, batch_first=True)
            )
            self.ffns.append(
                nn.Sequential(
                    nn.Linear(q_dim, ff_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(ff_dim, q_dim),
                    nn.Dropout(dropout),
                    nn.LayerNorm(q_dim)
                )
            )
        self.layernorm = nn.LayerNorm(q_dim)

    def forward(self, image_tokens: torch.Tensor) -> torch.Tensor:
        """
        image_tokens: [B, V, C_v]
        returns: q_out [B, M, q_dim]
        """
        B, V, C_v = image_tokens.shape
        q = self.queries.unsqueeze(0).expand(B, -1, -1).contiguous()  
        if self.kv_proj is not None:
            k = self.kv_proj(image_tokens) 
            v = k
        else:
            k = image_tokens
            v = k
        x_q = q
        for attn, ffn in zip(self.cross_attn_layers, self.ffns):
            attn_out, _ = attn(x_q, k, v, need_weights=False)  # [B, M, q_dim]
            x_q = x_q + attn_out
            x_q = x_q + ffn(x_q)
            x_q = self.layernorm(x_q)
        return x_q  # [B, M, q_dim]

class BLIP2_T5(nn.Module):
    """
    BLIP-2-like wiring:
      ResNet18 (frozen) -> spatial tokens -> QFormer -> project -> prepend to T5 encoder embeddings
      T5 encoder is frozen; train Q-former, projector, and T5 decoder.
    """
    def __init__(self, t5_name: str = "t5-small", q_n: int = 4, q_dim: int = 256,
                 q_num_layers: int = 1, freeze_img_encoder: bool = True, freeze_t5_encoder: bool = True,
                 pretrained_img: bool = True):
        super().__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_name)
        self.tokenizer = T5TokenizerFast.from_pretrained(t5_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
        self.d_model = self.t5.config.d_model

        self.img_encoder = ImageEncoderResNet18(pretrained=pretrained_img, spatial=True)

        self.q_former = QFormer(n_q=q_n, q_dim=q_dim, kv_dim=self.img_encoder.out_dim,
                                num_layers=q_num_layers)

        self.q_to_t5 = nn.Sequential(
            nn.Linear(q_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.Dropout(0.1)
        )

        # learned positional embeddings for image tokens
        self.img_token_pos = nn.Parameter(torch.randn(q_n, self.d_model) * 0.02)

        # freeze flags (recommended for 8GB)
        if freeze_img_encoder:
            for p in self.img_encoder.parameters():
                p.requires_grad = False
        if freeze_t5_encoder:
            for n, p in self.t5.get_encoder().named_parameters():
                p.requires_grad = False

    def encode_image(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert pixel_values -> img_embeds [B, M, d_model] and pooled bottleneck repr q_repr [B, d_model]
        """
        img_feats = self.img_encoder(pixel_values)   # [B, V, C_v]
        q_out = self.q_former(img_feats)             # [B, M, q_dim]
        img_embeds = self.q_to_t5(q_out)             # [B, M, d_model]
        img_embeds = img_embeds + self.img_token_pos.unsqueeze(0).to(img_embeds.device).to(img_embeds.dtype)
        q_repr = img_embeds.mean(dim=1)              # pooled bottleneck for contrastive
        return img_embeds, q_repr

    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> Tuple[object, Dict]:
        B = pixel_values.size(0)
        img_embeds, q_repr = self.encode_image(pixel_values)  # [B, M, d], [B, d]
        # get token embeddings from T5 encoder embed_tokens
        embed_tokens = self.t5.get_encoder().embed_tokens
        text_embeds = embed_tokens(input_ids)  # [B, Lq, d_model]
        # concat and build attention mask
        encoder_inputs_embeds = torch.cat([img_embeds, text_embeds], dim=1)  # [B, M+Lq, d]
        img_mask = torch.ones(B, img_embeds.size(1), dtype=attention_mask.dtype, device=attention_mask.device)
        encoder_attention_mask = torch.cat([img_mask, attention_mask], dim=1)  # [B, M+Lq]
        # run T5 encoder with inputs_embeds
        encoder_outputs = self.t5.get_encoder()(inputs_embeds=encoder_inputs_embeds,
                                                attention_mask=encoder_attention_mask,
                                                return_dict=True)
        outputs = self.t5(encoder_outputs=encoder_outputs,
                          attention_mask=encoder_attention_mask,
                          labels=labels,
                          return_dict=True)
        aux = {"encoder_outputs": encoder_outputs, "q_repr": q_repr, "img_embeds": img_embeds, "text_embeds": text_embeds}
        return outputs, aux

    def generate(self, pixel_values: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                 max_length: int = 64, num_beams: int = 3, **gen_kwargs):
        img_embeds, _ = self.encode_image(pixel_values)
        text_embeds = self.t5.get_encoder().embed_tokens(input_ids)
        encoder_inputs_embeds = torch.cat([img_embeds, text_embeds], dim=1)
        img_mask = torch.ones(pixel_values.size(0), img_embeds.size(1),
                              dtype=attention_mask.dtype, device=attention_mask.device)
        encoder_attention_mask = torch.cat([img_mask, attention_mask], dim=1)
        encoder_outputs = self.t5.get_encoder()(inputs_embeds=encoder_inputs_embeds,
                                                attention_mask=encoder_attention_mask,
                                                return_dict=True)
        generated_ids = self.t5.generate(encoder_outputs=encoder_outputs,
                                        attention_mask=encoder_attention_mask,
                                        max_length=max_length,
                                        num_beams=num_beams,
                                        **gen_kwargs)
        return generated_ids


# ---- Instantiate model, optimizer, sample training step (smoke) ----
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# instantiate model (this will download t5-small if not cached)
model_blip = BLIP2_T5(t5_name="t5-small", q_n=4, q_dim=256, q_num_layers=1,
                      freeze_img_encoder=True, freeze_t5_encoder=True, pretrained_img=True)
model_blip.to(device)

trainable = [p for p in model_blip.parameters() if p.requires_grad]
print("Trainable param groups:", sum(p.numel() for p in trainable), "total params:", sum(p.numel() for p in model_blip.parameters()))

# Example optimizer: train Q-former, q_to_t5, and T5 decoder (decoder params are trainable while encoder frozen)
optimizer = torch.optim.AdamW(trainable, lr=5e-5, weight_decay=0.01)

# Check train_loader existence and run a single combined step
if "train_loader" not in globals():
    print("train_loader not found. Model defined and ready. Run your dataloader cell and then use the training loop cell I'll provide next.")
else:
    batch = next(iter(train_loader))
    pixel_values = batch["pixel_values"].to(device)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    use_amp = (device == "cuda")
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    model_blip.train()
    with torch.cuda.amp.autocast(enabled=use_amp):
        outputs, aux = model_blip(pixel_values, input_ids, attention_mask, labels=labels)
        ce_loss = outputs.loss  # cross-entropy
        # text pooled repr: mean of token embeddings BEFORE encoder (we used embed_tokens)
        # project text embedding pooled to d_model (they already are d_model since we used embed_tokens)
        text_repr = aux["text_embeds"].mean(dim=1)  # [B, d_model]
        img_repr = aux["q_repr"]                    # [B, d_model]
        contrastive_loss = compute_contrastive_loss(img_repr, text_repr, temp=0.07)
        alpha = 0.1  # contrastive weight (tune: 0.05 - 0.2)
        loss = ce_loss + alpha * contrastive_loss

    # backward + step
    optimizer.zero_grad()
    if use_amp:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    print(f"CE loss: {float(ce_loss.item()):.4f}, contrastive: {float(contrastive_loss.item()):.4f}, total: {float(loss.item()):.4f}")
    # generation smoke test
    model_blip.eval()
    with torch.no_grad():
        gen_ids = model_blip.generate(pixel_values, input_ids, attention_mask, max_length=64, num_beams=3)
        decoded = model_blip.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        print("Sample generated (first 4):")
        for i, s in enumerate(decoded[:4]):
            print(f"[{i}] {s}")

