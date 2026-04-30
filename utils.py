import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_contrastive_loss(img_repr: torch.Tensor, text_repr: torch.Tensor, temp: float = 0.07) -> torch.Tensor:
    """
    img_repr: [B, d], text_repr: [B, d] (both projected to same d_model)
    returns symmetric InfoNCE scalar loss
    """
    img_n = F.normalize(img_repr, dim=-1)
    txt_n = F.normalize(text_repr, dim=-1)
    logits = (img_n @ txt_n.t()) / temp   # [B, B]
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_i + loss_t)