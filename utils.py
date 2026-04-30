import torch
import torch.nn as nn
import torch.nn.functional as F

CONTRASTIVE_TEMP = 0.07

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

def symmetric_contrastive_loss(img_repr, txt_repr, temp=CONTRASTIVE_TEMP):
    img_n = F.normalize(img_repr, dim=-1)
    txt_n = F.normalize(txt_repr, dim=-1)
    logits = (img_n @ txt_n.t()) / temp
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_i + loss_t)

# checkpoint save
def save_checkpoint(epoch, step, model, optimizer, scheduler, best_metric, path):
    state = {
        "epoch": epoch,
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_metric": best_metric
    }
    torch.save(state, path)

# metrics helpers (simple)
def normalize_text(s: str) -> str:
    return " ".join(str(s).lower().strip().split())

def exact_match(pred, ref):
    return int(normalize_text(pred) == normalize_text(ref))

def rouge_l_score(pred, ref):
    pred_tokens = normalize_text(pred).split()
    ref_tokens = normalize_text(ref).split()
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0
    m, n = len(pred_tokens), len(ref_tokens)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m):
        for j in range(n):
            if pred_tokens[i] == ref_tokens[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    lcs = dp[m][n]
    recall = lcs / n if n > 0 else 0.0
    prec = lcs / m if m > 0 else 0.0
    if recall + prec == 0:
        return 0.0
    beta2 = 1.0
    return (1 + beta2) * prec * recall / (recall + beta2 * prec)
