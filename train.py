# === BLIP-2 training loop with contrastive loss, AMP, accumulation, validation, checkpointing ===
import os, math, time
from tqdm.auto import tqdm
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import torch.nn.functional as F
from dataloader import train_loader, val_loader  
from model import model_blip  

# ------------- CONFIG -------------
EPOCHS = 5
ACCUMULATION_STEPS = 4
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 200
SAVE_DIR = "checkpoints_blip2_t5"
os.makedirs(SAVE_DIR, exist_ok=True)

MAX_GEN_LEN = 64
NUM_BEAMS = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LOG_EVERY_N_STEPS = 50
EVAL_EVERY_EPOCHS = 1

ALPHA = 0.1  # try 0.05 - 0.2
CONTRASTIVE_TEMP = 0.07
# ----------------------------------

# Resolve model instance: prefer model_blip, else model_q/model, else instantiate if class is present
if "model_blip" in globals():
    model = globals()["model_blip"]
    print("Using existing model_blip.")
elif "model_q" in globals():
    model = globals()["model_q"]
    print("Using model_q (renamed locally to model).")
elif "model" in globals():
    model = globals()["model"]
    print("Using model.")
else:
    if "BLIP2_T5" in globals():
        print("Instantiating BLIP2_T5 with defaults (q_n=4, q_dim=256).")
        model = globals()["BLIP2_T5"](t5_name="t5-small", q_n=4, q_dim=256, q_num_layers=1,
                                       freeze_img_encoder=True, freeze_t5_encoder=True, pretrained_img=True)
        globals()["model_blip"] = model
    else:
        raise RuntimeError("No model found. Define/instantiate model_blip, model_q or BLIP2_T5 class first.")

# Check dataloaders
if "train_loader" not in globals() or "val_loader" not in globals():
    raise RuntimeError("train_loader or val_loader not found. Run dataloader cells first.")

model.to(DEVICE)

# Trainable params (automatically selects those with requires_grad)
trainable_params = [p for p in model.parameters() if p.requires_grad]
if len(trainable_params) == 0:
    raise RuntimeError("No trainable parameters. Unfreeze Q-former / T5 decoder or check model init.")

optimizer = AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

steps_per_epoch = math.ceil(len(train_loader) / ACCUMULATION_STEPS)
max_train_steps = EPOCHS * steps_per_epoch
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=max_train_steps)

use_amp = DEVICE == "cuda"
scaler = torch.cuda.amp.GradScaler() if use_amp else None

# helper: symmetric InfoNCE
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

# training loop
best_rouge = -1.0
global_step = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    optimizer.zero_grad()
    train_iter = enumerate(train_loader)
    progress = tqdm(train_iter, total=len(train_loader), desc=f"Epoch {epoch}/{EPOCHS} - training")
    for step, batch in progress:
        pixel_values = batch["pixel_values"].to(DEVICE)
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs, aux = model(pixel_values, input_ids, attention_mask, labels=labels)
            ce_loss = outputs.loss
            # compute contrastive only when effective batch > 1
            bsz = aux["q_repr"].size(0)
            if bsz > 1:
                # text embeddings were stored in aux as token embeddings before encoder
                text_repr = aux["text_embeds"].mean(dim=1)  # [B, d_model]
                img_repr = aux["q_repr"]                    # [B, d_model]
                contrastive_loss = symmetric_contrastive_loss(img_repr, text_repr)
            else:
                contrastive_loss = torch.tensor(0.0, device=ce_loss.device)

            total_loss = ce_loss + ALPHA * contrastive_loss

            # normalize for gradient accumulation
            loss_to_backprop = total_loss / ACCUMULATION_STEPS

        if use_amp:
            scaler.scale(loss_to_backprop).backward()
        else:
            loss_to_backprop.backward()

        epoch_loss += float(total_loss.detach().cpu().item())

        if (step + 1) % ACCUMULATION_STEPS == 0 or (step + 1) == len(train_loader):
            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        # logging
        if (step + 1) % LOG_EVERY_N_STEPS == 0:
            avg_loss = epoch_loss / (step + 1)
            lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else None
            progress.set_postfix({"avg_loss": f"{avg_loss:.4f}", "lr": lr})

    avg_epoch_loss = epoch_loss / len(train_loader)
    print(f"\nEpoch {epoch} training finished. avg_loss: {avg_epoch_loss:.4f}")

    # validation
    if epoch % EVAL_EVERY_EPOCHS == 0:
        model.eval()
        val_loss = 0.0
        all_preds, all_refs = [], []
        with torch.no_grad():
            val_progress = tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Epoch {epoch} - validating")
            for vstep, batch in val_progress:
                pixel_values = batch["pixel_values"].to(DEVICE)
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs, aux = model(pixel_values, input_ids, attention_mask, labels=labels)
                    vloss = outputs.loss
                val_loss += float(vloss.detach().cpu().item())

                # generation
                gen_ids = model.generate(pixel_values, input_ids, attention_mask, max_length=MAX_GEN_LEN, num_beams=NUM_BEAMS)
                decoded_preds = model.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

                labels_cpu = labels.clone().detach().cpu()
                labels_cpu[labels_cpu == -100] = model.tokenizer.pad_token_id
                decoded_refs = model.tokenizer.batch_decode(labels_cpu, skip_special_tokens=True)

                all_preds.extend(decoded_preds)
                all_refs.extend(decoded_refs)

        em_count = sum(exact_match(p, r) for p, r in zip(all_preds, all_refs))
        em = em_count / len(all_preds) if all_preds else 0.0
        rouge_l_vals = [rouge_l_score(p, r) for p, r in zip(all_preds, all_refs)]
        rouge_l = sum(rouge_l_vals) / len(rouge_l_vals) if rouge_l_vals else 0.0
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0

        print(f"\nValidation -- avg_loss: {avg_val_loss:.4f}, EM: {em:.4f}, ROUGE-L: {rouge_l:.4f}")
        # print some examples
        for i in range(min(3, len(all_preds))):
            print(" PRED:", all_preds[i])
            print(" TRUE:", all_refs[i])
            print("----")

        # save best
        if rouge_l > best_rouge:
            best_rouge = rouge_l
            ckpt_path = os.path.join(SAVE_DIR, f"best_epoch{epoch}_rouge{rouge_l:.4f}.pt")
            save_checkpoint(epoch, global_step, model, optimizer, scheduler, best_rouge, ckpt_path)
            print("Saved best model to", ckpt_path)

print("\nTraining finished. Best ROUGE-L:", best_rouge)
final_path = os.path.join(SAVE_DIR, "final_model.pt")
save_checkpoint(EPOCHS, global_step, model, optimizer, scheduler, best_rouge, final_path)
print("Final model saved to:", final_path)
