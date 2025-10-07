# train.py
import os
import time
import json
from pathlib import Path
from typing import Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# imports from your project files (assume they are in same folder)
from image_encoder import ImageEncoder
from text_encoder import QuestionEncoder
from model_and_run import FusionBlock  # fusion block implemented earlier
from dataloader import EasyVQAPackageDataset, collate_fn_batch

# optional niceties
try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x: x


def compute_classification_metrics(y_true: List[int], y_pred: List[int], num_classes: int, labels: List[str] = None):
    """
    Compute accuracy, per-class precision/recall/f1, and macro f1.
    Returns a dict with numbers.
    """
    import numpy as np
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    correct = (y_true == y_pred).sum()
    total = len(y_true)
    accuracy = float(correct) / float(total) if total > 0 else 0.0

    precisions = []
    recalls = []
    f1s = []
    per_class = {}

    for c in range(num_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

        label_name = labels[c] if (labels is not None and c < len(labels)) else str(c)
        per_class[label_name] = {"precision": prec, "recall": rec, "f1": f1, "support": int((y_true == c).sum())}

    macro_f1 = float(sum(f1s) / len(f1s)) if len(f1s) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "precision_per_class": precisions,
        "recall_per_class": recalls,
        "f1_per_class": f1s,
        "total": total
    }


class ClassifierModel(nn.Module):
    """
    Simple classification model:
      image_encoder -> (512)
      question_encoder -> (768)
      fusion -> (512)
      classifier -> (num_answers)
    """
    def __init__(self,
                 vocab_size_q: int,
                 num_answers: int,
                 q_emb_dim: int = 300,
                 q_lstm_hidden: int = 384,
                 q_lstm_layers: int = 2,
                 pretrained_resnet: bool = True,
                 freeze_resnet: bool = False):
        super().__init__()
        self.image_encoder = ImageEncoder(pretrained=pretrained_resnet, freeze_backbone=freeze_resnet)
        self.question_encoder = QuestionEncoder(vocab_size=vocab_size_q, emb_dim=q_emb_dim,
                                                lstm_hidden=q_lstm_hidden, lstm_layers=q_lstm_layers)
        self.fusion = FusionBlock(in_dim=512 + 2 * q_lstm_hidden, fused_dim=512, dropout_p=0.35)
        self.classifier = nn.Linear(512, num_answers)

    def forward(self, images: torch.Tensor, question_tokens: torch.Tensor, question_lengths: torch.Tensor) -> torch.Tensor:
        # returns logits (B, num_answers)
        img_feat = self.image_encoder(images)          # (B,512)
        q_feat = self.question_encoder(question_tokens, lengths=question_lengths)  # (B, 2*q_lstm_hidden)
        fused = self.fusion(img_feat, q_feat)          # (B,512)
        logits = self.classifier(fused)                # (B, num_answers)
        return logits


def train_one_epoch(model: nn.Module, dataloader: DataLoader, optimizer, criterion, device: torch.device, epoch: int, grad_clip: float = None):
    model.train()
    running_loss = 0.0
    y_trues = []
    y_preds = []
    it = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, (images, q_ids, q_lens, ans_idx) in it:
        images = images.to(device)
        q_ids = q_ids.to(device)
        q_lens = q_lens.to(device)
        ans_idx = ans_idx.to(device)

        optimizer.zero_grad()
        logits = model(images, q_ids, q_lens)  # (B, num_answers)
        loss = criterion(logits, ans_idx)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = torch.argmax(logits, dim=1).detach().cpu().tolist()
        truths = ans_idx.detach().cpu().tolist()
        y_trues.extend(truths)
        y_preds.extend(preds)

        if (i + 1) % 50 == 0:
            it.set_description(f"Epoch {epoch} batch {i+1} loss={loss.item():.4f}")

    avg_loss = running_loss / len(dataloader.dataset)
    metrics = compute_classification_metrics(y_trues, y_preds, num_classes=model.classifier.out_features)
    return avg_loss, metrics


def evaluate(model: nn.Module, dataloader: DataLoader, criterion, device: torch.device):
    model.eval()
    running_loss = 0.0
    y_trues = []
    y_preds = []
    with torch.no_grad():
        for images, q_ids, q_lens, ans_idx in tqdm(dataloader):
            images = images.to(device)
            q_ids = q_ids.to(device)
            q_lens = q_lens.to(device)
            ans_idx = ans_idx.to(device)

            logits = model(images, q_ids, q_lens)
            loss = criterion(logits, ans_idx)
            running_loss += loss.item() * images.size(0)

            preds = torch.argmax(logits, dim=1).cpu().tolist()
            truths = ans_idx.cpu().tolist()
            y_trues.extend(truths)
            y_preds.extend(preds)

    avg_loss = running_loss / len(dataloader.dataset)
    metrics = compute_classification_metrics(y_trues, y_preds, num_classes=model.classifier.out_features)
    return avg_loss, metrics


def save_checkpoint(state: dict, path: str):
    torch.save(state, path)


def load_checkpoint(path: str, device: torch.device = None):
    map_loc = device if device is not None else None
    return torch.load(path, map_location=map_loc)


def main_train(
    out_dir: str = "checkpoints",
    epochs: int = 6,
    batch_size: int = 64,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    num_workers: int = 4,
    pretrained_resnet: bool = True,
    freeze_resnet: bool = False,
    max_q_len: int = 16
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # build datasets
    print("Preparing datasets...")
    train_ds = EasyVQAPackageDataset(split="train", max_q_len=max_q_len)
    test_ds = EasyVQAPackageDataset(split="test", max_q_len=max_q_len, vocab=train_ds.word2idx)  # share vocab

    vocab_size_q = len(train_ds.word2idx)
    num_answers = len(train_ds.all_answers)
    print(f"vocab_size_q={vocab_size_q}  num_answers={num_answers}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_batch, num_workers=num_workers)
    val_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_batch, num_workers=num_workers)

    # model
    model = ClassifierModel(vocab_size_q=vocab_size_q, num_answers=num_answers,
                            q_lstm_hidden=384, q_lstm_layers=2,
                            pretrained_resnet=pretrained_resnet, freeze_resnet=freeze_resnet).to(device)

    # optimizer & loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None

    print("Starting training on device:", device)
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        t1 = time.time()

        train_acc = train_metrics["accuracy"]
        val_acc = val_metrics["accuracy"]
        print(f"Epoch {epoch} time {(t1-t0):.1f}s  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  val_macro_f1={val_metrics['macro_f1']:.4f}")

        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "vocab": train_ds.word2idx,
                "answers": train_ds.all_answers
            }
            save_path = os.path.join(out_dir, "best_model.pt")
            save_checkpoint(best_state, save_path)
            print(f"Saved best model to {save_path} (val_acc={val_acc:.4f})")

    # final save
    final_path = os.path.join(out_dir, "final_model.pt")
    save_checkpoint({
        "epoch": epochs,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "vocab": train_ds.word2idx,
        "answers": train_ds.all_answers
    }, final_path)
    print("Training finished. Best val_acc:", best_val_acc)


if __name__ == "__main__":
    # example default run (adapt args as you like)
    main_train(
        out_dir="checkpoints",
        epochs=10,
        batch_size=64,   # reduce if GPU OOM or use 32/64
        lr=1e-4,
        weight_decay=1e-5,
        num_workers=4
    )
