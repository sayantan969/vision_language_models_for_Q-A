import random
from collections import defaultdict
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from transformers import T5TokenizerFast


CSV_PATH = "kvasir_vqa_subset/kvasir_vqa_subset_info_normalized.csv"
IMG_DIR = Path("kvasir_vqa_subset/images")


NUM_WORKERS = 0      # IMPORTANT: avoid multiprocessing hangs in Jupyter on Windows
BATCH_SIZE = 16       # small for quick check
IMG_SIZE = 224
VAL_IMAGE_SPLIT = 0.05
TOKENIZER_NAME = "t5-small"
PREFIX = "<image> "

# Load normalized CSV
df = pd.read_csv(CSV_PATH)

if "resolved_img_path" in df.columns:
    df["img_path"] = df["resolved_img_path"]
elif "img_path" not in df.columns and "img_id" in df.columns:
    df["img_path"] = df["img_id"].astype(str).apply(lambda x: str(IMG_DIR / f"{x}.jpg"))

# drop rows with missing image files (defensive)
from pathlib import Path
df["img_path"] = df["img_path"].astype(str)
df = df[df["img_path"].apply(lambda p: Path(p).exists())].reset_index(drop=True)
print("Rows after resolving image paths:", len(df))

# Build rows for dataset (use train/test splits already present in your CSV if available)
# If your CSV contains 'split' column (train/test) use that; else do image-level split
if "split" in df.columns:
    df_train = df[df["split"] == "train"].reset_index(drop=True)
    df_test  = df[df["split"] == "test"].reset_index(drop=True)
    print("Using 'split' column in CSV: train rows:", len(df_train), "test rows:", len(df_test))
else:
    # image-level split to create a validation set (no leakage)
    grouped = defaultdict(list)
    for _, r in df.iterrows():
        grouped[r["img_id"]].append(r.to_dict())
    img_ids = list(grouped.keys())
    random.seed(42)
    random.shuffle(img_ids)
    num_val = max(1, int(len(img_ids) * VAL_IMAGE_SPLIT))
    val_ids = set(img_ids[:num_val])
    train_ids = set(img_ids[num_val:])
    train_rows = [row for img in train_ids for row in grouped[img]]
    val_rows   = [row for img in val_ids for row in grouped[img]]
    # for convenience, treat val_rows as test set here
    df_train = pd.DataFrame(train_rows)
    df_test  = pd.DataFrame(val_rows)
    print("Created train/test splits by image. Train rows:", len(df_train), "Test rows:", len(df_test))

# now create rows list used by Dataset
def rows_from_df(dfin):
    rows = []
    for _, r in dfin.iterrows():
        rows.append({
            "img_path": r["img_path"],
            "img_id": str(r["img_id"]),
            "question": (r.get("question") or "").strip(),
            "answer": (r.get("answer") or "").strip()
        })
    return rows

train_rows = rows_from_df(df_train)
test_rows  = rows_from_df(df_test)

print(f"Train images (unique): {len(set(r['img_id'] for r in train_rows))}, Train rows: {len(train_rows)}")
print(f"Test images  (unique): {len(set(r['img_id'] for r in test_rows))}, Test rows: {len(test_rows)}")

# tokenizer + transforms
tokenizer = T5TokenizerFast.from_pretrained(TOKENIZER_NAME)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<pad>"})

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
val_transforms = transforms.Compose([
    transforms.Resize(int(IMG_SIZE*1.14)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

class KvasirVQADataset(Dataset):
    def __init__(self, rows, tokenizer, transforms, max_q_len=64, max_a_len=64, prefix=PREFIX):
        self.rows = rows
        self.tokenizer = tokenizer
        self.transforms = transforms
        self.max_q_len = max_q_len
        self.max_a_len = max_a_len
        self.prefix = prefix
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, idx):
        r = self.rows[idx]
        img = Image.open(r["img_path"]).convert("RGB")
        img = self.transforms(img)
        q = f"{self.prefix}{r['question']}"
        q_tok = self.tokenizer(q, truncation=True, max_length=self.max_q_len, padding=False)
        a_tok = self.tokenizer(r['answer'], truncation=True, max_length=self.max_a_len, padding=False)
        return {
            "pixel_values": img,
            "input_ids": torch.tensor(q_tok["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(q_tok["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(a_tok["input_ids"], dtype=torch.long),
            "img_id": r["img_id"]
        }

def collate_fn(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch], dim=0)
    input_ids = [b["input_ids"] for b in batch]
    attention_mask = [b["attention_mask"] for b in batch]
    labels = [b["labels"] for b in batch]
    tok_batch = tokenizer.pad({"input_ids": input_ids, "attention_mask": attention_mask},
                              padding="longest", return_tensors="pt")
    labels_batch = tokenizer.pad({"input_ids": labels}, padding="longest", return_tensors="pt")["input_ids"]
    labels_batch[labels_batch == tokenizer.pad_token_id] = -100
    return {
        "pixel_values": pixel_values,
        "input_ids": tok_batch["input_ids"],
        "attention_mask": tok_batch["attention_mask"],
        "labels": labels_batch,
        "img_ids": [b["img_id"] for b in batch]
    }

train_ds = KvasirVQADataset(train_rows, tokenizer, train_transforms)
val_ds   = KvasirVQADataset(test_rows, tokenizer, val_transforms)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                          collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True)

print("Dataloaders created. Train batches:", len(train_loader), "Val/Test batches:", len(val_loader))

# Quick sanity-check (fetch a single batch)
batch = next(iter(train_loader))
print("Sample batch shapes:", batch["pixel_values"].shape, batch["input_ids"].shape, batch["labels"].shape)
print("Decoded Q:", tokenizer.decode(batch["input_ids"][0], skip_special_tokens=True))
print("Decoded A:", tokenizer.decode(batch["labels"][0].masked_fill(batch["labels"][0]==-100, tokenizer.pad_token_id), skip_special_tokens=True))
