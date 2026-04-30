
import os, time, json, hashlib
from pathlib import Path
from io import BytesIO
from tqdm.auto import tqdm

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from PIL import Image

from datasets import load_dataset
import pandas as pd

# --------- User config (edit if needed) ----------
DATA_DIR = Path("./kvasir_vqa_subset")
IMG_DIR = DATA_DIR / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

HF_DATASET_ID = "SimulaMet/Kvasir-VQA-x1"
SPLITS = ["train", "test"]
SAVE_JSONL_DIR = DATA_DIR
SAVE_CSV_PATH = DATA_DIR / "kvasir_vqa_all_collected.csv"

HF_TOKEN = os.environ.get("HF_TOKEN", None)

# Network & retry config
REQUEST_TIMEOUT = 15
RETRY_TOTAL = 3
RETRY_BACKOFF = 0.5
SLEEP_ON_FAIL = 0.5

# Safety limits (set to None to run full split)
MAX_ROWS_PER_SPLIT = None   # e.g. 200000 to cap; None means no cap (process whole split)
VERBOSE = True
# --------------------------------------------------

# Setup requests session with retries
session = requests.Session()
retries = Retry(total=RETRY_TOTAL, backoff_factor=RETRY_BACKOFF, status_forcelist=[429, 500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))
session.mount("http://", HTTPAdapter(max_retries=retries))
if HF_TOKEN:
    session.headers.update({"Authorization": f"Bearer {HF_TOKEN}"})

def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def image_md5_bytes_from_pil(pil_img: Image.Image) -> str:
    buf = BytesIO()
    pil_img.convert("RGB").save(buf, format="JPEG", quality=90)
    return hashlib.md5(buf.getvalue()).hexdigest()

def to_pil(img_obj):
    """Robust converter like earlier: bytes, dict (HF image), url string, local path, PIL image."""
    if isinstance(img_obj, Image.Image):
        return img_obj
    if isinstance(img_obj, (bytes, bytearray)):
        return Image.open(BytesIO(img_obj))
    if isinstance(img_obj, dict):
        # common keys
        for key in ("path", "uri", "url"):
            v = img_obj.get(key)
            if isinstance(v, str) and v:
                return to_pil(v)
        if "bytes" in img_obj and img_obj["bytes"]:
            return Image.open(BytesIO(img_obj["bytes"]))
    if isinstance(img_obj, str):
        s = img_obj.strip()
        if s.startswith("http://") or s.startswith("https://"):
            resp = session.get(s, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return Image.open(BytesIO(resp.content))
        else:
            return Image.open(s)
    # last resort
    return Image.open(img_obj)

# Discover existing images on disk
print(f"Scanning existing images in {IMG_DIR} ...")
seen_ids = {}        # name/stem -> filepath
existing_hashes = {} # md5 -> filepath
for f in IMG_DIR.iterdir():
    if not f.is_file(): 
        continue
    if f.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
        continue
    stem = f.stem
    seen_ids[stem] = str(f)
    try:
        md5 = file_md5(f)
        if md5 not in existing_hashes:
            existing_hashes[md5] = str(f)
    except Exception:
        pass

print(f"Found {len(seen_ids)} files and {len(existing_hashes)} content-hashes in {IMG_DIR}.\n")

# Will collect rows across splits
all_collected_rows = []

# Process each split
for split in SPLITS:
    print(f"Loading dataset split: {split} ...")
    ds = load_dataset(HF_DATASET_ID, split=split)
    total_rows = len(ds)
    print(f"Split {split} contains {total_rows} rows.")
    collected_rows = []
    rows_scanned = 0
    saved_images = 0

    iter_rows = enumerate(tqdm(ds, desc=f"Scanning {split} rows"))
    for i, row in iter_rows:
        rows_scanned += 1
        if MAX_ROWS_PER_SPLIT and rows_scanned > MAX_ROWS_PER_SPLIT:
            if VERBOSE: print(f"Reached user cap of {MAX_ROWS_PER_SPLIT} rows for split {split}.")
            break

        # skip rows without image field
        if "image" not in row or row["image"] is None:
            continue

        # attempt to obtain stable img_id
        img_id = row.get("img_id") or row.get("image_id") or row.get("image_id_str")

        # reuse if known on disk by id
        if img_id is not None and str(img_id) in seen_ids:
            saved_path = seen_ids[str(img_id)]
            collected_rows.append({
                "split": split,
                "img_path": saved_path,
                "img_id": str(img_id),
                "question": (row.get("question") or "").strip(),
                "answer": (row.get("answer") or "").strip(),
                "complexity": row.get("complexity", None),
                "question_class": row.get("question_class", None),
            })
            continue

        # Try convert image field to PIL (may download)
        try:
            pil_img = to_pil(row["image"])
        except Exception as e:
            if VERBOSE:
                print(f"[{split}] Warning: failed to load image for row {i}: {e}")
            time.sleep(SLEEP_ON_FAIL)
            continue

        # compute content hash for naming if needed
        try:
            img_md5 = image_md5_bytes_from_pil(pil_img)
        except Exception:
            img_md5 = None

        if img_id is None:
            if img_md5 is not None:
                img_id = f"hash_{img_md5}"
            else:
                img_id = f"rowidx_{rows_scanned}"

        # if content already exists, reuse existing file
        if img_md5 is not None and img_md5 in existing_hashes:
            saved_path = existing_hashes[img_md5]
            seen_ids[str(img_id)] = saved_path
            collected_rows.append({
                "split": split,
                "img_path": saved_path,
                "img_id": str(img_id),
                "question": (row.get("question") or "").strip(),
                "answer": (row.get("answer") or "").strip(),
                "complexity": row.get("complexity", None),
                "question_class": row.get("question_class", None),
            })
            continue

        # Save to disk if not already saved
        out_path = IMG_DIR / f"{img_id}.jpg"
        cnt = 1
        while out_path.exists():
            out_path = IMG_DIR / f"{img_id}_{cnt}.jpg"
            cnt += 1
        try:
            pil_img.convert("RGB").save(out_path, format="JPEG", quality=90)
            saved_images += 1
            saved_path_str = str(out_path)
            seen_ids[str(img_id)] = saved_path_str
            if img_md5 is not None:
                existing_hashes[img_md5] = saved_path_str
        except Exception as e:
            if VERBOSE:
                print(f"[{split}] Warning: failed to save image {img_id}: {e}")
            continue

        # append QA row
        collected_rows.append({
            "split": split,
            "img_path": saved_path_str,
            "img_id": str(img_id),
            "question": (row.get("question") or "").strip(),
            "answer": (row.get("answer") or "").strip(),
            "complexity": row.get("complexity", None),
            "question_class": row.get("question_class", None),
        })

    # end split
    print(f"\n=== {split} summary ===")
    print(f"Rows scanned: {rows_scanned}")
    print(f"Images saved this run: {saved_images}")
    print(f"Collected QA rows from split: {len(collected_rows)}")
    # save split jsonl
    out_jsonl = SAVE_JSONL_DIR / f"kvasir_vqa_{split}_collected.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in collected_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {split} JSONL -> {out_jsonl}\n")

    all_collected_rows.extend(collected_rows)

# Save combined CSV
if len(all_collected_rows) > 0:
    df = pd.DataFrame(all_collected_rows)
    df.to_csv(SAVE_CSV_PATH, index=False)
    print(f"Saved combined CSV with {len(df)} rows -> {SAVE_CSV_PATH}")
else:
    print("No rows collected. Check logs.")
