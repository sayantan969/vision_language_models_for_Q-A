# Visual Question Answering (VQA) — Project README

Welcome — this README explains the entire VQA project, the dataset, every file in the repo, how to run training/evaluation, and suggestions for improvements. It’s written so a collaborator (or future you) can reproduce experiments and extend the project quickly.

---

# Table of contents

1. Project overview
2. Quick start (run in 5 minutes)
3. Dataset (easy-VQA) — how we use it
4. Model architectures (encoder(s), fusion, decoder/classifier)
5. Files in this repo — detailed explanation of each file
6. Installation and environment setup
7. Data download (Option A: package, Option B: repo + generate)
8. Dataloader details & tokenization
9. Training (`train.py`) — how it works & how to run
10. Evaluation & metrics (what we compute and why)
11. Inference / example usage
12. Tips, troubleshooting & common pitfalls
13. Extensions & future work
14. License & acknowledgements

---

# 1 — Project overview

This repository implements a Visual Question Answering pipeline for the **easy-VQA** dataset. The code includes:

* Image encoder (ResNet-based)
* Text/question encoder (Embedding + BiLSTM)
* Fusion block to combine image + question features
* Two usage modes:

  * **Classification head** (fused features → 13-class linear classifier) — implemented in `train.py` and suitable for easy-VQA single-token answers.
  * **Autoregressive decoder** (LSTM-based generative decoder) — implemented in `model_and_run.py` (original architecture you requested).
* Data loader for easy-VQA (package-based).
* Utilities for training, evaluation, and saving checkpoints.

Goal: a reproducible baseline you can train quickly on CPU/GPU and extend for attention, transformers, or RAG-style retrieval.

---

# 2 — Quick start (run in ~5 minutes)

```bash
# 1. Create/activate your conda env (optional)
conda create -n pt python=3.10 -y
conda activate pt

# 2. Install packages
pip install torch torchvision easy-vqa tqdm pillow

# 3. Copy dataset locally (recommended)
python download_easy_vqa.py

# 4. Run a quick test to check dataloaders + model
python model_and_run.py

# 5. Start training (small example)
python train.py
```

`model_and_run.py` contains a quick shape-sanity test under `if __name__ == "__main__":`. `train.py` runs the classifier training loop and saves `checkpoints/best_model.pt`.

---

# 3 — Dataset: easy-VQA

We use the **easy-VQA** dataset (synthetic VQA). Highlights:

* Train images: 4,000 (64×64)
* Train questions: ~38,575
* Test images: 1,000
* Test questions: ~9,673
* Answers: 13 possible tokens (colors, shapes, yes/no, etc.)

We recommend using the package `easy-vqa` (pip) for convenience or generating your own dataset from the repository. The dataloader and `download_easy_vqa.py` arrange the files into `easy_vqa_data/` with `train/`, `test/`, `answers.txt`.

Note: the model uses ResNet (ImageNet pretrained) which expects larger images (224×224). The dataloader resizes 64×64 images to 224×224 for compatibility with the ResNet backbone. If you prefer to keep 64×64, either use a smaller backbone or set `Resize((64,64))` in the transform.

---

# 4 — Model architecture (summary)

There are two main pipelines included:

A. **Classifier** (used by `train.py`)

* Image encoder: `ImageEncoder` — ResNet34 backbone (final avg-pooled vector, 512-dimensional).
* Question encoder: `QuestionEncoder` — Embedding (default 300) + 2-layer BiLSTM (hidden 384 per direction) → 768-D question vector (concat top-layer forward/back).
* Fusion: `FusionBlock` — concatenates image (512) + question (768) → FC (1280 → 512) + ReLU + dropout.
* Classifier: Linear(512 → 13) → CrossEntropyLoss (single-token answers).

B. **Autoregressive decoder** (available in `model_and_run.py`)

* Same encoders + fusion as above.
* Decoder: `DecoderLSTM` — embedding + 2-layer LSTM (hidden 512) whose initial hidden/cell are projected from fused vector. Generates tokens autoregressively (teacher forcing at train time).

The repo defaults to the **classifier** approach for training because easy-VQA answers are a small fixed vocabulary and single-token, which is faster and simpler.

---

# 5 — Files in this repo (detailed)

Below is a file-by-file explanation of purpose, key functions/classes, and usage.

### `image_encoder.py`

* Class: `ImageEncoder(pretrained: bool=True, freeze_backbone: bool=False)`
* Uses `torchvision.models.resnet34(pretrained=...)`, strips final FC and returns a 512-D vector after global pooling.
* Use `freeze_backbone=True` to freeze ResNet parameters (fine for fast experiments).

### `text_encoder.py`

* Class: `QuestionEncoder(vocab_size, emb_dim=300, lstm_hidden=384, lstm_layers=2)`
* Embedding layer plus BiLSTM. `forward(token_ids, lengths=None)` returns (B, 2*lstm_hidden) where we concatenate top-layer forward and backward states.

### `model_and_run.py`

* `FusionBlock`, `DecoderLSTM`, and `VQAModel` (the autoregressive VQA model).
* `VQAModel.forward(...)` supports both training (with `answer_tokens`) and inference (no `answer_tokens`) mode.
* Contains a small `if __name__ == "__main__":` test to validate shapes — handy for quick sanity checks.

### `dataloaders.py` (package-based loader)

* `EasyVQAPackageDataset` — wraps the `easy_vqa` package API:

  * Returns `img_tensor`, `question_ids` (padded), `question_length`, `answer_idx`.
  * Builds a simple whitespace vocabulary if none provided.
  * Uses default transforms (Resize to 224, ToTensor, ImageNet normalization).
* `build_vocab`, `encode_question`, and `collate_fn_batch` helpers included.

### `download_easy_vqa.py`

* Utility script that imports `easy_vqa` package and copies images + questions into a local `easy_vqa_data/` directory, producing:

  ```
  easy_vqa_data/
    train/images/
    train/questions.json
    test/images/
    test/questions.json
    answers.txt
  ```
* Run this to create a local dataset copy.

### `train.py`

* Trains a **classification** model (`ClassifierModel`) that uses `ImageEncoder` + `QuestionEncoder` + `FusionBlock` + `Linear` classifier.
* Main features:

  * Data loading via `EasyVQAPackageDataset`.
  * AdamW optimizer, CrossEntropyLoss.
  * Compute metrics: accuracy, per-class precision/recall/f1, macro-F1 (custom NumPy code — no sklearn required).
  * Save best checkpoint as `checkpoints/best_model.pt` and `final_model.pt`.
  * Progress printed with `tqdm`.

### `.gitignore`

* Configured to ignore dataset folders (`easy_vqa_data/`, `easy_vqa/data/`), model checkpoints (`checkpoints/`, `*.pt`), and common env/editor files.
* **Do not upload dataset or large checkpoints to GitHub.**

---

# 6 — Installation & environment

Recommended: create a conda env `pt` as you already have.

```bash
conda create -n pt python=3.10 -y
conda activate pt

# Install core packages
pip install torch torchvision tqdm pillow easy-vqa

# Optional for dev
pip install matplotlib jupyterlab
```

If you have CUDA GPU, install the appropriate `torch` build (see pytorch.org for the command matching your CUDA version).

---

# 7 — Data download

Two options:

### Option A — Package (recommended)

```bash
pip install easy-vqa
python download_easy_vqa.py
```

This copies package assets locally into `easy_vqa_data/`.

### Option B — Generate from repository

```bash
git clone https://github.com/vzhou842/easy-VQA.git
cd easy-VQA
pip install -r gen_data/requirements.txt
python gen_data/generate_data.py
```

Edit `NUM_TRAIN` / `NUM_TEST` inside `generate_data.py` if you want more/less data. Then either use the package-based loader or the manual loader skeleton provided earlier.

---

# 8 — Dataloader & tokenization details

* Tokenization: simple whitespace lowercasing (quick and interpretable). The `build_vocab` reserves indices:

  ```
  0: <pad>
  1: <unk>
  2: <sos>
  3: <eos>
  ```
* `encode_question` pads/truncates to `max_q_len` (default 16–20).
* Collation yields `(images, question_ids, question_lengths, answer_idx)`.
* `EasyVQAPackageDataset` also provides `all_answers` (list of 13 answer strings) and `answer2idx` mapping.

If you want better tokenization (BPE, WordPiece, or pretrained `transformers` tokenizer), I can integrate that — useful if you later use transformer encoders.

---

# 9 — Training (`train.py`)

### How it trains

* Model: `ClassifierModel` (encoder + fusion + FC 13-way).
* Loss: `CrossEntropyLoss`
* Optimizer: `AdamW`
* Metrics: accuracy, per-class precision/recall/f1, macro-F1
* Checkpointing: saves best model by validation accuracy to `checkpoints/best_model.pt`

### Run

```bash
python train.py
```

You can edit `main_train(...)` call at the bottom or modify `main_train` arguments:

* epochs (default 6), batch_size (default 128), lr (default 1e-4), num_workers.

### Notes

* If GPU OOM, reduce `batch_size` (try 32 or 64) or set `pretrained_resnet=False` to skip downloading/resizing overhead.
* For smaller experiments set `freeze_resnet=True` to freeze the backbone.

---

# 10 — Evaluation & metrics

`train.py` computes:

* **Accuracy**: overall exact-match accuracy for single-token answers.
* **Per-class precision / recall / F1**: computed without sklearn to avoid extra deps.
* **Macro-F1**: average F1 across classes, useful for imbalanced classes (like colors vs yes/no).

Why these metrics?

* easy-VQA is a classification-style dataset (single-token answers). Accuracy measures correctness; macro-F1 balances classes that differ in support.

---

# 11 — Inference / example usage

Load the saved checkpoint and run the model on a batch or a single example:

```python
# inference_example.py
import torch
from model_and_run import FusionBlock  # or import ClassifierModel from train.py if you saved it
from image_encoder import ImageEncoder
from text_encoder import QuestionEncoder
from train import ClassifierModel  # if train.py is in same folder and defines ClassifierModel
from dataloaders import EasyVQAPackageDataset, collate_fn_batch

# load dataset to get vocab & answer list
ds = EasyVQAPackageDataset(split="test", max_q_len=16)
vocab = ds.word2idx
answers = ds.all_answers

# instantiate model
model = ClassifierModel(vocab_size_q=len(vocab), num_answers=len(answers))
ckpt = torch.load("checkpoints/best_model.pt", map_location="cpu")
model.load_state_dict(ckpt["model_state"])
model.eval()

# pick a sample
img, q_ids, q_len, ans_idx = ds[0]
img = img.unsqueeze(0)  # B=1
q_ids = q_ids.unsqueeze(0)
q_len = q_len.unsqueeze(0)

with torch.no_grad():
    logits = model(img, q_ids, q_len)
    pred_idx = logits.argmax(dim=-1).item()
    print("Predicted answer:", answers[pred_idx], "Ground truth:", answers[ans_idx.item()])
```

For autoregressive generation using `VQAModel`, use `model_and_run.VQAModel` and call forward with `answer_tokens=None` to generate token ids.

---

# 12 — Tips, troubleshooting & common pitfalls

* **Image size mismatch**: ResNet expects larger inputs — dataloader resizes to 224×224. If you get poor performance, try training a small CNN on 64×64.
* **OOM on GPU**: reduce batch size or freeze backbone layers.
* **Pretrained weights warnings**: torchvision changed `pretrained` to `weights`. The code uses `pretrained=True` for compatibility; ignore the deprecation warnings or update to `weights=ResNet34_Weights.DEFAULT`.
* **Reproducibility**: set random seeds for `numpy`, `torch`, and `random` if exact reproducibility is important.
* **Accidentally committed data**: add dataset paths to `.gitignore` and use `git rm --cached` to untrack them. Use BFG or `git filter-repo` to purge large files from history if needed.

---

# 13 — Extensions & future work

Ideas to improve accuracy / research:

* Replace BiLSTM question encoder with a transformer (e.g., DistilBERT) for better language understanding.
* Use spatial image features (take ResNet features before global pooling) + attention mechanism (visual attention or co-attention) instead of a global image vector.
* Replace decoder/classifier with multi-task head: classification + generative answer (helpful for longer answers).
* Data augmentation (color jitter, random cropping, rotations) — careful with synthetic shapes, preserve semantics.
* Use beam search for autoregressive decoder.
* Logging with TensorBoard or Weights & Biases.
* Use curriculum training / curriculum difficulty sampling.

---

# 14 — License & acknowledgements

* This project code is released under [MIT License] (if you want a license file, I can add it).
* Dataset: easy-VQA by vzhou842 (see their repo and license). Acknowledge their dataset in any publication.

---

# Contact / Next steps

If you want I can:

* Convert `train.py` to use `argparse` so hyperparameters are CLI configurable.
* Add TensorBoard logging and a small notebook to visualize predictions.
* Swap the classifier with the autoregressive decoder training loop (teacher forcing) and give a demo.
* Add a `requirements.txt` or `environment.yml`.

Would you like me to:

1. Add an `argparse` wrapper to `train.py` now?
2. Convert the repo to train the autoregressive decoder instead?
3. Create a small Jupyter notebook that demonstrates data loading, training 1 epoch, and visualizing 10 predictions?

Pick one and I’ll produce the code.
