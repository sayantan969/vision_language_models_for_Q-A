# dataloaders.py  (Package-based easy-VQA loader)
import os
from typing import List, Tuple, Dict, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import numpy as np

# Optional: import easy_vqa package (install via `pip install easy-vqa`)
try:
    import easy_vqa
    HAS_EASY_VQA = True
except Exception:
    HAS_EASY_VQA = False

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"

def build_vocab(questions: List[str], min_freq: int = 1, max_size: Optional[int] = None) -> Tuple[Dict[str,int], Dict[int,str]]:
    """Simple whitespace tokenizer -> builds word2idx with reserved tokens."""
    freq = {}
    for q in questions:
        for w in q.lower().strip().split():
            freq[w] = freq.get(w, 0) + 1
    items = [w for w, c in freq.items() if c >= min_freq]
    items.sort(key=lambda w: (-freq[w], w))
    if max_size:
        items = items[:max_size]
    # reserve indices
    word2idx = {PAD_TOKEN: 0, UNK_TOKEN: 1, SOS_TOKEN: 2, EOS_TOKEN: 3}
    idx2word = {0: PAD_TOKEN, 1: UNK_TOKEN, 2: SOS_TOKEN, 3: EOS_TOKEN}
    idx = 4
    for w in items:
        if w not in word2idx:
            word2idx[w] = idx
            idx2word[idx] = w
            idx += 1
    return word2idx, idx2word

def encode_question(q: str, word2idx: Dict[str,int], max_len: int) -> Tuple[List[int], int]:
    """Returns token ids (no SOS/EOS here; collate can add if you want)."""
    toks = [word2idx.get(w, word2idx[UNK_TOKEN]) for w in q.lower().strip().split()]
    length = min(len(toks), max_len)
    if len(toks) >= max_len:
        toks = toks[:max_len]
    else:
        toks = toks + [word2idx[PAD_TOKEN]] * (max_len - len(toks))
    return toks, length

class EasyVQAPackageDataset(Dataset):
    """
    Dataset powered by the easy_vqa package.
    Each __getitem__ returns: image_tensor, question_ids (padded), question_length, answer_idx
    """
    def __init__(self,
                 split: str = "train",
                 max_q_len: int = 20,
                 vocab: Optional[Dict[str,int]] = None,
                 transform: Optional[transforms.Compose] = None):
        """
        split: "train" or "test"
        max_q_len: max token length (questions are short in easy-VQA)
        vocab: if None, will be built from the entire train questions if split=="train"
        """
        assert split in ("train", "test")
        if not HAS_EASY_VQA:
            raise RuntimeError("easy_vqa package not found. Install with `pip install easy-vqa`")

        if split == "train":
            questions, answers, image_ids = easy_vqa.get_train_questions()
            image_paths_map = easy_vqa.get_train_image_paths()
        else:
            questions, answers, image_ids = easy_vqa.get_test_questions()
            image_paths_map = easy_vqa.get_test_image_paths()

        self.questions = questions
        self.answers = answers
        self.image_ids = image_ids
        self.image_paths_map = image_paths_map
        self.max_q_len = max_q_len

        # answers list (all possible answers)
        self.all_answers = easy_vqa.get_answers()
        self.answer2idx = {a: i for i, a in enumerate(self.all_answers)}

        # build vocab if not provided
        if vocab is None:
            # build from train set (recommended). If user requests test split only and vocab None,
            # we build from the questions of this split.
            if split == "train":
                all_train_qs, _, _ = easy_vqa.get_train_questions()
                self.word2idx, self.idx2word = build_vocab(all_train_qs, min_freq=1)
            else:
                self.word2idx, self.idx2word = build_vocab(self.questions, min_freq=1)
        else:
            self.word2idx = vocab
            self.idx2word = {i:w for w,i in vocab.items()}

        # default transform: resize to 224 for ResNet, convert to tensor and normalize
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        q_text = self.questions[idx]
        ans = self.answers[idx]           # string (e.g., 'circle' or 'yes')
        img_id = self.image_ids[idx]      # integer id into image paths map
        img_path = self.image_paths_map[img_id]

        # load image
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        # encode question
        q_ids, q_len = encode_question(q_text, self.word2idx, max_len=self.max_q_len)

        # answer -> idx
        ans_idx = self.answer2idx[ans]

        return img, torch.tensor(q_ids, dtype=torch.long), torch.tensor(q_len, dtype=torch.long), torch.tensor(ans_idx, dtype=torch.long)

def collate_fn_batch(batch):
    """
    Collate to tensors: images (B,3,H,W), questions (B,T), lengths (B,), answers (B,)
    """
    imgs, q_ids, q_lens, ans_idx = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    q_ids = torch.stack(q_ids, dim=0)
    q_lens = torch.stack(q_lens, dim=0)
    ans_idx = torch.stack(ans_idx, dim=0)
    return imgs, q_ids, q_lens, ans_idx

# Example usage
if __name__ == "__main__":
    if not HAS_EASY_VQA:
        print("Install easy-vqa: pip install easy-vqa")
    else:
        ds = EasyVQAPackageDataset(split="train", max_q_len=16)
        from torch.utils.data import DataLoader
        dl = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate_fn_batch, num_workers=2)
        images, q_ids, q_lens, ans = next(iter(dl))
        print("images", images.shape)
        print("q_ids", q_ids.shape, "q_lens", q_lens)
        print("ans", ans.shape)
