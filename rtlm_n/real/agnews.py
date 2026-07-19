"""AG News: 120k real news texts, 4 classes (World, Sports, Business,
Sci/Tech).

Downloaded once from the public CharCnn_Keras mirror on GitHub and cached
under ~/.cache/rtlmn. A word-level tokenizer (vocabulary built from the
training corpus) feeds the torch backbones.
"""

from __future__ import annotations

import csv
import os
import re
import urllib.request
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

CACHE = os.path.expanduser(os.environ.get("RTLMN_CACHE", "~/.cache/rtlmn"))
BASE_URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv"
N_CLASSES = 4
CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]

PAD, UNK, MASK = 0, 1, 2
_word_re = re.compile(r"[a-z0-9']+")


def _fetch(name: str) -> str:
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name)
    if not os.path.exists(path):
        urllib.request.urlretrieve(f"{BASE_URL}/{name.replace('agnews_', '')}", path)
    return path


def _read(path: str) -> Tuple[List[str], np.ndarray]:
    texts, labels = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            labels.append(int(row[0]) - 1)
            texts.append((row[1] + " " + row[2]).replace("\\", " "))
    return texts, np.array(labels, dtype=int)


def tokenize(text: str) -> List[str]:
    return _word_re.findall(text.lower())


@dataclass
class Vocab:
    stoi: dict
    seq_len: int

    @property
    def size(self) -> int:
        return len(self.stoi) + 3  # + PAD/UNK/MASK

    def encode(self, texts: List[str]) -> np.ndarray:
        out = np.full((len(texts), self.seq_len), PAD, dtype=np.int64)
        for i, t in enumerate(texts):
            ids = [self.stoi.get(w, UNK - 3) + 3 for w in tokenize(t)[: self.seq_len]]
            ids = [j if j >= 3 else UNK for j in ids]
            out[i, : len(ids)] = ids
        return out


def build_vocab(texts: List[str], max_words: int = 12000, seq_len: int = 24) -> Vocab:
    counts = Counter(w for t in texts for w in tokenize(t))
    stoi = {w: i for i, (w, _) in enumerate(counts.most_common(max_words))}
    return Vocab(stoi=stoi, seq_len=seq_len)


def load_agnews(
    n_train: int = 20000, n_test: int = 4000, seed: int = 0, seq_len: int = 24
) -> dict:
    """Load a reproducible subsample, build the vocab from training text."""
    train_texts, train_labels = _read(_fetch("agnews_train.csv"))
    test_texts, test_labels = _read(_fetch("agnews_test.csv"))
    rng = np.random.default_rng(seed)
    tr = rng.permutation(len(train_texts))[:n_train]
    te = rng.permutation(len(test_texts))[:n_test]
    train_texts = [train_texts[i] for i in tr]
    test_texts = [test_texts[i] for i in te]
    vocab = build_vocab(train_texts, seq_len=seq_len)
    return {
        "vocab": vocab,
        "train_tokens": vocab.encode(train_texts),
        "train_labels": train_labels[tr],
        "test_tokens": vocab.encode(test_texts),
        "test_labels": test_labels[te],
        "n_classes": N_CLASSES,
    }
