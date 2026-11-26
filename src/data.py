"""
Carga, partición reproducible, normalización y vocabularios.

La función `load_and_split` replica EXACTAMENTE el bloque obligatorio de la
Sección 3 del enunciado (filtro >= 8 palabras EN, <= 120 palabras, semilla 23,
partición 80/10/10 y huella SHA-256 del test).
"""
import hashlib
import random
import re
import unicodedata
from collections import Counter

import torch
from torch.utils.data import DataLoader, Dataset

PAD, SOS, EOS, UNK = "<pad>", "<sos>", "<eos>", "<unk>"
PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3
SPECIALS = [PAD, SOS, EOS, UNK]

DATA_URL = "https://www.manythings.org/anki/spa-eng.zip"


# ---------------------------------------------------------------------------
# 1. Protocolo obligatorio (Listing 1 del enunciado)
# ---------------------------------------------------------------------------
def load_and_split(path="spa.txt", verbose=True):
    with open(path, "r", encoding="utf-8") as f:
        raw_pairs = [line.strip().split("\t")[:2] for line in f if "\t" in line]

    filtered_pairs = [
        (en, es) for en, es in raw_pairs
        if 8 <= len(en.split()) <= 120 and len(es.split()) <= 120
    ]

    rng = random.Random(23)
    rng.shuffle(filtered_pairs)

    n = len(filtered_pairs)
    n_train = int(0.80 * n)
    n_val = int(0.10 * n)

    train_pairs = filtered_pairs[:n_train]
    val_pairs = filtered_pairs[n_train:n_train + n_val]
    test_pairs = filtered_pairs[n_train + n_val:]

    test_hash = hashlib.sha256(
        "\n".join([p[0] for p in test_pairs]).encode()
    ).hexdigest()[:12]
    if verbose:
        print(f"Train: {len(train_pairs)} | Val: {len(val_pairs)} | Test: {len(test_pairs)}")
        print(f"SHA-256 Test Audit ID: {test_hash}")
    return train_pairs, val_pairs, test_pairs, test_hash


# ---------------------------------------------------------------------------
# 2. Normalización simétrica (se usa en entrenamiento desde cero Y en BLEU)
# ---------------------------------------------------------------------------
_APOS = re.compile(r"[’'`´]")
_NON_ALNUM = re.compile(r"[^a-z0-9\s]")
_SPACES = re.compile(r"\s+")


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def normalize(s: str) -> str:
    """minúsculas -> sin acentos (ñ->n) -> sin puntuación -> espacios colapsados.

    Esta MISMA función se aplica a: (a) entradas/salidas de los modelos A, B, C;
    (b) predicciones de los 4 modelos y (c) referencias de test antes de BLEU.
    """
    s = strip_accents(s.lower())
    s = _APOS.sub("", s)            # don't -> dont
    s = _NON_ALNUM.sub(" ", s)      # ¿ ¡ . , ! ? etc. -> espacio
    return _SPACES.sub(" ", s).strip()


def tokenize(s: str):
    return normalize(s).split()


def length_bucket(en_sentence: str) -> str:
    """Bucket por longitud del ORIGEN en palabras (mismo criterio que el filtro)."""
    n = len(en_sentence.split())
    return "medium" if n <= 18 else "long"   # el filtro garantiza n >= 8


# 3. Vocabularios 
def build_vocab(sentences, min_freq=1, max_size=None):
    counter = Counter(tok for s in sentences for tok in tokenize(s))
    words = [w for w, c in counter.most_common(max_size) if c >= min_freq]
    word2idx = {w: i for i, w in enumerate(SPECIALS)}
    for w in words:
        if w not in word2idx:
            word2idx[w] = len(word2idx)
    idx2word = {i: w for w, i in word2idx.items()}
    return word2idx, idx2word


def build_vocabs(train_pairs, min_freq=1, max_size=None):
    eng_word2idx, eng_idx2word = build_vocab([p[0] for p in train_pairs], min_freq, max_size)
    spa_word2idx, spa_idx2word = build_vocab([p[1] for p in train_pairs], min_freq, max_size)
    return eng_word2idx, eng_idx2word, spa_word2idx, spa_idx2word


def encode(sentence, word2idx, add_sos_eos=False, max_len=None):
    ids = [word2idx.get(t, UNK_IDX) for t in tokenize(sentence)]   
    if max_len is not None:
        ids = ids[:max_len]
    if add_sos_eos:
        ids = [SOS_IDX] + ids + [EOS_IDX]
    return ids


def decode(ids, idx2word):
    out = []
    for i in ids:
        i = int(i)
        if i == EOS_IDX:
            break
        if i in (PAD_IDX, SOS_IDX):
            continue
        out.append(idx2word.get(i, UNK))
    return " ".join(out)


def oov_rate(pairs, eng_word2idx, spa_word2idx):
    def rate(sents, vocab):
        toks = [t for s in sents for t in tokenize(s)]
        return 100.0 * sum(t not in vocab for t in toks) / max(1, len(toks))
    return {"en_oov_%": rate([p[0] for p in pairs], eng_word2idx),
            "es_oov_%": rate([p[1] for p in pairs], spa_word2idx)}


# 4. Dataset / DataLoader
class TranslationDataset(Dataset):
    def __init__(self, pairs, eng_word2idx, spa_word2idx, max_len=None):
        self.data = [
            (encode(en, eng_word2idx, add_sos_eos=False, max_len=max_len) or [UNK_IDX],
             encode(es, spa_word2idx, add_sos_eos=True,
                    max_len=None if max_len is None else max_len))
            for en, es in pairs
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def collate_batch(batch):
    srcs, tgts = zip(*batch)
    src_lens = torch.tensor([len(s) for s in srcs], dtype=torch.long)
    S, T = max(src_lens).item(), max(len(t) for t in tgts)
    src = torch.full((len(batch), S), PAD_IDX, dtype=torch.long)
    tgt = torch.full((len(batch), T), PAD_IDX, dtype=torch.long)
    for i, (s, t) in enumerate(zip(srcs, tgts)):
        src[i, :len(s)] = torch.tensor(s)
        tgt[i, :len(t)] = torch.tensor(t)
    return src, src_lens, tgt


def make_loader(pairs, eng_word2idx, spa_word2idx, batch_size=64, shuffle=False,
                max_len=None, seed=23):
    ds = TranslationDataset(pairs, eng_word2idx, spa_word2idx, max_len=max_len)
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate_batch, generator=g)
