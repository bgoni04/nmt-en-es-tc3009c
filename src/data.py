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


