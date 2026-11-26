import os

import pytest
import sacrebleu
import torch

from src.data import (EOS_IDX, PAD_IDX, SOS_IDX, UNK_IDX, build_vocabs, decode, encode,
                      length_bucket, load_and_split, make_loader, normalize, oov_rate,
                      tokenize)
from src.utils import count_trainable_params

SPA = os.path.join(os.path.dirname(__file__), "..", "spa.txt")
pytestmark = pytest.mark.skipif(not os.path.exists(SPA), reason="falta spa.txt en la raíz")


@pytest.fixture(scope="module")
def splits():
    return load_and_split(SPA, verbose=False)


@pytest.fixture(scope="module")
def vocabs(splits):
    return build_vocabs(splits[0], min_freq=2)


def test_particion_y_audit_id(splits):
    train, val, test, h = splits
    assert h == "dbac04ac5648"
    assert (len(train), len(val), len(test)) == (29130, 3641, 3642)
    assert all(en and es for en, es in train + val + test)


def test_normalize():
    assert normalize("¿Qué año nació Él?") == "que ano nacio el"
    assert normalize("I don't think Tom's ready.") == "i dont think toms ready"
    assert normalize("Él comió el pingüino.") == "el comio el pinguino"


def test_normalize_idempotente(splits):
    for _, es in splits[2]:
        assert normalize(normalize(es)) == normalize(es)


def test_tokenize(splits):
    en = splits[2][0][0]
    assert tokenize(en) == normalize(en).split()


def test_length_bucket(splits):
    buckets = [length_bucket(en) for en, _ in splits[2]]
    assert (buckets.count("medium"), buckets.count("long")) == (3611, 31)


def test_vocabularios(vocabs):
    e2i, i2e, s2i, _ = vocabs
    assert (len(e2i), len(s2i)) == (5529, 8026)
    assert [e2i[w] for w in ["<pad>", "<sos>", "<eos>", "<unk>"]] == [0, 1, 2, 3]
    assert all(i2e[i] == w for w, i in e2i.items())


def test_oov(splits, vocabs):
    e2i, _, s2i, _ = vocabs
    rates = oov_rate(splits[2], e2i, s2i)
    assert 0 < rates["en_oov_%"] < 5 and 0 < rates["es_oov_%"] < 10


def test_encode_decode(splits, vocabs):
    e2i, _, s2i, i2s = vocabs
    en, es = splits[2][0]
    assert encode(en, e2i) == [7, 20, 5, 354, 4, 1290, 3, 4227]
    ids = encode(es, s2i, add_sos_eos=True)
    assert ids[0] == SOS_IDX and ids[-1] == EOS_IDX
    expected = " ".join(t if t in s2i else "<unk>" for t in tokenize(es))
    assert decode(ids, i2s) == expected
    assert encode("qwertyzzz", e2i) == [UNK_IDX]
    assert len(encode(es, s2i, add_sos_eos=True, max_len=3)) == 5


def test_loader(splits, vocabs):
    e2i, _, s2i, _ = vocabs
    loader = make_loader(splits[0], e2i, s2i, batch_size=64, shuffle=True)
    src, lens, tgt = next(iter(loader))
    assert src.shape[0] == tgt.shape[0] == 64
    assert torch.equal(lens, (src != PAD_IDX).sum(1))
    assert int(lens.min()) >= 1
    assert lens.device.type == "cpu"
    again = next(iter(make_loader(splits[0], e2i, s2i, batch_size=64, shuffle=True)))[0]
    assert torch.equal(src, again)


def test_largo_maximo(splits, vocabs):
    e2i, _, s2i, _ = vocabs
    pairs = [p for split in splits[:3] for p in split]
    assert max(len(encode(en, e2i)) for en, _ in pairs) <= 128
    assert max(len(encode(es, s2i, add_sos_eos=True)) for _, es in pairs) <= 128


def test_bleu_identidad(splits):
    refs = [normalize(es) for _, es in splits[2][:200]]
    assert sacrebleu.corpus_bleu(refs, [refs]).score == pytest.approx(100.0)


def test_count_params():
    assert count_trainable_params(torch.nn.Linear(10, 5)) == 55
