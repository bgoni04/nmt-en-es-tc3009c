import json
import os
import random
import time
from contextlib import contextmanager

import numpy as np
import torch


def set_seed(seed=23):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


@contextmanager
def timer(device=None):
    out = {}
    if device is not None:
        sync(device)
    start = time.perf_counter()
    yield out
    if device is not None:
        sync(device)
    out["s"] = time.perf_counter() - start


def results_dir(base=None):
    path = base or os.environ.get("NMT_RESULTS_DIR", "results")
    os.makedirs(path, exist_ok=True)
    return path


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def gpu_name():
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
