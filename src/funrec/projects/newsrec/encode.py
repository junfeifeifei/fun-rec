from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class IdMap:
    """1-based id encoder with 0 reserved for padding/unknown."""

    name: str
    classes_: np.ndarray
    offset: int = 1
    unknown_value: int = 0

    def __post_init__(self) -> None:
        pd.Index(self.classes_)  # validate serializable / hashable

    @property
    def vocab_size(self) -> int:
        return int(len(self.classes_) + self.offset)

    def transform(self, values: Any) -> np.ndarray:
        index = pd.Index(self.classes_)
        arr = np.asarray(values)
        flat = arr.reshape(-1)
        idx = index.get_indexer(flat)
        out = idx.astype(np.int64) + self.offset
        out[idx < 0] = self.unknown_value
        return out.reshape(arr.shape).astype(np.int32)

    def inverse_transform(self, indices: Any) -> np.ndarray:
        arr = np.asarray(indices)
        flat = arr.reshape(-1).astype(np.int64)
        out = np.empty_like(flat, dtype=object)
        unknown_mask = flat == self.unknown_value
        valid = ~unknown_mask
        out[unknown_mask] = None
        if valid.any():
            raw = flat[valid] - self.offset
            out[valid] = self.classes_[raw]
        return out.reshape(arr.shape)

    @classmethod
    def fit(cls, name: str, values: Iterable[Any], offset: int = 1) -> "IdMap":
        uniq = pd.unique(pd.Series(list(values)))
        try:
            uniq = np.array(sorted(uniq))
        except Exception:
            uniq = np.array(list(uniq))
        return cls(name=name, classes_=uniq, offset=offset)

    def dump(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "IdMap":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, IdMap):
            raise TypeError(f"Unexpected object type in {path}: {type(obj)}")
        return obj


def pad_sequences(
    sequences: Iterable[Iterable[int]],
    max_len: int,
    pad_value: int = 0,
    dtype: str = "int32",
) -> np.ndarray:
    seq_list = [list(seq) for seq in sequences]
    out = np.full((len(seq_list), max_len), pad_value, dtype=dtype)
    for i, seq in enumerate(seq_list):
        if not seq:
            continue
        seq = seq[-max_len:]
        out[i, -len(seq) :] = np.asarray(seq, dtype=dtype)
    return out
