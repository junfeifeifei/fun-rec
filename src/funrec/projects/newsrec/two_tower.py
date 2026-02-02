from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import faiss
import numpy as np
import pandas as pd

from funrec.features.feature_column import FeatureColumn
from funrec.training.trainer import train_model

from .encode import IdMap


@dataclass(frozen=True)
class TwoTowerArtifacts:
    user_id_map: IdMap
    item_id_map: IdMap
    model_path: Path
    user_tower_path: Path
    item_tower_path: Path
    item_embeddings_path: Path
    faiss_index_path: Path


def load_two_tower_runtime(save_dir: Path):
    """Load trained two-tower runtime artifacts (id maps, towers, FAISS index)."""
    import tensorflow as tf

    user_id_map = IdMap.load(save_dir / "user_id_map.pkl")
    item_id_map = IdMap.load(save_dir / "item_id_map.pkl")

    user_model = tf.keras.models.load_model(save_dir / "user_tower", compile=False)
    item_model = tf.keras.models.load_model(save_dir / "item_tower", compile=False)
    index = faiss.read_index(str(save_dir / "faiss_index.bin"))
    return user_id_map, item_id_map, user_model, item_model, index


def build_two_tower_recall_candidates(
    *,
    user_model,
    index: faiss.Index,
    train_hist: pd.DataFrame,
    user_id_map: IdMap,
    item_id_map: IdMap,
    max_seq_len: int,
    topk: int = 100,
    batch_size: int = 4096,
) -> pd.DataFrame:
    """
    Generate recall candidates with a trained two-tower model + FAISS index.

    Output columns are aligned with the Chapter 5 project:
    - user_id (raw)
    - article_id (raw)
    - recall_score
    - recall_rank
    """
    train_hist = train_hist.sort_values(["user_id", "click_timestamp"])
    user_hist_series = (
        train_hist.groupby("user_id")["click_article_id"].apply(list)
    )

    users_raw = user_hist_series.index.to_numpy()
    hist_raw_list = user_hist_series.to_list()

    user_enc = user_id_map.transform(users_raw)
    valid_mask = user_enc > 0
    if not valid_mask.all():
        users_raw = users_raw[valid_mask]
        user_enc = user_enc[valid_mask]
        hist_raw_list = [seq for i, seq in enumerate(hist_raw_list) if bool(valid_mask[i])]

    # Fast raw->encoded mapping for items
    item_raw_to_enc = {int(v): int(i + item_id_map.offset) for i, v in enumerate(item_id_map.classes_)}

    hist_enc = np.zeros((len(users_raw), max_seq_len), dtype=np.int32)
    for i, seq in enumerate(hist_raw_list):
        if not seq:
            continue
        enc_seq = [item_raw_to_enc.get(int(x), 0) for x in seq][-max_seq_len:]
        hist_enc[i, -len(enc_seq) :] = np.asarray(enc_seq, dtype=np.int32)

    rows: List[Tuple[int, int, float, int]] = []
    search_k = int(topk + max_seq_len + 10)

    for start in range(0, len(users_raw), batch_size):
        end = min(start + batch_size, len(users_raw))
        feats = {"user_id": user_enc[start:end], "hist_article_id": hist_enc[start:end]}
        user_embs = user_model.predict(feats, batch_size=4096, verbose=0).astype(np.float32)
        faiss.normalize_L2(user_embs)

        D, I = index.search(user_embs, search_k)
        for local_idx, (scores, item_ids) in enumerate(zip(D, I)):
            user_raw = int(users_raw[start + local_idx])
            hist_set = set(hist_enc[start + local_idx][hist_enc[start + local_idx] > 0].tolist())

            kept: List[Tuple[int, float]] = []
            for item_id, score in zip(item_ids.tolist(), scores.tolist()):
                if item_id <= 0:
                    continue
                if int(item_id) in hist_set:
                    continue
                kept.append((int(item_id), float(score)))
                if len(kept) >= topk:
                    break

            for rank, (enc_item, score) in enumerate(kept, start=1):
                raw_item = int(item_id_map.classes_[enc_item - item_id_map.offset])
                rows.append((user_raw, raw_item, score, rank))

    return pd.DataFrame(rows, columns=["user_id", "article_id", "recall_score", "recall_rank"])


def _build_last_item_samples(
    train_hist: pd.DataFrame, max_seq_len: int
) -> pd.DataFrame:
    """One training sample per user: predict user's last clicked item in train_hist."""
    train_hist = train_hist.sort_values(["user_id", "click_timestamp"])
    rows: List[Tuple[int, List[int], int]] = []
    for user_id, group in train_hist.groupby("user_id"):
        items = group["click_article_id"].tolist()
        if len(items) < 2:
            continue
        target = items[-1]
        hist = items[:-1]
        hist = hist[-max_seq_len:]
        rows.append((int(user_id), hist, int(target)))
    return pd.DataFrame(rows, columns=["user_id", "hist_article_id", "article_id"])


def _encode_hist_sequences(
    sequences: Iterable[List[int]],
    item_id_map: IdMap,
    max_seq_len: int,
) -> np.ndarray:
    seq_list: List[List[int]] = []
    for seq in sequences:
        if not seq:
            seq_list.append([])
            continue
        seq_list.append(item_id_map.transform(np.asarray(seq)).tolist())

    out = np.zeros((len(seq_list), max_seq_len), dtype=np.int32)
    for i, seq in enumerate(seq_list):
        if not seq:
            continue
        seq = seq[-max_seq_len:]
        out[i, -len(seq) :] = np.asarray(seq, dtype=np.int32)
    return out


def train_two_tower_dssm_inbatch(
    *,
    train_hist: pd.DataFrame,
    valid_last: pd.DataFrame,
    articles: pd.DataFrame,
    save_dir: Path,
    max_seq_len: int = 30,
    emb_dim: int = 32,
    dnn_units: Optional[List[int]] = None,
    temperature: float = 0.05,
    epochs: int = 3,
    batch_size: int = 1024,
    seed: int = 42,
    verbose: int = 0,
) -> Tuple[TwoTowerArtifacts, Dict[str, float]]:
    """
    Train a two-tower retrieval model with in-batch negatives (InfoNCE).

    Notes:
    - Uses one training instance per user (predict last item in train_hist).
    - Encodes ids as 1-based, with 0 reserved for padding/unknown.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1) Build train samples
    train_df = _build_last_item_samples(train_hist, max_seq_len=max_seq_len)

    # 2) Fit encoders (include all items from articles for stable vocab)
    user_id_map = IdMap.fit("user_id", train_df["user_id"].tolist(), offset=1)
    item_id_map = IdMap.fit("article_id", articles["article_id"].tolist(), offset=1)

    # 3) Encode train features
    train_features = {
        "user_id": user_id_map.transform(train_df["user_id"].values),
        "hist_article_id": _encode_hist_sequences(
            train_df["hist_article_id"].tolist(), item_id_map, max_seq_len=max_seq_len
        ),
        "article_id": item_id_map.transform(train_df["article_id"].values),
    }
    dummy_labels = np.zeros(len(train_df), dtype=np.float32)

    # 4) Encode test features (one row per user)
    valid_last = valid_last.copy()
    valid_last["user_id_enc"] = user_id_map.transform(valid_last["user_id"].values)
    valid_last["article_id_enc"] = item_id_map.transform(
        valid_last["click_article_id"].values
    )

    # Build user history sequence for evaluation (from full train_hist)
    train_hist_sorted = train_hist.sort_values(["user_id", "click_timestamp"])
    user_hist = (
        train_hist_sorted.groupby("user_id")["click_article_id"].apply(list).to_dict()
    )

    test_users: List[int] = []
    test_hist: List[List[int]] = []
    test_labels: List[int] = []
    for u, t in zip(valid_last["user_id"].tolist(), valid_last["article_id_enc"].tolist()):
        if u not in user_hist:
            continue
        seq = user_hist[u][-max_seq_len:]
        test_users.append(u)
        test_hist.append(seq)
        test_labels.append(int(t))

    test_features = {
        "user_id": user_id_map.transform(np.asarray(test_users, dtype=np.int64)),
        "hist_article_id": _encode_hist_sequences(
            test_hist, item_id_map, max_seq_len=max_seq_len
        ),
        # item_id_col for evaluator (not used by our FAISS eval)
        "article_id": np.asarray(test_labels, dtype=np.int32),
    }

    # 5) Feature columns
    feature_columns = [
        FeatureColumn(
            name="user_id",
            group=["user"],
            type="sparse",
            vocab_size=user_id_map.vocab_size,
            emb_dim=emb_dim,
        ),
        FeatureColumn(
            name="hist_article_id",
            emb_name="article_id",
            group=["user"],
            type="varlen_sparse",
            max_len=max_seq_len,
            combiner="mean",
            vocab_size=item_id_map.vocab_size,
            emb_dim=emb_dim,
        ),
        FeatureColumn(
            name="article_id",
            group=["item"],
            type="sparse",
            vocab_size=item_id_map.vocab_size,
            emb_dim=emb_dim,
        ),
    ]

    # 6) Train model
    training_config = {
        "build_function": "funrec.models.dssm.build_dssm_inbatch_model",
        "model_params": {
            "dnn_units": dnn_units or [128, 64, 32],
            "dropout_rate": 0.1,
            "temperature": temperature,
        },
        "optimizer": "adam",
        "optimizer_params": {"learning_rate": 2e-4},
        "loss": "contrastive_loss",
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "verbose": int(verbose),
        "validation_split": 0.0,
    }

    processed_data = {
        "train": {"features": train_features, "labels": dummy_labels},
        "test": {"features": test_features, "labels": None, "eval_data": {"label_list": test_labels}},
        "feature_dict": {"user_id": user_id_map.vocab_size, "item_id": item_id_map.vocab_size},
        # 0 is reserved for padding/unknown; do not index it.
        "all_items": {"article_id": np.arange(1, item_id_map.vocab_size, dtype=np.int32)},
    }

    model, user_model, item_model = train_model(training_config, feature_columns, processed_data)

    # 7) Export embeddings + build FAISS
    item_inputs = processed_data["all_items"]
    item_embs = item_model.predict(item_inputs, batch_size=4096, verbose=0).astype(np.float32)
    user_inputs = {k: v for k, v in test_features.items() if k in ["user_id", "hist_article_id"]}
    user_embs = user_model.predict(user_inputs, batch_size=4096, verbose=0).astype(np.float32)

    faiss.normalize_L2(item_embs)
    faiss.normalize_L2(user_embs)

    # Use explicit ids so FAISS returns encoded item ids (not row positions).
    item_ids = item_inputs["article_id"].astype(np.int64)
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(item_embs.shape[1]))
    index.add_with_ids(item_embs, item_ids)

    # 8) Offline recall evaluation (filter seen items)
    hits_at_20 = 0
    ndcg_at_20 = 0.0
    k = 20
    for i in range(len(test_users)):
        hist_set = set(item_id_map.transform(np.asarray(test_hist[i], dtype=np.int64)).tolist())
        D, I = index.search(user_embs[i : i + 1], k + len(hist_set) + 1)
        recs: List[int] = []
        for item_id in I[0].tolist():
            if item_id <= 0:
                continue
            if item_id in hist_set:
                continue
            recs.append(int(item_id))
            if len(recs) >= k:
                break
        target = int(test_labels[i])
        if target in recs:
            hits_at_20 += 1
            rank = recs.index(target)
            ndcg_at_20 += 1.0 / np.log2(rank + 2)

    metrics = {
        "hit_rate@20": hits_at_20 / max(1, len(test_users)),
        "ndcg@20": ndcg_at_20 / max(1, len(test_users)),
        "num_users_eval": float(len(test_users)),
        "num_items": float(item_id_map.vocab_size - 1),
    }

    # 9) Persist artifacts
    # Use SavedModel directories for maximum compatibility with custom layers.
    model_path = save_dir / "dssm_inbatch_model"
    user_tower_path = save_dir / "user_tower"
    item_tower_path = save_dir / "item_tower"
    item_embeddings_path = save_dir / "item_embeddings.npy"
    faiss_index_path = save_dir / "faiss_index.bin"

    model.save(model_path)
    user_model.save(user_tower_path)
    item_model.save(item_tower_path)
    np.save(item_embeddings_path, item_embs)
    faiss.write_index(index, str(faiss_index_path))

    user_id_map.dump(save_dir / "user_id_map.pkl")
    item_id_map.dump(save_dir / "item_id_map.pkl")
    with open(save_dir / "metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)

    artifacts = TwoTowerArtifacts(
        user_id_map=user_id_map,
        item_id_map=item_id_map,
        model_path=model_path,
        user_tower_path=user_tower_path,
        item_tower_path=item_tower_path,
        item_embeddings_path=item_embeddings_path,
        faiss_index_path=faiss_index_path,
    )
    return artifacts, metrics
