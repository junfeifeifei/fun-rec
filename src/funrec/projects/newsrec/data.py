from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OfflineSplit:
    train_hist: pd.DataFrame
    valid_last: pd.DataFrame
    user_hist: Dict[int, List[int]]
    valid_last_map: Dict[int, int]


def build_offline_split_last_click(click_df: pd.DataFrame) -> OfflineSplit:
    click_df = click_df.sort_values(["user_id", "click_timestamp"])
    last_click = click_df.groupby("user_id").tail(1)
    hist = click_df.drop(last_click.index)

    valid_users = hist["user_id"].unique()
    hist = hist[hist["user_id"].isin(valid_users)]
    last_click = last_click[last_click["user_id"].isin(valid_users)]

    hist = hist.reset_index(drop=True)
    last_click = last_click.reset_index(drop=True)

    user_hist = (
        hist.sort_values(["user_id", "click_timestamp"])
        .groupby("user_id")["click_article_id"]
        .apply(list)
        .to_dict()
    )
    valid_last_map = dict(zip(last_click["user_id"], last_click["click_article_id"]))

    return OfflineSplit(
        train_hist=hist, valid_last=last_click, user_hist=user_hist, valid_last_map=valid_last_map
    )


def sample_users(
    df: pd.DataFrame, max_users: int, seed: int = 42
) -> pd.DataFrame:
    if max_users <= 0:
        return df
    users = df["user_id"].unique()
    if len(users) <= max_users:
        return df
    rng = np.random.default_rng(seed)
    sample_users = rng.choice(users, size=max_users, replace=False)
    return df[df["user_id"].isin(sample_users)]

