from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from funrec.utils import load_env_with_fallback


@dataclass(frozen=True)
class NewsRecPaths:
    data_path: Path
    project_path: Path

    @property
    def artifacts_path(self) -> Path:
        path = self.project_path / "artifacts"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def two_tower_path(self) -> Path:
        path = self.artifacts_path / "two_tower"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def ranking_path(self) -> Path:
        path = self.artifacts_path / "ranking"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def sequence_path(self) -> Path:
        path = self.artifacts_path / "sequence"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def multitask_path(self) -> Path:
        path = self.artifacts_path / "multitask"
        path.mkdir(parents=True, exist_ok=True)
        return path


def resolve_newsrec_paths() -> NewsRecPaths:
    load_env_with_fallback()

    raw_data_path = Path(os.getenv("FUNREC_RAW_DATA_PATH"))
    processed_data_path = Path(os.getenv("FUNREC_PROCESSED_DATA_PATH"))

    data_path = raw_data_path / "dataset" / "news_recommendation"
    if not data_path.exists():
        data_path = raw_data_path / "news_recommendation"

    project_path = processed_data_path / "projects" / "news_recommendation_system"
    project_path.mkdir(parents=True, exist_ok=True)

    return NewsRecPaths(data_path=data_path, project_path=project_path)

