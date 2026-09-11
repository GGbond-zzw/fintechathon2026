"""Central path registry sourced only from config.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    train: Path
    test: Path
    interim: Path
    processed: Path
    cache: Path
    models: Path
    experiments: Path
    submissions: Path
    reports: Path
    logs: Path

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ProjectPaths":
        values = config["paths"]
        return cls(**{field: Path(values[field]) for field in cls.__annotations__})

    def create_generated_directories(self) -> None:
        """Create project-owned outputs; never create or alter raw input files."""
        for path in (
            self.interim,
            self.processed,
            self.cache,
            self.models,
            self.experiments,
            self.submissions,
            self.reports,
            self.logs,
        ):
            path.mkdir(parents=True, exist_ok=True)

