"""Configuration helpers for the peak seeking project."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional


@dataclass
class EnvironmentConfig:
    size: int = 20
    max_steps: int = 100
    threshold_ratio: float = 0.98
    peak_value_min: float = 5.0
    peak_value_max: float = 10.0
    render_mode: Optional[str] = None


@dataclass
class TrainingConfig:
    episodes: int = 200
    learning_rate: float = 0.2
    discount_factor: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.99
    seed: Optional[int] = 42
    eval_interval: int = 20
    log_interval: int = 10


@dataclass
class ProjectConfig:
    env: EnvironmentConfig
    training: TrainingConfig

    def to_dict(self) -> Dict[str, Any]:
        return {
            "env": vars(self.env),
            "training": vars(self.training),
        }


def parse_config_from_dict(data: Mapping[str, Any] | None) -> ProjectConfig:
    """Convert a raw mapping (typically YAML/JSON) into ProjectConfig."""

    payload: Mapping[str, Any] = data or {}
    env_data = payload.get("env", {})
    training_data = payload.get("training", {})

    env_cfg = EnvironmentConfig(**env_data)
    training_cfg = TrainingConfig(**training_data)
    return ProjectConfig(env=env_cfg, training=training_cfg)


__all__ = [
    "EnvironmentConfig",
    "TrainingConfig",
    "ProjectConfig",
    "parse_config_from_dict",
]
