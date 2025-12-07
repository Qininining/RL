"""Configuration helpers for the snake game project."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional


@dataclass
class EnvironmentConfig:
    size: int = 6
    max_steps: int = 200


@dataclass
class TrainingConfig:
    episodes: int = 20000
    learning_rate: float = 0.1
    discount_factor: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.9995
    seed: Optional[int] = 42
    eval_interval: int = 500
    log_interval: int = 500
    render_mode: Optional[str] = None


@dataclass
class EvaluationConfig:
    episodes: int = 5
    render_mode: Optional[str] = "human"


@dataclass
class ProjectConfig:
    env: EnvironmentConfig
    training: TrainingConfig
    evaluation: EvaluationConfig

    def to_dict(self) -> Dict[str, Any]:
        return {
            "env": vars(self.env),
            "training": vars(self.training),
            "evaluation": vars(self.evaluation),
        }


def parse_config_from_dict(data: Mapping[str, Any] | None) -> ProjectConfig:
    """Convert a raw mapping (typically YAML/JSON) into ProjectConfig."""

    payload: Mapping[str, Any] = data or {}
    env_data = payload.get("env", {})
    training_data = payload.get("training", {})
    evaluation_data = payload.get("evaluation", {})

    if "render_mode" in env_data:
        del env_data["render_mode"]

    env_cfg = EnvironmentConfig(**env_data)
    training_cfg = TrainingConfig(**training_data)
    evaluation_cfg = EvaluationConfig(**evaluation_data)
    return ProjectConfig(env=env_cfg, training=training_cfg, evaluation=evaluation_cfg)


__all__ = [
    "EnvironmentConfig",
    "TrainingConfig",
    "EvaluationConfig",
    "ProjectConfig",
    "parse_config_from_dict",
]
