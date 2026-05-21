from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml


@dataclass
class ProjectPaths:
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    model_dir: Path = Path("models")
    result_dir: Path = Path("results")
    figure_dir: Path = Path("figures")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectPaths":
        return cls(**{k: Path(v) for k, v in data.items()})

    def make_dirs(self) -> None:
        for path in [self.raw_dir, self.processed_dir, self.model_dir, self.result_dir, self.figure_dir]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    project_name: str = "ham10000_trustworthy_skin_ai"
    seed: int = 42
    image_size: int = 224
    batch_size: int = 8
    backbone: str = "ResNet50"
    loss_type: str = "weighted_ce"
    use_metadata: bool = False
    head_epochs: int = 8
    finetune_epochs: int = 20
    finetune_last_n_layers: int = 40
    dropout_rate: float = 0.30
    head_learning_rate: float = 1e-3
    finetune_learning_rate: float = 1e-5
    early_stopping_patience: int = 5
    focal_gamma: float = 2.0
    class_balanced_beta: float = 0.9999
    n_bins_ece: int = 15
    bootstrap_reps: int = 1000
    min_subgroup_n: int = 30
    mc_dropout_passes: int = 20
    class_names: list[str] = field(default_factory=lambda: ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"])
    paths: ProjectPaths = field(default_factory=ProjectPaths)

    @property
    def class_to_idx(self) -> dict[str, int]:
        return {name: idx for idx, name in enumerate(self.class_names)}


def load_config(path: str | Path = "configs/default.yaml") -> Config:
    data = yaml.safe_load(Path(path).read_text())
    paths = ProjectPaths.from_dict(data.pop("paths", {}))
    cfg = Config(**data, paths=paths)
    cfg.paths.make_dirs()
    return cfg
