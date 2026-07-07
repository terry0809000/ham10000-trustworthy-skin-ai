from pathlib import Path

from ham10000_ai.config import load_config


def test_load_config_accepts_grouped_v4_yaml(tmp_path):
    cfg_path = tmp_path / "grouped.yaml"
    cfg_path.write_text(
        f"""
project_name: grouped_v4
reproducibility:
  seed: 123
  force_retrain: true
model:
  image_size: 128
  batch_size: 4
  backbone: EfficientNetB0
  class_names: [akiec, bcc]
imbalance:
  loss_type: focal
  focal_gamma: 3.0
evaluation:
  n_bins_ece: 10
unknown_notebook_only_section:
  reviewer_switch: true
paths:
  raw_dir: {tmp_path / "raw"}
  processed_dir: {tmp_path / "processed"}
  model_dir: {tmp_path / "models"}
  result_dir: {tmp_path / "results"}
  figure_dir: {tmp_path / "figures"}
""",
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)

    assert cfg.project_name == "grouped_v4"
    assert cfg.seed == 123
    assert cfg.image_size == 128
    assert cfg.batch_size == 4
    assert cfg.backbone == "EfficientNetB0"
    assert cfg.loss_type == "focal"
    assert cfg.focal_gamma == 3.0
    assert cfg.n_bins_ece == 10
    assert cfg.class_names == ["akiec", "bcc"]
    assert cfg.paths.raw_dir == Path(tmp_path / "raw")
    assert cfg.paths.raw_dir.exists()


def test_load_config_accepts_original_flat_yaml(tmp_path):
    cfg_path = tmp_path / "flat.yaml"
    cfg_path.write_text(
        f"""
project_name: flat_config
seed: 42
image_size: 224
batch_size: 8
backbone: ResNet50
paths:
  raw_dir: {tmp_path / "raw"}
  processed_dir: {tmp_path / "processed"}
  model_dir: {tmp_path / "models"}
  result_dir: {tmp_path / "results"}
  figure_dir: {tmp_path / "figures"}
""",
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)

    assert cfg.project_name == "flat_config"
    assert cfg.seed == 42
    assert cfg.image_size == 224
    assert cfg.backbone == "ResNet50"
    assert cfg.paths.figure_dir.exists()
