from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import kagglehub
import yaml


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")
SPLITS = ("train", "valid", "test")


@dataclass(frozen=True)
class Config:
    dataset_handle: str
    dataset_dir: Path | None
    project: Path
    run_name: str
    weights: str
    epochs: int
    image_size: int
    batch_size: int
    device: str | int | None
    workers: int
    seed: int
    patience: int
    eval_split: str
    save_json: bool


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path.resolve()}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return data


def section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"Config section {name!r} must be a mapping.")
    return value


def optional_path(value: Any) -> Path | None:
    if value is None or not str(value).strip():
        return None
    return Path(str(value))


def optional_device(value: Any) -> str | int | None:
    if value is None or str(value).strip() == "":
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    return int(text) if text.isdigit() else text


def load_config(config_path: Path) -> Config:
    raw = load_yaml(config_path)
    dataset = section(raw, "dataset")
    output = section(raw, "output")
    model = section(raw, "model")
    training = section(raw, "training")
    evaluation = section(raw, "evaluation")

    return Config(
        dataset_handle=str(
            dataset.get(
                "handle",
                "ahsan71/ecodetect-recyclable-waste-detection-dataset",
            )
        ),
        dataset_dir=optional_path(dataset.get("dir")),
        project=Path(str(output.get("project", "artifacts/ecodetect/yolov11"))),
        run_name=str(output.get("name", "train")),
        weights=str(model.get("weights", "yolo11n.pt")),
        epochs=int(training.get("epochs", 50)),
        image_size=int(training.get("image_size", 640)),
        batch_size=int(training.get("batch_size", 16)),
        device=optional_device(training.get("device", 0)),
        workers=int(training.get("workers", 2)),
        seed=int(training.get("seed", 42)),
        patience=int(training.get("patience", 20)),
        eval_split=str(evaluation.get("split", "test")),
        save_json=bool(evaluation.get("save_json", True)),
    )


def parse_config_path() -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args, _ = parser.parse_known_args()
    return args.config


def parse_args(config: Config, config_path: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate YOLOv11 on the EcoDetect detection dataset."
    )
    parser.add_argument("--config", type=Path, default=config_path)
    parser.add_argument("--dataset-dir", type=Path, default=config.dataset_dir)
    parser.add_argument("--project", type=Path, default=config.project)
    parser.add_argument("--name", default=config.run_name)
    parser.add_argument("--weights", default=config.weights)
    parser.add_argument("--epochs", type=int, default=config.epochs)
    parser.add_argument("--imgsz", type=int, default=config.image_size)
    parser.add_argument("--batch", type=int, default=config.batch_size)
    parser.add_argument("--device", default=config.device)
    parser.add_argument("--workers", type=int, default=config.workers)
    parser.add_argument("--seed", type=int, default=config.seed)
    parser.add_argument("--patience", type=int, default=config.patience)
    parser.add_argument("--eval-split", default=config.eval_split)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate dataset paths and write the prepared data.yaml without training.",
    )
    return parser.parse_args()


def find_cached_kaggle_dataset(handle: str) -> Path | None:
    owner, dataset = handle.split("/", maxsplit=1)
    versions_dir = (
        Path.home()
        / ".cache"
        / "kagglehub"
        / "datasets"
        / owner
        / dataset
        / "versions"
    )
    if not versions_dir.is_dir():
        return None

    version_dirs = [path for path in versions_dir.iterdir() if path.is_dir()]
    if not version_dirs:
        return None

    def version_key(path: Path) -> tuple[int, str]:
        return (int(path.name), path.name) if path.name.isdigit() else (-1, path.name)

    return max(version_dirs, key=version_key)


def find_yolo_root(root: Path) -> Path | None:
    candidates = [root, *[path for path in root.rglob("*") if path.is_dir()]]
    for candidate in candidates:
        if all(
            (candidate / split / "images").is_dir()
            and (candidate / split / "labels").is_dir()
            for split in SPLITS
        ):
            return candidate
    return None


def resolve_dataset_dir(dataset_dir: Path | None, handle: str) -> Path:
    root = dataset_dir or find_cached_kaggle_dataset(handle)
    if root is None:
        root = Path(kagglehub.dataset_download(handle))
    root = root.expanduser().resolve()

    yolo_root = find_yolo_root(root)
    if yolo_root is None:
        raise FileNotFoundError(f"Could not find YOLO train/valid/test folders under {root}.")
    return yolo_root


def read_class_names(dataset_dir: Path) -> list[str]:
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.is_file():
        return ["aluminum", "paper", "plastic"]
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    names = data.get("names", ["aluminum", "paper", "plastic"])
    if isinstance(names, dict):
        return [names[index] for index in sorted(names)]
    return list(names)


def count_files(dataset_dir: Path) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        image_dir = dataset_dir / split / "images"
        label_dir = dataset_dir / split / "labels"
        counts[split] = {
            "images": sum(1 for path in image_dir.iterdir() if path.is_file()),
            "labels": sum(1 for path in label_dir.iterdir() if path.is_file()),
        }
    return counts


def write_prepared_data_yaml(dataset_dir: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    class_names = read_class_names(dataset_dir)
    prepared = {
        "path": str(dataset_dir),
        "train": str((dataset_dir / "train" / "images").resolve()),
        "val": str((dataset_dir / "valid" / "images").resolve()),
        "test": str((dataset_dir / "test" / "images").resolve()),
        "nc": len(class_names),
        "names": class_names,
    }
    data_yaml = output_dir / "ecodetect_data.yaml"
    data_yaml.write_text(yaml.safe_dump(prepared, sort_keys=False), encoding="utf-8")
    return data_yaml


def metrics_to_jsonable(metrics: Any) -> dict[str, Any]:
    box = getattr(metrics, "box", None)
    return {
        "map50_95": float(getattr(box, "map", 0.0)) if box is not None else None,
        "map50": float(getattr(box, "map50", 0.0)) if box is not None else None,
        "map75": float(getattr(box, "map75", 0.0)) if box is not None else None,
        "mean_precision": float(getattr(box, "mp", 0.0)) if box is not None else None,
        "mean_recall": float(getattr(box, "mr", 0.0)) if box is not None else None,
    }


def main() -> None:
    config_path = parse_config_path()
    config = load_config(config_path)
    args = parse_args(config, config_path)

    dataset_dir = resolve_dataset_dir(args.dataset_dir, config.dataset_handle)
    run_root = args.project / args.name
    data_yaml = write_prepared_data_yaml(dataset_dir, run_root)
    counts = count_files(dataset_dir)

    print(f"Dataset root: {dataset_dir}")
    print(f"Prepared data.yaml: {data_yaml}")
    print(f"Classes: {read_class_names(dataset_dir)}")
    for split, split_counts in counts.items():
        print(f"{split}: {split_counts['images']} images, {split_counts['labels']} labels")

    if args.check_only:
        print("Check complete. No training was run.")
        return

    from ultralytics import YOLO

    model = YOLO(args.weights)
    train_results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        seed=args.seed,
        patience=args.patience,
        plots=True,
    )

    save_dir = Path(train_results.save_dir)
    best_weights = save_dir / "weights" / "best.pt"
    eval_model = YOLO(str(best_weights if best_weights.is_file() else save_dir / "weights" / "last.pt"))
    metrics = eval_model.val(
        data=str(data_yaml),
        split=args.eval_split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(args.project),
        name=f"{args.name}_{args.eval_split}",
        exist_ok=True,
        plots=True,
        save_json=config.save_json,
    )

    summary = {
        "dataset_dir": str(dataset_dir),
        "data_yaml": str(data_yaml),
        "train_save_dir": str(save_dir),
        "eval_save_dir": str(metrics.save_dir),
        "best_weights": str(best_weights),
        "counts": counts,
        "metrics": metrics_to_jsonable(metrics),
    }
    (save_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary["metrics"], indent=2))
    print(f"YOLOv11 artifacts saved under: {save_dir}")
    print(f"Evaluation artifacts saved under: {metrics.save_dir}")


if __name__ == "__main__":
    main()
