from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as torch_F
import yaml
from PIL import Image, ImageDraw, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import boxes as box_ops
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.roi_heads import maskrcnn_inference


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")
TACO10_OTHER = "Other Litter"
TACO10_CATEGORY_MAP = {
    "Aerosol": "Can",
    "Aluminium foil": TACO10_OTHER,
    "Battery": TACO10_OTHER,
    "Aluminium blister pack": TACO10_OTHER,
    "Carded blister pack": TACO10_OTHER,
    "Clear plastic bottle": "Bottle",
    "Glass bottle": "Bottle",
    "Other plastic bottle": "Bottle",
    "Plastic bottle cap": "Bottle cap",
    "Metal bottle cap": "Bottle cap",
    "Broken glass": TACO10_OTHER,
    "Drink can": "Can",
    "Food Can": "Can",
    "Corrugated carton": TACO10_OTHER,
    "Drink carton": TACO10_OTHER,
    "Egg carton": TACO10_OTHER,
    "Meal carton": TACO10_OTHER,
    "Other carton": TACO10_OTHER,
    "Paper cup": "Cup",
    "Disposable plastic cup": "Cup",
    "Foam cup": "Cup",
    "Glass cup": "Cup",
    "Other plastic cup": "Cup",
    "Food waste": TACO10_OTHER,
    "Plastic lid": "Lid",
    "Metal lid": "Lid",
    "Magazine paper": TACO10_OTHER,
    "Tissues": TACO10_OTHER,
    "Wrapping paper": TACO10_OTHER,
    "Normal paper": TACO10_OTHER,
    "Paper bag": TACO10_OTHER,
    "Plastified paper bag": TACO10_OTHER,
    "Pizza box": TACO10_OTHER,
    "Garbage bag": "Plastic bag + wrapper",
    "Single-use carrier bag": "Plastic bag + wrapper",
    "Polypropylene bag": "Plastic bag + wrapper",
    "Produce bag": "Plastic bag + wrapper",
    "Cereal bag": "Plastic bag + wrapper",
    "Bread bag": "Plastic bag + wrapper",
    "Plastic film": "Plastic bag + wrapper",
    "Crisp packet": "Plastic bag + wrapper",
    "Other plastic wrapper": "Plastic bag + wrapper",
    "Retort pouch": "Plastic bag + wrapper",
    "Spread tub": TACO10_OTHER,
    "Tupperware": TACO10_OTHER,
    "Disposable food container": TACO10_OTHER,
    "Foam food container": TACO10_OTHER,
    "Other plastic container": TACO10_OTHER,
    "Plastic glooves": TACO10_OTHER,
    "Plastic utensils": TACO10_OTHER,
    "Pop tab": "Pop tab",
    "Rope & strings": TACO10_OTHER,
    "Scrap metal": TACO10_OTHER,
    "Shoe": TACO10_OTHER,
    "Six pack rings": "Plastic bag + wrapper",
    "Squeezable tube": TACO10_OTHER,
    "Plastic straw": "Straw",
    "Paper straw": "Straw",
    "Styrofoam piece": TACO10_OTHER,
    "Toilet tube": TACO10_OTHER,
    "Unlabeled litter": TACO10_OTHER,
    "Glass jar": TACO10_OTHER,
    "Other plastic": TACO10_OTHER,
    "Cigarette": "Cigarette",
}
TACO10_CLASS_NAMES = [
    "Bottle",
    "Bottle cap",
    "Can",
    "Cigarette",
    "Cup",
    "Lid",
    TACO10_OTHER,
    "Plastic bag + wrapper",
    "Pop tab",
    "Straw",
]


@dataclass(frozen=True)
class Config:
    dataset_dir: Path | None
    annotation_file: str
    image_extensions: frozenset[str]
    taxonomy: str
    category_field: str
    output_dir: Path
    pretrained: bool
    weights: str | None
    batch_size: int
    epochs: int
    learning_rate: float
    momentum: float
    weight_decay: float
    workers: int
    seed: int
    val_fraction: float
    test_fraction: float
    patience: int
    device: str | None
    horizontal_flip_probability: float
    rotation_degrees: float
    object_crop_probability: float
    object_crop_scale: tuple[float, float]
    brightness: float
    contrast: float
    saturation: float
    hue: float
    blur_probability: float
    blur_kernel_size: int
    noise_probability: float
    noise_std: float
    evaluation_score_threshold: float
    cross_validation: bool
    folds: int


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


def optional_text(value: Any) -> str | None:
    if value is None or not str(value).strip():
        return None
    return str(value)


def extensions(value: Any) -> frozenset[str]:
    if isinstance(value, str):
        parts = value.split(",")
    else:
        parts = value
    return frozenset(str(part).strip().lower() for part in parts if str(part).strip())


def float_pair(value: Any, default: tuple[float, float]) -> tuple[float, float]:
    if value is None:
        return default
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ValueError(f"Expected a two-value list, got: {value!r}")
    return float(value[0]), float(value[1])


def load_config(config_path: Path) -> Config:
    raw = load_yaml(config_path)
    dataset = section(raw, "dataset")
    output = section(raw, "output")
    model = section(raw, "model")
    training = section(raw, "training")
    augmentation = section(raw, "augmentation")
    evaluation = section(raw, "evaluation")
    cross_validation = section(raw, "cross_validation")

    return Config(
        dataset_dir=optional_path(dataset.get("dir")),
        annotation_file=str(dataset.get("annotation_file", "annotations.json")),
        image_extensions=extensions(dataset.get("image_extensions", [".jpg", ".jpeg", ".png"])),
        taxonomy=str(dataset.get("taxonomy", "taco10")),
        category_field=str(dataset.get("category_field", "supercategory")),
        output_dir=Path(str(output.get("dir", "artifacts/taco/maskrcnn_taco10_cv_70_15_15_coco_metrics"))),
        pretrained=bool(model.get("pretrained", True)),
        weights=optional_text(model.get("weights", "DEFAULT")),
        batch_size=int(training.get("batch_size", 2)),
        epochs=int(training.get("epochs", 20)),
        learning_rate=float(training.get("learning_rate", 0.005)),
        momentum=float(training.get("momentum", 0.9)),
        weight_decay=float(training.get("weight_decay", 0.0005)),
        workers=int(training.get("workers", 0)),
        seed=int(training.get("seed", 42)),
        val_fraction=float(training.get("val_fraction", 0.15)),
        test_fraction=float(training.get("test_fraction", 0.15)),
        patience=int(training.get("patience", 5)),
        device=optional_text(training.get("device")),
        horizontal_flip_probability=float(augmentation.get("horizontal_flip_probability", 0.5)),
        rotation_degrees=float(augmentation.get("rotation_degrees", 0.0)),
        object_crop_probability=float(augmentation.get("object_crop_probability", 0.0)),
        object_crop_scale=float_pair(augmentation.get("object_crop_scale"), (0.65, 1.0)),
        brightness=float(augmentation.get("brightness", 0.0)),
        contrast=float(augmentation.get("contrast", 0.0)),
        saturation=float(augmentation.get("saturation", 0.0)),
        hue=float(augmentation.get("hue", 0.0)),
        blur_probability=float(augmentation.get("blur_probability", 0.0)),
        blur_kernel_size=int(augmentation.get("blur_kernel_size", 5)),
        noise_probability=float(augmentation.get("noise_probability", 0.0)),
        noise_std=float(augmentation.get("noise_std", 0.0)),
        evaluation_score_threshold=float(evaluation.get("score_threshold", 0.001)),
        cross_validation=bool(cross_validation.get("enabled", False)),
        folds=int(cross_validation.get("folds", 4)),
    )


def parse_config_path() -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args, _ = parser.parse_known_args()
    return args.config


def parse_args(config: Config, config_path: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Torchvision Mask R-CNN on the TACO instance-segmentation dataset."
    )
    parser.add_argument("--config", type=Path, default=config_path)
    parser.add_argument("--dataset-dir", type=Path, default=config.dataset_dir)
    parser.add_argument("--annotation-file", default=config.annotation_file)
    parser.add_argument("--output-dir", type=Path, default=config.output_dir)
    parser.add_argument(
        "--taxonomy",
        choices=["taco10", "category-field"],
        default=config.taxonomy,
        help="Use the paper-style TACO-10 mapping or group by --category-field.",
    )
    parser.add_argument("--category-field", default=config.category_field)
    parser.add_argument("--batch-size", type=int, default=config.batch_size)
    parser.add_argument("--epochs", type=int, default=config.epochs)
    parser.add_argument("--learning-rate", type=float, default=config.learning_rate)
    parser.add_argument("--workers", type=int, default=config.workers)
    parser.add_argument("--seed", type=int, default=config.seed)
    parser.add_argument("--val-fraction", type=float, default=config.val_fraction)
    parser.add_argument("--test-fraction", type=float, default=config.test_fraction)
    parser.add_argument("--patience", type=int, default=config.patience)
    parser.add_argument("--device", default=config.device)
    parser.add_argument(
        "--cross-validation",
        action=argparse.BooleanOptionalAction,
        default=config.cross_validation,
        help="Train/evaluate rotated folds instead of one random train/val/test split.",
    )
    parser.add_argument("--folds", type=int, default=config.folds)
    parser.add_argument(
        "--evaluation-score-threshold",
        type=float,
        default=config.evaluation_score_threshold,
        help="Minimum prediction score included in COCO AP evaluation.",
    )
    parser.add_argument(
        "--paper-score-eval-only",
        action="store_true",
        help=(
            "Load saved fold checkpoints and evaluate TACO-paper-style "
            "class, litter, and ratio prediction scores without retraining."
        ),
    )
    parser.add_argument(
        "--checkpoint-name",
        default="best_model.pth",
        help="Checkpoint filename to load inside each fold directory for --paper-score-eval-only.",
    )
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str | None) -> torch.device:
    if device_name is None or device_name.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested device {device_name!r}, but CUDA is not available. "
            "Use device: auto or --device cpu."
        )
    return device


def describe_device(device: torch.device) -> str:
    if device.type != "cuda":
        return str(device)
    index = device.index if device.index is not None else torch.cuda.current_device()
    return f"{device} ({torch.cuda.get_device_name(index)})"


def resolve_dataset_dir(dataset_dir: Path | None, annotation_file: str) -> Path:
    if dataset_dir is None:
        raise ValueError(
            "Set dataset.dir in TACO/config.yaml or pass --dataset-dir pointing to "
            "the TACO folder that contains annotations.json."
        )
    root = dataset_dir.expanduser().resolve()
    if not (root / annotation_file).is_file():
        raise FileNotFoundError(f"Could not find {annotation_file} under {root}.")
    return root


def load_coco_annotations(dataset_dir: Path, annotation_file: str) -> dict[str, Any]:
    path = dataset_dir / annotation_file
    data = json.loads(path.read_text(encoding="utf-8"))
    required = {"images", "annotations", "categories"}
    missing = required.difference(data)
    if missing:
        raise ValueError(f"{path} is missing COCO keys: {sorted(missing)}")
    return data


def category_name(category: dict[str, Any], field: str) -> str:
    value = category.get(field)
    if value is None or not str(value).strip():
        value = category.get("name")
    if value is None or not str(value).strip():
        raise ValueError(f"Category is missing both {field!r} and 'name': {category}")
    return str(value).strip()


def build_category_map(
    categories: list[dict[str, Any]],
    taxonomy: str,
    category_field: str,
) -> tuple[dict[int, int], list[str]]:
    if taxonomy == "taco10":
        class_to_id = {name: index + 1 for index, name in enumerate(TACO10_CLASS_NAMES)}
        raw_id_to_label = {}
        for category in categories:
            name = str(category["name"]).strip()
            if name not in TACO10_CATEGORY_MAP:
                raise ValueError(f"TACO-10 mapping is missing category: {name}")
            raw_id_to_label[int(category["id"])] = class_to_id[TACO10_CATEGORY_MAP[name]]
        return raw_id_to_label, ["background", *TACO10_CLASS_NAMES]

    raw_id_to_group: dict[int, str] = {}
    for category in categories:
        raw_id_to_group[int(category["id"])] = category_name(category, category_field)

    class_names = sorted(set(raw_id_to_group.values()))
    class_to_id = {name: index + 1 for index, name in enumerate(class_names)}
    raw_id_to_label = {
        raw_id: class_to_id[group_name] for raw_id, group_name in raw_id_to_group.items()
    }
    return raw_id_to_label, ["background", *class_names]


def collect_annotations_by_image(
    annotations: list[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        if int(annotation.get("iscrowd", 0)):
            continue
        if float(annotation.get("area", 0.0)) <= 0:
            continue
        if not has_valid_polygon(annotation.get("segmentation")):
            continue
        grouped[int(annotation["image_id"])].append(annotation)
    return grouped


def collect_image_records(
    coco: dict[str, Any],
    dataset_dir: Path,
    image_extensions: frozenset[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for image in coco["images"]:
        file_name = str(image["file_name"])
        image_path = dataset_dir / file_name
        if image_path.suffix.lower() not in image_extensions:
            continue
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image referenced by annotations: {image_path}")
        records.append(image)
    return records


def has_valid_polygon(segmentation: Any) -> bool:
    if not isinstance(segmentation, list):
        return False
    return any(
        isinstance(polygon, list)
        and len(polygon) >= 6
        and len(polygon) % 2 == 0
        for polygon in segmentation
    )


def split_records(
    records: list[dict[str, Any]],
    annotations_by_image: dict[int, list[dict[str, Any]]],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    annotated = [record for record in records if annotations_by_image.get(int(record["id"]))]
    if not annotated:
        raise ValueError("No annotated TACO images were found.")
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError("--val-fraction and --test-fraction must be nonnegative and sum below 1.")

    rng = random.Random(seed)
    shuffled = list(annotated)
    rng.shuffle(shuffled)
    total = len(shuffled)
    test_count = max(1, int(round(total * test_fraction))) if test_fraction else 0
    val_count = max(1, int(round(total * val_fraction))) if val_fraction else 0
    train_count = total - val_count - test_count
    if train_count < 1:
        raise ValueError("Split fractions leave no training images.")

    train = shuffled[:train_count]
    val = shuffled[train_count : train_count + val_count]
    test = shuffled[train_count + val_count :]
    return train, val, test


def record_label_counts(
    record: dict[str, Any],
    annotations_by_image: dict[int, list[dict[str, Any]]],
    raw_id_to_label: dict[int, int],
    class_count: int,
) -> list[int]:
    counts = [0] * class_count
    for annotation in annotations_by_image.get(int(record["id"]), []):
        raw_category_id = int(annotation.get("category_id", -1))
        label = raw_id_to_label.get(raw_category_id)
        if label is not None and 0 <= label < class_count:
            counts[label] += 1
    return counts


def split_records_into_folds(
    records: list[dict[str, Any]],
    annotations_by_image: dict[int, list[dict[str, Any]]],
    raw_id_to_label: dict[int, int],
    class_count: int,
    folds: int,
    seed: int,
) -> list[list[dict[str, Any]]]:
    annotated = [record for record in records if annotations_by_image.get(int(record["id"]))]
    if folds < 2:
        raise ValueError("--folds must be at least 2.")
    if len(annotated) < folds:
        raise ValueError(f"Need at least {folds} annotated images for {folds}-fold evaluation.")

    rng = random.Random(seed)
    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in annotated:
        counts = record_label_counts(record, annotations_by_image, raw_id_to_label, class_count)
        bucket = max(range(1, class_count), key=lambda index: counts[index]) if sum(counts) else 0
        buckets[bucket].append(record)

    fold_records: list[list[dict[str, Any]]] = [[] for _ in range(folds)]
    for bucket_index, bucket_records in sorted(buckets.items()):
        rng.shuffle(bucket_records)
        start_fold = bucket_index % folds
        for offset, record in enumerate(bucket_records):
            fold_index = min(
                range(folds),
                key=lambda index: (
                    len(fold_records[index]),
                    (index - start_fold - offset) % folds,
                ),
            )
            fold_records[fold_index].append(record)

    if any(not fold for fold in fold_records):
        raise ValueError("Could not create non-empty folds. Try fewer folds.")
    return [sorted(fold, key=lambda record: int(record["id"])) for fold in fold_records]


def split_records_preserving_distribution(
    records: list[dict[str, Any]],
    annotations_by_image: dict[int, list[dict[str, Any]]],
    raw_id_to_label: dict[int, int],
    class_count: int,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError("--val-fraction and --test-fraction must be nonnegative and sum below 1.")

    annotated = [record for record in records if annotations_by_image.get(int(record["id"]))]
    if not annotated:
        raise ValueError("No annotated TACO images were found.")

    rng = random.Random(seed)
    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in annotated:
        counts = record_label_counts(record, annotations_by_image, raw_id_to_label, class_count)
        bucket = max(range(1, class_count), key=lambda index: counts[index]) if sum(counts) else 0
        buckets[bucket].append(record)

    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for bucket_records in buckets.values():
        rng.shuffle(bucket_records)
        total = len(bucket_records)
        test_count = int(round(total * test_fraction)) if test_fraction else 0
        val_count = int(round(total * val_fraction)) if val_fraction else 0
        if total >= 3:
            if test_fraction and test_count == 0:
                test_count = 1
            if val_fraction and val_count == 0:
                val_count = 1
        if test_count + val_count >= total:
            overflow = test_count + val_count - total + 1
            if val_count >= test_count:
                val_count = max(0, val_count - overflow)
            else:
                test_count = max(0, test_count - overflow)

        test.extend(bucket_records[:test_count])
        val.extend(bucket_records[test_count : test_count + val_count])
        train.extend(bucket_records[test_count + val_count :])

    if not train or not val or not test:
        return split_records(annotated, annotations_by_image, val_fraction, test_fraction, seed)

    return (
        sorted(train, key=lambda record: int(record["id"])),
        sorted(val, key=lambda record: int(record["id"])),
        sorted(test, key=lambda record: int(record["id"])),
    )


def summarize_record_distribution(
    records: list[dict[str, Any]],
    annotations_by_image: dict[int, list[dict[str, Any]]],
    raw_id_to_label: dict[int, int],
    class_names: list[str],
) -> dict[str, Any]:
    counts = [0] * len(class_names)
    for record in records:
        record_counts = record_label_counts(
            record,
            annotations_by_image,
            raw_id_to_label,
            len(class_names),
        )
        for index, count in enumerate(record_counts):
            counts[index] += count
    return {
        "images": len(records),
        "annotations_per_class": {
            class_names[index]: count
            for index, count in enumerate(counts)
            if index != 0
        },
    }


def polygon_to_mask(
    segmentation: Any,
    width: int,
    height: int,
) -> torch.Tensor | None:
    if not isinstance(segmentation, list):
        return None

    mask_image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_image)
    for polygon in segmentation:
        if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2 != 0:
            continue
        points = [(float(polygon[index]), float(polygon[index + 1])) for index in range(0, len(polygon), 2)]
        draw.polygon(points, outline=1, fill=1)

    mask = torch.from_numpy(np.array(mask_image, dtype=np.uint8))
    if int(mask.sum().item()) == 0:
        return None
    return mask


def boxes_from_masks(masks: torch.Tensor) -> torch.Tensor:
    boxes = []
    for mask in masks:
        y_indices, x_indices = torch.where(mask > 0)
        if len(x_indices) == 0 or len(y_indices) == 0:
            boxes.append(torch.tensor([0.0, 0.0, 1.0, 1.0]))
            continue
        x_min = torch.min(x_indices).float()
        x_max = torch.max(x_indices).float()
        y_min = torch.min(y_indices).float()
        y_max = torch.max(y_indices).float()
        boxes.append(torch.stack([x_min, y_min, x_max, y_max]))
    return torch.stack(boxes) if boxes else torch.zeros((0, 4), dtype=torch.float32)


def valid_box_mask(boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.bool)
    return (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])


def filter_instances_with_valid_boxes(
    masks: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    boxes = boxes_from_masks(masks)
    keep = valid_box_mask(boxes)
    masks = masks[keep]
    labels = labels[keep]
    boxes = boxes[keep]
    areas = masks.flatten(1).sum(dim=1).to(torch.float32)
    return masks, labels, boxes, areas


def keep_nonempty_masks(
    masks: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    keep = masks.flatten(1).sum(dim=1) > 0
    return masks[keep], labels[keep]


def random_adjust_factor(amount: float) -> float:
    if amount <= 0:
        return 1.0
    return random.uniform(max(0.0, 1.0 - amount), 1.0 + amount)


def apply_object_centered_crop(
    image: torch.Tensor,
    masks: torch.Tensor,
    labels: torch.Tensor,
    crop_scale: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _, height, width = image.shape
    if height <= 1 or width <= 1:
        return image, masks, labels

    boxes = boxes_from_masks(masks)
    object_index = random.randrange(masks.shape[0])
    x_min, y_min, x_max, y_max = boxes[object_index].tolist()
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    scale = random.uniform(crop_scale[0], crop_scale[1])
    crop_height = max(1, min(height, int(round(height * scale))))
    crop_width = max(1, min(width, int(round(width * scale))))

    jitter_x = random.uniform(-0.15, 0.15) * crop_width
    jitter_y = random.uniform(-0.15, 0.15) * crop_height
    left = int(round(center_x - crop_width / 2.0 + jitter_x))
    top = int(round(center_y - crop_height / 2.0 + jitter_y))
    left = max(0, min(width - crop_width, left))
    top = max(0, min(height - crop_height, top))

    cropped_image = F.crop(image, top, left, crop_height, crop_width)
    cropped_masks = F.crop(masks, top, left, crop_height, crop_width)
    cropped_masks, cropped_labels = keep_nonempty_masks(cropped_masks, labels)
    if cropped_masks.numel() == 0:
        return image, masks, labels
    return cropped_image, cropped_masks, cropped_labels


def apply_train_augmentations(
    image: torch.Tensor,
    masks: torch.Tensor,
    labels: torch.Tensor,
    *,
    flip_probability: float,
    rotation_degrees: float,
    object_crop_probability: float,
    object_crop_scale: tuple[float, float],
    brightness: float,
    contrast: float,
    saturation: float,
    hue: float,
    blur_probability: float,
    blur_kernel_size: int,
    noise_probability: float,
    noise_std: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if random.random() < flip_probability:
        image = F.hflip(image)
        masks = torch.flip(masks, dims=[2])

    if rotation_degrees > 0:
        angle = random.uniform(-rotation_degrees, rotation_degrees)
        image = F.rotate(
            image,
            angle,
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        )
        masks = F.rotate(
            masks,
            angle,
            interpolation=InterpolationMode.NEAREST,
            fill=0,
        )
        masks, labels = keep_nonempty_masks(masks, labels)

    if masks.numel() and random.random() < object_crop_probability:
        image, masks, labels = apply_object_centered_crop(
            image,
            masks,
            labels,
            object_crop_scale,
        )

    if brightness > 0:
        image = F.adjust_brightness(image, random_adjust_factor(brightness))
    if contrast > 0:
        image = F.adjust_contrast(image, random_adjust_factor(contrast))
    if saturation > 0:
        image = F.adjust_saturation(image, random_adjust_factor(saturation))
    if hue > 0:
        image = F.adjust_hue(image, random.uniform(-hue, hue))
    if blur_probability > 0 and random.random() < blur_probability:
        kernel_size = blur_kernel_size if blur_kernel_size % 2 == 1 else blur_kernel_size + 1
        image = F.gaussian_blur(image, kernel_size=[max(3, kernel_size), max(3, kernel_size)])
    if noise_probability > 0 and noise_std > 0 and random.random() < noise_probability:
        image = torch.clamp(image + torch.randn_like(image) * noise_std, 0.0, 1.0)

    return image, masks, labels



class TacoMaskDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        annotations_by_image: dict[int, list[dict[str, Any]]],
        raw_id_to_label: dict[int, int],
        dataset_dir: Path,
        train: bool,
        flip_probability: float,
        rotation_degrees: float = 0.0,
        object_crop_probability: float = 0.0,
        object_crop_scale: tuple[float, float] = (0.65, 1.0),
        brightness: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0,
        hue: float = 0.0,
        blur_probability: float = 0.0,
        blur_kernel_size: int = 5,
        noise_probability: float = 0.0,
        noise_std: float = 0.0,
    ) -> None:
        self.annotations_by_image = annotations_by_image
        self.raw_id_to_label = raw_id_to_label
        self.dataset_dir = dataset_dir
        self.train = train
        self.flip_probability = flip_probability
        self.rotation_degrees = rotation_degrees
        self.object_crop_probability = object_crop_probability
        self.object_crop_scale = object_crop_scale
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.blur_probability = blur_probability
        self.blur_kernel_size = blur_kernel_size
        self.noise_probability = noise_probability
        self.noise_std = noise_std

        # Filter out images whose annotations cannot produce at least one valid mask.
        self.records, self.skipped_invalid_mask_count = self.filter_records_with_valid_masks(records)

    def record_has_valid_mask(self, record: dict[str, Any]) -> bool:
        image_id = int(record["id"])
        image_path = self.dataset_dir / str(record["file_name"])

        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image)
            width, height = image.size

        for annotation in self.annotations_by_image.get(image_id, []):
            raw_category_id = int(annotation.get("category_id", -1))
            if raw_category_id not in self.raw_id_to_label:
                continue

            segmentation = annotation.get("segmentation")
            if not has_valid_polygon(segmentation):
                continue

            mask = polygon_to_mask(annotation.get("segmentation"), width, height)
            if mask is None:
                continue
            boxes = boxes_from_masks(mask.unsqueeze(0))
            if bool(valid_box_mask(boxes).item()):
                return True

        return False

    def filter_records_with_valid_masks(
        self,
        records: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], int]:
        valid_records = [
            record
            for record in records
            if self.record_has_valid_mask(record)
        ]
        skipped = len(records) - len(valid_records)

        if not valid_records:
            raise ValueError("No images with valid polygon masks were found.")

        return valid_records, skipped

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        for offset in range(len(self.records)):
            try:
                return self.load_item((index + offset) % len(self.records))
            except ValueError:
                continue
        raise ValueError("No images with valid positive-area boxes were found.")

    def load_item(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        record = self.records[index]
        image_id = int(record["id"])
        image_path = self.dataset_dir / str(record["file_name"])
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            width, height = image.size
            image_tensor = F.to_tensor(image)

        masks: list[torch.Tensor] = []
        labels: list[int] = []
        for annotation in self.annotations_by_image.get(image_id, []):
            if not has_valid_polygon(annotation.get("segmentation")):
                continue
            mask = polygon_to_mask(annotation.get("segmentation"), width, height)
            if mask is None:
                continue
            raw_category_id = int(annotation["category_id"])
            if raw_category_id not in self.raw_id_to_label:
                raise ValueError(f"Unknown category id {raw_category_id} in image {image_id}.")
            masks.append(mask)
            labels.append(self.raw_id_to_label[raw_category_id])

        if not masks:
            raise ValueError(f"No valid polygon masks found for {image_path}.")


        mask_tensor = torch.stack(masks)
        label_tensor = torch.tensor(labels, dtype=torch.int64)
        mask_tensor, label_tensor, box_tensor, area_tensor = filter_instances_with_valid_boxes(
            mask_tensor,
            label_tensor,
        )
        if mask_tensor.numel() == 0:
            raise ValueError(f"No valid positive-area boxes found for {image_path}.")

        if self.train:
            original_image = image_tensor
            original_masks = mask_tensor
            original_labels = label_tensor
            image_tensor, mask_tensor, label_tensor = apply_train_augmentations(
                image_tensor,
                mask_tensor,
                label_tensor,
                flip_probability=self.flip_probability,
                rotation_degrees=self.rotation_degrees,
                object_crop_probability=self.object_crop_probability,
                object_crop_scale=self.object_crop_scale,
                brightness=self.brightness,
                contrast=self.contrast,
                saturation=self.saturation,
                hue=self.hue,
                blur_probability=self.blur_probability,
                blur_kernel_size=self.blur_kernel_size,
                noise_probability=self.noise_probability,
                noise_std=self.noise_std,
            )
            if mask_tensor.numel() == 0:
                image_tensor = original_image
                mask_tensor = original_masks
                label_tensor = original_labels

        mask_tensor, label_tensor, box_tensor, area_tensor = filter_instances_with_valid_boxes(
            mask_tensor,
            label_tensor,
        )
        if mask_tensor.numel() == 0:
            raise ValueError(f"No valid positive-area boxes found after augmentation for {image_path}.")
        iscrowd = torch.zeros((len(label_tensor),), dtype=torch.int64)

        target = {
            "boxes": box_tensor,
            "labels": label_tensor,
            "masks": mask_tensor,
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "area": area_tensor,
            "iscrowd": iscrowd,
        }
        return image_tensor, target


def collate_batch(batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]]) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    images, targets = zip(*batch)
    return list(images), list(targets)


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
    )


def build_model(num_classes: int, pretrained: bool, weights_name: str | None) -> torch.nn.Module:

    weights = None
    if pretrained:
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        if weights_name and weights_name.upper() != "DEFAULT":
            weights = MaskRCNN_ResNet50_FPN_Weights[weights_name]

    model = maskrcnn_resnet50_fpn(weights=weights, weights_backbone=None)
    box_in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(box_in_features, num_classes)
    mask_in_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        mask_in_features,
        hidden_layer,
        num_classes,
    )
    return model


def move_targets_to_device(
    targets: list[dict[str, torch.Tensor]],
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    return [{key: value.to(device) for key, value in target.items()} for target in targets]


def filter_targets_with_valid_boxes(
    images: list[torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    filtered_images: list[torch.Tensor] = []
    filtered_targets: list[dict[str, torch.Tensor]] = []
    for image, target in zip(images, targets):
        keep = valid_box_mask(target["boxes"])
        if not bool(keep.any().item()):
            continue
        if bool(keep.all().item()):
            filtered_images.append(image)
            filtered_targets.append(target)
            continue

        filtered_target = target.copy()
        filtered_target["boxes"] = target["boxes"][keep]
        filtered_target["labels"] = target["labels"][keep]
        filtered_target["masks"] = target["masks"][keep]
        filtered_target["area"] = target["area"][keep]
        filtered_target["iscrowd"] = target["iscrowd"][keep]
        filtered_images.append(image)
        filtered_targets.append(filtered_target)

    return filtered_images, filtered_targets


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    batches = 0
    for images, targets in loader:
        images = [image.to(device) for image in images]
        targets = move_targets_to_device(targets, device)
        images, targets = filter_targets_with_valid_boxes(images, targets)
        if not targets:
            continue
        optimizer.zero_grad(set_to_none=True)
        losses = model(images, targets)
        loss = sum(loss_value for loss_value in losses.values())
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        batches += 1
        if batches % 20 == 0:
            print(f"epoch {epoch} batch {batches}: loss={loss.item():.4f}")
    return total_loss / max(1, batches)


@torch.no_grad()
def evaluate_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    was_training = model.training
    model.train()
    total_loss = 0.0
    batches = 0
    for images, targets in loader:
        images = [image.to(device) for image in images]
        targets = move_targets_to_device(targets, device)
        images, targets = filter_targets_with_valid_boxes(images, targets)
        if not targets:
            continue
        losses = model(images, targets)
        loss = sum(loss_value for loss_value in losses.values())
        total_loss += float(loss.item())
        batches += 1
    model.train(was_training)
    return total_loss / max(1, batches)


@torch.no_grad()
def save_sample_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    class_names: list[str],
    output_dir: Path,
    device: torch.device,
    score_threshold: float = 0.5,
) -> None:
    model.eval()
    images, targets = next(iter(loader))
    predictions = model([image.to(device) for image in images])
    summary = []
    for target, prediction in zip(targets, predictions):
        keep = prediction["scores"].detach().cpu() >= score_threshold
        labels = prediction["labels"].detach().cpu()[keep].tolist()
        scores = prediction["scores"].detach().cpu()[keep].tolist()
        summary.append(
            {
                "image_id": int(target["image_id"].item()),
                "detections": [
                    {"label": class_names[int(label)], "score": float(score)}
                    for label, score in zip(labels, scores)
                ],
            }
        )
    (output_dir / "sample_predictions.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def coco_ground_truth(
    records: list[dict[str, Any]],
    annotations_by_image: dict[int, list[dict[str, Any]]],
    raw_id_to_label: dict[int, int],
    class_names: list[str],
) -> dict[str, Any]:
    image_ids = {int(record["id"]) for record in records}
    images = [
        {
            "id": int(record["id"]),
            "file_name": str(record["file_name"]),
            "width": int(record["width"]),
            "height": int(record["height"]),
        }
        for record in records
    ]
    annotations = []
    annotation_id = 1
    for image_id in sorted(image_ids):
        for annotation in annotations_by_image.get(image_id, []):
            raw_category_id = int(annotation["category_id"])
            if raw_category_id not in raw_id_to_label:
                continue
            segmentation = annotation.get("segmentation")
            if not isinstance(segmentation, list) or not segmentation:
                continue
            bbox = annotation.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": raw_id_to_label[raw_category_id],
                    "segmentation": segmentation,
                    "bbox": [float(value) for value in bbox],
                    "area": float(annotation.get("area", 0.0)),
                    "iscrowd": int(annotation.get("iscrowd", 0)),
                }
            )
            annotation_id += 1

    return {
        "info": {"description": "TACO test split remapped for Mask R-CNN evaluation"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": index, "name": name}
            for index, name in enumerate(class_names)
            if index != 0
        ],
    }


def prediction_to_coco_results(
    prediction: dict[str, torch.Tensor],
    image_id: int,
    score_threshold: float,
) -> list[dict[str, Any]]:
    try:
        from pycocotools import mask as mask_utils
    except ImportError as error:
        raise ImportError(
            "pycocotools is required for COCO mask AP. Install requirements.txt "
            "again, or in Colab run: !pip install pycocotools"
        ) from error

    boxes = prediction["boxes"].detach().cpu()
    labels = prediction["labels"].detach().cpu()
    scores = prediction["scores"].detach().cpu()
    masks = prediction["masks"].detach().cpu()
    results = []
    for box, label, score, mask in zip(boxes, labels, scores, masks):
        score_value = float(score.item())
        if score_value < score_threshold:
            continue
        mask_array = (mask[0].numpy() >= 0.5).astype(np.uint8)
        if int(mask_array.sum()) == 0:
            continue
        rle = mask_utils.encode(np.asfortranarray(mask_array))
        rle["counts"] = rle["counts"].decode("utf-8")
        x_min, y_min, x_max, y_max = [float(value) for value in box.tolist()]
        results.append(
            {
                "image_id": image_id,
                "category_id": int(label.item()),
                "bbox": [x_min, y_min, max(0.0, x_max - x_min), max(0.0, y_max - y_min)],
                "segmentation": rle,
                "score": score_value,
            }
        )
    return results


def prediction_to_coco_results_with_scores(
    prediction: dict[str, torch.Tensor],
    scores: torch.Tensor,
    image_id: int,
    score_threshold: float,
) -> list[dict[str, Any]]:
    try:
        from pycocotools import mask as mask_utils
    except ImportError as error:
        raise ImportError(
            "pycocotools is required for COCO mask AP. Install requirements.txt "
            "again, or in Colab run: !pip install pycocotools"
        ) from error

    boxes = prediction["boxes"].detach().cpu()
    labels = prediction["labels"].detach().cpu()
    masks = prediction["masks"].detach().cpu()
    scores = scores.detach().cpu()
    results = []
    for box, label, score, mask in zip(boxes, labels, scores, masks):
        score_value = float(score.item())
        if score_value < score_threshold:
            continue
        mask_array = (mask[0].numpy() >= 0.5).astype(np.uint8)
        if int(mask_array.sum()) == 0:
            continue
        rle = mask_utils.encode(np.asfortranarray(mask_array))
        rle["counts"] = rle["counts"].decode("utf-8")
        x_min, y_min, x_max, y_max = [float(value) for value in box.tolist()]
        results.append(
            {
                "image_id": image_id,
                "category_id": int(label.item()),
                "bbox": [x_min, y_min, max(0.0, x_max - x_min), max(0.0, y_max - y_min)],
                "segmentation": rle,
                "score": score_value,
            }
        )
    return results


def postprocess_paper_score_detections(
    model: torch.nn.Module,
    class_logits: torch.Tensor,
    box_regression: torch.Tensor,
    proposals: list[torch.Tensor],
    image_shapes: list[tuple[int, int]],
    score_threshold: float,
) -> dict[str, list[dict[str, torch.Tensor]]]:
    roi_heads = model.roi_heads
    device = class_logits.device
    num_classes = class_logits.shape[-1]
    boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
    pred_boxes = roi_heads.box_coder.decode(box_regression, proposals)
    pred_scores = torch_F.softmax(class_logits, -1)

    pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
    pred_scores_list = pred_scores.split(boxes_per_image, 0)
    outputs: dict[str, list[dict[str, torch.Tensor]]] = {
        "class_score": [],
        "litter_score": [],
        "ratio_score": [],
    }

    for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
        if boxes.numel() == 0:
            empty_prediction = {
                "boxes": torch.empty((0, 4), device=device),
                "labels": torch.empty((0,), dtype=torch.int64, device=device),
                "scores": torch.empty((0,), device=device),
                "class_score": torch.empty((0,), device=device),
                "litter_score": torch.empty((0,), device=device),
                "ratio_score": torch.empty((0,), device=device),
            }
            for score_name in outputs:
                outputs[score_name].append(empty_prediction.copy())
            continue

        class_ids = scores.argmax(dim=1)
        row_indices = torch.arange(scores.shape[0], device=device)
        class_scores = scores[row_indices, class_ids]
        background_scores = scores[:, 0]
        litter_scores = 1.0 - background_scores
        ratio_scores = class_scores / (background_scores + 0.0001)

        boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
        selected_boxes = boxes[row_indices, class_ids]

        keep = torch.where(class_ids > 0)[0]
        keep = keep[torch.where(class_scores[keep] >= score_threshold)[0]]
        keep = keep[box_ops.remove_small_boxes(selected_boxes[keep], min_size=1e-2)]

        if keep.numel() > 0:
            nms_keep = box_ops.batched_nms(
                selected_boxes[keep],
                class_scores[keep],
                class_ids[keep],
                roi_heads.nms_thresh,
            )
            keep = keep[nms_keep]

        score_values = {
            "class_score": class_scores,
            "litter_score": litter_scores,
            "ratio_score": ratio_scores,
        }
        for score_name, variant_scores in score_values.items():
            variant_keep = keep
            if variant_keep.numel() > 0:
                _, order = variant_scores[variant_keep].sort(descending=True)
                variant_keep = variant_keep[order[: roi_heads.detections_per_img]]
            outputs[score_name].append(
                {
                    "boxes": selected_boxes[variant_keep],
                    "labels": class_ids[variant_keep],
                    "scores": variant_scores[variant_keep],
                    "class_score": class_scores[variant_keep],
                    "litter_score": litter_scores[variant_keep],
                    "ratio_score": ratio_scores[variant_keep],
                }
            )

    return outputs


@torch.no_grad()
def predict_paper_score_variants(
    model: torch.nn.Module,
    images: list[torch.Tensor],
    device: torch.device,
    score_threshold: float,
) -> dict[str, list[dict[str, torch.Tensor]]]:
    model.eval()
    original_image_sizes = [tuple(image.shape[-2:]) for image in images]
    images = [image.to(device) for image in images]
    transformed_images, _ = model.transform(images, None)
    features = model.backbone(transformed_images.tensors)
    if isinstance(features, torch.Tensor):
        features = {"0": features}
    proposals, _ = model.rpn(transformed_images, features, None)

    box_features = model.roi_heads.box_roi_pool(
        features,
        proposals,
        transformed_images.image_sizes,
    )
    box_features = model.roi_heads.box_head(box_features)
    class_logits, box_regression = model.roi_heads.box_predictor(box_features)
    variants = postprocess_paper_score_detections(
        model,
        class_logits,
        box_regression,
        proposals,
        transformed_images.image_sizes,
        score_threshold,
    )

    for detections in variants.values():
        mask_proposals = [prediction["boxes"] for prediction in detections]
        mask_features = model.roi_heads.mask_roi_pool(
            features,
            mask_proposals,
            transformed_images.image_sizes,
        )
        mask_features = model.roi_heads.mask_head(mask_features)
        mask_logits = model.roi_heads.mask_predictor(mask_features)
        labels = [prediction["labels"] for prediction in detections]
        masks_probs = maskrcnn_inference(mask_logits, labels)
        for mask_prob, prediction in zip(masks_probs, detections):
            prediction["masks"] = mask_prob

    return {
        score_name: model.transform.postprocess(
            detections,
            transformed_images.image_sizes,
            original_image_sizes,
        )
        for score_name, detections in variants.items()
    }


@torch.no_grad()
def collect_paper_score_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    score_threshold: float,
) -> dict[str, list[dict[str, Any]]]:
    results = {
        "class_score": [],
        "litter_score": [],
        "ratio_score": [],
    }
    for images, targets in loader:
        variants = predict_paper_score_variants(model, images, device, score_threshold)
        for target_index, target in enumerate(targets):
            image_id = int(target["image_id"].item())
            for score_name, predictions in variants.items():
                prediction = predictions[target_index]
                results[score_name].extend(
                    prediction_to_coco_results_with_scores(
                        prediction,
                        prediction[score_name],
                        image_id,
                        score_threshold,
                    )
                )
    return results


@torch.no_grad()
def collect_coco_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    score_threshold: float,
) -> list[dict[str, Any]]:
    model.eval()
    results = []
    for images, targets in loader:
        predictions = model([image.to(device) for image in images])
        for target, prediction in zip(targets, predictions):
            image_id = int(target["image_id"].item())
            results.extend(prediction_to_coco_results(prediction, image_id, score_threshold))
    return results


def summarize_coco_eval(stats: np.ndarray) -> dict[str, float]:
    names = [
        "AP",
        "AP50",
        "AP75",
        "AP_small",
        "AP_medium",
        "AP_large",
        "AR_max1",
        "AR_max10",
        "AR_max100",
        "AR_small",
        "AR_medium",
        "AR_large",
    ]
    return {name: float(value) for name, value in zip(names, stats.tolist())}


def run_coco_eval(
    ground_truth_path: Path,
    predictions_path: Path,
    iou_type: str,
) -> dict[str, float]:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as error:
        raise ImportError(
            "pycocotools is required for COCO AP metrics. Install requirements.txt "
            "again, or in Colab run: !pip install pycocotools"
        ) from error

    coco_gt = COCO(str(ground_truth_path))
    coco_predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    if not coco_predictions:
        return {name: 0.0 for name in summarize_coco_eval(np.zeros(12)).keys()}
    coco_dt = coco_gt.loadRes(str(predictions_path))
    evaluator = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return summarize_coco_eval(evaluator.stats)


def evaluate_coco_metrics(
    model: torch.nn.Module,
    test_loader: DataLoader,
    test_records: list[dict[str, Any]],
    annotations_by_image: dict[int, list[dict[str, Any]]],
    raw_id_to_label: dict[int, int],
    class_names: list[str],
    output_dir: Path,
    device: torch.device,
    score_threshold: float,
) -> dict[str, dict[str, float]]:
    ground_truth = coco_ground_truth(
        test_records,
        annotations_by_image,
        raw_id_to_label,
        class_names,
    )
    ground_truth_path = output_dir / "coco_test_ground_truth.json"
    predictions_path = output_dir / "coco_test_predictions.json"
    ground_truth_path.write_text(json.dumps(ground_truth), encoding="utf-8")

    predictions = collect_coco_predictions(model, test_loader, device, score_threshold)
    predictions_path.write_text(json.dumps(predictions), encoding="utf-8")

    metrics = {
        "segm": run_coco_eval(ground_truth_path, predictions_path, "segm"),
        "bbox": run_coco_eval(ground_truth_path, predictions_path, "bbox"),
    }
    (output_dir / "coco_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    return metrics


def evaluate_paper_score_metrics(
    model: torch.nn.Module,
    test_loader: DataLoader,
    test_records: list[dict[str, Any]],
    annotations_by_image: dict[int, list[dict[str, Any]]],
    raw_id_to_label: dict[int, int],
    class_names: list[str],
    output_dir: Path,
    device: torch.device,
    score_threshold: float,
) -> dict[str, dict[str, dict[str, float]]]:
    ground_truth = coco_ground_truth(
        test_records,
        annotations_by_image,
        raw_id_to_label,
        class_names,
    )
    ground_truth_path = output_dir / "paper_score_ground_truth.json"
    ground_truth_path.write_text(json.dumps(ground_truth), encoding="utf-8")

    predictions_by_score = collect_paper_score_predictions(
        model,
        test_loader,
        device,
        score_threshold,
    )
    metrics: dict[str, dict[str, dict[str, float]]] = {}
    for score_name, predictions in predictions_by_score.items():
        predictions_path = output_dir / f"paper_score_{score_name}_predictions.json"
        predictions_path.write_text(json.dumps(predictions), encoding="utf-8")
        metrics[score_name] = {
            "segm": run_coco_eval(ground_truth_path, predictions_path, "segm"),
            "bbox": run_coco_eval(ground_truth_path, predictions_path, "bbox"),
        }

    (output_dir / "paper_score_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    return metrics


def save_split_summary(
    output_dir: Path,
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    class_names: list[str],
    train_dataset: TacoMaskDataset | None = None,
    val_dataset: TacoMaskDataset | None = None,
    test_dataset: TacoMaskDataset | None = None,
) -> None:
    summary = {
        "train_images": len(train_records),
        "val_images": len(val_records),
        "test_images": len(test_records),
        "classes": class_names,
    }
    if train_dataset is not None and val_dataset is not None and test_dataset is not None:
        summary.update(
            {
                "usable_train_images": len(train_dataset),
                "usable_val_images": len(val_dataset),
                "usable_test_images": len(test_dataset),
                "skipped_invalid_train_masks": train_dataset.skipped_invalid_mask_count,
                "skipped_invalid_val_masks": val_dataset.skipped_invalid_mask_count,
                "skipped_invalid_test_masks": test_dataset.skipped_invalid_mask_count,
            }
        )
    (output_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def make_taco_dataset(
    records: list[dict[str, Any]],
    annotations_by_image: dict[int, list[dict[str, Any]]],
    raw_id_to_label: dict[int, int],
    dataset_dir: Path,
    config: Config,
    train: bool,
) -> TacoMaskDataset:
    return TacoMaskDataset(
        records,
        annotations_by_image,
        raw_id_to_label,
        dataset_dir,
        train=train,
        flip_probability=config.horizontal_flip_probability if train else 0.0,
        rotation_degrees=config.rotation_degrees if train else 0.0,
        object_crop_probability=config.object_crop_probability if train else 0.0,
        object_crop_scale=config.object_crop_scale,
        brightness=config.brightness if train else 0.0,
        contrast=config.contrast if train else 0.0,
        saturation=config.saturation if train else 0.0,
        hue=config.hue if train else 0.0,
        blur_probability=config.blur_probability if train else 0.0,
        blur_kernel_size=config.blur_kernel_size,
        noise_probability=config.noise_probability if train else 0.0,
        noise_std=config.noise_std if train else 0.0,
    )


def average_coco_metrics(fold_summaries: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    averaged: dict[str, dict[str, float]] = {}
    for metric_type in ("segm", "bbox"):
        keys = sorted(
            {
                key
                for summary in fold_summaries
                for key in summary["coco_metrics"][metric_type]
            }
        )
        averaged[metric_type] = {
            key: float(
                sum(summary["coco_metrics"][metric_type][key] for summary in fold_summaries)
                / len(fold_summaries)
            )
            for key in keys
        }
    return averaged


def average_paper_score_metrics(
    fold_summaries: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, float]]]:
    averaged: dict[str, dict[str, dict[str, float]]] = {}
    score_names = sorted(
        {
            score_name
            for summary in fold_summaries
            for score_name in summary["paper_score_metrics"]
        }
    )
    for score_name in score_names:
        averaged[score_name] = {}
        for metric_type in ("segm", "bbox"):
            keys = sorted(
                {
                    key
                    for summary in fold_summaries
                    for key in summary["paper_score_metrics"][score_name][metric_type]
                }
            )
            averaged[score_name][metric_type] = {
                key: float(
                    sum(
                        summary["paper_score_metrics"][score_name][metric_type][key]
                        for summary in fold_summaries
                    )
                    / len(fold_summaries)
                )
                for key in keys
            }
    return averaged


def run_paper_score_evaluation_split(
    args: argparse.Namespace,
    config: Config,
    dataset_dir: Path,
    annotations_by_image: dict[int, list[dict[str, Any]]],
    raw_id_to_label: dict[int, int],
    class_names: list[str],
    test_records: list[dict[str, Any]],
    output_dir: Path,
    device: torch.device,
    split_name: str,
) -> dict[str, Any]:
    checkpoint_path = output_dir / args.checkpoint_name
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Could not find checkpoint for {split_name}: {checkpoint_path}")

    test_dataset = make_taco_dataset(
        test_records,
        annotations_by_image,
        raw_id_to_label,
        dataset_dir,
        config,
        train=False,
    )
    test_loader = make_loader(test_dataset, args.batch_size, shuffle=False, workers=args.workers)
    model = build_model(
        num_classes=len(class_names),
        pretrained=False,
        weights_name=None,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    print(f"{split_name} paper-score evaluation using {checkpoint_path}")
    paper_score_metrics = evaluate_paper_score_metrics(
        model,
        test_loader,
        test_records,
        annotations_by_image,
        raw_id_to_label,
        class_names,
        output_dir,
        device,
        score_threshold=0.0,
    )
    for score_name, metrics in paper_score_metrics.items():
        print(f"{split_name} {score_name} mask AP: {metrics['segm']['AP']:.4f}")

    return {
        "output_dir": str(output_dir),
        "checkpoint": str(checkpoint_path),
        "paper_score_metrics": paper_score_metrics,
    }


def run_training_split(
    args: argparse.Namespace,
    config: Config,
    dataset_dir: Path,
    annotations_by_image: dict[int, list[dict[str, Any]]],
    raw_id_to_label: dict[int, int],
    class_names: list[str],
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    output_dir: Path,
    device: torch.device,
    split_name: str,
) -> dict[str, Any]:
    train_dataset = make_taco_dataset(
        train_records,
        annotations_by_image,
        raw_id_to_label,
        dataset_dir,
        config,
        train=True,
    )
    val_dataset = make_taco_dataset(
        val_records,
        annotations_by_image,
        raw_id_to_label,
        dataset_dir,
        config,
        train=False,
    )
    test_dataset = make_taco_dataset(
        test_records,
        annotations_by_image,
        raw_id_to_label,
        dataset_dir,
        config,
        train=False,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_split_summary(
        output_dir,
        train_records,
        val_records,
        test_records,
        class_names,
        train_dataset,
        val_dataset,
        test_dataset,
    )
    (output_dir / "labels.json").write_text(
        json.dumps(class_names, indent=2),
        encoding="utf-8",
    )

    model = build_model(
        num_classes=len(class_names),
        pretrained=config.pretrained and not args.no_pretrained,
        weights_name=config.weights,
    ).to(device)

    train_loader = make_loader(train_dataset, args.batch_size, shuffle=True, workers=args.workers)
    val_loader = make_loader(val_dataset, args.batch_size, shuffle=False, workers=args.workers)
    test_loader = make_loader(test_dataset, args.batch_size, shuffle=False, workers=args.workers)

    print(
        f"{split_name} split sizes: "
        f"train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)}"
    )
    skipped_invalid = (
        train_dataset.skipped_invalid_mask_count
        + val_dataset.skipped_invalid_mask_count
        + test_dataset.skipped_invalid_mask_count
    )
    if skipped_invalid:
        print(
            f"{split_name} skipped images without valid polygon masks: "
            f"train={train_dataset.skipped_invalid_mask_count} "
            f"val={val_dataset.skipped_invalid_mask_count} "
            f"test={test_dataset.skipped_invalid_mask_count}"
        )

    images, targets = next(iter(train_loader))
    with torch.no_grad():
        model.train()
        smoke_images = [image.to(device) for image in images[:1]]
        smoke_targets = move_targets_to_device(targets[:1], device)
        smoke_images, smoke_targets = filter_targets_with_valid_boxes(smoke_images, smoke_targets)
        if not smoke_targets:
            raise ValueError("Smoke batch did not contain any valid positive-area boxes.")
        smoke_losses = model(smoke_images, smoke_targets)
    print(f"{split_name} smoke loss keys: {sorted(smoke_losses)}")
    if args.check_only:
        return {
            "output_dir": str(output_dir),
            "split_summary": {
                "train_images": len(train_dataset),
                "val_images": len(val_dataset),
                "test_images": len(test_dataset),
            },
        }

    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    history = {"loss": [], "val_loss": []}
    best_val_loss = math.inf
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss = evaluate_loss(model, val_loader, device)
        scheduler.step()
        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(
            f"{split_name} epoch {epoch}/{args.epochs}: "
            f"loss={train_loss:.4f} val_loss={val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "best_model.pth")
        else:
            patience_counter += 1
            if 0 < args.patience <= patience_counter:
                print(f"{split_name} early stopping after {epoch} epochs.")
                break

    if (output_dir / "best_model.pth").is_file():
        model.load_state_dict(torch.load(output_dir / "best_model.pth", map_location=device))
    test_loss = evaluate_loss(model, test_loader, device)
    coco_metrics = evaluate_coco_metrics(
        model,
        test_loader,
        test_records,
        annotations_by_image,
        raw_id_to_label,
        class_names,
        output_dir,
        device,
        args.evaluation_score_threshold,
    )
    torch.save(model.state_dict(), output_dir / "taco_maskrcnn.pth")
    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )
    (output_dir / "test_metrics.json").write_text(
        json.dumps({"loss": float(test_loss), "coco": coco_metrics}, indent=2),
        encoding="utf-8",
    )
    save_sample_predictions(model, test_loader, class_names, output_dir, device)
    print(f"{split_name} test loss: {test_loss:.4f}")
    print(f"{split_name} mask AP: {coco_metrics['segm']['AP']:.4f}")
    print(f"{split_name} mask AP50: {coco_metrics['segm']['AP50']:.4f}")
    print(f"{split_name} mask AP75: {coco_metrics['segm']['AP75']:.4f}")
    print(f"{split_name} artifacts saved to {output_dir.resolve()}")

    return {
        "output_dir": str(output_dir),
        "epochs_ran": len(history["loss"]),
        "best_val_loss": float(best_val_loss),
        "test_loss": float(test_loss),
        "coco_metrics": coco_metrics,
    }


def run_cross_validation(
    args: argparse.Namespace,
    config: Config,
    dataset_dir: Path,
    records: list[dict[str, Any]],
    annotations_by_image: dict[int, list[dict[str, Any]]],
    raw_id_to_label: dict[int, int],
    class_names: list[str],
    device: torch.device,
) -> None:
    folds = split_records_into_folds(
        records,
        annotations_by_image,
        raw_id_to_label,
        len(class_names),
        args.folds,
        args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_summaries: list[dict[str, Any]] = []

    for fold_index in range(args.folds):
        fold_records = folds[fold_index]
        train_records, val_records, test_records = split_records_preserving_distribution(
            fold_records,
            annotations_by_image,
            raw_id_to_label,
            len(class_names),
            args.val_fraction,
            args.test_fraction,
            args.seed + fold_index,
        )
        split_distribution = {
            "fold": summarize_record_distribution(
                fold_records,
                annotations_by_image,
                raw_id_to_label,
                class_names,
            ),
            "train": summarize_record_distribution(
                train_records,
                annotations_by_image,
                raw_id_to_label,
                class_names,
            ),
            "val": summarize_record_distribution(
                val_records,
                annotations_by_image,
                raw_id_to_label,
                class_names,
            ),
            "test": summarize_record_distribution(
                test_records,
                annotations_by_image,
                raw_id_to_label,
                class_names,
            ),
        }
        fold_output_dir = args.output_dir / f"fold{fold_index + 1}"
        print(
            f"Fold {fold_index + 1}/{args.folds}: "
            f"{len(fold_records)} total, "
            f"{len(train_records)} train, {len(val_records)} val, {len(test_records)} test images"
        )
        if args.paper_score_eval_only:
            summary = run_paper_score_evaluation_split(
                args,
                config,
                dataset_dir,
                annotations_by_image,
                raw_id_to_label,
                class_names,
                test_records,
                fold_output_dir,
                device,
                split_name=f"fold {fold_index + 1}",
            )
        else:
            summary = run_training_split(
                args,
                config,
                dataset_dir,
                annotations_by_image,
                raw_id_to_label,
                class_names,
                train_records,
                val_records,
                test_records,
                fold_output_dir,
                device,
                split_name=f"fold {fold_index + 1}",
            )
        summary.update(
            {
                "fold": fold_index + 1,
                "split_distribution": split_distribution,
            }
        )
        fold_summaries.append(summary)

    if args.paper_score_eval_only:
        completed = [summary for summary in fold_summaries if "paper_score_metrics" in summary]
        average_metrics = average_paper_score_metrics(completed) if completed else {}
        cv_summary = {
            "folds": args.folds,
            "val_fraction": args.val_fraction,
            "test_fraction": args.test_fraction,
            "classes": class_names,
            "fold_summaries": fold_summaries,
            "average_paper_score_metrics": average_metrics,
        }
        (args.output_dir / "paper_score_summary.json").write_text(
            json.dumps(cv_summary, indent=2),
            encoding="utf-8",
        )
        for score_name, metrics in average_metrics.items():
            print(f"Average {score_name} mask AP: {metrics['segm']['AP']:.4f}")
        print(f"Paper-score summary saved to {args.output_dir / 'paper_score_summary.json'}")
        return

    completed = [summary for summary in fold_summaries if "coco_metrics" in summary]
    average_metrics = average_coco_metrics(completed) if completed else {}
    cv_summary = {
        "folds": args.folds,
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
        "classes": class_names,
        "fold_summaries": fold_summaries,
        "average_coco_metrics": average_metrics,
        "average_mask_ap": average_metrics.get("segm", {}).get("AP"),
        "average_bbox_ap": average_metrics.get("bbox", {}).get("AP"),
    }
    (args.output_dir / "cross_validation_summary.json").write_text(
        json.dumps(cv_summary, indent=2),
        encoding="utf-8",
    )
    if args.check_only:
        print("Cross-validation check complete. No training was run.")
    else:
        print(f"Average mask AP: {cv_summary['average_mask_ap']:.4f}")
        print(f"Average bbox AP: {cv_summary['average_bbox_ap']:.4f}")
    print(f"Cross-validation summary saved to {args.output_dir / 'cross_validation_summary.json'}")


def main() -> None:
    config_path = parse_config_path()
    config = load_config(config_path)
    args = parse_args(config, config_path)
    set_seed(args.seed)

    dataset_dir = resolve_dataset_dir(args.dataset_dir, args.annotation_file)
    coco = load_coco_annotations(dataset_dir, args.annotation_file)
    raw_id_to_label, class_names = build_category_map(
        coco["categories"],
        args.taxonomy,
        args.category_field,
    )
    annotations_by_image = collect_annotations_by_image(coco["annotations"])
    records = collect_image_records(coco, dataset_dir, config.image_extensions)

    device = resolve_device(args.device)
    print(f"Dataset root: {dataset_dir}")
    print(f"Device: {describe_device(device)}")
    print(f"Classes ({len(class_names)} including background): {class_names}")

    if args.cross_validation:
        run_cross_validation(
            args,
            config,
            dataset_dir,
            records,
            annotations_by_image,
            raw_id_to_label,
            class_names,
            device,
        )
        return

    train_records, val_records, test_records = split_records(
        records,
        annotations_by_image,
        args.val_fraction,
        args.test_fraction,
        args.seed,
    )
    run_training_split(
        args,
        config,
        dataset_dir,
        annotations_by_image,
        raw_id_to_label,
        class_names,
        train_records,
        val_records,
        test_records,
        args.output_dir,
        device,
        split_name="single split",
    )
    if args.check_only:
        print("Check complete. No training was run.")


if __name__ == "__main__":
    main()
