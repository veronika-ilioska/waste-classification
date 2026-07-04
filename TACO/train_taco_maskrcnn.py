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
import yaml
from PIL import Image, ImageDraw, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


@dataclass(frozen=True)
class Config:
    dataset_dir: Path | None
    annotation_file: str
    image_extensions: frozenset[str]
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


def load_config(config_path: Path) -> Config:
    raw = load_yaml(config_path)
    dataset = section(raw, "dataset")
    output = section(raw, "output")
    model = section(raw, "model")
    training = section(raw, "training")
    augmentation = section(raw, "augmentation")

    return Config(
        dataset_dir=optional_path(dataset.get("dir")),
        annotation_file=str(dataset.get("annotation_file", "annotations.json")),
        image_extensions=extensions(dataset.get("image_extensions", [".jpg", ".jpeg", ".png"])),
        category_field=str(dataset.get("category_field", "supercategory")),
        output_dir=Path(str(output.get("dir", "artifacts/taco/maskrcnn"))),
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
    category_field: str,
) -> tuple[dict[int, int], list[str]]:
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


class TacoMaskDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        annotations_by_image: dict[int, list[dict[str, Any]]],
        raw_id_to_label: dict[int, int],
        dataset_dir: Path,
        train: bool,
        flip_probability: float,
    ) -> None:
        self.annotations_by_image = annotations_by_image
        self.raw_id_to_label = raw_id_to_label
        self.dataset_dir = dataset_dir
        self.train = train
        self.flip_probability = flip_probability
        self.records, self.skipped_invalid_mask_count = self.filter_records_with_valid_masks(records)

    def record_has_valid_mask(self, record: dict[str, Any]) -> bool:
        image_id = int(record["id"])
        image_path = self.dataset_dir / str(record["file_name"])
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image)
            width, height = image.size

        for annotation in self.annotations_by_image.get(image_id, []):
            if not has_valid_polygon(annotation.get("segmentation")):
                continue
            if int(annotation["category_id"]) not in self.raw_id_to_label:
                continue
            if polygon_to_mask(annotation.get("segmentation"), width, height) is not None:
                return True
        return False

    def filter_records_with_valid_masks(
        self,
        records: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], int]:
        valid_records = [record for record in records if self.record_has_valid_mask(record)]
        skipped = len(records) - len(valid_records)
        if not valid_records:
            raise ValueError("No images with valid polygon masks were found.")
        return valid_records, skipped

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        record = self.records[index]
        image_id = int(record["id"])
        image_path = self.dataset_dir / str(record["file_name"])
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            width, height = image.size
            image_tensor = F.to_tensor(image)

        masks: list[torch.Tensor] = []
        labels: list[int] = []
        areas: list[float] = []
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
            areas.append(float(annotation.get("area", mask.sum().item())))

        if not masks:
            raise ValueError(f"No valid polygon masks found for {image_path}.")

        mask_tensor = torch.stack(masks)
        box_tensor = boxes_from_masks(mask_tensor)
        label_tensor = torch.tensor(labels, dtype=torch.int64)
        area_tensor = torch.tensor(areas, dtype=torch.float32)
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        if self.train and random.random() < self.flip_probability:
            image_tensor = F.hflip(image_tensor)
            mask_tensor = torch.flip(mask_tensor, dims=[2])
            x_min = width - box_tensor[:, 2]
            x_max = width - box_tensor[:, 0]
            box_tensor[:, 0] = x_min
            box_tensor[:, 2] = x_max

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
    from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
    from torchvision.models.detection import maskrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

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


def main() -> None:
    config_path = parse_config_path()
    config = load_config(config_path)
    args = parse_args(config, config_path)
    set_seed(args.seed)

    dataset_dir = resolve_dataset_dir(args.dataset_dir, args.annotation_file)
    coco = load_coco_annotations(dataset_dir, args.annotation_file)
    raw_id_to_label, class_names = build_category_map(coco["categories"], args.category_field)
    annotations_by_image = collect_annotations_by_image(coco["annotations"])
    records = collect_image_records(coco, dataset_dir, config.image_extensions)
    train_records, val_records, test_records = split_records(
        records,
        annotations_by_image,
        args.val_fraction,
        args.test_fraction,
        args.seed,
    )

    train_dataset = TacoMaskDataset(
        train_records,
        annotations_by_image,
        raw_id_to_label,
        dataset_dir,
        train=True,
        flip_probability=config.horizontal_flip_probability,
    )
    val_dataset = TacoMaskDataset(
        val_records,
        annotations_by_image,
        raw_id_to_label,
        dataset_dir,
        train=False,
        flip_probability=0.0,
    )
    test_dataset = TacoMaskDataset(
        test_records,
        annotations_by_image,
        raw_id_to_label,
        dataset_dir,
        train=False,
        flip_probability=0.0,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_split_summary(
        args.output_dir,
        train_records,
        val_records,
        test_records,
        class_names,
        train_dataset,
        val_dataset,
        test_dataset,
    )
    (args.output_dir / "labels.json").write_text(
        json.dumps(class_names, indent=2),
        encoding="utf-8",
    )

    device = resolve_device(args.device)
    model = build_model(
        num_classes=len(class_names),
        pretrained=config.pretrained and not args.no_pretrained,
        weights_name=config.weights,
    ).to(device)

    train_loader = make_loader(train_dataset, args.batch_size, shuffle=True, workers=args.workers)
    val_loader = make_loader(val_dataset, args.batch_size, shuffle=False, workers=args.workers)
    test_loader = make_loader(test_dataset, args.batch_size, shuffle=False, workers=args.workers)

    print(f"Dataset root: {dataset_dir}")
    print(f"Device: {describe_device(device)}")
    print(f"Classes ({len(class_names)} including background): {class_names}")
    print(
        "Split sizes: "
        f"train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)}"
    )
    skipped_invalid = (
        train_dataset.skipped_invalid_mask_count
        + val_dataset.skipped_invalid_mask_count
        + test_dataset.skipped_invalid_mask_count
    )
    if skipped_invalid:
        print(
            "Skipped images without valid polygon masks: "
            f"train={train_dataset.skipped_invalid_mask_count} "
            f"val={val_dataset.skipped_invalid_mask_count} "
            f"test={test_dataset.skipped_invalid_mask_count}"
        )

    images, targets = next(iter(train_loader))
    with torch.no_grad():
        model.train()
        smoke_losses = model(
            [image.to(device) for image in images[:1]],
            move_targets_to_device(targets[:1], device),
        )
    print(f"Smoke loss keys: {sorted(smoke_losses)}")
    if args.check_only:
        print("Check complete. No training was run.")
        return

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
            f"epoch {epoch}/{args.epochs}: "
            f"loss={train_loss:.4f} val_loss={val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), args.output_dir / "best_model.pth")
        else:
            patience_counter += 1
            if args.patience > 0 and patience_counter >= args.patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    if (args.output_dir / "best_model.pth").is_file():
        model.load_state_dict(torch.load(args.output_dir / "best_model.pth", map_location=device))
    test_loss = evaluate_loss(model, test_loader, device)
    torch.save(model.state_dict(), args.output_dir / "taco_maskrcnn.pth")
    (args.output_dir / "history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "test_metrics.json").write_text(
        json.dumps({"loss": float(test_loss)}, indent=2),
        encoding="utf-8",
    )
    save_sample_predictions(model, test_loader, class_names, args.output_dir, device)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Saved Mask R-CNN artifacts to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
