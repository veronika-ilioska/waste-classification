from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import kagglehub
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn as nn
import yaml
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")
BACKGROUND_CLASS = "background"
SPLITS = ("train", "valid", "test")


@dataclass(frozen=True)
class Config:
    dataset_handle: str
    dataset_dir: Path | None
    output_dir: Path
    model_name: str
    image_size: tuple[int, int]
    image_extensions: frozenset[str]
    batch_size: int
    epochs: int
    fine_tune_epochs: int
    fine_tune_layers: int
    learning_rate: float
    fine_tune_learning_rate: float
    seed: int
    workers: int
    use_pretrained_weights: bool
    dropout_rate: float
    random_rotation: float
    random_resized_crop_scale: tuple[float, float]
    color_jitter: float
    early_stopping_patience: int
    misclassified_examples: int


def load_yaml_config(config_path: Path) -> dict:
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path.resolve()}")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping.")
    return data


def config_section(config: dict, name: str) -> dict:
    value = config.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"Config section {name!r} must be a mapping.")
    return value


def config_path_value(value: str | None) -> Path | None:
    if value is None or not str(value).strip():
        return None
    return Path(str(value))


def config_extensions(value: list[str] | str) -> frozenset[str]:
    if isinstance(value, str):
        parts = value.split(",")
    else:
        parts = value
    return frozenset(extension.strip().lower() for extension in parts if extension.strip())


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> Config:
    raw_config = load_yaml_config(config_path)
    dataset = config_section(raw_config, "dataset")
    output = config_section(raw_config, "output")
    model = config_section(raw_config, "model")
    training = config_section(raw_config, "training")
    augmentation = config_section(raw_config, "augmentation")
    callbacks = config_section(raw_config, "callbacks")
    evaluation = config_section(raw_config, "evaluation")
    crop_scale = augmentation.get("random_resized_crop_scale", [0.85, 1.0])

    return Config(
        dataset_handle=str(
            dataset.get(
                "handle",
                "ahsan71/ecodetect-recyclable-waste-detection-dataset",
            )
        ),
        dataset_dir=config_path_value(dataset.get("dir")),
        output_dir=Path(str(output.get("dir", "artifacts/ecodetect/mobilenetv4"))),
        model_name=str(model.get("name", "mobilenetv4_conv_small")),
        image_size=(
            int(model.get("image_height", 224)),
            int(model.get("image_width", 224)),
        ),
        image_extensions=config_extensions(
            dataset.get("image_extensions", [".bmp", ".gif", ".jpeg", ".jpg", ".png"])
        ),
        batch_size=int(training.get("batch_size", 32)),
        epochs=int(training.get("epochs", 10)),
        fine_tune_epochs=int(training.get("fine_tune_epochs", 5)),
        fine_tune_layers=int(training.get("fine_tune_layers", 30)),
        learning_rate=float(training.get("learning_rate", 0.001)),
        fine_tune_learning_rate=float(training.get("fine_tune_learning_rate", 0.00001)),
        seed=int(training.get("seed", 42)),
        workers=int(training.get("workers", 0)),
        use_pretrained_weights=bool(model.get("use_pretrained_weights", True)),
        dropout_rate=float(model.get("dropout_rate", 0.25)),
        random_rotation=float(augmentation.get("random_rotation", 8)),
        random_resized_crop_scale=(float(crop_scale[0]), float(crop_scale[1])),
        color_jitter=float(augmentation.get("color_jitter", 0.1)),
        early_stopping_patience=int(callbacks.get("early_stopping_patience", 3)),
        misclassified_examples=int(evaluation.get("misclassified_examples", 25)),
    )


def parse_config_path() -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args, _ = parser.parse_known_args()
    return args.config


def parse_args(config: Config, config_path: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a timm MobileNetV4 image classifier on EcoDetect."
    )
    parser.add_argument("--config", type=Path, default=config_path)
    parser.add_argument("--dataset-dir", type=Path, default=config.dataset_dir)
    parser.add_argument("--output-dir", type=Path, default=config.output_dir)
    parser.add_argument("--model-name", default=config.model_name)
    parser.add_argument("--batch-size", type=int, default=config.batch_size)
    parser.add_argument("--epochs", type=int, default=config.epochs)
    parser.add_argument("--fine-tune-epochs", type=int, default=config.fine_tune_epochs)
    parser.add_argument("--fine-tune-layers", type=int, default=config.fine_tune_layers)
    parser.add_argument("--learning-rate", type=float, default=config.learning_rate)
    parser.add_argument(
        "--fine-tune-learning-rate",
        type=float,
        default=config.fine_tune_learning_rate,
    )
    parser.add_argument("--seed", type=int, default=config.seed)
    parser.add_argument("--workers", type=int, default=config.workers)
    parser.add_argument(
        "--no-pretrained-weights",
        action="store_true",
        help="Initialize randomly instead of loading timm pretrained weights.",
    )
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--misclassified-examples",
        type=int,
        default=config.misclassified_examples,
    )
    return parser.parse_args()


def find_cached_kaggle_dataset(handle: str) -> Path | None:
    owner, dataset = handle.split("/", maxsplit=1)
    versions_dir = Path.home() / ".cache" / "kagglehub" / "datasets" / owner / dataset / "versions"
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


def resolve_dataset_dir(dataset_dir: Path | None, config: Config) -> Path:
    root = dataset_dir or find_cached_kaggle_dataset(config.dataset_handle)
    if root is None:
        root = Path(kagglehub.dataset_download(config.dataset_handle))
    root = root.expanduser().resolve()
    yolo_root = find_yolo_root(root)
    if yolo_root is None:
        expected = ", ".join(f"{split}/images and {split}/labels" for split in SPLITS)
        raise FileNotFoundError(f"Could not find YOLO split folders ({expected}) under {root}.")
    return yolo_root


def read_yolo_class_names(dataset_dir: Path) -> list[str]:
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.is_file():
        return ["aluminum", "paper", "plastic"]
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    names = data.get("names", ["aluminum", "paper", "plastic"])
    if isinstance(names, dict):
        return [names[index] for index in sorted(names)]
    return list(names)


def read_image_label(label_path: Path, detection_names: list[str]) -> str:
    if not label_path.is_file():
        return BACKGROUND_CLASS
    rows = [line.split() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return BACKGROUND_CLASS

    best_class_id = None
    best_area = -1.0
    for row in rows:
        if len(row) < 5:
            continue
        class_id = int(float(row[0]))
        area = float(row[3]) * float(row[4])
        if area > best_area:
            best_class_id = class_id
            best_area = area

    if best_class_id is None:
        return BACKGROUND_CLASS
    return detection_names[best_class_id]


def split_contains_background(
    dataset_dir: Path,
    split: str,
    image_extensions: frozenset[str],
    detection_names: list[str],
) -> bool:
    image_dir = dataset_dir / split / "images"
    label_dir = dataset_dir / split / "labels"
    for image_path in image_dir.iterdir():
        if not image_path.is_file() or image_path.suffix.lower() not in image_extensions:
            continue
        if read_image_label(label_dir / f"{image_path.stem}.txt", detection_names) == BACKGROUND_CLASS:
            return True
    return False


def collect_split_examples(
    dataset_dir: Path,
    split: str,
    image_extensions: frozenset[str],
    detection_names: list[str],
    class_to_label: dict[str, int],
) -> tuple[list[Path], np.ndarray]:
    image_dir = dataset_dir / split / "images"
    label_dir = dataset_dir / split / "labels"
    paths: list[Path] = []
    labels: list[int] = []

    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in image_extensions:
            continue
        class_name = read_image_label(label_dir / f"{image_path.stem}.txt", detection_names)
        paths.append(image_path)
        labels.append(class_to_label[class_name])

    if not paths:
        raise ValueError(f"No images found in {image_dir}.")
    return paths, np.array(labels, dtype=np.int64)


def collect_datasets(
    dataset_dir: Path,
    config: Config,
) -> tuple[list[Path], np.ndarray, list[Path], np.ndarray, list[Path], np.ndarray, list[str]]:
    detection_names = read_yolo_class_names(dataset_dir)
    has_train_background = split_contains_background(
        dataset_dir,
        "train",
        config.image_extensions,
        detection_names,
    )
    has_any_background = any(
        split_contains_background(dataset_dir, split, config.image_extensions, detection_names)
        for split in SPLITS
    )
    if has_any_background and not has_train_background:
        raise ValueError(
            "Background images were found outside train, but none were found in train/images."
        )

    class_names = sorted(detection_names + ([BACKGROUND_CLASS] if has_train_background else []))
    class_to_label = {class_name: index for index, class_name in enumerate(class_names)}
    train_paths, train_labels = collect_split_examples(
        dataset_dir,
        "train",
        config.image_extensions,
        detection_names,
        class_to_label,
    )
    val_paths, val_labels = collect_split_examples(
        dataset_dir,
        "valid",
        config.image_extensions,
        detection_names,
        class_to_label,
    )
    test_paths, test_labels = collect_split_examples(
        dataset_dir,
        "test",
        config.image_extensions,
        detection_names,
        class_to_label,
    )
    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels, class_names


class EcoDetectDataset(Dataset):
    def __init__(
        self,
        paths: list[Path],
        labels: np.ndarray,
        transform: transforms.Compose,
    ) -> None:
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, torch.tensor(int(self.labels[index]), dtype=torch.long)


def make_transforms(config: Config) -> tuple[transforms.Compose, transforms.Compose]:
    height, width = config.image_size
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                (height, width),
                scale=config.random_resized_crop_scale,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(config.random_rotation),
            transforms.ColorJitter(
                brightness=config.color_jitter,
                contrast=config.color_jitter,
                saturation=config.color_jitter,
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, eval_transform


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
    )


def calculate_class_weights(class_counts: np.ndarray) -> torch.Tensor:
    if np.any(class_counts == 0):
        raise ValueError(f"At least one class has no training images: {class_counts}")
    total = int(class_counts.sum())
    weights = [total / (len(class_counts) * int(count)) for count in class_counts]
    return torch.tensor(weights, dtype=torch.float32)


def build_model(
    model_name: str,
    num_classes: int,
    pretrained: bool,
    dropout_rate: float,
) -> nn.Module:
    try:
        return timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout_rate,
        )
    except RuntimeError as error:
        if not pretrained:
            raise
        print(f"Could not load pretrained weights for {model_name}: {error}")
        print("Falling back to random initialization.")
        return timm.create_model(
            model_name,
            pretrained=False,
            num_classes=num_classes,
            drop_rate=dropout_rate,
        )


def set_backbone_trainable(model: nn.Module, trainable: bool, fine_tune_layers: int = 0) -> None:
    parameters = list(model.parameters())
    for parameter in parameters:
        parameter.requires_grad = trainable
    classifier = model.get_classifier()
    for parameter in classifier.parameters():
        parameter.requires_grad = True

    if trainable and fine_tune_layers > 0:
        for parameter in parameters:
            parameter.requires_grad = False
        for parameter in parameters[-fine_tune_layers:]:
            parameter.requires_grad = True
        for parameter in classifier.parameters():
            parameter.requires_grad = True


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_examples += batch_size

    return total_loss / total_examples, total_correct / total_examples


def train_stage(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    learning_rate: float,
    epochs: int,
    patience: int,
    output_dir: Path,
    stage: str,
) -> dict[str, list[float]]:
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
    )
    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    best_val_loss = math.inf
    best_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, device)
        history["loss"].append(train_loss)
        history["accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        print(
            f"{stage} epoch {epoch}/{epochs}: "
            f"loss={train_loss:.4f} accuracy={train_accuracy:.4f} "
            f"val_loss={val_loss:.4f} val_accuracy={val_accuracy:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save(best_state, output_dir / f"best_{stage}.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping {stage} after {epoch} epochs.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def merge_histories(
    first: dict[str, list[float]],
    second: dict[str, list[float]] | None,
) -> dict[str, list[float]]:
    history = {key: list(values) for key, values in first.items()}
    if second:
        for key, values in second.items():
            history.setdefault(key, []).extend(values)
    return history


def plot_training_history(history: dict[str, list[float]], output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(history.get("loss", []), label="train_loss")
    axes[0].plot(history.get("val_loss", []), label="val_loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(history.get("accuracy", []), label="train_accuracy")
    axes[1].plot(history.get("val_accuracy", []), label="val_accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(output_dir / "training_curves.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str],
    output_path: Path,
    normalized: bool,
) -> None:
    values = matrix.astype(float)
    if normalized:
        row_totals = values.sum(axis=1, keepdims=True)
        values = np.divide(values, row_totals, out=np.zeros_like(values), where=row_totals != 0)

    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(values, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title="Normalized confusion matrix" if normalized else "Confusion matrix",
    )
    threshold = values.max() / 2 if values.size else 0
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            text = f"{values[row, column]:.2f}" if normalized else str(matrix[row, column])
            axis.text(
                column,
                row,
                text,
                ha="center",
                va="center",
                color="white" if values[row, column] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            logits = model(images)
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    probability_array = np.concatenate(probabilities, axis=0)
    return probability_array, probability_array.argmax(axis=1)


def save_misclassified_gallery(
    file_paths: list[Path],
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    confidences: np.ndarray,
    class_names: list[str],
    output_path: Path,
    limit: int,
) -> int:
    incorrect = np.flatnonzero(true_labels != predicted_labels)
    if not len(incorrect) or limit <= 0:
        return 0

    selected = incorrect[np.argsort(confidences[incorrect])[::-1]][:limit]
    columns = min(5, len(selected))
    rows = math.ceil(len(selected) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows))
    axes = np.atleast_1d(axes).ravel()

    for axis, index in zip(axes, selected):
        image = Image.open(file_paths[index]).convert("RGB")
        axis.imshow(image)
        axis.set_title(
            f"True: {class_names[true_labels[index]]}\n"
            f"Pred: {class_names[predicted_labels[index]]} "
            f"({confidences[index]:.1%})"
        )
        axis.axis("off")

    for axis in axes[len(selected) :]:
        axis.axis("off")

    figure.suptitle("Most confident misclassifications", fontsize=16)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return len(selected)


def save_evaluation_outputs(
    model: nn.Module,
    test_loader: DataLoader,
    test_paths: list[Path],
    test_labels: np.ndarray,
    class_names: list[str],
    output_dir: Path,
    device: torch.device,
    misclassified_examples: int,
) -> None:
    probabilities, predicted_labels = predict(model, test_loader, device)
    confidences = probabilities.max(axis=1)
    label_ids = list(range(len(class_names)))
    report = classification_report(
        test_labels,
        predicted_labels,
        labels=label_ids,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(test_labels, predicted_labels, labels=label_ids)

    (output_dir / "classification_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    (output_dir / "classification_report.txt").write_text(
        classification_report(
            test_labels,
            predicted_labels,
            labels=label_ids,
            target_names=class_names,
            zero_division=0,
        ),
        encoding="utf-8",
    )
    np.savetxt(
        output_dir / "confusion_matrix.csv",
        matrix,
        delimiter=",",
        fmt="%d",
        header=",".join(class_names),
        comments="",
    )
    plot_confusion_matrix(matrix, class_names, output_dir / "confusion_matrix.png", normalized=False)
    plot_confusion_matrix(
        matrix,
        class_names,
        output_dir / "confusion_matrix_normalized.png",
        normalized=True,
    )

    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            ["file", "true_label", "predicted_label", "confidence", "correct"]
            + [f"probability_{name}" for name in class_names]
        )
        for path, true_id, predicted_id, confidence, row in zip(
            test_paths,
            test_labels,
            predicted_labels,
            confidences,
            probabilities,
        ):
            writer.writerow(
                [
                    path,
                    class_names[true_id],
                    class_names[predicted_id],
                    float(confidence),
                    bool(true_id == predicted_id),
                    *[float(value) for value in row],
                ]
            )

    gallery_count = save_misclassified_gallery(
        test_paths,
        test_labels,
        predicted_labels,
        confidences,
        class_names,
        output_dir / "misclassified_examples.png",
        misclassified_examples,
    )
    print("\nClassification report:")
    print((output_dir / "classification_report.txt").read_text(encoding="utf-8"))
    print(f"Misclassified images: {int((test_labels != predicted_labels).sum())}")
    print(f"Gallery examples saved: {gallery_count}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    config_path = parse_config_path()
    config = load_config(config_path)
    args = parse_args(config, config_path)
    if args.epochs < 1 or args.fine_tune_epochs < 0:
        raise ValueError("--epochs must be positive and --fine-tune-epochs nonnegative.")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_dir = resolve_dataset_dir(args.dataset_dir, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    (
        train_paths,
        train_labels,
        val_paths,
        val_labels,
        test_paths,
        test_labels,
        class_names,
    ) = collect_datasets(dataset_dir, config)
    train_counts = np.bincount(train_labels, minlength=len(class_names))
    class_weights = calculate_class_weights(train_counts).to(device)
    train_transform, eval_transform = make_transforms(config)
    train_loader = make_loader(
        EcoDetectDataset(train_paths, train_labels, train_transform),
        args.batch_size,
        shuffle=True,
        workers=args.workers,
    )
    val_loader = make_loader(
        EcoDetectDataset(val_paths, val_labels, eval_transform),
        args.batch_size,
        shuffle=False,
        workers=args.workers,
    )
    test_loader = make_loader(
        EcoDetectDataset(test_paths, test_labels, eval_transform),
        args.batch_size,
        shuffle=False,
        workers=args.workers,
    )

    print(f"Dataset root: {dataset_dir}")
    print(f"Model: {args.model_name}")
    print(f"Device: {device}")
    print(f"Classes: {class_names}")
    print(f"Training subset counts: {dict(zip(class_names, train_counts.tolist()))}")
    if BACKGROUND_CLASS not in class_names:
        print(
            "No background training samples were found, so this run will train "
            "only on annotated waste classes."
        )
    print(
        "Class weights: "
        f"{dict(zip(class_names, [float(value) for value in class_weights.cpu().tolist()]))}"
    )

    model = build_model(
        args.model_name,
        len(class_names),
        config.use_pretrained_weights and not args.no_pretrained_weights,
        config.dropout_rate,
    ).to(device)
    print(f"Trainable parameters: {sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad):,}")
    print(f"Total parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")

    if args.check_only:
        images, _ = next(iter(train_loader))
        model.eval()
        with torch.no_grad():
            output = model(images[:1].to(device))
        print(f"Smoke test output shape: {tuple(output.shape)}")
        return

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    set_backbone_trainable(model, trainable=False)
    head_history = train_stage(
        model,
        train_loader,
        val_loader,
        criterion,
        device,
        args.learning_rate,
        args.epochs,
        config.early_stopping_patience,
        args.output_dir,
        "head",
    )

    fine_tune_history = None
    if args.fine_tune_epochs:
        set_backbone_trainable(model, trainable=True, fine_tune_layers=args.fine_tune_layers)
        fine_tune_history = train_stage(
            model,
            train_loader,
            val_loader,
            criterion,
            device,
            args.fine_tune_learning_rate,
            args.fine_tune_epochs,
            config.early_stopping_patience,
            args.output_dir,
            "fine_tuned",
        )

    test_loss, test_accuracy = run_epoch(model, test_loader, criterion, device)
    test_metrics = {"loss": test_loss, "accuracy": test_accuracy}
    print(f"Test metrics: {test_metrics}")

    history = merge_histories(head_history, fine_tune_history)
    plot_training_history(history, args.output_dir)
    save_evaluation_outputs(
        model,
        test_loader,
        test_paths,
        test_labels,
        class_names,
        args.output_dir,
        device,
        args.misclassified_examples,
    )

    torch.save(model.state_dict(), args.output_dir / "ecodetect_mobilenetv4.pth")
    (args.output_dir / "labels.json").write_text(
        json.dumps(class_names, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "test_metrics.json").write_text(
        json.dumps({key: float(value) for key, value in test_metrics.items()}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved model artifacts to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
