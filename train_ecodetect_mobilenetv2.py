from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import kagglehub
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml
from sklearn.metrics import classification_report, confusion_matrix


DEFAULT_CONFIG_PATH = Path("config.yaml")
BACKGROUND_CLASS = "background"
SPLITS = ("train", "valid", "test")


@dataclass(frozen=True)
class Config:
    dataset_handle: str
    dataset_dir: Path | None
    output_dir: Path
    image_size: tuple[int, int]
    image_extensions: frozenset[str]
    batch_size: int
    epochs: int
    fine_tune_epochs: int
    fine_tune_layers: int
    learning_rate: float
    fine_tune_learning_rate: float
    seed: int
    use_imagenet_weights: bool
    random_rotation: float
    random_zoom: float
    random_contrast: float
    dropout_rate: float
    early_stopping_patience: int
    lr_reduction_factor: float
    lr_reduction_patience: int
    min_learning_rate: float
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

    return Config(
        dataset_handle=str(
            dataset.get(
                "handle",
                "ahsan71/ecodetect-recyclable-waste-detection-dataset",
            )
        ),
        dataset_dir=config_path_value(dataset.get("dir")),
        output_dir=Path(str(output.get("dir", "artifacts/ecodetect"))),
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
        use_imagenet_weights=bool(model.get("use_imagenet_weights", True)),
        random_rotation=float(augmentation.get("random_rotation", 0.08)),
        random_zoom=float(augmentation.get("random_zoom", 0.1)),
        random_contrast=float(augmentation.get("random_contrast", 0.1)),
        dropout_rate=float(model.get("dropout_rate", 0.25)),
        early_stopping_patience=int(callbacks.get("early_stopping_patience", 3)),
        lr_reduction_factor=float(callbacks.get("lr_reduction_factor", 0.2)),
        lr_reduction_patience=int(callbacks.get("lr_reduction_patience", 2)),
        min_learning_rate=float(callbacks.get("min_learning_rate", 0.0000001)),
        misclassified_examples=int(evaluation.get("misclassified_examples", 25)),
    )


def parse_config_path() -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args, _ = parser.parse_known_args()
    return args.config


def parse_args(config: Config, config_path: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train MobileNetV2 on the EcoDetect YOLO dataset by converting "
            "image annotations into image-level labels."
        )
    )
    parser.add_argument("--config", type=Path, default=config_path)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=config.dataset_dir,
        help="Dataset root containing train/valid/test images and labels.",
    )
    parser.add_argument("--output-dir", type=Path, default=config.output_dir)
    parser.add_argument("--batch-size", type=int, default=config.batch_size)
    parser.add_argument("--epochs", type=int, default=config.epochs)
    parser.add_argument(
        "--fine-tune-epochs", type=int, default=config.fine_tune_epochs
    )
    parser.add_argument(
        "--fine-tune-layers",
        type=int,
        default=config.fine_tune_layers,
        help="Number of final MobileNetV2 layers to unfreeze.",
    )
    parser.add_argument("--learning-rate", type=float, default=config.learning_rate)
    parser.add_argument(
        "--fine-tune-learning-rate",
        type=float,
        default=config.fine_tune_learning_rate,
    )
    parser.add_argument("--seed", type=int, default=config.seed)
    parser.add_argument(
        "--no-imagenet-weights",
        action="store_true",
        help="Initialize randomly instead of downloading ImageNet weights.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate the data and build the model without training.",
    )
    parser.add_argument(
        "--misclassified-examples",
        type=int,
        default=config.misclassified_examples,
        help="Number of incorrect test predictions to save in the gallery.",
    )
    return parser.parse_args()


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


def find_cached_kaggle_dataset(handle: str) -> Path | None:
    owner, dataset = handle.split("/", maxsplit=1)
    cache_root = Path.home() / ".cache" / "kagglehub" / "datasets" / owner / dataset
    versions_dir = cache_root / "versions"
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


def read_yolo_class_names(dataset_dir: Path) -> list[str]:
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.is_file():
        return ["aluminum", "paper", "plastic"]

    for line in data_yaml.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("names:"):
            names_text = stripped.partition(":")[2].strip()
            return [
                name.strip().strip("'\"")
                for name in names_text.strip("[]").split(",")
                if name.strip()
            ]
    return ["aluminum", "paper", "plastic"]


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
        width = float(row[3])
        height = float(row[4])
        area = width * height
        if area > best_area:
            best_class_id = class_id
            best_area = area

    if best_class_id is None:
        return BACKGROUND_CLASS
    return detection_names[best_class_id]


def collect_split_examples(
    dataset_dir: Path,
    split: str,
    image_extensions: frozenset[str],
    detection_names: list[str],
    class_to_label: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image_dir = dataset_dir / split / "images"
    label_dir = dataset_dir / split / "labels"
    paths: list[str] = []
    labels: list[int] = []
    class_names: list[str] = []

    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in image_extensions:
            continue
        class_name = read_image_label(label_dir / f"{image_path.stem}.txt", detection_names)
        paths.append(str(image_path))
        labels.append(class_to_label[class_name])
        class_names.append(class_name)

    if not paths:
        raise ValueError(f"No images found in {image_dir}.")

    return (
        np.array(paths),
        np.array(labels, dtype=np.int64),
        np.array(class_names),
    )


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


def make_dataset(
    paths: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    image_size: tuple[int, int],
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        dataset = dataset.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)

    def load_image(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image = tf.io.read_file(path)
        image = tf.io.decode_image(
            image,
            channels=3,
            expand_animations=False,
        )
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, image_size)
        image = tf.cast(image, tf.float32)
        return image, label

    return (
        dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def load_datasets(
    dataset_dir: Path,
    batch_size: int,
    seed: int,
    config: Config,
) -> tuple[
    tf.data.Dataset,
    tf.data.Dataset,
    tf.data.Dataset,
    list[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
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
            "Background images were found outside train, but none were found in "
            "train/images. Add background samples to train/images so the model can "
            "learn the background class."
        )

    class_names = sorted(detection_names + ([BACKGROUND_CLASS] if has_train_background else []))
    class_to_label = {class_name: index for index, class_name in enumerate(class_names)}

    train_paths, train_labels, _ = collect_split_examples(
        dataset_dir,
        "train",
        config.image_extensions,
        detection_names,
        class_to_label,
    )
    val_paths, val_labels, _ = collect_split_examples(
        dataset_dir,
        "valid",
        config.image_extensions,
        detection_names,
        class_to_label,
    )
    test_paths, test_labels, _ = collect_split_examples(
        dataset_dir,
        "test",
        config.image_extensions,
        detection_names,
        class_to_label,
    )

    train_ds = make_dataset(
        train_paths,
        train_labels,
        batch_size,
        config.image_size,
        shuffle=True,
        seed=seed,
    )
    val_ds = make_dataset(
        val_paths,
        val_labels,
        batch_size,
        config.image_size,
        shuffle=False,
        seed=seed,
    )
    test_ds = make_dataset(
        test_paths,
        test_labels,
        batch_size,
        config.image_size,
        shuffle=False,
        seed=seed,
    )
    train_counts = np.bincount(train_labels, minlength=len(class_names))
    return train_ds, val_ds, test_ds, class_names, train_counts, test_paths, test_labels


def calculate_class_weights(class_counts: np.ndarray) -> dict[int, float]:
    if np.any(class_counts == 0):
        raise ValueError(f"At least one class has no training images: {class_counts}")
    total = int(class_counts.sum())
    return {
        index: total / (len(class_counts) * int(count))
        for index, count in enumerate(class_counts)
    }


def build_model(
    num_classes: int,
    learning_rate: float,
    use_imagenet_weights: bool,
    config: Config,
) -> tuple[tf.keras.Model, tf.keras.Model]:
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(config.random_rotation),
            tf.keras.layers.RandomZoom(config.random_zoom),
            tf.keras.layers.RandomContrast(config.random_contrast),
        ],
        name="augmentation",
    )
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*config.image_size, 3),
        include_top=False,
        weights="imagenet" if use_imagenet_weights else None,
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(*config.image_size, 3), name="image")
    x = augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(config.dropout_rate)(x)
    outputs = tf.keras.layers.Dense(
        num_classes, activation="softmax", name="predictions"
    )(x)
    model = tf.keras.Model(inputs, outputs, name="ecodetect_mobilenetv2")
    compile_model(model, learning_rate)
    return model, base_model


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        ],
    )


def make_callbacks(
    output_dir: Path,
    stage: str,
    config: Config,
) -> list[tf.keras.callbacks.Callback]:
    return [
        tf.keras.callbacks.ModelCheckpoint(
            output_dir / f"best_{stage}.keras",
            monitor="val_loss",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.early_stopping_patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.lr_reduction_factor,
            patience=config.lr_reduction_patience,
            min_lr=config.min_learning_rate,
        ),
    ]


def merge_histories(
    first: tf.keras.callbacks.History,
    second: tf.keras.callbacks.History | None,
) -> dict[str, list[float]]:
    history = {key: list(values) for key, values in first.history.items()}
    if second:
        for key, values in second.history.items():
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
        values = np.divide(
            values,
            row_totals,
            out=np.zeros_like(values),
            where=row_totals != 0,
        )

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


def save_misclassified_gallery(
    file_paths: np.ndarray,
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
        image = tf.keras.utils.load_img(str(file_paths[index]))
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
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    test_paths: np.ndarray,
    test_labels: np.ndarray,
    class_names: list[str],
    output_dir: Path,
    misclassified_examples: int,
) -> None:
    probabilities = model.predict(test_ds, verbose=1)
    predicted_labels = probabilities.argmax(axis=1)
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
    plot_confusion_matrix(
        matrix,
        class_names,
        output_dir / "confusion_matrix.png",
        normalized=False,
    )
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


def main() -> None:
    config_path = parse_config_path()
    config = load_config(config_path)
    args = parse_args(config, config_path)
    if args.epochs < 1 or args.fine_tune_epochs < 0:
        raise ValueError("--epochs must be positive and --fine-tune-epochs nonnegative.")

    tf.keras.utils.set_random_seed(args.seed)
    dataset_dir = resolve_dataset_dir(args.dataset_dir, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    (
        train_ds,
        val_ds,
        test_ds,
        class_names,
        class_counts,
        test_paths,
        test_labels,
    ) = load_datasets(
        dataset_dir,
        args.batch_size,
        args.seed,
        config,
    )
    class_weights = calculate_class_weights(class_counts)
    print(f"Dataset root: {dataset_dir}")
    print(f"Classes: {class_names}")
    print(f"Training subset counts: {dict(zip(class_names, class_counts.tolist()))}")
    if BACKGROUND_CLASS not in class_names:
        print(
            "No background training samples were found, so this run will train "
            "only on annotated waste classes."
        )
    print(f"Class weights: {class_weights}")

    model, base_model = build_model(
        len(class_names),
        args.learning_rate,
        config.use_imagenet_weights and not args.no_imagenet_weights,
        config,
    )
    model.summary()

    if args.check_only:
        images, _ = next(iter(train_ds))
        predictions = model(images[:1], training=False)
        print(f"Smoke test output shape: {predictions.shape}")
        return

    head_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=make_callbacks(args.output_dir, "head", config),
    )

    fine_tune_history = None
    if args.fine_tune_epochs:
        base_model.trainable = True
        freeze_until = max(0, len(base_model.layers) - args.fine_tune_layers)
        for layer in base_model.layers[:freeze_until]:
            layer.trainable = False
        for layer in base_model.layers[freeze_until:]:
            layer.trainable = not isinstance(
                layer, tf.keras.layers.BatchNormalization
            )

        compile_model(model, args.fine_tune_learning_rate)
        fine_tune_history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.fine_tune_epochs,
            class_weight=class_weights,
            callbacks=make_callbacks(args.output_dir, "fine_tuned", config),
        )

    test_metrics = model.evaluate(test_ds, return_dict=True)
    print(f"Test metrics: {test_metrics}")

    history = merge_histories(head_history, fine_tune_history)
    plot_training_history(history, args.output_dir)
    save_evaluation_outputs(
        model,
        test_ds,
        test_paths,
        test_labels,
        class_names,
        args.output_dir,
        args.misclassified_examples,
    )

    model_path = args.output_dir / "ecodetect_mobilenetv2.keras"
    model.save(model_path)
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

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    (args.output_dir / "ecodetect_mobilenetv2.tflite").write_bytes(tflite_model)
    print(f"Saved model artifacts to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
