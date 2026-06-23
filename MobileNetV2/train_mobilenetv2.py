from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import kagglehub
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv


load_dotenv()


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false, got {value!r}.")


@dataclass(frozen=True)
class Config:
    dataset_handle: str
    dataset_dir: Path | None
    train_subdir: str
    test_subdir: str
    output_dir: Path
    image_size: tuple[int, int]
    image_extensions: frozenset[str]
    batch_size: int
    epochs: int
    fine_tune_epochs: int
    fine_tune_layers: int
    learning_rate: float
    fine_tune_learning_rate: float
    validation_split: float
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


def load_config() -> Config:
    dataset_dir = os.getenv("DATASET_DIR", "").strip()
    extensions = frozenset(
        extension.strip().lower()
        for extension in os.getenv(
            "IMAGE_EXTENSIONS", ".bmp,.gif,.jpeg,.jpg,.png"
        ).split(",")
        if extension.strip()
    )
    return Config(
        dataset_handle=os.getenv(
            "DATASET_HANDLE", "shubhamdivakar/waste-classification-dataset"
        ),
        dataset_dir=Path(dataset_dir) if dataset_dir else None,
        train_subdir=os.getenv("TRAIN_SUBDIR", "TRAIN"),
        test_subdir=os.getenv("TEST_SUBDIR", "TEST"),
        output_dir=Path(os.getenv("OUTPUT_DIR", "artifacts")),
        image_size=(
            int(os.getenv("IMAGE_HEIGHT", "224")),
            int(os.getenv("IMAGE_WIDTH", "224")),
        ),
        image_extensions=extensions,
        batch_size=int(os.getenv("BATCH_SIZE", "32")),
        epochs=int(os.getenv("EPOCHS", "10")),
        fine_tune_epochs=int(os.getenv("FINE_TUNE_EPOCHS", "5")),
        fine_tune_layers=int(os.getenv("FINE_TUNE_LAYERS", "30")),
        learning_rate=float(os.getenv("LEARNING_RATE", "0.001")),
        fine_tune_learning_rate=float(
            os.getenv("FINE_TUNE_LEARNING_RATE", "0.00001")
        ),
        validation_split=float(os.getenv("VALIDATION_SPLIT", "0.2")),
        seed=int(os.getenv("RANDOM_SEED", "42")),
        use_imagenet_weights=env_bool("USE_IMAGENET_WEIGHTS", True),
        random_rotation=float(os.getenv("RANDOM_ROTATION", "0.08")),
        random_zoom=float(os.getenv("RANDOM_ZOOM", "0.1")),
        random_contrast=float(os.getenv("RANDOM_CONTRAST", "0.1")),
        dropout_rate=float(os.getenv("DROPOUT_RATE", "0.25")),
        early_stopping_patience=int(os.getenv("EARLY_STOPPING_PATIENCE", "3")),
        lr_reduction_factor=float(os.getenv("LR_REDUCTION_FACTOR", "0.2")),
        lr_reduction_patience=int(os.getenv("LR_REDUCTION_PATIENCE", "2")),
        min_learning_rate=float(os.getenv("MIN_LEARNING_RATE", "0.0000001")),
    )


def parse_args(config: Config) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=config.dataset_dir,
        help="Dataset root containing TRAIN and TEST. Downloads it when omitted.",
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
    parser.add_argument(
        "--validation-split", type=float, default=config.validation_split
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
    return parser.parse_args()


def resolve_dataset_dir(dataset_dir: Path | None, config: Config) -> Path:
    root = dataset_dir or Path(kagglehub.dataset_download(config.dataset_handle))
    root = root.expanduser().resolve()
    if not (root / config.train_subdir).is_dir() or not (
        root / config.test_subdir
    ).is_dir():
        raise FileNotFoundError(
            f"{root} must contain {config.train_subdir} and "
            f"{config.test_subdir} directories."
        )
    return root


def load_datasets(
    dataset_dir: Path,
    batch_size: int,
    validation_split: float,
    seed: int,
    config: Config,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, list[str], np.ndarray]:
    common = {
        "directory": dataset_dir / config.train_subdir,
        "validation_split": validation_split,
        "seed": seed,
        "image_size": config.image_size,
        "batch_size": batch_size,
        "label_mode": "int",
    }
    train_ds = tf.keras.utils.image_dataset_from_directory(
        subset="training",
        shuffle=True,
        **common,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        subset="validation",
        shuffle=False,
        **common,
    )
    class_names = train_ds.class_names

    test_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir / config.test_subdir,
        image_size=config.image_size,
        batch_size=batch_size,
        label_mode="int",
        shuffle=False,
        class_names=class_names,
    )

    class_counts = np.array(
        [
            sum(
                path.is_file() and path.suffix.lower() in config.image_extensions
                for path in (
                    dataset_dir / config.train_subdir / class_name
                ).rglob("*")
            )
            for class_name in class_names
        ],
        dtype=np.int64,
    )

    options = tf.data.Options()
    options.experimental_deterministic = False

    train_ds = train_ds.with_options(options).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds, test_ds, class_names, class_counts


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
    model = tf.keras.Model(inputs, outputs, name="waste_mobilenetv2")
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


def main() -> None:
    config = load_config()
    args = parse_args(config)
    if args.epochs < 1 or args.fine_tune_epochs < 0:
        raise ValueError("--epochs must be positive and --fine-tune-epochs nonnegative.")
    if not 0 < args.validation_split < 1:
        raise ValueError("--validation-split must be between 0 and 1.")

    tf.keras.utils.set_random_seed(args.seed)
    dataset_dir = resolve_dataset_dir(args.dataset_dir, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds, class_names, class_counts = load_datasets(
        dataset_dir,
        args.batch_size,
        args.validation_split,
        args.seed,
        config,
    )
    class_weights = calculate_class_weights(class_counts)
    print(f"Classes: {class_names}")
    print(f"Training subset counts: {dict(zip(class_names, class_counts.tolist()))}")
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

    model_path = args.output_dir / "waste_mobilenetv2.keras"
    model.save(model_path)
    (args.output_dir / "labels.json").write_text(
        json.dumps(class_names, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "history.json").write_text(
        json.dumps(merge_histories(head_history, fine_tune_history), indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "test_metrics.json").write_text(
        json.dumps({key: float(value) for key, value in test_metrics.items()}, indent=2),
        encoding="utf-8",
    )

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    (args.output_dir / "waste_mobilenetv2.tflite").write_bytes(tflite_model)
    print(f"Saved model artifacts to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
