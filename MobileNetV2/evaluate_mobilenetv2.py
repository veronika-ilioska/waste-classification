from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from sklearn.metrics import classification_report, confusion_matrix

from train_mobilenetv2 import load_config, resolve_dataset_dir


load_dotenv()


def parse_args() -> argparse.Namespace:
    config = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=config.dataset_dir)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(os.getenv("MODEL_PATH", "artifacts/waste_mobilenetv2.keras")),
    )
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=Path(os.getenv("LABELS_PATH", "artifacts/labels.json")),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.getenv("EVALUATION_DIR", "artifacts/evaluation")),
    )
    parser.add_argument("--batch-size", type=int, default=config.batch_size)
    parser.add_argument(
        "--misclassified-examples",
        type=int,
        default=int(os.getenv("MISCLASSIFIED_EXAMPLES", "25")),
    )
    return parser.parse_args()


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
    file_paths: list[str],
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
        image = tf.keras.utils.load_img(file_paths[index])
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


def main() -> None:
    args = parse_args()
    config = load_config()
    dataset_dir = resolve_dataset_dir(args.dataset_dir, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.model_path.is_file():
        raise FileNotFoundError(f"Model not found: {args.model_path.resolve()}")
    if not args.labels_path.is_file():
        raise FileNotFoundError(f"Labels not found: {args.labels_path.resolve()}")

    class_names = json.loads(args.labels_path.read_text(encoding="utf-8"))
    test_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir / config.test_subdir,
        image_size=config.image_size,
        batch_size=args.batch_size,
        label_mode="int",
        shuffle=False,
        class_names=class_names,
    )
    file_paths = list(test_ds.file_paths)
    true_labels = np.concatenate([labels.numpy() for _, labels in test_ds])

    model = tf.keras.models.load_model(args.model_path)
    probabilities = model.predict(test_ds, verbose=1)
    predicted_labels = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)

    label_ids = list(range(len(class_names)))
    report = classification_report(
        true_labels,
        predicted_labels,
        labels=label_ids,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(true_labels, predicted_labels, labels=label_ids)

    (args.output_dir / "classification_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "classification_report.txt").write_text(
        classification_report(
            true_labels,
            predicted_labels,
            labels=label_ids,
            target_names=class_names,
            zero_division=0,
        ),
        encoding="utf-8",
    )
    np.savetxt(
        args.output_dir / "confusion_matrix.csv",
        matrix,
        delimiter=",",
        fmt="%d",
        header=",".join(class_names),
        comments="",
    )
    plot_confusion_matrix(
        matrix,
        class_names,
        args.output_dir / "confusion_matrix.png",
        normalized=False,
    )
    plot_confusion_matrix(
        matrix,
        class_names,
        args.output_dir / "confusion_matrix_normalized.png",
        normalized=True,
    )

    with (args.output_dir / "predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            ["file", "true_label", "predicted_label", "confidence", "correct"]
            + [f"probability_{name}" for name in class_names]
        )
        for path, true_id, predicted_id, confidence, row in zip(
            file_paths,
            true_labels,
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
        file_paths,
        true_labels,
        predicted_labels,
        confidences,
        class_names,
        args.output_dir / "misclassified_examples.png",
        args.misclassified_examples,
    )

    print("\nClassification report:")
    print((args.output_dir / "classification_report.txt").read_text(encoding="utf-8"))
    print(f"Misclassified images: {int((true_labels != predicted_labels).sum())}")
    print(f"Gallery examples saved: {gallery_count}")
    print(f"Evaluation saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
