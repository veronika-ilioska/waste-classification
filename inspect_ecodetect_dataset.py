from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import kagglehub
import yaml


DEFAULT_HANDLE = "ahsan71/ecodetect-recyclable-waste-detection-dataset"
SPLITS = ("train", "valid", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count images and YOLO annotation rows in the EcoDetect dataset."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Dataset root. Uses the Kaggle cache or downloads the dataset when omitted.",
    )
    parser.add_argument("--handle", default=DEFAULT_HANDLE)
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
        return []
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    names = data.get("names", [])
    if isinstance(names, dict):
        return [names[index] for index in sorted(names)]
    return list(names)


def count_split(dataset_dir: Path, split: str, class_names: list[str]) -> dict:
    image_dir = dataset_dir / split / "images"
    label_dir = dataset_dir / split / "labels"
    images = [path for path in image_dir.iterdir() if path.is_file()]
    labels = [path for path in label_dir.iterdir() if path.is_file()]
    annotation_rows = 0
    empty_labels = 0
    missing_labels = 0
    class_counts: Counter[str] = Counter()

    for image_path in images:
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            missing_labels += 1
            continue

        rows = [
            line.split()
            for line in label_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not rows:
            empty_labels += 1
            continue

        annotation_rows += len(rows)
        for row in rows:
            if not row:
                continue
            class_id = int(float(row[0]))
            class_name = class_names[class_id] if class_id < len(class_names) else str(class_id)
            class_counts[class_name] += 1

    return {
        "images": len(images),
        "label_files": len(labels),
        "annotation_rows": annotation_rows,
        "empty_labels": empty_labels,
        "missing_labels": missing_labels,
        "class_counts": class_counts,
    }


def main() -> None:
    args = parse_args()
    dataset_dir = resolve_dataset_dir(args.dataset_dir, args.handle)
    class_names = read_class_names(dataset_dir)

    print(f"Dataset root: {dataset_dir}")
    print(f"Classes: {class_names or 'not found'}")
    print()

    totals = Counter()
    total_class_counts: Counter[str] = Counter()
    for split in SPLITS:
        counts = count_split(dataset_dir, split, class_names)
        totals.update(
            {
                "images": counts["images"],
                "label_files": counts["label_files"],
                "annotation_rows": counts["annotation_rows"],
                "empty_labels": counts["empty_labels"],
                "missing_labels": counts["missing_labels"],
            }
        )
        total_class_counts.update(counts["class_counts"])

        print(f"{split}:")
        print(f"  images: {counts['images']}")
        print(f"  label files: {counts['label_files']}")
        print(f"  annotation rows: {counts['annotation_rows']}")
        print(f"  empty label files: {counts['empty_labels']}")
        print(f"  missing label files: {counts['missing_labels']}")
        for class_name, count in counts["class_counts"].items():
            print(f"  {class_name}: {count}")
        print()

    print("total:")
    print(f"  images: {totals['images']}")
    print(f"  label files: {totals['label_files']}")
    print(f"  annotation rows: {totals['annotation_rows']}")
    print(f"  empty label files: {totals['empty_labels']}")
    print(f"  missing label files: {totals['missing_labels']}")
    for class_name, count in total_class_counts.items():
        print(f"  {class_name}: {count}")


if __name__ == "__main__":
    main()
