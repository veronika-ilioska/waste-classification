"""Render saved TACO test masks alongside original images and annotations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycocotools.coco import COCO


COLORS = [(255, 90, 90), (60, 200, 255), (255, 200, 50), (150, 100, 255),
          (60, 230, 140), (255, 100, 210), (255, 150, 60), (100, 220, 220)]


def overlay(image, annotations, coco):
    pixels = np.array(image).copy()
    for index, annotation in enumerate(annotations):
        mask = Image.fromarray(coco.annToMask(annotation) * 255)
        mask = np.asarray(mask.resize(image.size, Image.Resampling.NEAREST)) > 0
        color = np.array(COLORS[index % len(COLORS)])
        pixels[mask] = (0.55 * pixels[mask] + 0.45 * color).astype(np.uint8)
    result = Image.fromarray(pixels)
    draw = ImageDraw.Draw(result)
    font = ImageFont.load_default(size=14)
    info = coco.imgs[annotations[0]["image_id"]] if annotations else None
    for index, annotation in enumerate(annotations):
        x, y, width, height = annotation["bbox"]
        sx, sy = image.width / info["width"], image.height / info["height"]
        x, y, right, bottom = x * sx, y * sy, (x + width) * sx, (y + height) * sy
        color = COLORS[index % len(COLORS)]
        draw.rectangle((x, y, right, bottom), outline=color, width=2)
        label = coco.cats[annotation["category_id"]]["name"]
        if "score" in annotation:
            label += f" {annotation['score']:.0%}"
        box = draw.textbbox((0, 0), label, font=font)
        x = max(0, min(x, image.width - (box[2] + 8)))
        y = max(0, y - 22)
        draw.rectangle((x, y, x + box[2] + 8, y + 22), fill=(20, 25, 35))
        draw.text((x + 4, y + 2), label, font=font, fill=color)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="A single split folder.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets/TACO/data"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    args = parser.parse_args()
    if args.count < 1 or not 0 <= args.score_threshold <= 1:
        parser.error("Count must be positive and score threshold must be between 0 and 1.")
    coco = COCO(str(args.run_dir / "coco_test_ground_truth.json"))
    predictions = json.loads((args.run_dir / "coco_test_predictions.json").read_text())
    grouped = {}
    for prediction in predictions:
        if prediction["score"] >= args.score_threshold:
            grouped.setdefault(prediction["image_id"], []).append(prediction)
    selected = random.Random(args.seed).sample(sorted(coco.imgs), min(args.count, len(coco.imgs)))
    output = args.output_dir or args.run_dir / "segmentation_images"
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for image_id in selected:
        info = coco.imgs[image_id]
        with Image.open(args.dataset_dir / info["file_name"]) as source:
            original = source.convert("RGB")
        original.thumbnail((600, 600))
        gt = coco.imgToAnns.get(image_id, [])
        pred = sorted(grouped.get(image_id, []), key=lambda ann: ann["score"], reverse=True)
        panels = [original, overlay(original, gt, coco), overlay(original, pred, coco)]
        canvas = Image.new("RGB", (original.width * 3, original.height + 64), (20, 25, 35))
        draw = ImageDraw.Draw(canvas)
        titles = ["Original", f"Ground truth ({len(gt)} objects)",
                  f"Predictions ({len(pred)} objects)"]
        for index, (panel, title) in enumerate(zip(panels, titles)):
            canvas.paste(panel, (index * original.width, 64))
            draw.text((index * original.width + 10, 10), title,
                      font=ImageFont.load_default(size=17), fill="white")
        draw.text((10, 37), info["file_name"], font=ImageFont.load_default(size=14), fill="white")
        draw.text((2 * original.width + 10, 37), f"Confidence >= {args.score_threshold:.0%}",
                  font=ImageFont.load_default(size=14), fill="white")
        filename = f"image_{image_id:05d}.jpg"
        canvas.save(output / filename, quality=94)
        manifest.append({"image_id": image_id, "source": info["file_name"], "file": filename})
    (output / "manifest.json").write_text(json.dumps({"seed": args.seed,
        "score_threshold": args.score_threshold, "images": manifest}, indent=2))
    print(f"Saved {len(manifest)} segmentation comparisons to {output.resolve()}")


if __name__ == "__main__":
    main()
