import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from pycocotools import mask as mask_utils

from iou_metrics import evaluate_instance_iou


class InstanceIoUTest(unittest.TestCase):
    def evaluate(self, predictions):
        mask = mask_utils.encode(np.ones((4, 4), dtype=np.uint8, order="F"))
        mask["counts"] = mask["counts"].decode()
        annotation = {"id": 1, "image_id": 1, "category_id": 1,
                      "segmentation": mask, "bbox": [0, 0, 4, 4], "iscrowd": 0}
        gt = {"images": [{"id": 1, "height": 4, "width": 4}],
              "categories": [{"id": 1, "name": "Can"}, {"id": 2, "name": "Cup"}],
              "annotations": [annotation, dict(annotation, id=2)]}
        pred = [dict(annotation, **overrides) for overrides in predictions]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "gt.json").write_text(json.dumps(gt))
            (root / "pred.json").write_text(json.dumps(pred))
            return evaluate_instance_iou(root / "gt.json", root / "pred.json")

    def test_missed_object_counts_zero(self):
        report = self.evaluate([{"score": 0.9}])
        for kind in ("segm", "bbox"):
            result = report[kind]["overall"]
            self.assertEqual(result["matched_mean_iou"], 1)
            self.assertEqual(result["gt_mean_iou"], 0.5)
            self.assertEqual(result["recall_at_iou50"], 0.5)

    def test_duplicate_predictions_cannot_reuse_gt(self):
        report = self.evaluate([{"score": 0.9}, {"score": 0.8}, {"score": 0.7}])
        self.assertEqual(report["segm"]["overall"]["matched_objects"], 2)
        self.assertAlmostEqual(report["segm"]["overall"]["precision_at_iou50"], 2 / 3)

    def test_wrong_class_low_confidence_and_empty_predictions(self):
        for predictions in ([], [{"score": 0.4}], [{"score": 0.9, "category_id": 2}]):
            report = self.evaluate(predictions)
            self.assertIsNone(report["segm"]["overall"]["matched_mean_iou"])
            self.assertEqual(report["segm"]["overall"]["gt_mean_iou"], 0)


if __name__ == "__main__":
    unittest.main()
