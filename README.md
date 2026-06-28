# Waste Classification

A MobileNetV2 image classifier for sorting waste into three categories:

- `E`: electronic waste
- `O`: organic waste
- `R`: recyclable waste

The model uses transfer learning with ImageNet weights and is trained on the
[Waste Classification Dataset](https://www.kaggle.com/datasets/shubhamdivakar/waste-classification-dataset).

## Test Results

The final model was evaluated on 2,539 test images.

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Electronic (`E`) | 0.79 | 1.00 | 0.88 | 26 |
| Organic (`O`) | 0.93 | 0.97 | 0.95 | 1,401 |
| Recyclable (`R`) | 0.96 | 0.90 | 0.93 | 1,112 |
| **Accuracy** |  |  | **0.94** | **2,539** |
| **Macro average** | **0.89** | **0.96** | **0.92** | **2,539** |
| **Weighted average** | **0.94** | **0.94** | **0.94** | **2,539** |

The model correctly classified all 26 electronic-waste test images. However,
the electronic class has very limited support, so its metrics are less reliable
than those for the organic and recyclable classes. More electronic-waste data
is needed before relying on this model in production.

### Confusion Matrix

| Actual \ Predicted | E | O | R |
|---|---:|---:|---:|
| E | 26 | 0 | 0 |
| O | 0 | 1,361 | 40 |
| R | 7 | 106 | 999 |

### Misclassified Examples

The following gallery shows the 25 most confident incorrect predictions. Each
image includes its true class, predicted class, and the model's confidence in
the incorrect prediction.

![Most confident misclassified examples](artifacts/evaluation/misclassified_examples.png)

## Setup

Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and adjust the settings when needed. Leaving
`DATASET_DIR` empty lets `kagglehub` download or locate the dataset
automatically.

## Training

```powershell
.\.venv\Scripts\python.exe MobileNetV2\train_mobilenetv2.py
```

Training produces the following files under `artifacts/`:

- `waste_mobilenetv2.keras`: final Keras model
- `waste_mobilenetv2.tflite`: TensorFlow Lite export
- `best_head.keras`: best classifier-head checkpoint
- `best_fine_tuned.keras`: best fine-tuning checkpoint
- `labels.json`: model class order
- `history.json`: epoch-by-epoch training metrics
- `test_metrics.json`: final test loss and accuracy

## Evaluation

```powershell
.\.venv\Scripts\python.exe MobileNetV2\evaluate_mobilenetv2.py
```

Evaluation outputs are written to `artifacts/evaluation/`:

- Classification report in text and JSON formats
- Raw and normalized confusion-matrix images
- Predictions for every test image in CSV format
- A gallery of the most confident misclassifications

## EcoDetect MobileNetV2

The EcoDetect dataset is YOLO-formatted, so
`MobileNetV2/train_ecodetect_mobilenetv2.py` converts each image's bounding-box
annotations into one image-level class before training a MobileNetV2 classifier.
If an image has multiple object classes, the largest bounding box decides the
image label.

Run a smoke check:

```powershell
.\.venv\Scripts\python.exe MobileNetV2\train_ecodetect_mobilenetv2.py --check-only
```

Train the model:

```powershell
.\.venv\Scripts\python.exe MobileNetV2\train_ecodetect_mobilenetv2.py
```

The EcoDetect run saves the model, TensorFlow Lite export, training curves,
classification report, confusion matrices, predictions CSV, and misclassified
example gallery under `artifacts/ecodetect/mobilenetv2`.

The MobileNetV2 EcoDetect pipeline supports extra imbalance controls under
`training.class_weight_multipliers`. The current config applies an additional
`1.5x` multiplier to `aluminum` after inverse-frequency class weighting, raising
the training weight for aluminum from `1.9444` to `2.9167`. The script also
supports optional focal loss through `training.focal_loss_gamma`; it is disabled
by default because the saved focal-loss experiment reduced test performance.

## EcoDetect MobileNetV3

`MobileNetV3/train_ecodetect_mobilenetv3.py` uses the same EcoDetect
classification pipeline with a MobileNetV3Small backbone.

Run a smoke check:

```powershell
.\.venv\Scripts\python.exe MobileNetV3\train_ecodetect_mobilenetv3.py --check-only
```

Train and evaluate MobileNetV3:

```powershell
.\.venv\Scripts\python.exe MobileNetV3\train_ecodetect_mobilenetv3.py
```

Outputs are saved under `artifacts/ecodetect/mobilenetv3`.

## EcoDetect MobileNetV4

`MobileNetV4/train_ecodetect_mobilenetv4.py` trains a MobileNetV4 classifier
through PyTorch and `timm`, using the same largest-bounding-box image label
conversion as the MobileNetV2 and MobileNetV3 pipelines.

Run a smoke check:

```powershell
.\.venv\Scripts\python.exe MobileNetV4\train_ecodetect_mobilenetv4.py --check-only
```

Train and evaluate MobileNetV4:

```powershell
.\.venv\Scripts\python.exe MobileNetV4\train_ecodetect_mobilenetv4.py
```

Outputs are saved under `artifacts/ecodetect/mobilenetv4`.

## EcoDetect YOLOv11

`YOLOv11/train_ecodetect_yolov11.py` trains object detection directly on the
EcoDetect YOLO labels. It prepares a local `data.yaml` with correct paths, then
uses Ultralytics YOLOv11.

Run a smoke check:

```powershell
.\.venv\Scripts\python.exe YOLOv11\train_ecodetect_yolov11.py --check-only
```

Train and evaluate YOLOv11:

```powershell
.\.venv\Scripts\python.exe YOLOv11\train_ecodetect_yolov11.py
```

Outputs are saved under `artifacts/ecodetect/yolov11/train`, and test
evaluation outputs are saved under `artifacts/ecodetect/yolov11/train_test`.
Ultralytics writes plots such as `results.png`, `confusion_matrix.png`, and
`confusion_matrix_normalized.png` in those run folders.

## EcoDetect Model Comparison

The EcoDetect experiments compare two image-level classifiers and one object
detector on the same dataset split. MobileNetV2 and MobileNetV3 convert each
YOLO image into a single class label using the largest bounding box, so their
main metric is classification accuracy. YOLOv11 keeps the original bounding-box
labels, so its main metrics are detection precision, recall, and mAP.

| Model | Task | Test images | Main result | Notes |
|---|---|---:|---:|---|
| MobileNetV2 | Image classification | 75 | 69.33% accuracy | Best classifier in this run; weighted F1-score was 0.69. |
| MobileNetV3Small | Image classification | 75 | 40.00% accuracy | Underperformed MobileNetV2; weighted F1-score was 0.42. |
| MobileNetV4 Conv Small | Image classification | 75 | 48.00% accuracy | Better aluminum recall than MobileNetV2, but much lower overall accuracy; weighted F1-score was 0.50. |
| YOLOv11 | Object detection | 75 | 46.74% mAP50 | Mean precision was 37.63%, mean recall was 59.40%, and mAP50-95 was 33.47%. |

### MobileNetV2 Aluminum Imbalance Check

MobileNetV2 remained the strongest EcoDetect classifier, so it was retrained
with stronger aluminum weighting. The safer class-weight multiplier run kept
overall accuracy and weighted F1 essentially unchanged, but did not improve
aluminum precision or recall. A focal-loss follow-up run was also saved for
comparison, but it hurt aluminum precision and overall accuracy.

| Run | Aluminum handling | Aluminum precision | Aluminum recall | Accuracy | Weighted F1 |
|---|---|---:|---:|---:|---:|
| Baseline MobileNetV2 | inverse-frequency class weights | 0.50 | 0.44 | 0.6933 | 0.6913 |
| Aluminum multiplier | aluminum class weight `1.5x` after balancing | 0.50 | 0.44 | 0.6933 | 0.6915 |
| Aluminum multiplier + focal loss | aluminum class weight `1.5x`, focal gamma `1.5` | 0.33 | 0.44 | 0.6667 | 0.6732 |

Artifacts for the safer retrain are under
[`artifacts/ecodetect/mobilenetv2_aluminum_balanced`](artifacts/ecodetect/mobilenetv2_aluminum_balanced),
and the focal-loss experiment is under
[`artifacts/ecodetect/mobilenetv2_aluminum_focal`](artifacts/ecodetect/mobilenetv2_aluminum_focal).

### Training and Evaluation Artifacts

| Model | Training curves | Confusion matrix | Extra evaluation outputs |
|---|---|---|---|
| MobileNetV2 | [`training_curves.png`](artifacts/ecodetect/mobilenetv2/training_curves.png) | [`confusion_matrix.png`](artifacts/ecodetect/mobilenetv2/confusion_matrix.png), [`confusion_matrix_normalized.png`](artifacts/ecodetect/mobilenetv2/confusion_matrix_normalized.png) | [`misclassified_examples.png`](artifacts/ecodetect/mobilenetv2/misclassified_examples.png), [`classification_report.txt`](artifacts/ecodetect/mobilenetv2/classification_report.txt), [`predictions.csv`](artifacts/ecodetect/mobilenetv2/predictions.csv) |
| MobileNetV3Small | [`training_curves.png`](artifacts/ecodetect/mobilenetv3/training_curves.png) | [`confusion_matrix.png`](artifacts/ecodetect/mobilenetv3/confusion_matrix.png), [`confusion_matrix_normalized.png`](artifacts/ecodetect/mobilenetv3/confusion_matrix_normalized.png) | [`misclassified_examples.png`](artifacts/ecodetect/mobilenetv3/misclassified_examples.png), [`classification_report.txt`](artifacts/ecodetect/mobilenetv3/classification_report.txt), [`predictions.csv`](artifacts/ecodetect/mobilenetv3/predictions.csv) |
| MobileNetV4 Conv Small | [`training_curves.png`](artifacts/ecodetect/mobilenetv4/training_curves.png) | [`confusion_matrix.png`](artifacts/ecodetect/mobilenetv4/confusion_matrix.png), [`confusion_matrix_normalized.png`](artifacts/ecodetect/mobilenetv4/confusion_matrix_normalized.png) | [`misclassified_examples.png`](artifacts/ecodetect/mobilenetv4/misclassified_examples.png), [`classification_report.txt`](artifacts/ecodetect/mobilenetv4/classification_report.txt), [`predictions.csv`](artifacts/ecodetect/mobilenetv4/predictions.csv) |
| YOLOv11 | [`results.png`](artifacts/ecodetect/yolov11/runs/detect/artifacts/ecodetect/yolov11/train/results.png), [`results.csv`](artifacts/ecodetect/yolov11/runs/detect/artifacts/ecodetect/yolov11/train/results.csv) | [`confusion_matrix.png`](artifacts/ecodetect/yolov11/runs/detect/artifacts/ecodetect/yolov11/train_test/confusion_matrix.png), [`confusion_matrix_normalized.png`](artifacts/ecodetect/yolov11/runs/detect/artifacts/ecodetect/yolov11/train_test/confusion_matrix_normalized.png) | [`BoxPR_curve.png`](artifacts/ecodetect/yolov11/runs/detect/artifacts/ecodetect/yolov11/train_test/BoxPR_curve.png), [`BoxF1_curve.png`](artifacts/ecodetect/yolov11/runs/detect/artifacts/ecodetect/yolov11/train_test/BoxF1_curve.png), [`val_batch0_pred.jpg`](artifacts/ecodetect/yolov11/runs/detect/artifacts/ecodetect/yolov11/train_test/val_batch0_pred.jpg) |

#### MobileNetV2 Training Curves

![MobileNetV2 EcoDetect training curves](artifacts/ecodetect/mobilenetv2/training_curves.png)

#### MobileNetV3Small Training Curves

![MobileNetV3Small EcoDetect training curves](artifacts/ecodetect/mobilenetv3/training_curves.png)

#### MobileNetV4 Conv Small Training Curves

![MobileNetV4 Conv Small EcoDetect training curves](artifacts/ecodetect/mobilenetv4/training_curves.png)

#### YOLOv11 Training Curves

![YOLOv11 EcoDetect training results](artifacts/ecodetect/yolov11/runs/detect/artifacts/ecodetect/yolov11/train/results.png)

#### YOLOv11 Precision-Recall Curve

![YOLOv11 EcoDetect precision-recall curve](artifacts/ecodetect/yolov11/runs/detect/artifacts/ecodetect/yolov11/train_test/BoxPR_curve.png)

MobileNetV2 performed best for simple image-level waste classification on this
EcoDetect run. It was strongest on `plastic` and `paper`, but still struggled
with `aluminum`, where only 4 of 9 test images were classified correctly.
MobileNetV3Small did not learn the split as well, especially for `paper` and
`plastic`, and is not the recommended classifier based on the saved results.
MobileNetV4 Conv Small improved aluminum recall to 5 of 9 test images, but did
so by over-predicting aluminum and dropping overall accuracy to 48.00%, so it is
also not recommended over MobileNetV2 for image-level classification.

YOLOv11 solves a harder problem because it must localize waste objects as well
as classify them. Its recall was higher than its precision, meaning it found a
reasonable number of objects but produced more false positives. Use YOLOv11
when bounding boxes or object localization are required; use MobileNetV2 when
the goal is only to assign one waste class to each image. For production use,
the test set is small, so the next step should be to evaluate on a larger
held-out set and add more examples for the weaker classes.
