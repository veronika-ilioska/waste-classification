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
