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

![Most confident misclassified examples](artifacts/mobilenetv2_classifier_evaluation/misclassified_examples.png)

## Setup

Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and adjust the settings when needed.
`MobileNetV2/config.yaml` keeps shared MobileNetV2 defaults in `common`, with
classifier-specific and EcoDetect-specific overrides under `classifier` and
`ecodetect`. Values from `config.yaml` are used when present, environment
variables fill missing values, and command-line arguments can still override
both. Leaving `DATASET_DIR` empty lets `kagglehub` download or locate the
dataset automatically.
Shared MobileNet configuration helpers live in `MobileNetShared/config.py`.

## Training

```powershell
.\.venv\Scripts\python.exe MobileNetV2\train_mobilenetv2.py
```

Training produces the following files under
`artifacts/mobilenetv2_classifier_model/` by default:

- `waste_mobilenetv2.keras`: final Keras model
- `waste_mobilenetv2.tflite`: TensorFlow Lite export
- `best_model.keras`: best validation-loss checkpoint
- `labels.json`: model class order
- `history.json`: epoch-by-epoch training metrics
- `test_metrics.json`: final test loss and accuracy

## Evaluation

```powershell
.\.venv\Scripts\python.exe MobileNetV2\evaluate_mobilenetv2.py
```

Evaluation outputs are written to `artifacts/mobilenetv2_classifier_evaluation/`:

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

Disable early stopping when you want every requested epoch to run:

```powershell
.\.venv\Scripts\python.exe MobileNetV4\train_ecodetect_mobilenetv4.py --no-early-stopping
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

## TACO Mask R-CNN

The [TACO dataset](https://github.com/pedropro/TACO) provides COCO-style
instance-segmentation annotations for litter images. `TACO/train_taco_maskrcnn.py`
trains a Torchvision Mask R-CNN model from those polygon masks.
This setup follows the Mask R-CNN/TACO-10 direction from the
[TACO paper](https://arxiv.org/pdf/2003.06975).

First download TACO with the upstream repository instructions, then set
`dataset.dir` in `TACO/config.yaml` to the folder that contains
`annotations.json` and the `batch_*` image folders.

Run a smoke check:

```powershell
.\.venv\Scripts\python.exe TACO\train_taco_maskrcnn.py --check-only
```

Train Mask R-CNN:

```powershell
.\.venv\Scripts\python.exe TACO\train_taco_maskrcnn.py
```

Evaluate saved cross-validation checkpoints with the TACO paper's prediction
ranking scores without retraining:

```powershell
.\.venv\Scripts\python.exe TACO\train_taco_maskrcnn.py `
  --cross-validation `
  --paper-score-eval-only `
  --val-fraction 0.1 `
  --test-fraction 0.1 `
  --output-dir artifacts\taco\maskrcnn_taco10_cv_80_10_10_paper_metrics
```

The default `training.device: auto` uses CUDA automatically when a GPU is
available and falls back to CPU otherwise. In Colab, enable a GPU under
`Runtime > Change runtime type`, then run the script normally or force CUDA with
`--device cuda`.

By default the script uses the paper-style TACO-10 taxonomy: `Bottle`,
`Bottle cap`, `Can`, `Cigarette`, `Cup`, `Lid`, `Other Litter`,
`Plastic bag + wrapper`, `Pop tab`, and `Straw`. Change `dataset.taxonomy` to
`category-field` if you want to group categories by `dataset.category_field`
instead. Outputs are saved under the configured `output.dir`, which defaults to
`artifacts/taco/maskrcnn_taco10_cv_70_15_15_coco_metrics`.

Training augmentations are configured in `TACO/config.yaml`. The default setup
uses horizontal flips, small rotations, object-centered random crops, brightness
and contrast changes, saturation and hue jitter, Gaussian blur, and Gaussian
noise. Geometric augmentations are applied to both images and masks, then boxes
and areas are recomputed from the transformed masks.

After training, the script evaluates the test split with COCO-style metrics for
both masks and boxes. The main segmentation values are written to
`coco_metrics.json` and `test_metrics.json` as `segm.AP`, `segm.AP50`, and
`segm.AP75`. This requires `pycocotools`, which is included in
`requirements.txt`.

### TACO Mask R-CNN Cross-Validation Results

Two saved TACO-10 experiments use 4-fold cross validation with different
train/validation/test ratios. The 70/15/15 run is saved under
[`artifacts/taco/maskrcnn_taco10_cv_70_15_15_coco_metrics`](artifacts/taco/maskrcnn_taco10_cv_70_15_15_coco_metrics), and
the 80/10/10 run is saved under
[`artifacts/taco/maskrcnn_taco10_cv_80_10_10_coco_metrics`](artifacts/taco/maskrcnn_taco10_cv_80_10_10_coco_metrics).

The referenced TACO paper uses Mask R-CNN on TACO-10 with 4-fold cross
validation, an 80% training, 10% validation, and 10% test split inside each
fold, and mask Average Precision (AP) as the main metric. The regular
repository runs below use Torchvision's standard COCO-style prediction scores,
so they should be compared against each other rather than treated as direct
paper reproductions. A separate paper-style evaluation is included afterward.

#### 70/15/15 Split

| Fold | Epochs run | Mask AP | Mask AP50 | Mask AP75 | Bbox AP |
|---|---:|---:|---:|---:|---:|
| 1 | 18 | 19.79% | 26.09% | 21.93% | 17.31% |
| 2 | 14 | 22.58% | 33.08% | 25.84% | 22.03% |
| 3 | 13 | 28.64% | 36.75% | 31.08% | 27.51% |
| 4 | 10 | 15.27% | 21.00% | 16.97% | 13.86% |
| **Average** |  | **21.57%** | **29.23%** | **23.95%** | **20.18%** |

#### 80/10/10 Split

| Fold | Epochs run | Mask AP | Mask AP50 | Mask AP75 | Bbox AP |
|---|---:|---:|---:|---:|---:|
| 1 | 13 | 15.49% | 20.85% | 18.30% | 13.44% |
| 2 | 15 | 32.57% | 46.81% | 36.11% | 28.24% |
| 3 | 9 | 27.27% | 36.00% | 33.31% | 25.11% |
| 4 | 11 | 18.51% | 26.47% | 21.32% | 16.57% |
| **Average** |  | **23.46%** | **32.53%** | **27.26%** | **20.84%** |

#### Split Comparison

| Split | Train/val/test images across fold partitions | Mask AP | Mask AP50 | Mask AP75 | Bbox AP | Mask AR100 |
|---|---|---:|---:|---:|---:|---:|
| 70/15/15 | 1,054 / 223 / 223 | 21.57 +/- 5.59 | 29.23 +/- 7.05 | 23.95 +/- 5.97 | 20.18 +/- 5.92 | 43.19% |
| 80/10/10 | 1,196 / 152 / 152 | **23.46 +/- 7.86** | **32.53 +/- 11.39** | **27.26 +/- 8.77** | **20.84 +/- 6.98** | **46.14%** |

The 80/10/10 split has the better saved results. It improves the main mask AP
by 1.89 percentage points over the 70/15/15 split, with stronger AP50, AP75,
bbox AP, and mask recall. The tradeoff is that the 80/10/10 run evaluates on
fewer validation and test images per fold, and its fold-to-fold standard
deviation is higher, so the improvement should be treated as a promising but
not definitive margin.

#### Paper-Style Score Evaluation

The saved 80/10/10 checkpoints were also re-evaluated with the paper's three
prediction-ranking scores. Those artifacts are saved under
[`artifacts/taco/maskrcnn_taco10_cv_80_10_10_paper_metrics`](artifacts/taco/maskrcnn_taco10_cv_80_10_10_paper_metrics),
with the aggregate values in
[`paper_score_summary.json`](artifacts/taco/maskrcnn_taco10_cv_80_10_10_paper_metrics/paper_score_summary.json).

| Evaluation | Mask AP |
|---|---:|
| Paper, class score | 17.6 +/- 1.6 |
| This repository, class score | 15.70 +/- 4.13 |
| Paper, litter score | 18.4 +/- 1.5 |
| This repository, litter score | 16.00 +/- 3.69 |
| Paper, ratio score | **19.4 +/- 1.5** |
| This repository, ratio score | 16.89 +/- 4.40 |

With the paper-style scoring, the repository's best result is the ratio score
at 16.89 AP. That is lower than the paper's reported ratio-score result of
19.4 AP, so the paper-style comparison does not show an improvement over the
paper. The strongest conclusion from these artifacts is that 80/10/10 is the
better split under the repository's standard COCO-style evaluation, while ratio
score is the best of the three paper-style ranking methods for the saved
80/10/10 checkpoints.

## EcoDetect Model Comparison

The EcoDetect runs include image-level MobileNet classifiers and a YOLOv11
object detector. The MobileNet scripts convert each YOLO image into one class
label using the largest bounding box, so they are compared with classification
metrics. YOLOv11 keeps the original bounding boxes and is shown separately with
detection metrics.

### MobileNet Classifier Comparison

| Model | Test images | Accuracy | Weighted F1 | Notes |
|---|---:|---:|---:|---|
| MobileNetV2 | 75 | 69.33% | 0.6913 | Best saved image classifier. |
| MobileNetV3Small | 75 | 40.00% | 0.4203 | Weakest MobileNet run. |
| MobileNetV4 Conv Small | 75 | 48.00% | 0.5023 | Baseline V4 run; better than V3 but behind V2. |
| MobileNetV4 Conv Small, Colab best-F1 run | 75 | 53.33% | 0.5234 | Best saved V4 weighted F1, still below V2. |
| MobileNetV4 Conv Small, Colab extended-epochs run | 75 | 53.33% | 0.5179 | Similar accuracy to the best-F1 run, lower weighted F1. |

### YOLOv11 Detector Results

YOLOv11 solves a different task from the MobileNet classifiers: it must predict
both the class and the bounding-box location for each object.

| Metric | Value |
|---|---:|
| mAP50 | 46.74% |
| mAP50-95 | 33.47% |
| mAP75 | 40.01% |
| Mean precision | 37.63% |
| Mean recall | 59.40% |

These YOLO metrics evaluate both the predicted class and the predicted bounding
box location. `mAP50` is the more forgiving score: a prediction counts as
correct if its box overlaps the ground-truth box by at least 50% IoU
(`intersection over union`). The saved YOLOv11 run reached 46.74% mAP50, so it
finds some useful detections at this looser overlap threshold.

`mAP50-95` is stricter because it averages mAP across IoU thresholds from 50%
to 95%. The saved score is 33.47%, which means performance drops when the boxes
need to be placed more precisely. `mAP75` is an intermediate check at 75% IoU;
the saved run reached 40.01%.

Mean precision answers: "When YOLO predicts an object, how often is it right?"
The saved precision is 37.63%, so the detector produces many false positives.
Mean recall answers: "Out of all real objects, how many did YOLO find?" The
saved recall is 59.40%, so it finds more than half of the objects but still
misses a substantial number. In short, this YOLOv11 run is better at noticing
possible objects than being selective and confident.

### Class-Level Results

| Model | Aluminum F1 | Paper F1 | Plastic F1 | Accuracy |
|---|---:|---:|---:|---:|
| MobileNetV2 | 0.471 | 0.702 | 0.737 | 69.33% |
| MobileNetV3Small | 0.278 | 0.415 | 0.459 | 40.00% |
| MobileNetV4 Conv Small | 0.294 | 0.489 | 0.563 | 48.00% |
| MobileNetV4 Colab best F1 | 0.296 | 0.390 | 0.683 | 53.33% |
| MobileNetV4 Colab extended epochs | 0.250 | 0.381 | 0.690 | 53.33% |

MobileNetV2 is the best overall image classifier in the saved results. It has
the strongest balance across `paper` and `plastic`, but it still struggles with
`aluminum` because the test split has only 9 aluminum images. MobileNetV4's
newer Colab runs improve over the baseline V4 run, mostly by improving
`plastic`, but they still do not beat MobileNetV2. MobileNetV3Small is not
competitive on these artifacts.

YOLOv11 solves a harder task because it must find object locations as well as
classes. Its recall is higher than its precision, so it finds a reasonable
share of objects but produces more false positives. Use YOLOv11 when bounding
boxes are needed; use MobileNetV2 when the goal is one label per image.

### MobileNetV2 Aluminum Imbalance Check

MobileNetV2 was retrained with stronger aluminum weighting because it was the
strongest classifier overall. The safer class-weight multiplier run kept
overall accuracy and weighted F1 essentially unchanged, but did not improve
aluminum precision or recall. A focal-loss follow-up run hurt aluminum
precision and overall accuracy.

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

| Model | Training curves | Confusion matrix | Metrics and reports |
|---|---|---|---|
| MobileNetV2 | [`training_curves.png`](artifacts/ecodetect/mobilenetv2/training_curves.png) | [`confusion_matrix.png`](artifacts/ecodetect/mobilenetv2/confusion_matrix.png), [`confusion_matrix_normalized.png`](artifacts/ecodetect/mobilenetv2/confusion_matrix_normalized.png) | [`test_metrics.json`](artifacts/ecodetect/mobilenetv2/test_metrics.json), [`classification_report.txt`](artifacts/ecodetect/mobilenetv2/classification_report.txt), [`predictions.csv`](artifacts/ecodetect/mobilenetv2/predictions.csv) |
| MobileNetV3Small | [`training_curves.png`](artifacts/ecodetect/mobilenetv3/training_curves.png) | [`confusion_matrix.png`](artifacts/ecodetect/mobilenetv3/confusion_matrix.png), [`confusion_matrix_normalized.png`](artifacts/ecodetect/mobilenetv3/confusion_matrix_normalized.png) | [`test_metrics.json`](artifacts/ecodetect/mobilenetv3/test_metrics.json), [`classification_report.txt`](artifacts/ecodetect/mobilenetv3/classification_report.txt), [`predictions.csv`](artifacts/ecodetect/mobilenetv3/predictions.csv) |
| MobileNetV4 Conv Small | [`training_curves.png`](artifacts/ecodetect/mobilenetv4/training_curves.png) | [`confusion_matrix.png`](artifacts/ecodetect/mobilenetv4/confusion_matrix.png), [`confusion_matrix_normalized.png`](artifacts/ecodetect/mobilenetv4/confusion_matrix_normalized.png) | [`test_metrics.json`](artifacts/ecodetect/mobilenetv4/test_metrics.json), [`classification_report.txt`](artifacts/ecodetect/mobilenetv4/classification_report.txt), [`predictions.csv`](artifacts/ecodetect/mobilenetv4/predictions.csv) |
| MobileNetV4 Colab best F1 | [`training_curves.png`](artifacts/ecodetect/mobilenetv4_colab_best_f1/training_curves.png) | [`confusion_matrix.png`](artifacts/ecodetect/mobilenetv4_colab_best_f1/confusion_matrix.png), [`confusion_matrix_normalized.png`](artifacts/ecodetect/mobilenetv4_colab_best_f1/confusion_matrix_normalized.png) | [`test_metrics.json`](artifacts/ecodetect/mobilenetv4_colab_best_f1/test_metrics.json), [`classification_report.txt`](artifacts/ecodetect/mobilenetv4_colab_best_f1/classification_report.txt), [`predictions.csv`](artifacts/ecodetect/mobilenetv4_colab_best_f1/predictions.csv) |
| MobileNetV4 Colab extended epochs | [`training_curves.png`](artifacts/ecodetect/mobilenetv4_colab_extended_epochs/training_curves.png) | [`confusion_matrix.png`](artifacts/ecodetect/mobilenetv4_colab_extended_epochs/confusion_matrix.png), [`confusion_matrix_normalized.png`](artifacts/ecodetect/mobilenetv4_colab_extended_epochs/confusion_matrix_normalized.png) | [`test_metrics.json`](artifacts/ecodetect/mobilenetv4_colab_extended_epochs/test_metrics.json), [`classification_report.txt`](artifacts/ecodetect/mobilenetv4_colab_extended_epochs/classification_report.txt), [`predictions.csv`](artifacts/ecodetect/mobilenetv4_colab_extended_epochs/predictions.csv) |
| YOLOv11 | [`results.png`](artifacts/ecodetect/yolov11/train/results.png), [`results.csv`](artifacts/ecodetect/yolov11/train/results.csv) | [`confusion_matrix.png`](artifacts/ecodetect/yolov11/train_test/confusion_matrix.png), [`confusion_matrix_normalized.png`](artifacts/ecodetect/yolov11/train_test/confusion_matrix_normalized.png) | [`evaluation_summary.json`](artifacts/ecodetect/yolov11/train/evaluation_summary.json), [`BoxPR_curve.png`](artifacts/ecodetect/yolov11/train_test/BoxPR_curve.png), [`BoxF1_curve.png`](artifacts/ecodetect/yolov11/train_test/BoxF1_curve.png) |

#### MobileNetV2 Training Curves

![MobileNetV2 EcoDetect training curves](artifacts/ecodetect/mobilenetv2/training_curves.png)

#### MobileNetV3Small Training Curves

![MobileNetV3Small EcoDetect training curves](artifacts/ecodetect/mobilenetv3/training_curves.png)

#### MobileNetV4 Conv Small Training Curves

![MobileNetV4 Conv Small EcoDetect training curves](artifacts/ecodetect/mobilenetv4/training_curves.png)

#### Best Saved MobileNetV4 Training Curves

![MobileNetV4 Colab best-F1 training curves](artifacts/ecodetect/mobilenetv4_colab_best_f1/training_curves.png)

#### YOLOv11 Training Curves

![YOLOv11 EcoDetect training results](artifacts/ecodetect/yolov11/train/results.png)

#### YOLOv11 Precision-Recall Curve

![YOLOv11 EcoDetect precision-recall curve](artifacts/ecodetect/yolov11/train_test/BoxPR_curve.png)
