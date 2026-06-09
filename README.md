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
.\.venv\Scripts\python.exe train_mobilenetv2.py
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
.\.venv\Scripts\python.exe evaluate_mobilenetv2.py
```

Evaluation outputs are written to `artifacts/evaluation/`:

- Classification report in text and JSON formats
- Raw and normalized confusion-matrix images
- Predictions for every test image in CSV format
- A gallery of the most confident misclassifications
