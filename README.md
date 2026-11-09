# Respiratory Pattern Classification from mmWave Radar Signals

A 1D convolutional neural network that classifies breathing patterns from millimetre-wave (mmWave) radar signals into four classes: **bradypnea**, **eupnea**, **kussmaul** and **biot**.

The work sits in **integrated sensing and communication (ISAC)**: the same wireless signals used for communication are reused to sense people's breathing without contact. That makes respiratory monitoring possible without wearables.

## Problem

- **Input:** a preprocessed 600-sample radar signal per observation (1D).
- **Output:** one of 4 breathing patterns.
- **Data:** 200 samples, 50 per class. With this few samples, overfitting is the central challenge.

## Approach

Architecture: `Conv1D → Conv1D → MaxPool → [Flatten | GlobalAveragePooling] → Dense → (Dropout) → Softmax`

I trained and compared five configurations, varying filter counts, kernel sizes, dense width, learning rate, epochs and the classification head:

| Config | Conv filters / kernels | Head | Dense | Regularization | Train acc | Test acc |
|---|---|---|---|---|---|---|
| `baseline` | 32/7 → 64/5 | Flatten | 128 | none | 92.1% | 70.0% |
| `small` | 16/5 → 32/3 | Flatten | 64 | none | 89.3% | 71.7% |
| `small_kernels` | 32/3 → 64/3, lr 5e-4 | Flatten | 64 | none | 89.3% | 63.3% |
| `short_training` | 32/5 → 64/3, 10 epochs | Flatten | 128 | none | 91.4% | 68.3% |
| **`gap_dropout`** | **32/5 → 64/5** | **GlobalAvgPool** | **64** | **Dropout 0.3 + early stopping** | ~89–90% | **93.3–98.3%** |

*Single stratified 70/30 split (60 test samples). The final model's range covers two training runs.*

### Key findings

- **The Flatten head caused the overfitting.** Flattening 300 × 64 feature maps feeds about 19,200 inputs into the dense layer, 1.2–2.5M dense-layer parameters for 140 training samples. Every Flatten model hit 100% training accuracy while validation loss climbed.
- **Global average pooling fixed it.** Pooling each feature map to a single value cuts the head to about 4K parameters. Together with dropout and early stopping, test accuracy rose from about 70% to over 93%.
- **Moderate kernels (5) beat small ones (3).** A breathing cycle spans many time steps, and wider kernels capture more of each waveform's shape. Small kernels with a lower learning rate gave the worst result (63%).

### Evaluation note

With only 60 test samples, one sample is worth 1.7 percentage points, so single-split accuracy varies noticeably between runs. `train.py` therefore defaults to **repeated stratified 5-fold cross-validation** and reports mean ± std accuracy and macro-F1. The scaler is fitted on each training fold only, so no test information leaks into preprocessing.

## Usage

```bash
pip install -r requirements.txt

# Final model, repeated stratified 5-fold CV (3 repeats = 15 runs)
python train.py --data data/breath_detection_dataset.xlsx

# Compare all five configurations
python train.py --data data/breath_detection_dataset.xlsx --config all

# Single 70/30 split, as in the table above
python train.py --data data/breath_detection_dataset.xlsx --mode split
```

Results go to `results/summary.csv`, and the full metrics plus summed confusion matrices go to `results/results.json`.

**Data format:** a CSV or Excel file with one row per sample. The signal values come first, and the last column is the label.

**Dataset:** not included. The data belongs to the research group that collected it.

## Project structure

```
mmwave_cnn/
  model.py      CNNConfig dataclass, the 5 configurations, build_model()
  data.py       dataset loading, leak-free scaling and reshaping
train.py        CLI: cross-validation or single-split evaluation
tests/          smoke test on synthetic sinusoidal "breathing" signals
```

```bash
pytest -q   # runs the full training path on synthetic data; no dataset needed
```

## Possible extensions

- Data augmentation (time shift, amplitude scaling, added noise) to enlarge the 200-sample set
- Frequency-domain features (FFT or spectrogram) fused with the raw signal
- Small temporal models (TCN, 1D ResNet) and comparison against classical baselines (SVM on FFT features)
- Testing robustness across subjects, distances and environments

## Tech

Python · TensorFlow/Keras · scikit-learn · pandas · NumPy
