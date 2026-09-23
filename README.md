# MTR-Net

PyTorch implementation of **MTR-Net: Rainfall Detection Using Commercial Microwave Links Based on Multi-level Temporal Representation**.

MTR-Net is designed for high-temporal-resolution rainfall detection using paired bidirectional Commercial Microwave Link (CML) observations.

The model constructs multi-level temporal representations by jointly modeling:

- local short-term variations;
- overall temporal evolution;
- direction-specific dynamic responses of the two propagation directions.

During training, next-step forecasting, input-sequence reconstruction, and weighted auxiliary wet/dry classification are jointly optimized. During testing, only forecasting and reconstruction errors are used to construct the anomaly score, and a nonparametric Epsilon threshold is applied to obtain the final wet/dry decision.

---

## Overview

The overall architecture of MTR-Net is shown below.

<img width="1991" height="665" alt="image" src="https://github.com/user-attachments/assets/16d85fa9-e936-4e42-8461-8788cf6dfb4c" />



The model contains three main components:

1. **Multi-level temporal representation**
   - Conv1D for local temporal variations;
   - Time-oriented LSTM for overall temporal evolution;
   - Feature-oriented LSTM for direction-specific dynamic responses;
   - GRU for feature fusion and temporal compression.

2. **Supervision-guided multi-task learning**
   - next-step forecasting;
   - input-sequence reconstruction;
   - weighted auxiliary wet/dry classification.

3. **Anomaly-score-based rainfall detection**
   - forecasting error;
   - reconstruction error;
   - Epsilon threshold.

---

## Dataset

The experiments are based on the **OpenMRG** dataset:

**OpenMRG: Open data from Microwave links, Radar, and Gauges for rainfall quantification in Gothenburg, Sweden**

Dataset DOI:

```text
https://doi.org/10.5281/zenodo.7107689
```

OpenMRG contains:

```text
Commercial Microwave Link (CML) observations
NORDRAD weather-radar observations
Rain-gauge observations
```

This work uses:

```text
CML observations
NORDRAD weather-radar observations
```

The original OpenMRG dataset is **not included** in this repository. Please download it from the official OpenMRG dataset repository.

---

## Data Preprocessing

The preprocessing procedure follows and adapts the high-temporal-resolution CML wet/dry processing strategy described in:

**Technical Note: A simple feedforward artificial neural network for high-temporal-resolution rain event detection using signal attenuation from commercial microwave links**

The main preprocessing steps are as follows.

### 1. Calculate total signal loss

For each sublink:

```text
TL = TSL - RSL
```

where:

- `TSL` is the transmitted signal level;
- `RSL` is the received signal level;
- `TL` is the total signal loss.

### 2. Aggregate CML observations to 1-min resolution

The original CML observations have a temporal resolution of 10 s.

Valid observations within the same minute are averaged to obtain a 1-min CML time series.

### 3. Convert radar observations to rainfall intensity

NORDRAD observations are converted to rainfall intensity according to the radar-data processing procedure.

### 4. Calculate path-averaged radar rainfall intensity

Because a CML represents an integral observation along the microwave propagation path, radar rainfall intensities are spatially matched to each physical microwave link.

The rainfall intensity of the radar grids crossed by the microwave path is combined according to the path length within each grid to obtain the path-averaged rainfall intensity.

### 5. Interpolate radar rainfall to 1-min resolution

The original NORDRAD temporal resolution is 5 min.

The path-averaged radar rainfall series is linearly interpolated to 1-min resolution.

### 6. Synchronize CML and radar observations

The 1-min CML observations and 1-min radar rainfall series are aligned according to timestamps.

### 7. Generate wet/dry labels

Wet and dry labels are generated from the path-averaged radar rainfall intensity:

```text
rainfall > 0 mm/h  -> Wet (1)
rainfall = 0 mm/h  -> Dry (0)
```

### 8. Keep paired bidirectional observations

Only samples for which both propagation directions of the same physical microwave link contain valid observations are retained.

The two propagation directions form the two input features of MTR-Net.

### 9. Detrend CML observations

A 12-h rolling median is applied independently to each sublink to reduce long-term equipment drift and slowly varying propagation backgrounds.

### 10. Split data at the physical-link level

Training and test data are divided according to physical microwave links instead of randomly splitting sliding windows.

This prevents highly correlated observations from the same physical link from simultaneously appearing in both the training and test sets.

---

## Processed Data

After preprocessing, the following four files are generated:

```text
OpenMRG_train.pkl
OpenMRG_train_label.pkl
OpenMRG_test.pkl
OpenMRG_test_label.pkl
```

The input feature dimension is:

```text
2
```

corresponding to the two propagation directions of the same physical microwave link.

Place the processed files in:

```text
datasets/
└── data/
    └── processed/
        ├── OpenMRG_train.pkl
        ├── OpenMRG_train_label.pkl
        ├── OpenMRG_test.pkl
        └── OpenMRG_test_label.pkl
```

---

## Model Architecture

The main MTR-Net processing pipeline is:

```text
        Paired CML observations
                  |
                  v
                Conv1D
                  |
          +-------+-------+
          |               |
          v               v
 Time-oriented LSTM   Feature-oriented LSTM
          |               |
          +-------+-------+
                  |
                  v
                 GRU
                  |
        +---------+---------+
        |         |         |
        v         v         v
   Forecasting Reconstruction Auxiliary
      head        head      classifier
        |          |
        +-----+----+
              |
              v
        Anomaly Score
              |
              v
       Epsilon Threshold
              |
              v
          Wet / Dry
```

### Conv1D

Conv1D extracts local short-term variations from the paired CML observations.

### Time-oriented LSTM

The Time-oriented LSTM processes the complete two-dimensional CML sequence along the temporal dimension and models the overall temporal evolution of the paired sublinks.

### Feature-oriented LSTM

The Feature-oriented LSTM separately models the temporal dynamics of the two propagation directions using shared LSTM parameters.

This branch is designed to preserve direction-specific dynamic responses while maintaining a consistent feature-extraction mechanism across the two sublinks.

### GRU

The local representation, overall temporal representation, and direction-specific representation are concatenated and further compressed using a GRU.

The resulting shared latent representation is used by the three learning tasks.

---

## Multi-task Learning

MTR-Net jointly optimizes three tasks during training:

```text
Forecasting
Reconstruction
Auxiliary wet/dry classification
```

### Forecasting task

The forecasting branch predicts the next CML observation from the historical input window.

The forecasting loss is optimized using **Mean Squared Error (MSE)**.

### Reconstruction task

The reconstruction branch reconstructs the input sequence and preserves structural information from the CML observations.

The reconstruction loss is also optimized using **Mean Squared Error (MSE)**.

### Auxiliary wet/dry classification

Wet/dry labels are introduced only during training to provide rainfall-state supervision.

Because dry samples are much more frequent than wet samples in the training data, a positive-class weight is used:

```text
w_pos = min(N_dry / N_wet, 20)
```

The auxiliary classifier is optimized using weighted binary cross-entropy.

The auxiliary classification branch is used **only during training**.

Its output is **not directly used for the final rainfall decision during testing**.

---

## Training Objective

The total training loss consists of forecasting, reconstruction, and auxiliary classification losses:

```text
L_total = L_pred + lambda_rec * L_rec + lambda_cls * L_cls
```

where:

```text
lambda_rec = 1.0
lambda_cls = 0.5
```

The auxiliary classification task guides the shared representation toward rainfall-related features while preserving the forecasting and reconstruction capabilities of the model.

---

## Anomaly Score

During testing, only forecasting and reconstruction errors are used to construct the anomaly score.

For each propagation direction:

```text
score_k =
    |forecast_k - observation_k|
    +
    gamma * |reconstruction_k - observation_k|
```

The final global anomaly score is obtained by averaging the scores of the two propagation directions:

```text
S_t = mean(score_1, score_2)
```

where:

```text
gamma = 1.0
```

The auxiliary wet/dry classification probability is not used in this anomaly score.

---

## Epsilon Threshold

The final wet/dry decision is determined using a nonparametric Epsilon threshold based on the anomaly-score distribution of the training data.

Candidate thresholds are generated as:

```text
epsilon_z = mean(S_train) + z * std(S_train)
```

with:

```text
z = 2.5, 3.0, 3.5, ..., 11.5
```

and:

```text
step = 0.5
```

For different candidate thresholds, the changes in the mean and standard deviation of the anomaly-score distribution after removing high anomaly scores are evaluated.

The candidate producing the largest relative distribution change is selected as the final threshold.

The threshold is determined from **training anomaly scores only**.

Reference rainfall labels from the test set are not used for threshold selection.

---

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Run

Set the trained model ID in `result.py`:

```python
MODEL_ID = "25062026_213707"
```

Then run:

```bash
python result.py
```

---

## Main Parameters

```text
dataset                           OpenMRG
input feature dimension           2
lookback window                    100

Conv1D kernel size                 7

Time-oriented LSTM hidden dim      2
Feature-oriented LSTM hidden dim   32
GRU hidden dimension               300

batch size                         256
learning rate                      1e-4

reconstruction loss weight         1.0
auxiliary classification weight    0.5

gamma                              1.0

Epsilon z minimum                  2.5
Epsilon z maximum                  11.5
Epsilon step                       0.5
```

---

## Repository Structure

A typical project structure is:

```text
MTR-Net/
│
├── datasets/
│   └── data/
│       └── processed/
│           ├── OpenMRG_train.pkl
│           ├── OpenMRG_train_label.pkl
│           ├── OpenMRG_test.pkl
│           └── OpenMRG_test_label.pkl
│
├── figures/
│   ├── fig1_mtr_net_architecture.png
│   ├── fig2_rainfall_detection_rate.png
│   └── fig3_representative_rainfall_cases.png
│
├── output/
│   └── OpenMRG/
│
├── args.py
├── modules.py
├── mtad_gat.py
├── prediction.py
├── preprocessing.py
├── result.py
├── train.py
├── training.py
├── utils.py
├── requirements.txt
└── README.md
```

---

## Output

The experimental results are saved in:

```text
output/OpenMRG/<MODEL_ID>/
```

Main output files include:

```text
model.pt
summary.txt
config.txt
train_output.pkl
test_output.pkl
```

### `model.pt`

Trained MTR-Net parameters.

### `config.txt`

Configuration used for the experiment.

### `train_output.pkl`

Training-set anomaly scores and related outputs.

### `test_output.pkl`

Test-set forecasting, reconstruction, anomaly scores, thresholds, and final predictions.

### `summary.txt`

The main evaluation results are summarized in `summary.txt`.

Example:

```json
{
  "epsilon_result": {
    "f1": 0.7389519700814727,
    "precision": 0.8764611355108214,
    "recall": 0.6387465902154498,
    "TP": 800124.0,
    "TN": 7906886.0,
    "FP": 112779.0,
    "FN": 452523.0,
    "threshold": 0.020547835854813457
  }
}
```

The main evaluation metrics are:

```text
Precision
Recall
F1-score
```

---

## Results

The full MTR-Net model achieves approximately:

```text
Precision = 0.876
Recall    = 0.639
F1-score  = 0.739
```

---

### Rainfall Detection Under Different Rainfall Intensities

The rainfall detection rate increases as the path-averaged rainfall intensity increases.

When the path-averaged rainfall intensity exceeds **2 mm/h**, the rainfall detection rate remains above approximately **93%**.

<img width="985" height="886" alt="图2" src="https://github.com/user-attachments/assets/698ce80e-9861-4408-aac2-2ed3b2137788" />


The detection rates for different rainfall-intensity intervals are:

```text
Rainfall intensity      Detection rate

0–0.1 mm/h              36.0%
0.1–0.5 mm/h            60.0%
0.5–1 mm/h              79.5%
1–2 mm/h                89.4%
2–5 mm/h                93.1%
5–10 mm/h               95.2%
10–20 mm/h              94.7%
>20 mm/h                93.1%
```

These results indicate that stronger rainfall generally causes more pronounced deviations in the CML observations and is therefore more reliably detected.

Weak rainfall remains a more difficult detection scenario.

---

### Representative Rainfall Cases

Representative strong- and weak-rainfall events are shown below.

<img width="985" height="886" alt="图3" src="https://github.com/user-attachments/assets/b95569b3-f015-4cd3-854d-7e529bcaea8d" />

The examples show that strong rainfall events usually produce more pronounced and consistent signal-loss variations in both propagation directions.

For weak rainfall events, rainfall-induced variations may be closer to normal CML background fluctuations or other non-rainfall propagation disturbances, which makes wet/dry discrimination more difficult.

---

## Main Experimental Results

| Method | Precision | Recall | F1-score |
|---|---:|---:|---:|
| σ80 | 0.255 | 0.723 | 0.377 |
| MLP | 0.270 | 0.602 | 0.372 |
| LSTM | 0.586 | 0.505 | 0.542 |
| Conv1D-AE | 0.370 | 0.607 | 0.459 |
| MTAD-GAT | 0.869 | 0.483 | 0.620 |
| **MTR-Net** | **0.876** | **0.639** | **0.739** |

---

## Ablation Study

The main ablation-study results are:

| Model | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Baseline | 0.951 | 0.389 | 0.552 |
| Model-A | 0.881 | 0.596 | 0.711 |
| Model-B | 0.864 | 0.564 | 0.683 |
| Model-C | 0.868 | 0.584 | 0.698 |
| **Full MTR-Net** | **0.876** | **0.639** | **0.739** |

The ablation results indicate that:

- direction-specific temporal modeling improves the detection of real rainfall events;
- auxiliary wet/dry supervision improves the relationship between the shared representation and rainfall state;
- weighted auxiliary classification further improves sensitivity to minority wet samples;
- combining direction-specific representation learning with weighted auxiliary supervision provides complementary benefits.

---

## Figures

The figures used in this README are stored in:

```text
figures/
```

Required files:

```text
figures/fig1_mtr_net_architecture.png
figures/fig2_rainfall_detection_rate.png
figures/fig3_representative_rainfall_cases.png
```

---

## References

### OpenMRG Dataset

J. C. M. Andersson, J. Olsson, R. C. Z. Van de Beek, et al.

**OpenMRG: Open data from Microwave links, Radar, and Gauges for rainfall quantification in Gothenburg, Sweden.**

*Earth System Science Data*, 2022, 14(12): 5411–5426.

```text
https://doi.org/10.5194/essd-14-5411-2022
```

Dataset:

```text
https://doi.org/10.5281/zenodo.7107689
```

---

### High-Temporal-Resolution CML Wet/Dry Processing

E. Øydvin, M. Graf, C. Chwala, et al.

**Technical Note: A simple feedforward artificial neural network for high-temporal-resolution rain event detection using signal attenuation from commercial microwave links.**

*Hydrology and Earth System Sciences*, 2024, 28(23): 5163–5171.

```text
https://doi.org/10.5194/hess-28-5163-2024
```

---

### Prediction and Reconstruction-Based Anomaly Detection

H. Zhao, Y. Wang, J. Duan, et al.

**Multivariate time-series anomaly detection via graph attention network.**

*2020 IEEE International Conference on Data Mining (ICDM)*, 2020, pp. 841–850.

```text
https://doi.org/10.1109/ICDM50108.2020.00093
```

---

### Nonparametric Dynamic Threshold

K. Hundman, V. Constantinou, C. Laporte, et al.

**Detecting spacecraft anomalies using LSTMs and nonparametric dynamic thresholding.**

*Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, 2018, pp. 387–395.

```text
https://doi.org/10.1145/3219819.3219845
```

---

## Proposed Model

**MTR-Net: Rainfall Detection Using Commercial Microwave Links Based on Multi-level Temporal Representation**

GitHub repository:

```text
https://github.com/kongbaizhanglmxlmx/MTR-Net
```

---


