# MTR-Net

PyTorch implementation of **MTR-Net: Rainfall Detection Using Commercial Microwave Links Based on Multi-level Temporal Representation**.

MTR-Net is designed for high-temporal-resolution rainfall detection using bidirectional Commercial Microwave Link (CML) observations.

The model combines Conv1D, dual-branch LSTM and GRU for multi-level temporal representation. During training, forecasting, reconstruction and auxiliary wet/dry classification are jointly optimized. During testing, forecasting and reconstruction errors are combined to construct an anomaly score, and the Epsilon threshold is used for the final wet/dry decision.

---

## Dataset

The experiments are based on the OpenMRG dataset:

**OpenMRG: Open data from Microwave links, Radar, and Gauges for rainfall quantification in Gothenburg, Sweden**

Dataset:

```text
https://doi.org/10.5281/zenodo.7107689
```

OpenMRG contains CML, NORDRAD weather-radar and rain-gauge observations.

This work uses:

```text
CML observations
NORDRAD weather-radar observations
```

The original OpenMRG dataset is not included in this repository.

---

## Data Preprocessing

The preprocessing procedure follows and adapts the high-temporal-resolution CML wet/dry processing method described in:

**Technical Note: A simple feedforward artificial neural network for high temporal resolution classification of wet and dry periods using signal attenuation from commercial microwave links**

The main preprocessing steps are:

```text
1. Calculate total signal loss:

   TL = TSL - RSL

2. Aggregate the original 10-s CML observations to 1-min resolution
   by averaging valid observations within each minute.

3. Convert NORDRAD observations to rainfall intensity.

4. Calculate path-averaged radar rainfall intensity for each CML.

5. Linearly interpolate the 5-min radar rainfall series to 1 min.

6. Synchronize the 1-min CML and radar observations.

7. Generate wet/dry labels:

   rainfall > 0 mm/h -> Wet (1)
   rainfall = 0 mm/h -> Dry (0)

8. Keep samples for which both propagation directions are valid.

9. Apply 12-h rolling-median detrending to each sublink.

10. Split training and test data at the physical-link level.
```

After preprocessing, generate the following files:

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

Place the four files in:

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

## Model

The main MTR-Net architecture is:

```text
Bidirectional CML input
        |
        v
      Conv1D
        |
        +--------------------------+
        |                          |
        v                          v
Temporal-oriented LSTM    Feature-oriented LSTM
        |                          |
        +------------+-------------+
                     |
                     v
                    GRU
                     |
        +------------+------------+
        |            |            |
        v            v            v
   Forecasting   Reconstruction  Auxiliary
      head           head       classifier
        |            |
        +------+-----+
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

The auxiliary classifier is used only during training.

The final wet/dry decision is based on forecasting and reconstruction errors.

---

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run

run  `result.py`:

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
dataset                         OpenMRG
lookback                        100
Conv1D kernel size              7
Temporal LSTM hidden dimension  2
Feature LSTM hidden dimension   32
GRU hidden dimension            300
batch size                      256
learning rate                   1e-4
gamma                           1.0
```

---

## Output

The results are saved in:

```text
output/OpenMRG/<MODEL_ID>/
```

Main output files:

```text
model.pt
summary.txt
config.txt
train_output.pkl
test_output.pkl
```

`summary.txt` contains:

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

## References

### OpenMRG dataset

**OpenMRG: Open data from Microwave links, Radar, and Gauges for rainfall quantification in Gothenburg, Sweden**

```text
https://doi.org/10.5281/zenodo.7107689
```

### Data preprocessing

**Technical Note: A simple feedforward artificial neural network for high temporal resolution classification of wet and dry periods using signal attenuation from commercial microwave links**

### Proposed model

**Rainfall Detection Using Commercial Microwave Links Based on Multi-level Temporal Representation**
