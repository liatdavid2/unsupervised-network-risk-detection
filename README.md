# Unsupervised Network Risk Detection

This project implements an **unsupervised anomaly detection pipeline** for network flow data, designed to identify **risky and anomalous behavior** without relying on attack labels during training.

The pipeline is split into two clear stages:

1. **Feature preparation (label-agnostic)**
2. **Unsupervised model training and risk scoring**

## Setup

Clone the repository and enter the project directory:
```bash
git clone https://github.com/liatdavid2/unsupervised-network-risk-detection.git
cd unsupervised-network-risk-detection
````

Create and activate a virtual environment:

Windows (Git Bash):

```bash
py -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```
---

## Project Structure

```
unsupervised-network-risk-detection/
│
├── data/
│   ├── raw/
│   │   └── UNSW_Flow.parquet
│   └── processed/
│       └── UNSW_Flow_features.parquet
│
├── src/
│   ├── build_features.py
│   └── models/
│       └── train_anomaly_model.py
│
├── artifacts/
│   └── unsupervised_model/
│       └── <timestamped runs>
│
└── README.md
```

---

## Step 1 – Build Features (Label-Agnostic)

This step prepares **numeric, model-ready features**.
No labels are used for feature selection.

Run:

```bat
python src\build_features.py
```

Expected output:

```text
[1/5] Load data ............... OK (2,059,415 rows)
[2/5] Numeric features ........ 42
[3/5] Missing values handled
[4/5] Feature selection ....... 42 → 42 → 37
[5/5] Save features ........... data\processed\UNSW_Flow_features.parquet
```

What this means:

* Raw UNSW network flow data was loaded
* Only numeric behavioral features were kept
* Missing values were handled
* Variance + correlation filtering reduced features to 37
* Output file was saved to `data/processed/`

---

מעולה — הנה קטע **README מחובר, קצר, נקי ומקצועי**, בלי *Output Explained*, בדיוק לפי הבקשה שלך.
אפשר להדביק אותו כמו־שהוא.

---

## Step 2 – Train Unsupervised Anomaly Model

This step trains an **unsupervised anomaly detection model** on pre-engineered network flow features.

---

### Correct Way to Run Training (Recommended)

Run the model **as a Python module** from the project root:

```bat
python -m src.models.train_anomaly_model ^
  --input data\processed\UNSW_Flow_features.parquet ^
  --outdir artifacts\unsupervised_model ^
  --run_tag with_time_port_freq ^
  --contamination 0.01 ^
  --low_q 0.95 ^
  --high_q 0.99
```
---
### Parameter Explanation

* `--input data\processed\UNSW_Flow_features.parquet`
  Path to the processed feature file used for training.
  This dataset already includes all feature engineering steps (time-based, port-based, and frequency features).
  Labels may exist in the file but are **explicitly excluded** from training.

* `--outdir artifacts\unsupervised_model`
  Root directory where all artifacts from this training run are stored.
  Each run creates a dedicated subdirectory for reproducibility and experiment tracking.

* `--run_tag with_time_port_freq`
  Human-readable identifier for the run.
  Used to compare different feature sets, parameter choices, or modeling assumptions.

* `--contamination 0.01`
  Expected proportion of anomalies in the dataset (1%).
  This is a modeling assumption used internally by the unsupervised algorithm, not ground truth.

* `--low_q 0.95`
  Lower quantile threshold for anomaly scores.
  Flows below this threshold are considered **low risk**.

* `--high_q 0.99`
  Upper quantile threshold for anomaly scores.
  Flows above this threshold are considered **high risk**; values in between are treated as **medium risk**.
---

### Expected Output

```text
[0/6] Run directory ........ artifacts\unsupervised_model\with_time_port_freq
[1/6] Loaded data .......... (2059415, 39)
[2/6] Training features .... 37
[3/6] Training model ......
[4/6] Saving artifacts ...
[5/6] Done
[NOTE] AUC vs binary_label .... 0.9620 (sanity check only)
```

---

## What Happened in Training

### Data & Feature Safety

* ~2.06M network flows were loaded
* 39 columns total:

  * 37 numeric features
  * 2 label columns (`attack_label`, `binary_label`)
* Label columns were **explicitly excluded** from training
* Training is **strictly unsupervised**
* Non-numeric columns are rejected by design


### Risk Scoring

* Each flow receives an `anomaly_score`
* Scores are mapped to risk tiers:

  * **LOW**
  * **MEDIUM**
  * **HIGH**
    using quantile-based thresholds (`low_q`, `high_q`)
---

## Generated Artifacts

Each run creates a dedicated directory:

```text
artifacts/unsupervised_model/<run_id>/
├── pipeline.joblib        # Preprocessing + anomaly model
├── feature_names.json     # Features used for training
├── train_report.json      # Metrics, thresholds, run config
└── train_scored.parquet   # Input data + anomaly_score + risk_level
```


