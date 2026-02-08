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

## Project Structure (Relevant Parts)

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

## Step 2 – Train Unsupervised Anomaly Model

### Important: Required CLI Argument

If you run:

```bat
python src\models\train_unsupervised_model.py
```

You will get:

```text
error: the following arguments are required: --input
```

This is **expected behavior**.

The training script **requires** an explicit input file path to avoid accidental training on the wrong data.

---

### Correct Way to Run Training (Recommended)

Run the model **as a Python module** from the project root:

```bat
python -m src.models.train_anomaly_model ^
  --input data\processed\UNSW_Flow_features.parquet
```

(Use `/` instead of `\` in Git Bash.)

---

### Expected Output

```text
[0/6] Run directory ........ artifacts\unsupervised_model\20260208_105203
[1/6] Loaded data .......... (2059415, 39)
[2/6] Training features .... 37
[3/6] Training model ......
[4/6] Saving artifacts ...
[5/6] Done
[NOTE] AUC vs binary_label .... 0.9620 (sanity check only)
```
---

## What Happened in Training

### Data Loading

* ~2.06 million network flows loaded
* 39 columns total:

  * 37 numeric features
  * 2 label columns (`attack_label`, `binary_label`)

### Feature Safety

* Label columns were **explicitly removed from training features**
* Training is **strictly unsupervised**
* Non-numeric columns are rejected by design

### Model Training

* An Isolation Forest model was trained on behavioral features only
* No attack labels were used

### Risk Scoring

* Each flow received an `anomaly_score`
* Scores were converted into:

  * LOW
  * MEDIUM
  * HIGH risk tiers (quantile-based)

### Sanity Check (Post-hoc Only)

* `binary_label` was used **after training only**
* AUC ≈ **0.96** confirms that anomalous scores correlate strongly with known attacks
* This does **not** influence training or thresholds

---

## Generated Artifacts

Each run creates a timestamped directory:

```
artifacts/unsupervised_model/<run_id>/
├── pipeline.joblib        # Preprocessing + model
├── feature_names.json     # Features used for training
├── train_report.json      # Metrics and thresholds
└── train_scored.parquet   # Input data + anomaly_score + risk_level
```
---

## Key Design Principles

* Unsupervised by design (no label leakage)
* Clear separation:

  * `build_features.py` → data preparation
  * `train_anomaly_model.py` → modeling
* SOC-oriented output (risk levels, not raw predictions)
* Production-style CLI and artifacts

---

