# Execution Guide — Freddie Mac Credit Risk Scoring

## What This Project Does

Builds a credit score (300–900) for Freddie Mac mortgage borrowers that predicts the **probability of going 90+ days delinquent in the next 12 months**. Higher score = safer borrower.

---

## Step 0: Prerequisites

- Windows 10/11, Python 3.10+, Anaconda recommended
- At least **8 GB RAM** (servicing file is ~200MB, expands to ~1-2GB in memory)
- Your raw sample files extracted to one folder, e.g.:
  ```
  D:\Projects\Major Project\dataset\raw_extracted\
      sample_orig_1999.txt   (7.5 MB)
      sample_svcg_1999.txt   (202 MB)
      sample_orig_2000.txt
      sample_svcg_2000.txt
      ...
  ```

---

## Step 1: Create Conda Environment

```bash
conda create -n freddie_risk python=3.10 -y
conda activate freddie_risk
```

---

## Step 2: Install Dependencies

```bash
# navigate to project folder
cd "D:\Projects\Major Project\freddie_mac_credit_risk"

# install all requirements
pip install -r requirements.txt
```

If LightGBM install fails on Windows:
```bash
pip install lightgbm --prefer-binary
```

---

## Step 3: Configure Your Paths

Open `config/settings.py` and update the two path fields at the top:

```python
# Line ~15: where your raw txt files are
DEFAULT_RAW_DIR = r"D:\Projects\Major Project\dataset\raw_extracted"

# Line ~60 onwards: where outputs go (change username if needed)
@dataclass
class PathConfig:
    raw_dir:    str = r"D:\Projects\Major Project\dataset\raw_extracted"
    parquet_dir: str = r"D:\Projects\Major Project\freddie_mac_credit_risk\data\parquet"
    ...
```

**Important**: All output folders are created automatically — you only need to set the raw_dir.

---

## Step 4: Run the Full Pipeline

```bash
conda activate freddie_risk
cd "D:\Projects\Major Project\freddie_mac_credit_risk"
python pipeline.py
```

This runs all 6 stages in sequence. Expected runtime on sample files:
- Stage 1 (ingest):   5–15 min  (202MB servicing file is slow to parse)
- Stage 2 (targets):  5–10 min  (rolling window calculations)
- Stage 3 (features): 2–5 min
- Stage 4 (split):    < 1 min
- Stage 5 (train):    3–10 min  (4 models + ensemble)
- Stage 6 (evaluate): 2–5 min

**Total: ~20–45 minutes on first run.**

After the first run, stages 1–4 are cached in Parquet. Re-running just training is fast:
```bash
python pipeline.py --stages train evaluate
```

---

## Step 5: Run on a Specific Year Range

To use only 1999–2005 data (faster for testing):
```bash
python pipeline.py --start-year 1999 --end-year 2005
```

To run specific stages only:
```bash
python pipeline.py --stages ingest targets    # just data prep
python pipeline.py --stages train evaluate    # just modeling (if data ready)
```

---

## Step 6: View Results

After the pipeline finishes, outputs are in `data/reports/`:

```
data/reports/
    model_comparison.csv    -- AUC, Gini, KS for all models on OOS and OOT
    pipeline_summary.json   -- top-level summary with best model metrics
    roc_curves.png          -- ROC and PR curves for all models
    score_distribution.csv  -- default rate per score bucket (300-900 scale)
```

Open the comparison CSV to see which model won:
```python
import pandas as pd
df = pd.read_csv(r"data/reports/model_comparison.csv")
print(df.to_string())
```

---

## Step 7: Model Comparison Notebook

```bash
conda activate freddie_risk
cd "D:\Projects\Major Project\freddie_mac_credit_risk"
jupyter notebook model_comparison.ipynb
```

**Run all cells in order.** The notebook requires the pipeline to have completed at least through the `train` stage.

The notebook produces:
1. Bar charts comparing AUC and KS across all models
2. ROC and Precision-Recall curves side by side
3. KS decile table (standard bank validation format)
4. Feature importance charts for tree models
5. Credit score histogram (300-900) split by default vs non-default
6. Default rate per score bucket
7. OOS vs OOT performance drop (stability check)
8. PSI chart (population stability index)

---

## Understanding the Splits

| Split | Train Data    | Test Data     | Purpose                                    |
|-------|---------------|---------------|--------------------------------------------|
| OOS   | 70% random    | 30% random    | Checks for overfitting                     |
| OOT   | 1999–2007     | 2008–2012     | Checks temporal stability (GFC stress test) |

OOT is more meaningful for credit risk — if the model was only trained on pre-crisis loans and tested on crisis loans, and it still works, that's a strong model.

---

## Troubleshooting

**"No sample_orig_YYYY.txt files found"**
→ Check that `DEFAULT_RAW_DIR` in `config/settings.py` points to the right folder.
→ Check filenames: must be exactly `sample_orig_1999.txt` not `Sample_Orig_1999.TXT`.

**"Join produced 0 rows"**
→ The LOAN_SEQUENCE_NUMBER didn't match between origination and servicing.
→ This usually means the files are from different years or the raw files are corrupted.

**"Out of memory" on servicing load**
→ Reduce the year range: `python pipeline.py --start-year 1999 --end-year 2005`
→ Or add more RAM.

**LightGBM or XGBoost import error**
→ Run `pip install lightgbm xgboost --prefer-binary` in the conda env.

**"TARGET_12M: all zeros" or very low default rate**
→ Normal for good vintages. The sample data from 1999–2007 has ~2–8% default rate.
→ Run with more years that include 2008–2010 to get more defaults in training.

---

## Score Interpretation

| Score  | Risk Level       | Typical Action         |
|--------|------------------|------------------------|
| 760–900 | Low Risk         | Approve easily          |
| 720–760 | Low-Medium Risk  | Standard approval       |
| 660–720 | Medium Risk      | Normal underwriting     |
| 620–660 | Medium-High Risk | Enhanced scrutiny       |
| 580–620 | High Risk        | Conditional approval    |
| 300–580 | Very High Risk   | Decline or high rate    |

---

## Project Structure

```
freddie_mac_credit_risk/
├── config/
│   └── settings.py          # all constants, paths, model params
├── ingestion/
│   └── loader.py            # reads raw txt files, cleans data
├── features/
│   ├── targets.py           # builds TARGET_12M (no future leakage)
│   ├── engineering.py       # feature engineering (5 categories)
│   └── splitting.py         # OOS and OOT splits
├── models/
│   ├── trainer.py           # XGBoost, LightGBM, RF, LR, Ensemble
│   └── scorer.py            # probability → 300-900 credit score
├── validation/
│   └── evaluator.py         # AUC, KS, Gini, PSI, comparison table
├── utils/
│   └── io_utils.py          # load/save parquet and pickle
├── pipeline.py              # master orchestrator (6 stages)
├── model_comparison.ipynb   # comparison notebook
├── requirements.txt
└── EXECUTION_GUIDE.md
```