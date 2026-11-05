# Hybrid Optimization-Based Intrusion Detection System - Demonstration Dashboard

Interactive Streamlit dashboard for the research project
**"Algorithmic Optimization for Intrusion Detection: Multi-Objective Feature Selection Using Hybrid Global-Local Optimization"**.

It explains and demonstrates the methodology (Hybrid Firefly-PSO feature selection, Cuckoo Search, Random Forest,
adaptive class-wise thresholds) and presents the measured results from `../main-h-mooa.ipynb`.

> This is an **offline, batch analysis** tool for recorded datasets. It is not a real-time intrusion detection system.

## Result labels

Every result in the dashboard carries one of these labels:

| Label | Meaning |
|-------|---------|
| **Actual result** | Measured in the research notebook on the held-out test set (109,771 samples). Copied from saved notebook outputs in `ids_dashboard/results.py`; nothing is estimated. |
| **Preliminary experiment** | Measured notebook outputs of small-scale H-MOOA smoke tests (Firefly-PSO, Cuckoo Search, adaptive thresholds). Not final model performance. |
| **Live computation** | Computed now, in the dashboard, on the dataset you loaded. |
| **Illustrative demo** | Reduced live versions of the optimizers (few particles/nests and iterations) that show how the algorithms work. Not the complete original implementations. |
| **Synthetic data** | Generated demo data for when the real dataset is not available. Results on it are not research findings. |

## Modules

| Page | Contents |
|------|----------|
| Guided Demonstration | The 9-step presentation flow with buttons that jump to each module |
| Project Overview | Title, objective, dataset (CIC DataSenseIoT 2025: 71 features, 8 classes), algorithms, workflow |
| Dataset Analysis | CSV upload / local path / synthetic data, preview, dimensions, missing values, target-column detection, class distribution, feature statistics, correlation heatmap |
| Feature Selection | Firefly-PSO workflow and equations, notebook outputs (RF_9/RF_6 subsets, Pareto archive, five-seed stability), live reduced Firefly-PSO run |
| Cuckoo Search | Nests, Levy flights, nest updates, notebook outputs, Random Forest hyperparameter explorer, live reduced Cuckoo Search |
| Random Forest Classification | Leakage-safe split, training/testing, multiclass and benign-vs-attack results, confusion matrices, classification report, offline batch prediction on another CSV |
| Adaptive Thresholds | Class probabilities, class-specific thresholds, final decision, default vs adaptive on separate calibration/assessment data |
| Performance Evaluation | Accuracy, Macro Precision/Recall/F1, Macro FPR, feature reduction, training time, prediction latency, significance tests |
| Research Findings | Baseline comparison (RF_71 / RF_9 / RF_6), preliminary optimization experiments, limitations |

## Installation

Requires Python 3.10+.

```bash
cd dashboard
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

After installation the app needs no internet connection.

## Running

```bash
cd dashboard
streamlit run app.py
```

Streamlit opens the dashboard at http://localhost:8501.

## Using your dataset

1. Open **Dataset Analysis** and upload a CSV (max 4 GB, set in `.streamlit/config.toml`) or enter a local file path.
   Large files are read in chunks; use *Max rows* / *Sample fraction* to keep the laptop responsive.
2. Pick the **target column**. The dashboard suggests candidates (low-cardinality columns, benign-like values, names such as
   `label`) but never assumes a fixed name.
3. Review the **excluded columns**. Labels, identifiers, timestamps, IPs and MAC addresses are excluded by default to prevent
   leakage. Only numeric columns are used as features.

For a quick presentation with the full CIC-IIoT-2025 file (~3 GB), create a smaller stratified sample first:

```bash
python tools/make_sample.py combined_dataset.csv sample.csv --label <your label column> --per-class 5000
```

If you have no dataset available, click **Generate synthetic demo data**. All results on it are labelled as synthetic.

## Leakage prevention

- Group-aware splitting (default) keeps identical feature vectors in one partition, as in the notebook.
- Train / validation / test splits: thresholds are calibrated on one half of the validation split and assessed on the other;
  the test split is used only for the final classification metrics.
- Missing values are median-imputed inside the scikit-learn pipeline, fitted on training data only.

## Project layout

```
dashboard/
├── app.py                    # Streamlit entry point and sidebar navigation
├── requirements.txt
├── .streamlit/config.toml    # dark-blue theme, upload limit, offline settings
├── ids_dashboard/
│   ├── results.py            # measured notebook results (with notebook step references)
│   ├── data_utils.py         # CSV loading, target detection, leakage filtering, synthetic data
│   ├── modeling.py           # splitting, Random Forest pipeline, timing
│   ├── metrics.py            # Macro FPR and classification metrics
│   ├── pareto.py             # dominance, non-dominated sorting, crowding distance, archive
│   ├── firefly_pso.py        # reduced Binary Firefly-PSO demo
│   ├── cuckoo.py             # reduced Cuckoo Search demo with Levy flights
│   ├── thresholds.py         # adaptive class-wise threshold rule and calibration
│   ├── ui.py, state.py       # styling, cards, badges, session state
│   └── views/                # one module per dashboard page
├── tools/make_sample.py      # stratified sampling of large CSVs
└── tests/                    # pytest suite
```

## Tests

```bash
cd dashboard
pip install pytest
python -m pytest
```
