# H-MOOA: Multi-Objective Feature Selection for Intrusion Detection

**Algorithmic Optimization for Intrusion Detection: Multi-Objective Feature Selection Using Hybrid Global-Local Optimization**

This repository contains the full research notebook (`main-h-mooa.ipynb`) for building, optimizing and auditing a compact intrusion detection system (IDS) on the **CIC-IIoT-2025** dataset. The core idea is a Hybrid Multi-Objective Optimization Algorithm (H-MOOA) that searches for small feature subsets and model settings that keep detection performance high while reducing false positives and computational cost.

## Highlights

- **Dataset:** CIC-IIoT-2025 (`combined_dataset.csv`, ~3 GB), multiclass traffic labels: `benign`, `bruteforce`, `ddos`, `dos`, `malware`, `mitm`, `recon`, `web`.
- **Leakage-safe preprocessing:** label/identifier/timestamp columns are dropped, and group-aware splits keep identical feature vectors from crossing train/test boundaries.
- **Hybrid global-local optimization:**
  - Binary Firefly-PSO hybrid for feature selection with a non-dominated Pareto archive
  - Cuckoo Search (Levy flights) for Random Forest hyperparameters
  - Importance-guided iterative local search (71 -> 11 -> ... -> 6 features)
  - Adaptive class-specific decision-threshold optimization
- **Pareto utilities:** dominance checks, fast non-dominated sorting, crowding distance, bounded archive.
- **Additional model:** LLM-TinyTransformer-IDS (LLM-inspired tabular tokenization + tiny Transformer, TensorFlow) with XAI (SHAP, LIME and more) and offline LLM-style analysis reports. It does not call any external LLM API.
- **Rigorous evaluation:** five-seed robustness, feature-selection stability, ablation studies, permutation importance, bootstrap confidence intervals and McNemar tests with Holm correction.

## Key results (held-out test set, 109,771 samples)

| Model | Features | Macro F1 |
|-------|----------|----------|
| RF_71 (all features) | 71 | 0.9708 |
| RF_9 (best compact model) | 9 | 0.9579 |
| RF_6 | 6 | lower than RF_9 (see notebook) |

All pairwise differences (RF_71 / RF_9 / RF_6) are statistically significant: 6/6 bootstrap CIs exclude zero, and 3/3 McNemar tests are significant at 0.05.

## Notebook structure

| Phase | Content |
|-------|---------|
| 1 | Dataset inspection (schema, labels, missing values) |
| 2 | Data cleaning: duplicates, conflicting labels, quality analysis |
| 3 | Baseline models: Decision Tree, Random Forest (group-aware split) |
| 4 | Exploratory data analysis: class distribution, imbalance, correlations |
| 5-6 | Multi-objective feature selection, iterative reduction, final test evaluation, audits |
| 6.61-6.67 | H-MOOA experiments: fresh 70/15/15 split, Pareto utilities, Firefly-PSO, Cuckoo Search, adaptive thresholds |
| 7-8 | Confusion matrices, error analysis, computational cost |
| 9 | Robustness and validation (GA, multiple seeds) |
| 10 | Finalization: saved models, configs, figures and a reproducibility ZIP package |

## Requirements

Developed on Kaggle (Python 3.12, NVIDIA Tesla T4 GPU). Main libraries:

```
numpy pandas scikit-learn scipy matplotlib seaborn joblib
tensorflow shap lime
```

```bash
pip install numpy pandas scikit-learn scipy matplotlib seaborn joblib tensorflow shap lime
```

## How to run

1. Open `main-h-mooa.ipynb` on Kaggle (recommended) or in Jupyter.
2. Add the CIC-IIoT-2025 dataset. The notebook expects:
   ```
   /kaggle/input/datasets/muhammadirfangull/cic-iiot-2025/combined_dataset.csv
   ```
   If you run it elsewhere, update `DATA_PATH` / `DATASET_PATH`.
3. Run the cells in order. Outputs are written to `/kaggle/working/` (for example `final_research_artifacts/`, `hmooa_experiments_v2/`, and `final_research_artifacts.zip`).

## Outputs

- Trained models (`.joblib`, `.keras`) and selected-feature lists (JSON)
- Metric tables (`final_model_comparison.csv`, `final_classwise_metrics.csv`, significance tests, etc.)
- Research figures (class distribution, correlation matrix, feature importance, Pareto fronts, confusion matrices, adaptive thresholds, framework diagram)
