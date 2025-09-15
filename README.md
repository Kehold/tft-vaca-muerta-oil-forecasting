# ⚡ Temporal Fusion Transformer (TFT) for Oil Production Forecasting
Robust monthly production forecasts (with uncertainty) for wells in Argentina’s Vaca Muerta formation.

**Author:** Alexis Ortega  
**Project Status:** In development — Streamlit application for explainable production forecasting using the Temporal Fusion Transformer.

---

## 🎯 Goal
Deliver reliable probabilistic oil-rate forecasts (P10/P50/P90) at the well level so operators can make better technical, operational, and financial decisions.

## 🔎 What’s in this repo?
- Modeling: Temporal Fusion Transformer (TFT) with quantile regression (P10/P50/P90)

- Evaluation: rolling-window cross-validation that mimics “forecasting in the wild”

- Search: randomized search and Bayesian optimization (Optuna) with per-trial logging

- Explainability: local & global feature importance, temporal attention views

- Baselines: simple time-series models to benchmark TFT (now integrated in the app)

- Streamlit App: a viewer that explores data, CV metrics, baselines, predictions, and explanations

---

## 🧱 Data & Features (kept simple)

- Frequency: monthly

- Target: oil rate (e.g., *_bpd)

- Static covariates: operator/area/campaign/completion details, etc.

- Temporal covariates: well age and calendar encodings

- No DCA signals or “relative features” are used in the final model (they didn’t improve performance).

Folder paths and column names are configured in `src/vm_tft/cfg.py` (e.g., `TIME_COL`, `GROUP_COL`, `TARGET`, `FREQ`).

---

## 🧩 Project Structure

project-root/
├── app/                          # Streamlit app
│   ├── vm_app_.py                # Main app entry point
│   ├── pages/                    # Individual Streamlit pages
│   └── images/                   # Static images for UI
├── src/vm_tft/                   # Core TFT pipeline
│   ├── cli/                      # CLI tools (fit, search, test_predict, explain)
│   ├── features/                 # Covariate builders
│   ├── io_utils.py               # I/O helpers
│   ├── evaluate.py               # Metrics and evaluation
│   ├── explain.py                # Explainability exports
│   └── cfg.py                    # Configurations
├── artifacts/                    # Saved models, predictions, metrics
├── poetry.lock                   # Poetry lock file
├── pyproject.toml                # Poetry dependencies
└── README.md                     # Project documentation

---

## 🏗️ Environment

```bash
# create env (example)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -U pip

# install runtime deps
pip install -r requirements.txt
```

---

## 🚀 Quick Start

**1) Prepare data**

```bash
# your own script/CLI; adjust to your pipeline
poetry run vm-prepare
```

**2) Fit the model + cross-validation**

```bash
# your own script/CLI; adjust to your pipeline
poetry run vm-fit-cv --epochs 50 --batch-size 128 --hidden-size 64 --hidden-continuous-size 24 --lstm-layers 2 --heads 4 --dropout 0.05 --lr 0.00082 --run-tag baseline_rel_var --only-if-better --opt-metric rmse
```

**3) Bayesian optimization (Optuna) + per-trial saves**

```bash
# your own script/CLI; adjust to your pipeline
poetry run vm-search-bo --n-trials 10 --opt-metric mql_0_50 --samples 100 --stride 6 --sampler tpe --pruner none --run-tag bo_baseline_rel_var --progress
```

**4) Cross-validated test predictions & explainability**

```bash
# your own script/CLI; adjust to your pipeline
poetry run vm-test --case B --warm-months 24 --samples 200 --save-plots --run-tag search_model_best_bo_baseline_case_B --plot-all 

poetry run vm-test --case C --warm-months 24 --samples 200 --save-plots --run-tag search_model_best_bo_baseline_case_C --plot-all --export-csv --total-months 60

poetry run vm-explain --eval-set test --local-wells all --save-by well --attention-types all,heatmap 
```

**5) Baselines**
Run simple reference models with the same CV policy used for TFT:

```bash
# your own script/CLI; adjust to your pipeline
poetry run vm-baselines --h 12 --n-windows 6 --stride 12                                                                    
```

Baselines included:
- NaiveDrift
  
- ExponentialSmoothing (ETS, trend-only) – robust Holt–Winters without seasonality

- Theta – additive/no seasonality (auto-disabled if too short)

---

## 📊 Metrics & Validation

- Rolling historical_forecasts with retraining at each window (Darts)

- Key metrics: RMSE, MAE, sMAPE, MQL@0.50 (median quantile loss), MIC@10–90 (coverage)

- Selection metric: default is MQL@0.50 (lower is better).
If you care most about calibration, you can target MIC by minimizing |MIC − 0.80|.

---

## 🗂️ Artifacts & “Fixed pointers”

CLI commands write run-specific outputs under artifacts/ and also maintain stable pointers (files that “point” to the current best). Typical layout:

```bash
artifacts/
  metrics/
    metrics_overall.json              # TFT CV overall
    metrics_per_series_window.csv     # TFT CV granular
    search_trials.csv                 # randomized search results
    search_trials_bo.csv              # BO trials (per-trial saves)
    baselines_per_series.csv          # NEW
    baselines_overall.json            # NEW
  models/
    tft_best_from_search.pt
    tft_best_from_search_bo.pt
  predictions/
    plots/
      caseB_series_###.png
      caseC_series_###.png
      pred_well_<id>.csv
  explain/
    local/local_variable_importance.csv
    attention/attention_well_<id>.png
    global/global_variable_importance_pseudo.csv

```

## 🖥️ Streamlit App

Run locally:

```bash
streamlit run app/vm_app_.py                                                                 
```

**Pages**
**Home** – project summary, goal, methodology, tooling

**Data Explorer** – per-well time series with optional overlays (rolling mean/median, Holt–Winters line), ACF/PACF, STL

**Model CV** – overall metrics + distributions across wells/windows

**Randomized Search** – trial scatter/bubbles; best config table

**Baselines** – TFT vs NaiveDrift/ETS/Theta:

**Test Predictions** – Case B warm-start plots + Case C cumulative (P10/P50/P90) and KPIs

**Explainability** – local/global variable importance and temporal attention

**Cloud viewer**: the app auto-hides any compute that requires Darts/Torch and uses the pre-computed artifacts listed above.

---

## 🔍 Reproducibility

- Seeds are set across NumPy/PyTorch/Optuna, but exact bit-wise reproducibility is not guaranteed across different GPUs/CPUs, library versions, or multi-threading. Expect numerically close results, not identical arrays.

- Save trials as you go (both randomized and BO scripts do this), so you never lose progress.

---

## 🧰 CLI Reference (common flags)

`--samples` (MC samples for quantile eval)

`--stride` (CV step in months)

`--n-windows` and `--h` (window count and horizon)

`--opt-metric {mql_0_50, rmse, mae, smape, mic_10_90}`

BO: `--sampler {tpe, skopt}`, `--timeout`, `--n-trials`, `--pruner {none, median}`

---

## 🤝 Why simple baselines?

They keep us honest. The app shows where TFT clearly helps (lower errors, tighter calibrated bands) and where simple models already do fine. If a baseline wins on some wells, we investigate why.

---

## 🤝 Contributing

Contributions are welcome!
Open issues or PRs to improve the pipeline, explainability, or UI.

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙌 Acknowledgements

- Built with PyTorch Lightning, Darts, Optuna, Plotly, and Streamlit.

- Special thanks to WBS Coding School.

---

## 👩‍💻 Author

Alexis Ortega
Senior Petroleum Engineer & Data Scientist

---