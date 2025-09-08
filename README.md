# ⚡ Temporal Fusion Transformer (TFT) for Oil Production Forecasting

**Author:** Alexis Ortega  
**Project Status:** In development — Streamlit application for explainable production forecasting using the Temporal Fusion Transformer.

---

## 🚀 Project Overview

This project implements a complete end-to-end oil production forecasting pipeline based on the Temporal Fusion Transformer (TFT) architecture.

It enables:

- Training TFT models on multi-well production data with covariates.
- Cross-validation and hyperparameter search (Randomized Search).
- Explainability analysis (global/local variable importance & attention maps).
- Out-of-sample test predictions with uncertainty quantification.
- Long-term forecasting (Case C: up to 5 years ahead).
- Professional results visualization in an interactive Streamlit app.

Built with:

- 🐍 Python 3.12.x (managed via **Pyenv**)
- 🔥 PyTorch + Darts for deep learning forecasting
- 🚀 Streamlit for user interface
- 📊 Plotly & Matplotlib for visualization
- 📦 Scikit-learn for preprocessing & evaluation

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

## ⚙️ Features  

**Training & Search** 
- Cross-validation with rolling windows.
- Randomized hyperparameter search.

**Explainability** 
- Local and global variable importance.
- Attention maps for temporal relevance.

**Inference Modes** 
- Case B: Predict until end of history.
- Case C: Extend predictions up to 5 years ahead.

**Visualization** 
- Darts-style PNG forecasts.
- Per-well plots with uncertainty bands.
- Cumulative production forecasts.

**Streamlit UI** 
- Data Explorer (time series, ACF/PACF, STL).
- Randomized Search results.
- Test Predictions (Case B & Case C).
- Explainability dashboards.

---

## 💻 Installation

### 1. Install Python (with Pyenv)

Ensure you have Python 3.10.x installed via Pyenv:

```bash
pyenv install 3.12.1
pyenv local 3.12.1
```

### 2. Install Poetry

If you haven't installed Poetry yet:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Check Poetry version:

```bash
poetry --version
```

### 3. Install project dependencies

Install dependencies (Poetry will automatically create the virtual environment):

```bash
poetry install
```

Activate the virtual environment:

```bash
poetry shell
```

---

## 🚀 Running the Application

Step 1 — Start Streamlit inside Poetry shell

```bash
streamlit run vm_app_.py
```

Step 2 — Navigate between pages

- **Home**: Overview, methodology, tooling.

- **Data Explorer**: Explore well data, correlations, decompositions.

- **Randomized Search**: Hyperparameter search results.

- **Test Predictions**: Case B (history) and Case C (5-year forecast).

- **Explainability**: Local/global VI, attention plots.

➡️ You can navigate pages using the top-left menu inside the Streamlit app.

---

## 🗂️ Input Data Format

CSV with minimum required columns:

`date` — time index

`well_id` — well identifier

`oil_rate_bpd` — monthly oil rate (target)

`cum_oil_bbl` — cumulative oil (for Case C evaluation)

static and covariate columns as configured in `cfg.py`

---

📈 Example Outputs

✅ Darts-style forecast plots with uncertainty bands

✅ Randomized search scatter plots of metrics vs hyperparameters

✅ Local variable importance per well

✅ Attention heatmaps

✅ 5-year cumulative production forecast with P10/P50/P90

---

## 📦 Deployment

Option 1: Streamlit Cloud

1. Push your project to a public GitHub repository.

2. Connect your repo to Streamlit Cloud.

3. Define your main entry point as:

```bash
app/vm_app_.py
```

4. Deploy 🚀

Note: Poetry-managed projects work on Streamlit Cloud — ensure your pyproject.toml is complete!

Option 2: Docker (Optional for full control)

(Dockerfile can be provided on request!)

---

## 🧭 Roadmap

- Expand hyperparameter search with Bayesian optimization.

- Add baseline ARIMA/Prophet model comparison.

- Support multi-scenario forecasting.

- Deploy via Docker for production-ready pipelines.

---

## 🤝 Contributing

Contributions are welcome!
Open issues or PRs to improve the pipeline, explainability, or UI.

---

## 📄 License

This project is licensed under the MIT License.

---

## 👩‍💻 Author

Alexis Ortega
Senior Petroleum Engineer & Data Scientist

---