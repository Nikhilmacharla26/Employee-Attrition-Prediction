# Employee Turnover Risk Prediction and Early Intervention System

A stacked ensemble model (Random Forest + Decision Tree + KNN + Logistic
Regression → Logistic Regression meta-model) deployed as a Streamlit app,
organized as a proper Python project rather than a single script.

## Project structure

```
turnover_project/
├── config.py                 # paths, thresholds, shared constants
├── requirements.txt
├── Dockerfile
├── .streamlit/
│   └── config.toml           # native Streamlit theme
├── data/
│   └── HR-Employee-Attrition-New.csv   # <- put your dataset here
├── models/
│   └── model_bundle.pkl      # <- created by training, loaded by the app
├── src/                      # data science / ML logic (no UI code)
│   ├── __init__.py
│   ├── preprocessing.py      # IQRCapper transformer + pipeline/grid builders
│   ├── train.py              # training pipeline (CLI + importable function)
│   └── predict.py            # inference helpers: load bundle, predict, risk logic
├── app/                      # Streamlit UI (no ML logic)
│   ├── __init__.py
│   ├── theme.py               # CSS + hero header
│   ├── components.py          # metric strip, gauge, dashboard charts
│   └── main.py                 # page layout / entrypoint
└── tests/
    └── test_pipeline.py       # end-to-end smoke test (train → save → load → predict)
```

**Why this layout:** `src/` (training/inference) has no Streamlit imports and
can be reused in a notebook, a batch job, or a different UI. `app/` has no
ML logic — it only calls into `src/`. `config.py` is the single source of
truth for paths and thresholds, so both sides agree without duplication.

## 1. Install

```bash
pip install -r requirements.txt
```

## 2. Add your data

Put `HR-Employee-Attrition-New.csv` in `data/`.

## 3. Train

```bash
python -m src.train --data data/HR-Employee-Attrition-New.csv
```

This runs the full `GridSearchCV` (as in the original notebook) and saves
`models/model_bundle.pkl`. Add `--fast` for a quick smoke-test run with a
smaller grid. Re-run this command whenever your data changes.

## 4. Run the app

```bash
streamlit run app/main.py
```

Open the printed local URL (usually `http://localhost:8501`).

## 5. Run tests

```bash
pytest tests/ -q
```

## Deployment options

### Streamlit Community Cloud (fastest, free)
Push this folder to a GitHub repo (include `models/model_bundle.pkl` — use
Git LFS if it's large), go to https://share.streamlit.io → "New app", set
the main file to `app/main.py`, and deploy.

### Hugging Face Spaces
Create a new Space (SDK: Streamlit), upload/push this folder. It builds and
serves on port 8501 automatically.

### Docker (Render, Fly.io, Cloud Run, ECS, etc.)
```bash
docker build -t turnover-app .
docker run -p 8501:8501 turnover-app
```
Train before building the image (or mount `models/` as a volume/secret at
runtime), then point your platform's Docker deploy option at this repo or a
pushed image.

### Any VM
```bash
pip install -r requirements.txt
python -m src.train --data data/HR-Employee-Attrition-New.csv
nohup streamlit run app/main.py --server.port 8501 --server.address 0.0.0.0 &
```
Put nginx/Caddy in front for HTTPS and a custom domain.

## Notes

- The web form collects the most influential fields (Age, Monthly Income,
  OverTime, satisfaction/involvement/balance scores, Job Level, Job Role,
  tenure fields). Any column not on the form is auto-filled with its
  median (numeric) or most common value (categorical) from training data —
  stored in the bundle by `src/train.py`, used by `src/predict.py`.
- The dashboard tab reads metrics computed once at training time (stored in
  the bundle), so it loads instantly without recomputing anything.
