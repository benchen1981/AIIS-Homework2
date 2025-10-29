# AIIS-WH2 - COVID-19 Dataset
Repository: benchen1981/AIIS-WH2-Final

**Purpose**
This project demonstrates a CRISP-DM based workflow to analyze and model the COVID-19 dataset (Kaggle: imdevskp/corona-virus-report).
We train multiple regression models to predict target variables (e.g., `Confirmed`, `Deaths`, `Recovered`) with feature selection, model evaluation, and prediction plots with confidence/prediction intervals.

**Dataset**
Download the dataset from:
https://www.kaggle.com/datasets/imdevskp/corona-virus-report
Place the main CSV file as `data/corona.csv`.

**Contents**
- `app.py` — Streamlit app to explore data and run models (one-click on Replit).
- `notebooks/analysis.ipynb` — Jupyter notebook with CRISP-DM steps, feature engineering, models, plots (Plotly with CI).
- `src/` — Python modules (`data_utils.py`, `modeling.py`, `reporting.py`).
- `specs/` — Spec & SDD files, GitHub Actions workflow.
- `reports/` — Generated PDF and PPT templates; NotebookLM summary.
- `requirements.txt` — Python dependencies.
- `replit.nix`, `.replit` — Replit one-click run files.
- `deploy.sh` — Auto-deploy helper script.

**How to run locally**
1. Create a virtual environment.
2. `pip install -r requirements.txt`
3. Put dataset at `data/corona.csv`
4. Run notebook or `streamlit run app.py`.

**Replit**
This repo is Replit-ready; place it in your Replit and press Run. The `.replit` file runs `streamlit run app.py`.

