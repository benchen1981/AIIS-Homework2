# Software Design Document (SDD) - AIIS-WH2

## Overview
Provide interfaces for data loading, feature engineering, model training, evaluation and report generation.

## Inputs
- `data/corona.csv` — primary CSV from Kaggle.

## Outputs
- Model metrics (MSE, RMSE, R2)
- Plots (Plotly)
- PDF report (reports/AIIS_WH2_report.pdf)
- NotebookLM summary

## APIs (module functions)
- `src.data_utils.load_data(path)` -> DataFrame
- `src.modeling.run_models(df, target, test_size)` -> dict(metrics, fitted, selected_features)
- `src.reporting.create_report(output_path, summary_text)` -> pdf path
