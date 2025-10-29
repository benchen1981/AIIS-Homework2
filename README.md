# AIIS-WH2 - COVID-19 Dataset預測

## 👨‍💻 作者
**陳宥興 Ben Chen**
**學號：5114050015**
- GitHub: [@benchen1981](https://github.com/benchen1981/AIIS-WH2-Final)
- Email: benchen1981@gmail.com
- Supervising professor: HUAN CHEN

## 專案目標/簡介：
  以 Kaggle 的 COVID-19 dataset 為例，遵循 CRISP-DM 流程進行資料理解、特徵工程與模型建置。使用多種迴歸模型（線性、Ridge、Lasso、隨機森林）進行預測，並以相關係數進行特徵選擇。對線性模型使用統計方法估計信賴區間，對樹模型則採用殘差自助法估計預測區間來預測目標變數（例如，「確診」、「死亡」、「復健」）。  評估指標包含 RMSE、R2，並將結果匯入報告與 PPT 範本，方便教學與展示用途。

- **GPT 輔助內容**: 本專案的核心架構、程式碼實作、以及分析流程均由 GPT 輔助完成
- **NotebookLM 摘要**: 研究摘要基於 NotebookLM 對網路上線性回歸分析相關解法的研究
- **Kaggle Community**: 提供車輛資料集
- **Scikit-learn Team**: 提供優秀的機器學習庫
- **CRISP-DM Methodology**: 提供系統性的資料探勘流程

## 模型訓練摘要（本次合成資料運行結果）
  - 測試目標：Confirmed
  - 模型：Linear, Ridge, Lasso, RandomForest
  - 評估（範例）：
  - Linear — RMSE ≈ 67.76, R² ≈ 0.552
  - RandomForest — RMSE ≈ 60.05, R² ≈ 0.648

## Dataset
**資料集：** Vehicle Dataset from CarDekho (Kaggle)
**資料來源：** (https://www.kaggle.com/datasets/imdevskp/corona-virus-report)

## 目錄內容
- README.md — 專案說明與執行指引
- requirements.txt — 所需套件
- app.py — 用於探索資料和運行模型的 Streamlit 應用（一鍵運行 Replit）
- .github/workflows/ci.yml — GitHub Actions CI 範例
- .replit, replit.nix, run_streamlit.sh — Replit 一鍵啟動設定
- src/data_utils.py — 讀入與基礎清理
- src/modeling.py — 特徵工程、特徵選擇、模型訓練、模型評估、繪圖（含 95% band）
- src/reporting.py — ReportLab PDF 產生範例
- notebooks/analysis.ipynb — 包含 CRISP-DM 步驟、特徵工程、模型和圖表（Plotly 和 CI）的 Jupyter notebook
- specs/spec.md、specs/SDD.md — 規格與軟體設計文件
- data/corona.csv — 合成 COVID-like dataset（2,000 rows）
- reports/prediction_plot.png — 預測 vs 實際圖 + 95% band（Matplotlib）
- reports/AIIS_WH2_full_report.pdf — 含 NotebookLM 摘要、模型指標與圖表的 PDF 報告
- reports/NotebookLM_summary.md — NotebookLM 研究摘要（>100 字，已放入報告目錄）
- reports/presentation_template.txt — PPT 範本說明 (占位)
- replit.nix, .replit — Replit 一鍵執行檔案
- deploy.sh — 自動部署輔助Script

## ChatGPT Prompt 內容
請依照以下規格生成完整專案：
- 專案名稱：AIIS-WH2 - COVID-19 Dataset
- 資料來源：https://www.kaggle.com/datasets/imdevskp/corona-virus-report
- 遵循 CRISP-DM + SDD + Replit + GitHub Actions 流程
- 分析任務:使用各重回歸模型預測， 結果特徵選擇 (Feature Selection) 與模型評估 (Model Evaluation)及預測圖(加上信賴區間或預測區間)。
- 生成 Streamlit app、Jupyter notebook、Spec 文件、PDF/PPT 報告、README、Auto Deploy 腳本
- 生成NotebookLM 研究摘要，相同主題的解法進行研究，並撰寫一份 100 字以上的摘要，放入報告中。
- 建立 repo：benchen1981/AIIS-WH2-Final
- 可在 Replit 一鍵啟動 Streamlit
- 對話匯出 PDF（pdfCrowd / ReportLab）
- 打包成 AIIS-WH2-Final.zip

## How to run locally
1. Create a virtual environment.
2. `pip install -r requirements.txt`
3. Put dataset at `data/corona.csv`
4. Run notebook or `streamlit run app.py`.

## Replit
This repo is Replit-ready; place it in your Replit and press Run. The `.replit` file runs `streamlit run app.py`.

## MIT License
Copyright (c) 2025 Ben Chen
Permission is hereby granted, free of charge, to any person obtaining a copy...
