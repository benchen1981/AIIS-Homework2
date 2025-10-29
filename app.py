# app.py (Final Fixed)
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

st.set_page_config(layout="wide", page_title="AIIS-WH2 — CRISP-DM Regression Suite (Final Fixed)")

st.title("AIIS-WH2 — CRISP-DM 回歸分析（最終修正版）")
st.markdown("針對「訓練多個模型並進行評估」的錯誤做全面修正與防護。")

DATA_PATH = Path("/data/country_wise_latest.csv")

# ---------------------------
# 1) Upload / Load Data
# ---------------------------
st.header("1. 上傳或載入資料")
uploaded = st.file_uploader("上傳 CSV 檔案 (上傳後會自動執行)", type=["csv"])

if uploaded:
    try:
        save_dir = Path("data")
        save_dir.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(uploaded)
        df.to_csv(save_dir / "country_wise_latest.csv", index=False)
        st.success("已上傳並儲存為 data/country_wise_latest.csv")
    except Exception as e:
        st.error(f"讀取上傳檔案失敗：{e}")
        st.stop()
else:
    if DATA_PATH.exists():
        try:
            df = pd.read_csv(DATA_PATH)
            st.info(f"使用預設路徑載入：{DATA_PATH}")
        except Exception as e:
            st.error(f"讀取 {DATA_PATH} 失敗：{e}")
            st.stop()
    else:
        st.warning("尚未上傳檔案，請上傳 CSV 後自動執行流程。")
        st.stop()

st.write(f"資料大小：{df.shape[0]} rows × {df.shape[1]} cols")
st.dataframe(df.head(5))

# find date column if present
date_candidates = [c for c in df.columns if c.lower() in ("date", "observationdate", "observation_date", "report_date", "datetime")]
date_col = date_candidates[0] if date_candidates else None
if date_col:
    try:
        df[date_col] = pd.to_datetime(df[date_col])
    except Exception:
        st.warning(f"無法將 {date_col} 轉為 datetime，將保留原始值。")

# ---------------------------
# 2) Feature engineering
# ---------------------------
st.header("2. 特徵工程與候選特徵")

def feature_engineer(df_in):
    X = df_in.copy()
    if date_col and date_col in X.columns and np.issubdtype(X[date_col].dtype, np.datetime64):
        X["dayofyear"] = X[date_col].dt.dayofyear
        X["month"] = X[date_col].dt.month
        X["day"] = X[date_col].dt.day
        X = X.drop(columns=[date_col])
    # numeric columns only
    num = X.select_dtypes(include=[np.number]).copy()
    nunique = num.nunique()
    keep_cols = [c for c in num.columns if nunique[c] > 0]
    return num[keep_cols].dropna(axis=1, how='all')

X_all = feature_engineer(df)
if X_all.shape[1] == 0:
    st.error("找不到數值特徵可用於模型。請確認 CSV 含數值欄位（數字）。")
    st.stop()

st.write(f"發現 {X_all.shape[1]} 個數值型候選特徵")
st.dataframe(X_all.head())

# Sidebar controls
st.sidebar.header("模型設定")
target = st.sidebar.selectbox("目標變數 (target)", options=X_all.columns.tolist(), index=0)
test_size = st.sidebar.slider("測試集比例 (test_size)", min_value=0.05, max_value=0.5, value=0.2, step=0.05)
max_k = max(1, X_all.shape[1]-1)
k_features = st.sidebar.slider("SelectKBest: k (特徵數)", min_value=1, max_value=max_k, value=min(10, max_k))
auto_run = st.sidebar.checkbox("上傳後自動執行（若上傳則自動執行）", value=True)
run_btn = st.sidebar.button("手動執行一次")

should_run = auto_run or run_btn
if not should_run:
    st.info("請勾選『上傳後自動執行』或按下「手動執行一次」以啟動模型訓練。")
    st.stop()

# ---------------------------
# 3) 特徵選擇與資料切分
# ---------------------------
st.header("3. 特徵選擇與資料切分")

if target not in X_all.columns:
    st.error("選擇的 target 不在數值特徵中，請重新選擇。")
    st.stop()

y_all = X_all[target].copy()
X_candidates = X_all.drop(columns=[target]).fillna(0)
if X_candidates.shape[1] == 0:
    st.error("除 target 外無可用特徵，無法訓練模型。")
    st.stop()

def safe_select_kbest(X, y, k):
    try:
        k_use = min(k, X.shape[1])
        selector = SelectKBest(score_func=f_regression, k=k_use)
        selector.fit(X, y)
        mask = selector.get_support()
        chosen = X.columns[mask].tolist()
        scores = {col: float(score) for col, score in zip(X.columns[mask], selector.scores_[mask])}
        return chosen, scores
    except Exception as e:
        st.warning(f"SelectKBest 失敗 ({e})，改用相關係數選取前 k 特徵。")
        corrs = X.corrwith(y).abs().sort_values(ascending=False)
        chosen = corrs.head(k).index.tolist()
        return chosen, corrs.head(k).to_dict()

sel_features, sel_scores = safe_select_kbest(X_candidates, y_all, k_features)
st.write(f"選出 {len(sel_features)} 個特徵：", sel_features)
st.dataframe(pd.DataFrame.from_dict(sel_scores, orient='index', columns=['score']).sort_values('score', ascending=False))

X = X_candidates[sel_features]

# ---------------------------
# 4) 訓練多個模型並進行評估
# ---------------------------
st.header("4. 訓練多個模型並進行評估")

models_to_run = {
    "Linear": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42)
}
if HAS_XGB:
    models_to_run["XGBoost"] = XGBRegressor(n_estimators=200, random_state=42, verbosity=0)

metrics = {}
fitted = {}

try:
    X_train, X_test, y_train, y_test = train_test_split(X, y_all, test_size=test_size, random_state=42)
except Exception as e:
    st.error(f"資料切分失敗：{e}")
    st.stop()

for name, model in models_to_run.items():
    st.write(f"訓練：{name} ...")
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics[name] = {
            "MSE": mean_squared_error(y_test, y_pred),
            "RMSE": mean_squared_error(y_test, y_pred, squared=False),
            "R2": r2_score(y_test, y_pred)
        }
        fitted[name] = {"model": model, "X_test": X_test, "y_test": y_test, "y_pred": y_pred}
        st.success(f"{name} 訓練完成")
    except Exception as e:
        st.error(f"{name} 訓練失敗：{e}")

# ---------------------------
# 5) 評估與視覺化
# ---------------------------
st.header("5. 模型評估與視覺化")

metrics_df = pd.DataFrame(metrics).T
st.dataframe(metrics_df.style.format("{:.4f}"))

model_to_plot = st.selectbox("選擇要視覺化的模型", options=list(fitted.keys()))
entry = fitted[model_to_plot]
y_test_plot = np.array(entry["y_test"]).flatten()
y_pred_plot = np.array(entry["y_pred"]).flatten()

if y_test_plot.shape[0] != y_pred_plot.shape[0]:
    m = min(len(y_test_plot), len(y_pred_plot))
    y_test_plot = y_test_plot[:m]
    y_pred_plot = y_pred_plot[:m]
    st.warning("長度不符，已自動修正。")

idx = np.arange(len(y_test_plot))
resid = y_test_plot - y_pred_plot
se = np.std(resid)
lower = y_pred_plot - 1.96 * se
upper = y_pred_plot + 1.96 * se

fig = go.Figure()
fig.add_trace(go.Scatter(x=idx, y=y_test_plot, mode='markers+lines', name='Actual'))
fig.add_trace(go.Scatter(x=idx, y=y_pred_plot, mode='lines', name='Predicted'))
fig.add_trace(go.Scatter(
    x=np.concatenate([idx, idx[::-1]]),
    y=np.concatenate([upper, lower[::-1]]),
    fill='toself', name='95% band', opacity=0.2
))
fig.update_layout(title=f"{model_to_plot} — Actual vs Predicted", xaxis_title='Index', yaxis_title=target)
st.plotly_chart(fig, use_container_width=True)

st.success("✅ 模型訓練、評估與視覺化完成（Final Fixed）")
