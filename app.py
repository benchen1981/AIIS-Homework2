import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go

# 支援 XGBoost
try:
    from xgboost import XGBRegressor
    xgb_available = True
except ImportError:
    xgb_available = False

st.set_page_config(layout="wide", page_title="CRISP-DM Regression App")
st.title("CRISP-DM Regression Demo with COVID Dataset")
st.markdown("**CRISP-DM**流程：資料理解 → 特徵工程 → 建模 → 評估 → 部署")

DATA_PATH = "data/country_wise_latest.csv"
try:
    df = pd.read_csv(DATA_PATH)
except Exception:
    st.error(f"無法讀取檔案 {DATA_PATH}，請上傳正確 csv 檔案。")
    uploaded = st.file_uploader("上傳 country_wise_latest.csv", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        df.to_csv(DATA_PATH, index=False)
        st.success(f"已儲存至 {DATA_PATH}，請重新整理頁面。")
        st.stop()

#####################
# 資料理解
#####################
st.subheader("資料預覽與統計")
st.dataframe(df.head(20))
st.write(df.describe())
st.write("欄位：", df.columns.tolist())

#####################
# 特徵工程＋特徵選擇
#####################
def feature_engineer(df):
    X = df.copy()
    if 'ObservationDate' in X.columns:
        try:
            X['ObservationDate'] = pd.to_datetime(X['ObservationDate'])
            X['day'] = X['ObservationDate'].dt.dayofyear.fillna(0)
        except Exception:
            pass
    num_cols = X.select_dtypes(include='number').columns.tolist()
    return X[num_cols].dropna(axis=1, how='all')

def select_features(X, y, k=10):
    corrs = X.corrwith(y).abs().sort_values(ascending=False)
    selected = corrs.head(min(k, len(corrs))).index.tolist()
    return selected

#####################
# 模型訓練/評估
#####################
def run_models(df, target, test_size=0.2, random_state=42):
    X_all = feature_engineer(df)
    if target not in X_all.columns:
        raise ValueError(f"無法找到目標 {target}，請選擇數值欄位。")
    y = X_all[target]
    X = X_all.drop(columns=[target])
    sel = select_features(X, y, k=10)
    X = X[sel]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    models = {
        'Linear': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'RF': RandomForestRegressor(n_estimators=100, random_state=random_state)
    }
    if xgb_available:
        models['XGBoost'] = XGBRegressor(n_estimators=100, random_state=random_state)
    metrics, fitted = {}, {}

    for name, m in models.items():
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        metrics[name] = {
            'MSE': mean_squared_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),  # 修正: 用 np.sqrt 計算 RMSE
            'R2': r2_score(y_test, y_pred)
        }
        fitted[name] = {'model': m, 'X_test': X_test, 'y_test': y_test, 'y_pred': y_pred}

    return {'metrics': metrics, 'fitted': fitted, 'selected_features': sel}

#####################
# 視覺化（信賴區間）
#####################
def plot_with_ci(pred_entry, target):
    y_test = pred_entry['y_test']
    y_pred = pred_entry['y_pred']
    x_axis = np.arange(len(y_test))
    resid = y_test - y_pred
    se = np.std(resid)
    lower = y_pred - 1.96 * se
    upper = y_pred + 1.96 * se
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_axis, y=y_test, mode='markers+lines', name='Actual'))
    fig.add_trace(go.Scatter(x=x_axis, y=y_pred, mode='lines', name='Predicted'))
    fig.add_trace(go.Scatter(x=np.concatenate([x_axis, x_axis[::-1]]),
                            y=np.concatenate([upper, lower[::-1]]),
                            fill='toself', name='95% CI band', showlegend=True, opacity=0.2))
    fig.update_layout(title=f'預測與信賴區間 ({target})', xaxis_title='Test樣本', yaxis_title=target)
    return fig

#####################
# Streamlit UI
#####################
st.sidebar.header("模型控制")

num_cols = df.select_dtypes(include='number').columns.tolist()
target = st.sidebar.selectbox("預測目標", options=num_cols, index=0)
test_size = st.sidebar.slider("測試集比例", 0.1, 0.5, 0.2)
run_btn = st.sidebar.button("執行模型")

if run_btn:
    with st.spinner("模型訓練中..."):
        results = run_models(df, target=target, test_size=test_size)
    st.success("模型訓練完成")
    st.subheader("評估指標")
    st.dataframe(pd.DataFrame(results['metrics']).T)
    available_models = list(results['fitted'].keys())
    model_type = st.selectbox("選擇欲檢視的模型", available_models)
    st.subheader(f"預測結果與信賴區間 ({model_type})")
    fig = plot_with_ci(results['fitted'][model_type], target)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")
    st.caption("CRISP-DM Regression App © 2025")
