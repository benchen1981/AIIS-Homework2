import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go
import statsmodels.api as sm

def feature_engineer(df):
    # Simple engineered features: day of observation if date exists, population per country if exists, etc.
    X = df.copy()
    if 'ObservationDate' in X.columns:
        X['day'] = X['ObservationDate'].dt.dayofyear.fillna(0)
    # Numeric selection
    num_cols = X.select_dtypes(include='number').columns.tolist()
    return X[num_cols].dropna(axis=1, how='all')

def select_features(X, y, k=10):
    # Simple correlation-based selection
    corrs = X.corrwith(y).abs().sort_values(ascending=False)
    selected = corrs.head(min(k, len(corrs))).index.tolist()
    return selected

def run_models(df, target='Confirmed', test_size=0.2, random_state=42):
    X_all = feature_engineer(df)
    if target not in X_all.columns:
        raise ValueError(f"Target {target} not in numeric features. Choose another column.")
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

    metrics = {}
    fitted = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        metrics[name] = {
            'MSE': mean_squared_error(y_test, y_pred),
            'RMSE': mean_squared_error(y_test, y_pred, squared=False),
            'R2': r2_score(y_test, y_pred)
        }
        fitted[name] = {'model': m, 'X_test': X_test, 'y_test': y_test, 'y_pred': y_pred}

    return {'metrics': metrics, 'fitted': fitted, 'selected_features': sel}

def plot_with_ci(results, df, target='Confirmed'):
    # Use linear model for CI via statsmodels if available
    fitted = results['fitted']
    # Create a sample x-axis (index) and aggregate predictions as example
    model_entry = fitted.get('Linear') or list(fitted.values())[0]
    X_test = model_entry['X_test']
    y_test = model_entry['y_test']
    y_pred = model_entry['y_pred']

    # If statsmodels available and single feature, do OLS CI; otherwise plot predictions with simple band from residuals
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
                             fill='toself', name='95% band', showlegend=True, opacity=0.2))
    fig.update_layout(title=f'Predictions with 95% band ({target})', xaxis_title='sample', yaxis_title=target)
    return fig
