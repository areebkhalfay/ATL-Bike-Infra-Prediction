import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet


def prepare_data(file_path, use_log=True, pca_variance_threshold=0.95):
    df = pd.read_csv(file_path)
    print('Dataset loaded:', df.shape)

    # Extract features and target
    X = df.drop('bike_volume', axis=1)
    y = np.log1p(df['bike_volume']) if use_log else df['bike_volume']

    # Train/test split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # PCA
    pca = PCA(n_components=pca_variance_threshold)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    n_components = pca.n_components_
    print(f"PCA retained {n_components} components")

    # Save PCA and scaler info
    feature_names = X.columns.tolist()
    pca_df = pd.DataFrame({'Feature': feature_names})
    for i, comp in enumerate(pca.components_):
        pca_df[f'Component_{i + 1}'] = comp
    pca_df.to_csv('pca_parameters.csv', index=False)

    variance_df = pd.DataFrame({
        'Component': [f'Component_{i + 1}' for i in range(n_components)],
        'Explained_Variance_Ratio': pca.explained_variance_ratio_,
        'Cumulative_Variance': np.cumsum(pca.explained_variance_ratio_)
    })
    variance_df.to_csv('pca_variance.csv', index=False)

    scaler_df = pd.DataFrame({'Feature': feature_names, 'Mean': scaler.mean_, 'Scale': scaler.scale_})
    scaler_df.to_csv('scaler_parameters.csv', index=False)

    return X_train_pca, X_test_pca, y_train.values, y_test.values, n_components

def compute_r2_score(y_true, y_pred):
    return r2_score(y_true, y_pred)


def compute_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def regression_report(y_true, y_pred, dataset_label=""):
    """
    Prints a regression report with R², MAE, MSE, and RMSE.
    """
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    print(f"\n📊 Regression Report ({dataset_label})")
    print("-" * 40)
    print(f"R²:   {r2:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
def main():
    np.random.seed(42)

    file_path = 'merged_nyc_grid_data.csv'
    use_log = True
    pca_variance_threshold = 0.95

    X_train, X_test, y_train, y_test, pca_components = prepare_data(
                    file_path, use_log=use_log, pca_variance_threshold=pca_variance_threshold
                )
    return X_train, X_test, y_train, y_test, pca_components

if __name__ == "__main__":
    main()