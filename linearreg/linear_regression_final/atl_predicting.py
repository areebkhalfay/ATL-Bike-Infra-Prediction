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

def load_and_process_data(data_file, model_file, self_normalize=False):
    model = joblib.load(model_file)
    pca_components = model.n_features_in_
    print(f"Model expects {pca_components} PCA components")

    original_df = pd.read_csv(data_file)
    print(f"Dataset loaded with shape: {original_df.shape}")

    original_df.columns = [col.replace('demographics_', '') for col in original_df.columns]
    X = original_df.copy()
    X.columns = [col.replace('demographics_', '') for col in X.columns]

    if self_normalize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.values)
        print("Applied self normalization to features")
    else:
        X_scaled = X.values
        print("Did not normalize X (already normalized)")

    pca_params = pd.read_csv('pca_parameters.csv')
    components = np.zeros((pca_components, pca_params.shape[0]))
    for i in range(pca_components):
        col_name = f'Component_{i + 1}'
        if col_name in pca_params.columns:
            components[i] = pca_params[col_name].values
        else:
            print(f"Warning: {col_name} not found in PCA parameters")

    X_pca = np.dot(X_scaled, components.T)
    print(f"PCA-transformed shape: {X_pca.shape}")

    return X_pca, original_df


def predict_with_model(model_file, X_pca, use_log=True):
    model = joblib.load(model_file)
    print(f"Model loaded from {model_file}")

    predictions = model.predict(X_pca)

    if use_log:
        predictions = np.expm1(predictions)

    print(f"Generated {len(predictions)} predictions")
    return predictions


def save_predictions_to_csv(original_df, predictions, output_file):
    result_df = original_df.copy()
    result_df['bike_volume'] = predictions
    result_df.to_csv(output_file, index=False)
    print(f"Predictions saved to '{output_file}'")
    
def main():
    X_pca_atl, original_df_atl = load_and_process_data(
        "atl_input_data.csv", "linear_regression_ols_model.pkl", self_normalize=False
    )
    predictions_atl = predict_with_model(
        "linear_regression_ols_model.pkl", X_pca_atl, use_log=True
    )
    save_predictions_to_csv(
        original_df_atl, predictions_atl, "OLS_atl_predictions.csv"
    )
if __name__ == "__main__":
    main()