import pandas as pd
import numpy as np
import joblib
import os
import glob
from sklearn.preprocessing import StandardScaler

"""Runs inference on the ATL census grid dataset using decision tree regressors"""


def load_and_process_data(data_file, model_file, self_normalize=False):
    # Load model metadata (e.g. PCA component count)
    model = joblib.load(model_file)
    pca_components = model.n_features_in_
    print(f"Model expects {pca_components} PCA components")

    # Load raw dataset
    original_df = pd.read_csv(data_file)
    print(f"Dataset loaded with shape: {original_df.shape}")

    original_df.columns = [col.replace('demographics_', '') for col in original_df.columns]
    X = original_df.copy()
    X.columns = [col.replace('demographics_', '') for col in X.columns]

    # Normalize
    if self_normalize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.values)
        print("Applied self normalization to features")
    else:
        X_scaled = X.values
        print("Did not normalize X (already normalized)")

    # Load PCA parameters
    pca_params = pd.read_csv('pca_parameters.csv')
    components = np.zeros((pca_components, pca_params.shape[0]))
    for i in range(pca_components):
        col_name = f'Component_{i + 1}'
        if col_name in pca_params.columns:
            components[i] = pca_params[col_name].values
        else:
            print(f"Warning: {col_name} not found in PCA parameters")

    # Apply PCA transform
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
    data_folder = 'atl_data'
    model_file1 = 'decision_tree_depth10_minsplit5.pkl'
    model_file2 = 'decision_tree_depth15_minsplit5.pkl'
    use_log = True

    output_file1 = 'atl_preds_dt_model1.csv'
    output_file2 = 'atl_preds_dt_model2.csv'
    output_file_avg = 'atl_preds_dt_avg.csv'

    print("Checking model files:")
    print(f"  Model 1: {os.path.exists(model_file1)}")
    print(f"  Model 2: {os.path.exists(model_file2)}")

    data_files = glob.glob(os.path.join(data_folder, '*.csv'))
    if not data_files:
        print(f"No datasets found in '{data_folder}'")
        return

    print(f"Found {len(data_files)} datasets:")
    for data_file in data_files:
        print(f" - {os.path.basename(data_file)}")

    for data_file in data_files:
        file_name = os.path.basename(data_file)
        is_normalized = 'normalized' in file_name.lower()

        print(f"\n{'=' * 80}")
        print(f"Processing: {file_name} (normalized: {is_normalized})")
        print(f"{'=' * 80}")

        # Model 1
        print("\nRunning Model 1...")
        X_pca1, original_df = load_and_process_data(data_file, model_file1, self_normalize=not is_normalized)
        predictions1 = predict_with_model(model_file1, X_pca1, use_log=use_log)
        out1 = f"{os.path.splitext(output_file1)[0]}_{os.path.splitext(file_name)[0]}.csv"
        save_predictions_to_csv(original_df, predictions1, out1)

        # Model 2
        print("\nRunning Model 2...")
        X_pca2, _ = load_and_process_data(data_file, model_file2, self_normalize=not is_normalized)
        predictions2 = predict_with_model(model_file2, X_pca2, use_log=use_log)
        out2 = f"{os.path.splitext(output_file2)[0]}_{os.path.splitext(file_name)[0]}.csv"
        save_predictions_to_csv(original_df, predictions2, out2)

        # Average
        avg_predictions = [(p1 + p2) / 2.0 for p1, p2 in zip(predictions1, predictions2)]
        out_avg = f"{os.path.splitext(output_file_avg)[0]}_{os.path.splitext(file_name)[0]}.csv"
        save_predictions_to_csv(original_df, avg_predictions, out_avg)

        print(f"\nFinished processing: {file_name}")

    print("\n✅ All datasets processed successfully.")


if __name__ == "__main__":
    main()
