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


# ---------------------------- Data Preparation ----------------------------

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


# ---------------------------- Evaluation Metrics ----------------------------

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


# ---------------------------- Training and Evaluation ----------------------------

def train_decision_tree(X_train, y_train, X_test, y_test, max_depth=None, min_samples_split=2):
    model = DecisionTreeRegressor(random_state=42, max_depth=max_depth, min_samples_split=min_samples_split)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_r2 = compute_r2_score(y_train, y_train_pred)
    test_r2 = compute_r2_score(y_test, y_test_pred)
    train_mae = compute_mae(y_train, y_train_pred)
    test_mae = compute_mae(y_test, y_test_pred)

    return model, y_train_pred, y_test_pred, train_r2, test_r2, train_mae, test_mae


# ---------------------------- Main Experiment Loop ----------------------------

def main():
    np.random.seed(42)

    file_path = 'merged_nyc_grid_data.csv'
    use_log = True
    pca_variance_threshold = 0.95
    depths = [5, 10, 15]
    min_splits = [2, 5]
    all_results = []

    print("Starting Decision Tree sweep")
    for depth in depths:
        for min_split in min_splits:
            print(f"\nTraining Decision Tree (depth={depth}, min_samples_split={min_split})")

            X_train, X_test, y_train, y_test, pca_components = prepare_data(
                file_path, use_log=use_log, pca_variance_threshold=pca_variance_threshold
            )

            model, y_train_pred, y_test_pred, train_r2, test_r2, train_mae, test_mae = train_decision_tree(
                X_train, y_train, X_test, y_test, max_depth=depth, min_samples_split=min_split
            )

            print(f"Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
            print(f"Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")

            # Save model
            model_filename = f"decision_tree_depth{depth}_minsplit{min_split}.pkl"
            joblib.dump(model, model_filename)
            print(f"Model saved to {model_filename}")

            # Save results
            all_results.append({
                'max_depth': depth,
                'min_samples_split': min_split,
                'pca_components': pca_components,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_mae': train_mae,
                'test_mae': test_mae
            })

            # Optional: plot prediction scatter
            plt.figure(figsize=(6, 6))
            plt.scatter(y_test, y_test_pred, alpha=0.3)
            plt.xlabel('True Values')
            plt.ylabel('Predicted Values')
            plt.title(f'Decision Tree: Depth={depth}, MinSplit={min_split}')
            plt.grid(True)
            plt.savefig(f"dtree_scatter_depth{depth}_minsplit{min_split}.png")
            plt.close()

    results_df = pd.DataFrame(all_results)
    results_df.to_csv('decision_tree_results.csv', index=False)
    print("\nSweep complete. Top configurations:")
    print(results_df.sort_values('test_r2', ascending=False).head(5))
    regression_report(y_train, y_train_pred, dataset_label="Train")
    regression_report(y_test, y_test_pred, dataset_label="Test")

    # ---------------- Evaluate in Original Scale (optional) ----------------
    if use_log:
        y_test_original = np.expm1(y_test)
        y_pred_original = np.expm1(y_test_pred)

        r2_orig = r2_score(y_test_original, y_pred_original)
        mae_orig = mean_absolute_error(y_test_original, y_pred_original)
        mse_orig = mean_squared_error(y_test_original, y_pred_original)
        rmse_orig = np.sqrt(mse_orig)

        print("\n📊 Regression Report (Test - Original Scale)")
        print("-------------------------------------------")
        print(f"R²:   {r2_orig:.4f}")
        print(f"MAE:  {mae_orig:.2f} bikes")
        print(f"MSE:  {mse_orig:.2f} bikes²")
        print(f"RMSE: {rmse_orig:.2f} bikes")


if __name__ == '__main__':
    main()
