import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import os
import glob
from pathlib import Path


# Define the BikeVolumeNN class to match the one used for training
class BikeVolumeNN(nn.Module):
    def __init__(self, input_dim, num_layers=2, dropout_percent=0.3):
        super(BikeVolumeNN, self).__init__()

        assert num_layers in [2, 3]
        assert dropout_percent <= 1.0

        self.num_layers = num_layers
        self.dropout_percent = dropout_percent

        if num_layers == 2:
            self.model = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.BatchNorm1d(32, momentum=0.9),
                nn.ReLU(),
                nn.Dropout(self.dropout_percent),

                nn.Linear(32, 16),
                nn.BatchNorm1d(16, momentum=0.9),
                nn.ReLU(),
                nn.Dropout(self.dropout_percent - 0.1),

                nn.Linear(16, 1)
            )
        elif num_layers == 3:
            self.model = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.BatchNorm1d(64, momentum=0.9),
                nn.ReLU(),
                nn.Dropout(self.dropout_percent),

                nn.Linear(64, 32),
                nn.BatchNorm1d(32, momentum=0.9),
                nn.ReLU(),
                nn.Dropout(self.dropout_percent),

                nn.Linear(32, 16),
                nn.BatchNorm1d(16, momentum=0.9),
                nn.ReLU(),
                nn.Dropout(self.dropout_percent - 0.1),

                nn.Linear(16, 1)
            )

    def forward(self, x):
        return self.model(x)


# Custom Dataset class
class BikeDataset(Dataset):
    def __init__(self, features, targets=None):
        self.features = torch.tensor(features, dtype=torch.float32)
        if targets is not None:
            self.targets = torch.tensor(targets, dtype=torch.float32).reshape(-1, 1)
        else:
            self.targets = None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.targets is not None:
            return self.features[idx], self.targets[idx]
        else:
            return self.features[idx]


def volume_weighted_loss(outputs, targets, weight_factor=0.5):
    """
    Loss that gives more weight to high-volume errors
    """
    # Calculate relative weights based on target magnitude
    weights = (targets / targets.mean()) ** weight_factor

    # Apply weights to squared errors
    weighted_squared_errors = weights * ((outputs - targets) ** 2)

    # Mean of weighted errors
    loss = torch.mean(weighted_squared_errors)

    return loss


def compute_r2_score(y_true, y_pred):
    """Compute R² score (coefficient of determination)"""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)

    if ss_tot == 0:
        return 0

    return 1 - (ss_res / ss_tot)


def compute_mae(y_true, y_pred):
    """Compute Mean Absolute Error"""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    return np.mean(np.abs(y_true - y_pred))


def load_and_split_data(data_file, use_log=True):
    """Load the dataset and split into train/test sets using stratified sampling"""
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Load dataset
    df = pd.read_csv(data_file)
    print(f"Dataset loaded with shape: {df.shape}")

    # Extract features and target
    X = df.drop('bike_volume', axis=1)
    y = df['bike_volume']
    print(f"Features shape after dropping target: {X.shape}")

    # Cut off extreme values
    original_length = len(df)
    high_volume_mask = df['bike_volume'] <= 100000
    X = X[high_volume_mask]
    y = y[high_volume_mask]
    print(f"Removed {original_length - len(X)} observations with bike_volume > {str(100000)}")

    # Apply log transform if requested
    if use_log:
        y = np.log1p(y)
        print("Applied log1p transformation to target")

    # Create bins for stratified sampling
    num_bins = 10  # Can be adjusted based on dataset distribution
    y_binned = pd.qcut(y, num_bins, labels=False, duplicates='drop')

    # Split data using stratified sampling
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y_binned
    )

    print(f"Split data into training set ({len(X_train)} samples) and test set ({len(X_test)} samples)")

    return X_train, X_test, y_train, y_test


def apply_transformations(X_train, X_test, model_file):
    """Apply standardization and PCA transformation"""
    # Load model to get PCA components
    checkpoint = torch.load(model_file, weights_only=False)
    pca_components = checkpoint['pca_components']
    print(f"Model has {pca_components} PCA components")

    # Load scaler parameters
    scaler_params = pd.read_csv('scaler_parameters.csv')
    print(f"Scaler parameters shape: {scaler_params.shape}")

    # Check if feature names match
    feature_names = X_train.columns.tolist()
    scaler_features = scaler_params['Feature'].tolist()

    # Ensure X_train and X_test have the same columns as scaler_params in the same order
    def align_features(X, scaler_features):
        aligned_X = pd.DataFrame()
        for feature in scaler_features:
            if feature in X.columns:
                aligned_X[feature] = X[feature]
            else:
                # If feature missing in dataset, fill with zeros
                aligned_X[feature] = 0
                print(f"Added missing feature '{feature}' with zeros")
        return aligned_X

    X_train_aligned = align_features(X_train, scaler_features)
    X_test_aligned = align_features(X_test, scaler_features)
    print(f"Aligned features shape - train: {X_train_aligned.shape}, test: {X_test_aligned.shape}")

    # Apply standardization
    means = scaler_params['Mean'].values
    scales = scaler_params['Scale'].values
    X_train_scaled = (X_train_aligned.values - means) / scales
    X_test_scaled = (X_test_aligned.values - means) / scales

    # Load PCA parameters
    pca_params = pd.read_csv('pca_parameters.csv')
    print(f"PCA parameters shape: {pca_params.shape}")

    # Extract components for the number specified in the model
    components = np.zeros((pca_components, pca_params.shape[0]))
    for i in range(pca_components):
        col_name = f'Component_{i + 1}'
        if col_name in pca_params.columns:
            components[i] = pca_params[col_name].values
        else:
            print(f"Warning: {col_name} not found in PCA parameters")

    # Apply PCA transformation
    X_train_pca = np.dot(X_train_scaled, components.T)
    X_test_pca = np.dot(X_test_scaled, components.T)
    print(f"PCA-transformed features shape - train: {X_train_pca.shape}, test: {X_test_pca.shape}")

    return X_train_pca, X_test_pca


def create_dataloaders(X_train_pca, X_test_pca, y_train, y_test, batch_size=64):
    """Create DataLoader objects for training and test sets"""
    # Convert pandas Series to numpy arrays
    train_dataset = BikeDataset(X_train_pca, y_train.values)  # Add .values here
    test_dataset = BikeDataset(X_test_pca, y_test.values)  # Add .values here

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def predict_with_model(model_file, dataloader):
    """Run predictions with a model"""
    # Load the model
    checkpoint = torch.load(model_file, weights_only=False)
    input_dim = checkpoint['input_dim']
    num_layers = checkpoint['network_layers']

    # Create model with same architecture
    model = BikeVolumeNN(input_dim, num_layers=num_layers)

    # Load state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Run predictions
    all_predictions = []

    with torch.no_grad():
        for inputs, _ in dataloader:
            outputs = model(inputs)
            all_predictions.extend(outputs.numpy().flatten())

    return np.array(all_predictions)


def calculate_metrics(y_true, y_pred, use_log=True, set_name=""):
    """Calculate all metrics for the predictions"""
    # Calculate metrics in log scale
    r2_log = compute_r2_score(y_true, y_pred)
    mae_log = compute_mae(y_true, y_pred)

    # Volume weighted loss in log scale
    predictions_tensor = torch.tensor(y_pred, dtype=torch.float32).reshape(-1, 1)
    y_true_tensor = torch.tensor(y_true, dtype=torch.float32).reshape(-1, 1)
    vw_loss_log = volume_weighted_loss(predictions_tensor, y_true_tensor).item()

    # Convert to original scale if needed
    if use_log:
        y_true_orig = np.expm1(y_true)
        y_pred_orig = np.expm1(y_pred)
    else:
        y_true_orig = y_true
        y_pred_orig = y_pred

    # Calculate metrics in original scale
    r2_orig = compute_r2_score(y_true_orig, y_pred_orig)
    mae_orig = compute_mae(y_true_orig, y_pred_orig)

    # Volume weighted loss in original scale
    predictions_tensor = torch.tensor(y_pred_orig, dtype=torch.float32).reshape(-1, 1)
    y_true_tensor = torch.tensor(y_true_orig, dtype=torch.float32).reshape(-1, 1)
    vw_loss_orig = volume_weighted_loss(predictions_tensor, y_true_tensor).item()

    # Print metrics
    print(f"\n=== {set_name} Set Metrics ===")
    print(f"Log scale metrics:")
    print(f"  R² score: {r2_log:.4f}")
    print(f"  MAE: {mae_log:.4f}")
    print(f"  Volume weighted loss: {vw_loss_log:.4f}")

    print(f"Original scale metrics:")
    print(f"  R² score: {r2_orig:.4f}")
    print(f"  MAE: {mae_orig:.4f}")
    print(f"  Volume weighted loss: {vw_loss_orig:.4f}")

    return {
        'r2_log': r2_log,
        'mae_log': mae_log,
        'vw_loss_log': vw_loss_log,
        'r2_orig': r2_orig,
        'mae_orig': mae_orig,
        'vw_loss_orig': vw_loss_orig,
        'predictions_log': y_pred,
        'predictions_orig': y_pred_orig,
        'true_log': y_true,
        'true_orig': y_true_orig
    }


def plot_results(results, set_name):
    """Create a scatter plot of predictions vs actual values"""
    y_true = results['true_orig']
    y_pred = results['predictions_orig']
    r2 = results['r2_orig']

    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)

    # Add a perfect prediction line
    max_val = max(np.max(y_true), np.max(y_pred))
    min_val = min(np.min(y_true), np.min(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')

    plt.xlabel('Actual Bike Volume')
    plt.ylabel('Predicted Bike Volume')
    plt.title(f'Actual vs Predicted Bike Volume - {set_name} Set\nR²: {r2:.4f}')
    plt.tight_layout()

    # Save plot
    plot_file = f"ensemble_predictions_{set_name.lower()}.png"
    plt.savefig(plot_file)
    print(f"Saved {set_name} set prediction plot to: {plot_file}")

    return plot_file


def plot_comparison(train_results, test_results):
    """Create a scatter plot comparing train and test predictions"""
    # Create figure
    plt.figure(figsize=(12, 8))

    # Plot training data (blue)
    plt.scatter(train_results['true_orig'],
                train_results['predictions_orig'],
                alpha=0.9,
                color='blue',
                label='Training Data')

    # Plot test data (orange)
    plt.scatter(test_results['true_orig'],
                test_results['predictions_orig'],
                alpha=0.9,
                color='orange',
                label='Test Data')

    # Find global min and max for the perfect prediction line
    all_true = np.concatenate([train_results['true_orig'], test_results['true_orig']])
    all_pred = np.concatenate([train_results['predictions_orig'], test_results['predictions_orig']])
    max_val = max(np.max(all_true), np.max(all_pred))
    min_val = min(np.min(all_true), np.min(all_pred))

    # Add a perfect prediction line
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')

    # Add labels and title
    plt.xlabel('Actual Bike Volume')
    plt.ylabel('Predicted Bike Volume')
    plt.title('Actual vs Predicted Bike Volume\nBlue: Training Data, Orange: Test Data')

    # Add legend
    plt.legend()

    # Add R² score annotations
    train_r2 = train_results['r2_orig']
    test_r2 = test_results['r2_orig']
    plt.annotate(f'Training R²: {train_r2:.4f}',
                 xy=(0.05, 0.95),
                 xycoords='axes fraction',
                 fontsize=12,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))

    plt.annotate(f'Test R²: {test_r2:.4f}',
                 xy=(0.05, 0.90),
                 xycoords='axes fraction',
                 fontsize=12,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))

    plt.tight_layout()

    # Save plot
    plot_file = "ensemble_predictions_comparison.png"
    plt.savefig(plot_file)
    print(f"Saved comparison plot to: {plot_file}")

    return plot_file

def main():
    # Configuration
    data_file = 'merged_nyc_grid_data.csv'
    model_dir = 'models_toaverage'  # Current directory, change this if models are elsewhere
    model_pattern = '*.pth'  # Pattern to match model files
    use_log = True  # Same as in training

    # Check if data file exists
    print(f"Checking if data file exists: {os.path.exists(data_file)}")

    if not os.path.exists(data_file):
        print(f"ERROR: Data file '{data_file}' not found!")
        return

    # Find all model files in the directory
    model_files = glob.glob(os.path.join(model_dir, model_pattern))

    if not model_files:
        print(f"No model files found matching '{model_pattern}' in '{model_dir}'")
        return

    print(f"Found {len(model_files)} model files for ensemble:")
    for model_file in model_files:
        print(f"  - {os.path.basename(model_file)}")

    # Step 1: Load and split the data
    X_train, X_test, y_train, y_test = load_and_split_data(data_file, use_log=use_log)

    # Lists to store predictions from each model
    train_predictions_by_model = []
    test_predictions_by_model = []

    # Process each model
    for model_file in model_files:
        model_name = os.path.basename(model_file)
        print(f"\nProcessing model: {model_name}")

        # Step 2: Apply transformations specific to this model
        X_train_pca, X_test_pca = apply_transformations(X_train, X_test, model_file)

        # Step 3: Create dataloaders
        train_loader, test_loader = create_dataloaders(X_train_pca, X_test_pca, y_train, y_test)

        # Step 4: Run predictions
        train_predictions = predict_with_model(model_file, train_loader)
        test_predictions = predict_with_model(model_file, test_loader)

        train_predictions_by_model.append(train_predictions)
        test_predictions_by_model.append(test_predictions)

        print(f"Generated predictions for {model_name}")

    # Step 5: Create ensemble predictions by averaging
    train_ensemble_predictions = np.mean(train_predictions_by_model, axis=0)
    test_ensemble_predictions = np.mean(test_predictions_by_model, axis=0)

    print(f"\nCreated ensemble predictions by averaging {len(model_files)} models")

    # Step 6: Calculate metrics for train and test sets
    train_results = calculate_metrics(y_train.values, train_ensemble_predictions, use_log=use_log, set_name="Training")
    test_results = calculate_metrics(y_test.values, test_ensemble_predictions, use_log=use_log, set_name="Test")

    # Step 7: Plot results
    #train_plot = plot_results(train_results, "Training")
    #test_plot = plot_results(test_results, "Test")

    comparison_plot = plot_comparison(train_results, test_results)

    # Step 8: Save predictions to CSV
    try:
        train_df = pd.DataFrame({
            'actual_volume': train_results['true_orig'],
            'predicted_volume': train_results['predictions_orig'],
            'actual_log': train_results['true_log'],
            'predicted_log': train_results['predictions_log']
        })

        test_df = pd.DataFrame({
            'actual_volume': test_results['true_orig'],
            'predicted_volume': test_results['predictions_orig'],
            'actual_log': test_results['true_log'],
            'predicted_log': test_results['predictions_log']
        })

        train_df.to_csv("ensemble_predictions_train.csv", index=False)
        test_df.to_csv("ensemble_predictions_test.csv", index=False)
        print(f"Saved predictions to CSV files")
    except Exception as e:
        print(f"Error saving predictions to CSV: {str(e)}")

    # Print a summary comparison of train vs test performance
    print("\n=== Performance Summary ===")
    print(f"Metric                  | Training Set | Test Set")
    print(f"------------------------+-------------+----------")
    print(f"R² (log scale)          | {train_results['r2_log']:.4f}      | {test_results['r2_log']:.4f}")
    print(f"MAE (log scale)         | {train_results['mae_log']:.4f}      | {test_results['mae_log']:.4f}")
    print(f"VW Loss (log scale)     | {train_results['vw_loss_log']:.4f}      | {test_results['vw_loss_log']:.4f}")
    print(f"R² (original scale)     | {train_results['r2_orig']:.4f}      | {test_results['r2_orig']:.4f}")
    print(f"MAE (original scale)    | {train_results['mae_orig']:.2f}     | {test_results['mae_orig']:.2f}")
    print(f"VW Loss (original scale)| {train_results['vw_loss_orig']:.2f}     | {test_results['vw_loss_orig']:.2f}")


if __name__ == "__main__":
    main()