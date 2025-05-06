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

def load_and_process_data(data_file, model_file, use_log=True, batch_size=64):
    # Load model first to get PCA components
    checkpoint = torch.load(model_file, weights_only=False)
    pca_components = checkpoint['pca_components']
    print(f"Model has {pca_components} PCA components")

    # Load dataset
    df = pd.read_csv(data_file)
    print(f"Dataset loaded with shape: {df.shape}")

    # Extract features and target
    X = df.drop('bike_volume', axis=1)
    print(f"Features shape after dropping target: {X.shape}")

    # Cut off extreme values
    original_length = len(df)
    high_volume_mask = df['bike_volume'] <= 100000
    X = X[high_volume_mask]
    df = df[high_volume_mask]
    print(f"Removed {original_length - len(df)} observations with bike_volume > {str(100000)}")

    if use_log:
        y = np.log1p(df['bike_volume'])
    else:
        y = df['bike_volume']

    # Load scaler parameters
    scaler_params = pd.read_csv('scaler_parameters.csv')
    print(f"Scaler parameters shape: {scaler_params.shape}")

    # Check if feature names match
    feature_names = X.columns.tolist()
    scaler_features = scaler_params['Feature'].tolist()

    # Print feature differences for debugging
    print(f"Dataset features count: {len(feature_names)}")
    print(f"Scaler features count: {len(scaler_features)}")

    # Find the difference in features
    missing_in_dataset = set(scaler_features) - set(feature_names)
    extra_in_dataset = set(feature_names) - set(scaler_features)

    if missing_in_dataset:
        print(f"Features in scaler but missing in dataset: {missing_in_dataset}")
    if extra_in_dataset:
        print(f"Features in dataset but missing in scaler: {extra_in_dataset}")

    # Ensure X has the same columns as scaler_params in the same order
    # This is crucial for correct standardization
    aligned_X = pd.DataFrame()
    for feature in scaler_features:
        if feature in X.columns:
            aligned_X[feature] = X[feature]
        else:
            # If feature missing in dataset, fill with zeros
            aligned_X[feature] = 0
            print(f"Added missing feature '{feature}' with zeros")

    X = aligned_X
    print(f"Aligned features shape: {X.shape}")

    # Apply standardization
    means = scaler_params['Mean'].values
    scales = scaler_params['Scale'].values
    X_scaled = (X.values - means) / scales

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

    print(f"PCA components matrix shape: {components.shape}")
    print(f"X_scaled shape: {X_scaled.shape}")

    # Apply PCA transformation
    X_pca = np.dot(X_scaled, components.T)
    print(f"PCA-transformed features shape: {X_pca.shape}")

    # Create dataset and dataloader
    dataset = BikeDataset(X_pca, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return dataloader, y.values


def predict_with_model(model_file, dataloader, use_log=True):
    # Load the model
    checkpoint = torch.load(model_file, weights_only=False)
    input_dim = checkpoint['input_dim']
    num_layers = checkpoint['network_layers']

    print(f"Loading model with {num_layers} layers and input dimension {input_dim}")

    # Create model with same architecture
    model = BikeVolumeNN(input_dim, num_layers=num_layers)

    # Load state
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model state loaded successfully")

    # Set model to evaluation mode
    model.eval()

    # Run predictions
    all_predictions = []

    with torch.no_grad():
        for inputs, _ in dataloader:
            outputs = model(inputs)
            all_predictions.extend(outputs.numpy().flatten())

    print(f"Generated {len(all_predictions)} predictions")

    # Convert back from log scale if needed
    if use_log:
        all_predictions = np.expm1(all_predictions)

    return all_predictions


def main():
    torch.manual_seed(42)
    np.random.seed(42)

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

    # Store individual model predictions
    all_model_predictions_log = []

    # For storing metrics
    ensemble_results = {}

    # Load first model to get the dataloader
    first_model = model_files[0]
    dataloader, y_true = load_and_process_data(data_file, first_model, use_log=use_log)

    # Process each model
    for model_file in model_files:
        model_name = os.path.basename(model_file)
        print(f"\nRunning predictions for model: {model_name}")

        # If not the first model, we need to recreate the dataloader with this model's PCA components
        if model_file != first_model:
            dataloader, _ = load_and_process_data(data_file, model_file, use_log=use_log)

        # Run predictions (keeping in log scale)
        predictions_log = predict_with_model(model_file, dataloader, use_log=False)
        all_model_predictions_log.append(predictions_log)

        print(f"Generated {len(predictions_log)} predictions for {model_name}")

    # Calculate ensemble predictions (average of all models)
    ensemble_predictions_log = np.mean(all_model_predictions_log, axis=0)
    print(f"\nCreated ensemble predictions by averaging {len(all_model_predictions_log)} models")

    # Calculate metrics in log scale
    if use_log:
        y_true_log = y_true  # Already in log scale
        r2_log = compute_r2_score(y_true_log, ensemble_predictions_log)
        mae_log = compute_mae(y_true_log, ensemble_predictions_log)

        # Volume weighted loss in log scale
        predictions_tensor = torch.tensor(ensemble_predictions_log, dtype=torch.float32).reshape(-1, 1)
        y_true_tensor = torch.tensor(y_true_log, dtype=torch.float32).reshape(-1, 1)
        vw_loss_log = volume_weighted_loss(predictions_tensor, y_true_tensor).item()

        # Convert to original scale for original scale metrics
        ensemble_predictions_orig = np.expm1(ensemble_predictions_log)
        y_true_orig = np.expm1(y_true_log)
    else:
        # If not using log, then the values are already in original scale
        ensemble_predictions_orig = ensemble_predictions_log
        y_true_orig = y_true
        r2_log = None
        mae_log = None
        vw_loss_log = None

    # Calculate metrics in original scale
    r2_orig = compute_r2_score(y_true_orig, ensemble_predictions_orig)
    mae_orig = compute_mae(y_true_orig, ensemble_predictions_orig)

    # Volume weighted loss in original scale
    predictions_tensor = torch.tensor(ensemble_predictions_orig, dtype=torch.float32).reshape(-1, 1)
    y_true_tensor = torch.tensor(y_true_orig, dtype=torch.float32).reshape(-1, 1)
    vw_loss_orig = volume_weighted_loss(predictions_tensor, y_true_tensor).item()

    # Store ensemble results
    ensemble_results = {
        'model_name': 'Ensemble (Average)',
        'r2_log': r2_log,
        'mae_log': mae_log,
        'vw_loss_log': vw_loss_log,
        'r2_orig': r2_orig,
        'mae_orig': mae_orig,
        'vw_loss_orig': vw_loss_orig,
        'predictions_log': ensemble_predictions_log,
        'predictions_orig': ensemble_predictions_orig
    }

    # Print ensemble metrics
    print("\n=== Ensemble Model Results ===")
    if use_log:
        print(f"Log scale metrics:")
        print(f"  R² score: {r2_log:.4f}")
        print(f"  MAE: {mae_log:.4f}")
        print(f"  Volume weighted loss: {vw_loss_log:.4f}")

    print(f"Original scale metrics:")
    print(f"  R² score: {r2_orig:.4f}")
    print(f"  MAE: {mae_orig:.4f}")
    print(f"  Volume weighted loss: {vw_loss_orig:.4f}")

    # Create visualization for ensemble
    try:
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true_orig, ensemble_predictions_orig, alpha=0.5)

        # Add a perfect prediction line
        max_val = max(np.max(y_true_orig), np.max(ensemble_predictions_orig))
        min_val = min(np.min(y_true_orig), np.min(ensemble_predictions_orig))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')

        plt.xlabel('Actual Bike Volume')
        plt.ylabel('Predicted Bike Volume')
        plt.title(f'Actual vs Predicted Bike Volume\nEnsemble Model, R²: {r2_orig:.4f}')
        plt.tight_layout()

        # Save plot
        plot_file = "ensemble_predictions.png"
        plt.savefig(plot_file)
        print(f"\nSaved ensemble prediction plot to: {plot_file}")
    except Exception as e:
        print(f"Error creating plot: {str(e)}")

    # Also save the ensemble predictions to a CSV
    try:
        results_df = pd.DataFrame({
            'actual_volume': y_true_orig,
            'predicted_volume': ensemble_predictions_orig,
            'actual_log': y_true_log if use_log else np.log1p(y_true_orig),
            'predicted_log': ensemble_predictions_log
        })

        csv_file = "ensemble_predictions.csv"
        results_df.to_csv(csv_file, index=False)
        print(f"Saved ensemble predictions to: {csv_file}")
    except Exception as e:
        print(f"Error saving predictions to CSV: {str(e)}")


if __name__ == "__main__":
    main()