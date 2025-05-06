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


def plot_actual_vs_predicted(actual, predicted, model_name):
    plt.figure(figsize=(10, 8))

    # Plot scatter points
    plt.scatter(actual, predicted, alpha=0.5)

    # Find the range of values to set axes limits
    min_val = min(min(actual), min(predicted))
    max_val = max(max(actual), max(predicted))

    # Add padding to the limits
    padding = (max_val - min_val) * 0.05
    #plt.xlim(min_val - padding, max_val + padding)
    plt.xlim(0,100000)
    #plt.ylim(min_val - padding, max_val + padding)
    plt.ylim(0,100000)

    # Add diagonal line (y=x)
    plt.plot([min_val - padding, max_val + padding],
             [min_val - padding, max_val + padding],
             'r--', label='y=x')

    # Calculate R² score
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    ss_res = np.sum((actual - predicted) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    # Calculate MAE
    mae = np.mean(np.abs(actual - predicted))

    plt.title(f'Actual vs Predicted Bike Volume\nR² = {r2:.4f}, MAE = {mae:.4f}')
    plt.xlabel('Actual Bike Volume')
    plt.ylabel('Predicted Bike Volume')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add a text box with model details
    plt.text(0.05, 0.95, f'Model: {model_name}', transform=plt.gca().transAxes,
             fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))

    plt.tight_layout()
    plt.savefig(f'bike_volume_prediction_scatter_{model_name}.png')

    # Don't show the plot to avoid blocking when processing multiple models
    # plt.show()

    plt.close()

    return r2, mae


def plot_model_comparison(results):
    # Sort results by R² score
    sorted_results = sorted(results, key=lambda x: x['r2'], reverse=True)

    models = [r['model_name'] for r in sorted_results]
    r2_scores = [r['r2'] for r in sorted_results]
    mae_values = [r['mae'] for r in sorted_results]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot R² scores
    bars1 = ax1.bar(models, r2_scores, color='skyblue')
    ax1.set_title('Model Comparison - R² Score (higher is better)')
    ax1.set_xlabel('Model')
    ax1.set_ylabel('R² Score')
    ax1.set_ylim(0, max(r2_scores) * 1.1)  # Add 10% padding
    ax1.set_xticklabels(models, rotation=45, ha='right')

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=8)

    # Plot MAE values
    bars2 = ax2.bar(models, mae_values, color='salmon')
    ax2.set_title('Model Comparison - Mean Absolute Error (lower is better)')
    ax2.set_xlabel('Model')
    ax2.set_ylabel('MAE')
    ax2.set_ylim(0, max(mae_values) * 1.1)  # Add 10% padding
    ax2.set_xticklabels(models, rotation=45, ha='right')

    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.show()

    # Create a summary table and save to CSV
    summary_df = pd.DataFrame({
        'Model': models,
        'R²': r2_scores,
        'MAE': mae_values
    })

    summary_df.to_csv('model_comparison_results.csv', index=False)
    print(f"Summary saved to 'model_comparison_results.csv'")

    return summary_df


def main():
    # Configuration
    data_file = 'merged_nyc_grid_data.csv'
    model_dir = 'models'  # Current directory, change this if models are elsewhere
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

    print(f"Found {len(model_files)} model files to evaluate:")
    for model_file in model_files:
        print(f"  - {os.path.basename(model_file)}")

    # Store results for comparison
    results = []

    # Process each model
    for model_file in model_files:
        try:
            print(f"\n{'=' * 80}")
            print(f"Processing model: {os.path.basename(model_file)}")
            print(f"{'=' * 80}")

            # Extract model name for plotting
            model_name = Path(model_file).stem

            # Load data and process
            dataloader, actual_values = load_and_process_data(
                data_file, model_file, use_log=use_log)

            # Convert back from log if necessary
            if use_log:
                actual_values = np.expm1(actual_values)

            # Load model and get predictions
            predicted_values = predict_with_model(model_file, dataloader, use_log=use_log)

            # Plot results
            r2, mae = plot_actual_vs_predicted(actual_values, predicted_values, model_name)

            print(f"Model evaluation complete for {model_name}")
            print(f"R² Score: {r2:.4f}")
            print(f"Mean Absolute Error: {mae:.4f}")
            print(f"Visualization saved to 'bike_volume_prediction_scatter_{model_name}.png'")

            # Store results for comparison
            results.append({
                'model_name': model_name,
                'r2': r2,
                'mae': mae
            })

        except Exception as e:
            print(f"Error processing model {model_file}: {str(e)}")

    if results:
        print(f"\n{'=' * 80}")
        print(f"Generating model comparison")
        print(f"{'=' * 80}")

        # Create comparison visualization
        summary_df = plot_model_comparison(results)

        # Print sorted results
        print("\nModel Performance Summary (sorted by R² score):")
        print(summary_df.sort_values('R²', ascending=False).to_string(index=False))
    else:
        print("No results to compare. All model evaluations failed.")


if __name__ == "__main__":
    main()

# import torch
# import torch.nn as nn
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from torch.utils.data import Dataset, DataLoader
# import os
#
#
# # Define the BikeVolumeNN class to match the one used for training
# class BikeVolumeNN(nn.Module):
#     def __init__(self, input_dim, num_layers=2, dropout_percent=0.3):
#         super(BikeVolumeNN, self).__init__()
#
#         assert num_layers in [2, 3]
#         assert dropout_percent <= 1.0
#
#         self.num_layers = num_layers
#         self.dropout_percent = dropout_percent
#
#         if num_layers == 2:
#             self.model = nn.Sequential(
#                 nn.Linear(input_dim, 32),
#                 nn.BatchNorm1d(32, momentum=0.9),
#                 nn.ReLU(),
#                 nn.Dropout(self.dropout_percent),
#
#                 nn.Linear(32, 16),
#                 nn.BatchNorm1d(16, momentum=0.9),
#                 nn.ReLU(),
#                 nn.Dropout(self.dropout_percent - 0.1),
#
#                 nn.Linear(16, 1)
#             )
#         elif num_layers == 3:
#             self.model = nn.Sequential(
#                 nn.Linear(input_dim, 64),
#                 nn.BatchNorm1d(64, momentum=0.9),
#                 nn.ReLU(),
#                 nn.Dropout(self.dropout_percent),
#
#                 nn.Linear(64, 32),
#                 nn.BatchNorm1d(32, momentum=0.9),
#                 nn.ReLU(),
#                 nn.Dropout(self.dropout_percent),
#
#                 nn.Linear(32, 16),
#                 nn.BatchNorm1d(16, momentum=0.9),
#                 nn.ReLU(),
#                 nn.Dropout(self.dropout_percent - 0.1),
#
#                 nn.Linear(16, 1)
#             )
#
#     def forward(self, x):
#         return self.model(x)
#
#
# # Custom Dataset class
# class BikeDataset(Dataset):
#     def __init__(self, features, targets=None):
#         self.features = torch.tensor(features, dtype=torch.float32)
#         if targets is not None:
#             self.targets = torch.tensor(targets, dtype=torch.float32).reshape(-1, 1)
#         else:
#             self.targets = None
#
#     def __len__(self):
#         return len(self.features)
#
#     def __getitem__(self, idx):
#         if self.targets is not None:
#             return self.features[idx], self.targets[idx]
#         else:
#             return self.features[idx]
#
#
# def load_and_process_data(data_file, model_file, use_log=True, batch_size=64):
#     # Load model first to get PCA components
#     checkpoint = torch.load(model_file, weights_only=False)
#     pca_components = checkpoint['pca_components']
#     print(f"Model has {pca_components} PCA components")
#
#     # Load dataset
#     df = pd.read_csv(data_file)
#     print(f"Dataset loaded with shape: {df.shape}")
#
#     # Extract features and target
#     X = df.drop('bike_volume', axis=1)
#     print(f"Features shape after dropping target: {X.shape}")
#
#     if use_log:
#         y = np.log1p(df['bike_volume'])
#     else:
#         y = df['bike_volume']
#
#     # Load scaler parameters
#     scaler_params = pd.read_csv('scaler_parameters.csv')
#     print(f"Scaler parameters shape: {scaler_params.shape}")
#
#     # Check if feature names match
#     feature_names = X.columns.tolist()
#     scaler_features = scaler_params['Feature'].tolist()
#
#     # Print feature differences for debugging
#     print(f"Dataset features count: {len(feature_names)}")
#     print(f"Scaler features count: {len(scaler_features)}")
#
#     # Find the difference in features
#     missing_in_dataset = set(scaler_features) - set(feature_names)
#     extra_in_dataset = set(feature_names) - set(scaler_features)
#
#     if missing_in_dataset:
#         print(f"Features in scaler but missing in dataset: {missing_in_dataset}")
#     if extra_in_dataset:
#         print(f"Features in dataset but missing in scaler: {extra_in_dataset}")
#
#     # Ensure X has the same columns as scaler_params in the same order
#     # This is crucial for correct standardization
#     aligned_X = pd.DataFrame()
#     for feature in scaler_features:
#         if feature in X.columns:
#             aligned_X[feature] = X[feature]
#         else:
#             # If feature missing in dataset, fill with zeros
#             aligned_X[feature] = 0
#             print(f"Added missing feature '{feature}' with zeros")
#
#     X = aligned_X
#     print(f"Aligned features shape: {X.shape}")
#
#     # Apply standardization
#     means = scaler_params['Mean'].values
#     scales = scaler_params['Scale'].values
#     X_scaled = (X.values - means) / scales
#
#     # Load PCA parameters
#     pca_params = pd.read_csv('pca_parameters.csv')
#     print(f"PCA parameters shape: {pca_params.shape}")
#
#     # Extract components for the number specified in the model
#     components = np.zeros((pca_components, pca_params.shape[0]))
#     for i in range(pca_components):
#         col_name = f'Component_{i + 1}'
#         if col_name in pca_params.columns:
#             components[i] = pca_params[col_name].values
#         else:
#             print(f"Warning: {col_name} not found in PCA parameters")
#
#     print(f"PCA components matrix shape: {components.shape}")
#     print(f"X_scaled shape: {X_scaled.shape}")
#
#     # Apply PCA transformation
#     X_pca = np.dot(X_scaled, components.T)
#     print(f"PCA-transformed features shape: {X_pca.shape}")
#
#     # Create dataset and dataloader
#     dataset = BikeDataset(X_pca, y)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
#
#     return dataloader, y.values
#
#
# def predict_with_model(model_file, dataloader, use_log=True):
#     # Load the model
#     checkpoint = torch.load(model_file, weights_only=False)
#     input_dim = checkpoint['input_dim']
#     num_layers = checkpoint['network_layers']
#
#     print(f"Loading model with {num_layers} layers and input dimension {input_dim}")
#
#     # Create model with same architecture
#     model = BikeVolumeNN(input_dim, num_layers=num_layers)
#
#     # Load state
#     model.load_state_dict(checkpoint['model_state_dict'])
#     print("Model state loaded successfully")
#
#     # Set model to evaluation mode
#     model.eval()
#
#     # Run predictions
#     all_predictions = []
#
#     with torch.no_grad():
#         for inputs, _ in dataloader:
#             outputs = model(inputs)
#             all_predictions.extend(outputs.numpy().flatten())
#
#     print(f"Generated {len(all_predictions)} predictions")
#
#     # Convert back from log scale if needed
#     if use_log:
#         all_predictions = np.expm1(all_predictions)
#
#     return all_predictions
#
#
# def plot_actual_vs_predicted(actual, predicted, model_name):
#     plt.figure(figsize=(10, 8))
#
#     # Plot scatter points
#     plt.scatter(actual, predicted, alpha=0.5)
#
#     # Find the range of values to set axes limits
#     min_val = min(min(actual), min(predicted))
#     max_val = max(max(actual), max(predicted))
#
#     # Add padding to the limits
#     padding = (max_val - min_val) * 0.05
#     plt.xlim(min_val - padding, max_val + padding)
#     plt.ylim(min_val - padding, max_val + padding)
#
#     # Add diagonal line (y=x)
#     plt.plot([min_val - padding, max_val + padding],
#              [min_val - padding, max_val + padding],
#              'r--', label='y=x')
#
#     # Calculate R² score
#     ss_tot = np.sum((actual - np.mean(actual)) ** 2)
#     ss_res = np.sum((actual - predicted) ** 2)
#     r2 = 1 - (ss_res / ss_tot)
#
#     # Calculate MAE
#     mae = np.mean(np.abs(actual - predicted))
#
#     plt.title(f'Actual vs Predicted Bike Volume\nR² = {r2:.4f}, MAE = {mae:.4f}')
#     plt.xlabel('Actual Bike Volume')
#     plt.ylabel('Predicted Bike Volume')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#
#     # Add a text box with model details
#     plt.text(0.05, 0.95, f'Model: {model_name}', transform=plt.gca().transAxes,
#              fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
#
#     plt.tight_layout()
#     plt.savefig(f'new_bike_volume_prediction_scatter_{model_name}.png')
#     plt.show()
#
#     return r2, mae
#
#
# def main():
#
#     data_file = 'merged_nyc_grid_data.csv'
#     model_file = 'apr10_model_trnsfrmlog_cutoff2000000lossvolume_weighted_layers3_lr0.00005000_64batch_pca29.pth'
#
#     print(f"Checking if files exist:")
#     print(f"  Data file exists: {os.path.exists(data_file)}")
#     print(f"  Model file exists: {os.path.exists(model_file)}")
#
#     use_log = True  # Same as in training
#
#     # Extract model name for plotting
#     model_name = model_file.split('.')[0]
#
#     # Load data and process using the more robust approach
#     dataloader, actual_values = load_and_process_data(
#         data_file, model_file, use_log=use_log)
#
#     # Convert back from log if necessary
#     if use_log:
#         actual_values = np.expm1(actual_values)
#
#     # Load model and get predictions
#     predicted_values = predict_with_model(model_file, dataloader, use_log=use_log)
#
#     # Plot results
#     r2, mae = plot_actual_vs_predicted(actual_values, predicted_values, model_name)
#
#     print(f"Model evaluation complete")
#     print(f"R² Score: {r2:.4f}")
#     print(f"Mean Absolute Error: {mae:.4f}")
#     print(f"Visualization saved to 'new_bike_volume_prediction_scatter_{model_name}.png'")
#
#
# if __name__ == "__main__":
#     main()