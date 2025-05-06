import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import os
import glob

"""Runs inference on the ATL census grid dataset"""

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
    def __init__(self, features):
        self.features = torch.tensor(features, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


def load_and_process_data(data_file, model_file, use_log=True, batch_size=64, self_normalize=False):
    """
    Load and process data for inference

    Parameters:
    - data_file: Path to the dataset file
    - model_file: Path to the model file (.pth)
    - use_log: Whether to use log transformation for prediction
    - batch_size: Batch size for DataLoader
    - self_normalize: Whether to normalize the data using its own statistics instead of loaded scaler parameters

    Returns:
    - dataloader: DataLoader containing the processed features
    - original_df: The original dataframe for later use
    """
    # Load model first to get PCA components
    checkpoint = torch.load(model_file, weights_only=False)
    pca_components = checkpoint['pca_components']
    print(f"Model has {pca_components} PCA components")

    # Load dataset
    original_df = pd.read_csv(data_file)
    print(f"Dataset loaded with shape: {original_df.shape}")

    original_df.columns = [col.replace('demographics_', '') for col in original_df.columns]

    # Extract features (assuming all columns are features since there's no target)
    X = original_df.copy()
    X.columns = [col.replace('demographics_', '') for col in X.columns]
    print(f"Features shape: {X.shape}")

    # Apply standardization
    if self_normalize:
        # TODO: Implement z-score normalization using StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.values)
        print("Applied self normalization to features")
    else:
        X_scaled = X.values
        print("Did not normalize X (already normalized)")

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
    dataset = BikeDataset(X_pca)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return dataloader, original_df


def predict_with_model(model_file, dataloader, use_log=True):
    """
    Make predictions using the loaded model

    Parameters:
    - model_file: Path to the model file (.pth)
    - dataloader: DataLoader containing the processed features
    - use_log: Whether to use log transformation for prediction

    Returns:
    - all_predictions: List of predictions
    """
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
        for inputs in dataloader:
            outputs = model(inputs)
            all_predictions.extend(outputs.numpy().flatten())

    print(f"Generated {len(all_predictions)} predictions")

    # Convert back from log scale if needed
    if use_log:
        all_predictions = np.expm1(all_predictions)

    return all_predictions


def save_predictions_to_csv(original_df, predictions, output_file):
    """
    Save predictions to a CSV file

    Parameters:
    - original_df: Original dataframe
    - predictions: List of predictions
    - output_file: Path to save the output CSV
    """
    # Create a copy of the original dataframe
    result_df = original_df.copy()

    # Add predictions as a new column
    result_df['bike_volume'] = predictions

    # Save to CSV
    result_df.to_csv(output_file, index=False)
    print(f"Predictions saved to '{output_file}'")


def main():
    # Configuration
    data_folder = 'atl_data'  # Folder containing Atlanta datasets
    model_file1 = 'nn_finalmodel_1.pth'  # First model file
    model_file2 = 'nn_finalmodel_2.pth'  # Second model file
    use_log = True  # Same as in training

    # Output files
    output_file1 = 'atl_preds_model1.csv'
    output_file2 = 'atl_preds_model2.csv'
    output_file_avg = 'atl_preds_avg.csv'

    # Check if model files exist
    print(f"Checking if model files exist:")
    print(f"  Model 1 exists: {os.path.exists(model_file1)}")
    print(f"  Model 2 exists: {os.path.exists(model_file2)}")

    # Find all dataset files in the data folder
    data_files = glob.glob(os.path.join(data_folder, '*.csv'))

    if not data_files:
        print(f"No dataset files found in '{data_folder}'")
        return

    print(f"Found {len(data_files)} dataset files:")
    for data_file in data_files:
        print(f"  - {os.path.basename(data_file)}")

    # Process each dataset with both models
    for data_file in data_files:
        file_name = os.path.basename(data_file)
        is_normalized = 'normalized' in file_name.lower()

        print(f"\n{'=' * 80}")
        print(f"Processing dataset: {file_name}")
        print(f"Is normalized: {is_normalized}")
        print(f"{'=' * 80}")

        # Process with Model 1
        print("\nProcessing with Model 1...")
        dataloader1, original_df = load_and_process_data(
            data_file, model_file1, use_log=use_log, self_normalize=not is_normalized)

        predictions1 = predict_with_model(model_file1, dataloader1, use_log=use_log)

        # Save Model 1 predictions
        model1_output = f"{os.path.splitext(output_file1)[0]}_{os.path.splitext(file_name)[0]}.csv"
        save_predictions_to_csv(original_df, predictions1, model1_output)

        # Process with Model 2
        print("\nProcessing with Model 2...")
        dataloader2, _ = load_and_process_data(
            data_file, model_file2, use_log=use_log, self_normalize=not is_normalized)

        predictions2 = predict_with_model(model_file2, dataloader2, use_log=use_log)

        # Save Model 2 predictions
        model2_output = f"{os.path.splitext(output_file2)[0]}_{os.path.splitext(file_name)[0]}.csv"
        save_predictions_to_csv(original_df, predictions2, model2_output)

        # Average the predictions and save
        print("\nAveraging predictions...")
        avg_predictions = [(p1 + p2) / 2.0 for p1, p2 in zip(predictions1, predictions2)]

        # Save average predictions
        avg_output = f"{os.path.splitext(output_file_avg)[0]}_{os.path.splitext(file_name)[0]}.csv"
        save_predictions_to_csv(original_df, avg_predictions, avg_output)

        print(f"\nProcessing complete for {file_name}")

    print("\nAll datasets processed successfully.")


if __name__ == "__main__":
    main()