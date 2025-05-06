import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import dropout
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import wandb
import os
"""
This script:
1. Defines the BikeVolumeNN model
2. Loads the NYC training dataset (population features + bike volumes) from a .csv file
3. Applies PCA to the features, keeping components that explain 95% of variance
4. Saves PCA parameters to CSV for future use
5. Trains and saves the model  
6. Uses average of last 20 epochs for metrics to reduce noise
"""


class BikeVolumeNN(nn.Module):
    """
    BikeVolumeNN: Neural network that predicts a single continuous value: bike volume, which is the number of bike
    rides that start or end in a particular grid location.

    The input dataset consists of rows which correspond to a
    single grid location in the city, and columns which correspond to various features of that grid location
    (population density, age and socioeconomic demographics, whether the location is close to a tourist site, etc.)
    """

    def __init__(self, input_dim, num_layers = 2, dropout_percent = 0.3):
        super(BikeVolumeNN, self).__init__()

        assert num_layers in [2,3]
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
                nn.Dropout(self.dropout_percent-0.1),

                nn.Linear(16, 1)  # Output: Predicted bike volume (number of rides in and out of the grid location
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
                nn.Dropout(self.dropout_percent-0.1),

                nn.Linear(16, 1)  # Output: Predicted bike volume (number of rides in and out of the grid location
            )

    def forward(self, x):
        return self.model(x)



# Custom Dataset class
class BikeDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


# Data Loading and Processing
def prepare_data(file_path, batch_size=64, use_log=True, pca_variance_threshold=0.95):
    """
    Load the NYC data from a csv file, apply PCA to retain components explaining a specified proportion of variance,
    save PCA parameters, split into test and training set, and pass it into a dataloader for training
    
    Parameters:
    - file_path: Path to CSV file containing the dataset
    - batch_size: Batch size for training
    - use_log: Whether to log-transform the target variable
    - pca_variance_threshold: Proportion of variance to retain (e.g., 0.95 for 95%)
    
    Returns:
    - train_loader: DataLoader for training data
    - test_loader: DataLoader for test data
    - input_dim: Number of input features (PCA components)
    - pca_components: Number of PCA components used
    """

    df = pd.read_csv(file_path)
    print('Dataset loaded')
    print('Dataset shape: ', df.shape)

    # Check for non-numeric columns
    print("DataFrame info:")
    print(df.info())

    # Check for NaN values
    print("\nNaN values per column:")
    print(df.isna().sum())

    # Check column dtypes
    print("\nColumn dtypes:")
    print(df.dtypes)

    # Extract features and target
    X = df.drop('bike_volume', axis=1)
    
    # Transform target if using log
    if use_log:
        y = np.log1p(df['bike_volume'])
    else:
        y = df['bike_volume']
    
    # Split data before applying PCA to avoid data leakage
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Store original feature names
    feature_names = X.columns.tolist()
    
    # Standardize the data before PCA
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    
    # Apply PCA to retain components explaining the specified proportion of variance
    pca = PCA(n_components=pca_variance_threshold)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    # Get number of components used
    n_components = pca.n_components_
    print(f"Number of PCA components that explain {pca_variance_threshold*100:.1f}% of variance: {n_components}")
    
    # Calculate explained variance per component
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    # Save PCA parameters to CSV
    pca_params = {
        'Feature': feature_names,
    }
    
    # Add component loadings
    for i in range(n_components):
        pca_params[f'Component_{i+1}'] = pca.components_[i]
    
    # Create DataFrame and add explained variance as a separate row
    pca_df = pd.DataFrame(pca_params)
    
    # Save component variances as a separate CSV
    variance_df = pd.DataFrame({
        'Component': [f'Component_{i+1}' for i in range(n_components)],
        'Explained_Variance_Ratio': explained_variance_ratio,
        'Cumulative_Variance': cumulative_variance
    })
    
    # Save PCA parameters and variances
    pca_df.to_csv('pca_parameters.csv', index=False)
    variance_df.to_csv('pca_variance.csv', index=False)
    
    # Save scaler parameters
    scaler_params = pd.DataFrame({
        'Feature': feature_names,
        'Mean': scaler.mean_,
        'Scale': scaler.scale_
    })
    scaler_params.to_csv('scaler_parameters.csv', index=False)
    
    print(f"PCA parameters saved to 'pca_parameters.csv'")
    print(f"PCA variance information saved to 'pca_variance.csv'")
    print(f"Scaler parameters saved to 'scaler_parameters.csv'")
    
    # Create datasets
    train_dataset = BikeDataset(X_train_pca, y_train.values)
    test_dataset = BikeDataset(X_test_pca, y_test.values)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    input_dim = X_train_pca.shape[1]
    print('Input dim (number of PCA components):', input_dim)
    
    return train_loader, test_loader, input_dim, n_components


def train_model(model, train_loader, test_loader, num_epochs=100, learning_rate=0.001, weight_decay=0.001, patience=5000, window_size=20, pca_components=None, batch_size=None, dropout=None):
    """
    Train model and track metrics, using a window of recent epochs for smoothing results
    
    Parameters:
    - window_size: Number of epochs to average for stable metrics (default: 20)
    - pca_components: Number of PCA components used (for logging purposes)
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay)

    train_losses = []
    test_losses = []
    train_r2_scores = []
    test_r2_scores = []
    train_mae_scores = []
    test_mae_scores = []

    # Early stopping vars
    best_test_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    run = wandb.init(
    entity="ryanbowers166-gt",
    project="ml-bike-project",
    name = "pca4_lr"+str(learning_rate)+"_"+str(model.num_layers)+"layers_"+str(weight_decay)+"wd_"+str(batch_size)+"bs_"+str(dropout)+"dropout_"+str(pca_components)+"pca",
    config={
        "learning_rate": learning_rate,
        "num_layers": model.num_layers,
        "epochs": num_epochs,
        "weight_decay": weight_decay,
        "patience": patience,
        "pca_components": pca_components,
        "batch_size": batch_size,
        "dropout": dropout,
    },
)

    # Training loop
    for epoch in range(num_epochs):

        # 1. Train
        model.train()
        running_loss = 0.0
        all_train_preds = []
        all_train_targets = []

        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # For accuracy progress:
            all_train_preds.extend(outputs.detach().numpy())
            all_train_targets.extend(targets.numpy())

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        train_r2 = compute_r2_score(all_train_targets, all_train_preds)
        train_mae = compute_mae(all_train_targets, all_train_preds)
        train_r2_scores.append(train_r2)
        train_mae_scores.append(train_mae)

        # 2. Evaluate
        model.eval()
        running_test_loss = 0.0
        all_test_preds = []
        all_test_targets = []

        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                test_loss = criterion(outputs, targets)
                running_test_loss += test_loss.item()

                # For accuracy calculation
                all_test_preds.extend(outputs.numpy())
                all_test_targets.extend(targets.numpy())

        avg_test_loss = running_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        test_r2 = compute_r2_score(all_test_targets, all_test_preds)
        test_mae = compute_mae(all_test_targets, all_test_preds)
        test_r2_scores.append(test_r2)
        test_mae_scores.append(test_mae)
        
        # Calculate window averages for more stable metric reporting
        window_train_loss = np.mean(train_losses[-min(window_size, len(train_losses)):])
        window_test_loss = np.mean(test_losses[-min(window_size, len(test_losses)):])
        window_train_r2 = np.mean(train_r2_scores[-min(window_size, len(train_r2_scores)):])
        window_test_r2 = np.mean(test_r2_scores[-min(window_size, len(test_r2_scores)):])
        window_train_mae = np.mean(train_mae_scores[-min(window_size, len(train_mae_scores)):])
        window_test_mae = np.mean(test_mae_scores[-min(window_size, len(test_mae_scores)):])

        # Early stopping check - use window average for more stable stopping
        if window_test_loss < best_test_loss:
            best_test_loss = window_test_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        # Log to Wandb
        run.log({
            "window_train_loss": window_train_loss,
            "window_test_loss": window_test_loss,
            "window_train_r2": window_train_r2,
            "window_test_r2": window_test_r2,
            "window_train_mae": window_train_mae,
            "window_test_mae": window_test_mae
            })

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], '
                  f'Train Loss: {avg_train_loss:.4f} (Window: {window_train_loss:.4f}), '
                  f'Test Loss: {avg_test_loss:.4f} (Window: {window_test_loss:.4f}), '
                  f'Train R²: {train_r2:.4f} (Window: {window_train_r2:.4f}), '
                  f'Test R²: {test_r2:.4f} (Window: {window_test_r2:.4f}), '
                  f'Train MAE: {train_mae:.4f} (Window: {window_train_mae:.4f}), '
                  f'Test MAE: {test_mae:.4f} (Window: {window_test_mae:.4f})')

        # If patience is exceeded, stop training
        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            # Load the best model state
            model.load_state_dict(best_model_state)
            break

    # Calculate final averaged metrics over window_size
    final_metrics = {
        'train_loss': np.mean(train_losses[-window_size:]),
        'test_loss': np.mean(test_losses[-window_size:]),
        'train_r2': np.mean(train_r2_scores[-window_size:]),
        'test_r2': np.mean(test_r2_scores[-window_size:]),
        'train_mae': np.mean(train_mae_scores[-window_size:]),
        'test_mae': np.mean(test_mae_scores[-window_size:])
    }

    run.finish()

    return model, train_losses, test_losses, train_r2_scores, test_r2_scores, train_mae_scores, test_mae_scores, final_metrics


# Helper functions
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


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    # Hyperparameters for sweep
    layers = [3]
    dropout_percents = [0.4]
    learning_rates = [0.00005]
    batch_sizes = [64]
    num_epochs = 28000
    weight_decays = [0.003]
    window_size = 20  # Number of epochs to average for metrics

    use_log = True  # Whether to use log(target) instead of target
    file_path = 'merged_nyc_grid_data.csv'
    pca_variance_threshold = 0.95  # PCA threshold that explains 95% of variance

    num_conditions = len(layers) * len(dropout_percents) * len(learning_rates) * len(batch_sizes) * len(weight_decays)

    # Store results for comparison
    all_results = []
    
    print('Beginning sweep with ', num_conditions, ' conditions')
    print(f'Using {window_size}-epoch window for averaging metrics')
    print(f'Using PCA with {pca_variance_threshold*100:.1f}% variance threshold')

    # Perform grid search
    for num_layers in layers:
        for lr in learning_rates:
            for batch_size in batch_sizes:
                for weight_decay in weight_decays:
                    for dropout_percent in dropout_percents:
                        print(f"\n{'=' * 80}")
                        print(
                            f"Training configuration: Num layers={num_layers}, LR={lr}, Batch={batch_size}, PCA threshold={pca_variance_threshold}")
                        print(f"{'=' * 80}")

                        # Prepare data with PCA
                        train_loader, test_loader, input_dim, pca_components = prepare_data(
                            file_path, 
                            batch_size, 
                            use_log, 
                            pca_variance_threshold
                        )

                        # Instantiate model based on network type
                        model = BikeVolumeNN(input_dim, num_layers=num_layers, dropout_percent=dropout_percent)

                        # Train model
                        trained_model, train_losses, test_losses, train_r2_scores, test_r2_scores, train_mae_scores, test_mae_scores, final_metrics = train_model(
                            model, train_loader, test_loader, 
                            num_epochs=num_epochs, 
                            learning_rate=lr, 
                            weight_decay=weight_decay, 
                            pca_components=pca_components, 
                            batch_size=batch_size, 
                            patience=10000, 
                            window_size=window_size,
                            dropout=dropout_percent
                        )

                        # Generate descriptive filename
                        config_str = f"net_{num_layers}layers_lr_{lr:.8f}_batch_{batch_size}_pca_{pca_components}"
                        model_filename = f"pca4_bike_volume_model_{config_str}.pth"
                        plot_filename = f"pca4_training_progress_{config_str}.png"

                        # Save model with averaged metrics
                        torch.save({
                            'model_state_dict': trained_model.state_dict(),
                            'input_dim': input_dim,
                            'network_layers': num_layers,
                            'learning_rate': lr,
                            'batch_size': batch_size,
                            'pca_components': pca_components,
                            'epochs': num_epochs,
                            'window_size': window_size,
                            'final_test_r2': final_metrics['test_r2'],  # Averaged over window_size
                            'final_test_mae': final_metrics['test_mae'],  # Averaged over window_size
                            'final_test_loss': final_metrics['test_loss']  # Averaged over window_size
                        }, model_filename)
                        print(f"Model saved to '{model_filename}'")

                        # Visualize and save training progress
                        fig, axes = plt.subplots(5, 1, figsize=(12, 25))  # Added an extra subplot for window averages

                        # Add configuration as title
                        fig.suptitle(
                            f"Num_layers: {num_layers}, Learning Rate: {lr}, Batch Size: {batch_size}, PCA Components: {pca_components}",
                            fontsize=16
                        )

                        # Plot Loss
                        axes[0].plot(train_losses, label='Training Loss', alpha=0.5)
                        axes[0].plot(test_losses, label='Validation Loss', alpha=0.5)
                        axes[0].set_xlabel('Epochs')
                        axes[0].set_ylabel('Loss (MSE)')
                        axes[0].set_title('Training and Validation Loss (Raw)')
                        axes[0].legend()

                        # Plot R² Score
                        axes[1].plot(train_r2_scores, label='Training R²', alpha=0.5)
                        axes[1].plot(test_r2_scores, label='Validation R²', alpha=0.5)
                        axes[1].set_xlabel('Epochs')
                        axes[1].set_ylabel('R² Score')
                        axes[1].set_title('Training and Validation R² Score (Raw)')
                        axes[1].legend()

                        # Plot R² Score (Zoomed)
                        axes[2].plot(train_r2_scores, label='Training R²', alpha=0.5)
                        axes[2].plot(test_r2_scores, label='Validation R²', alpha=0.5)
                        axes[2].set_xlabel('Epochs')
                        axes[2].set_ylabel('R² Score')
                        axes[2].set_title('Training and Validation R² Score (Zoomed)')
                        axes[2].set_ylim(0, 1)
                        axes[2].legend()

                        # Plot MAE
                        axes[3].plot(train_mae_scores, label='Training MAE', alpha=0.5)
                        axes[3].plot(test_mae_scores, label='Validation MAE', alpha=0.5)
                        axes[3].set_xlabel('Epochs')
                        axes[3].set_ylabel('Mean Absolute Error')
                        axes[3].set_title('Training and Validation MAE (Raw)')
                        axes[3].legend()
                        
                        # Plot smoothed metrics with moving average
                        ma_train_r2 = np.convolve(train_r2_scores, np.ones(window_size)/window_size, mode='valid')
                        ma_test_r2 = np.convolve(test_r2_scores, np.ones(window_size)/window_size, mode='valid')
                        ma_train_loss = np.convolve(train_losses, np.ones(window_size)/window_size, mode='valid')
                        ma_test_loss = np.convolve(test_losses, np.ones(window_size)/window_size, mode='valid')
                        
                        axes[4].plot(range(window_size-1, len(train_r2_scores)), ma_train_r2, 
                                    label=f'Training R² ({window_size}-epoch avg)')
                        axes[4].plot(range(window_size-1, len(test_r2_scores)), ma_test_r2, 
                                    label=f'Validation R² ({window_size}-epoch avg)')
                        axes[4].set_xlabel('Epochs')
                        axes[4].set_ylabel('R² Score (Moving Average)')
                        axes[4].set_title(f'Smoothed R² Score ({window_size}-epoch Moving Average)')
                        axes[4].set_ylim(0, 1)
                        axes[4].legend()

                        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for the suptitle
                        plt.savefig(plot_filename)
                        print(f"Training visualization saved to '{plot_filename}'")

                        # Store results for comparison
                        result = {
                            "pca":True,
                            'num_layers': num_layers,
                            'learning_rate': lr,
                            'batch_size': batch_size,
                            'dropout_percent': dropout_percent,
                            'weight_decay': weight_decay,
                            'pca_components': pca_components,
                            'final_train_loss': final_metrics['train_loss'],  # Window averaged
                            'final_test_loss': final_metrics['test_loss'],    # Window averaged
                            'final_train_r2': final_metrics['train_r2'],      # Window averaged
                            'final_test_r2': final_metrics['test_r2'],        # Window averaged
                            'final_train_mae': final_metrics['train_mae'],    # Window averaged
                            'final_test_mae': final_metrics['test_mae'],      # Window averaged
                            'window_size': window_size
                        }
                        all_results.append(result)

    # Create a DataFrame with all results and save it
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('pca_parameter_sweep_results.csv', index=False)
    print(f"\nParameter sweep complete. Results saved to 'parameter_sweep_results.csv' using {window_size}-epoch window averaging and PCA")

    # Print top 5 configurations based on validation R² score
    top_configs = results_df.sort_values('final_test_r2', ascending=False).head(5)
    print("\nTop 5 configurations by validation R² score (averaged over last 20 epochs):")
    print(top_configs[
              ['num_layers', 'learning_rate', 'batch_size', 'dropout_percent', 'pca_components', 'weight_decay', 'final_test_r2', 'final_test_mae']])

if __name__ == "__main__":
    main()