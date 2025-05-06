import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_regression

# Load your neural network architecture
class BikeVolumeNN(torch.nn.Module):
    def __init__(self, input_dim):
        super(BikeVolumeNN, self).__init__()
        self.model = nn.Sequential(

            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64,momentum=0.9),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32, momentum=0.9),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16, momentum=0.9),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(16, 1) # Output: Predicted bike volume (number of rides in and out of the grid location
        )

    def forward(self, x):
        return self.model(x)
    
def prepare_data(file_path, use_log=True, select_features=25):
    """
    Modified data preparation for scikit-learn models
    Returns numpy arrays instead of DataLoaders
    """
    df = pd.read_csv(file_path)
    
    # Create log-transformed target if specified
    y = np.log1p(df['bike_volume']) if use_log else df['bike_volume']
    X = df.drop('bike_volume', axis=1)
    
    # Feature selection
    if select_features > 0:
        selector = SelectKBest(f_regression, k=select_features)
        X = selector.fit_transform(X, y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, use_log

def load_nn_model(model_path, input_dim):
    """Load trained neural network"""
    model = BikeVolumeNN(input_dim)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def compare_models():
    # Configuration
    data_path = 'merged_nyc_grid_data.csv'
    nn_model_path = 'bike_volume_model_run3.pth'
    use_log = True
    select_features = 25

    # Prepare data
    X_train, X_test, y_train, y_test, use_log = prepare_data(
        data_path, use_log, select_features
    )

    # Train Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)

    # Load and evaluate Neural Network
    nn_model = load_nn_model(nn_model_path, X_train.shape[1])
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        nn_pred = nn_model(X_test_tensor).numpy().flatten()

    # Convert back to original scale if using log
    if use_log:
        y_test_orig = np.expm1(y_test)
        lr_pred_orig = np.expm1(lr_pred)
        nn_pred_orig = np.expm1(nn_pred)

    # Calculate metrics
    metrics = {
        'Linear Regression': {
            'log': {
                'R²': r2_score(y_test, lr_pred),
                'MAE': mean_absolute_error(y_test, lr_pred),
                'MSE': mean_squared_error(y_test, lr_pred)
            },
            'original': {
                'MAE': mean_absolute_error(y_test_orig, lr_pred_orig),
                'MSE': mean_squared_error(y_test_orig, lr_pred_orig)
            } if use_log else None
        },
        'Neural Network': {
            'log': {
                'R²': r2_score(y_test, nn_pred),
                'MAE': mean_absolute_error(y_test, nn_pred),
                'MSE': mean_squared_error(y_test, nn_pred)
            },
            'original': {
                'MAE': mean_absolute_error(y_test_orig, nn_pred_orig),
                'MSE': mean_squared_error(y_test_orig, nn_pred_orig)
            } if use_log else None
        }
    }

    # Plotting
    plt.figure(figsize=(15, 10))
    
    # Metrics comparison
    metrics_to_plot = ['R²', 'MAE', 'MSE']
    for i, metric in enumerate(metrics_to_plot, 1):
        plt.subplot(2, 3, i)
        plt.bar(['Linear', 'NN'], 
                [metrics['Linear Regression']['log'][metric], 
                 metrics['Neural Network']['log'][metric]])
        plt.title(f'{metric} (log scale)')
        plt.ylabel(metric)

    if use_log:
        orig_metrics = ['MAE', 'MSE']
        for i, metric in enumerate(orig_metrics, 4):
            plt.subplot(2, 3, i)
            plt.bar(['Linear', 'NN'], 
                   [metrics['Linear Regression']['original'][metric], 
                    metrics['Neural Network']['original'][metric]])
            plt.title(f'{metric} (original scale)')
            plt.ylabel(metric)

    plt.tight_layout()
    plt.suptitle('Model Performance Comparison', y=1.02)
    plt.show()

    # Prediction scatter plots
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, lr_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual (log)')
    plt.ylabel('Predicted (log)')
    plt.title('Linear Regression Predictions')
    
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, nn_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual (log)')
    plt.ylabel('Predicted (log)')
    plt.title('Neural Network Predictions')
    
    plt.tight_layout()
    plt.show()

    # Print metrics
    print("Linear Regression Metrics:")
    print(f"Log Scale - R²: {metrics['Linear Regression']['log']['R²']:.3f}, MAE: {metrics['Linear Regression']['log']['MAE']:.3f}, MSE: {metrics['Linear Regression']['log']['MSE']:.3f}")
    if use_log:
        print(f"Original Scale - MAE: {metrics['Linear Regression']['original']['MAE']:.1f}, MSE: {metrics['Linear Regression']['original']['MSE']:.1f}")
    
    print("\nNeural Network Metrics:")
    print(f"Log Scale - R²: {metrics['Neural Network']['log']['R²']:.3f}, MAE: {metrics['Neural Network']['log']['MAE']:.3f}, MSE: {metrics['Neural Network']['log']['MSE']:.3f}")
    if use_log:
        print(f"Original Scale - MAE: {metrics['Neural Network']['original']['MAE']:.1f}, MSE: {metrics['Neural Network']['original']['MSE']:.1f}")

if __name__ == "__main__":
    compare_models()