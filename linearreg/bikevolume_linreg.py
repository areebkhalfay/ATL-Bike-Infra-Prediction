import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_regression
import joblib  # For saving the sklearn model

def prepare_data_sklearn(file_path, use_log=True, select_features=25):
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

def train_linear_regression(X_train, X_test, y_train, y_test, use_log):
    """
    Train and evaluate scikit-learn Linear Regression model
    """
    # Create and train model
    lr_model = LinearRegression(fit_intercept=True)
    lr_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = lr_model.predict(X_train)
    y_pred_test = lr_model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'train': {
            'r2': r2_score(y_train, y_pred_train),
            'mae': mean_absolute_error(y_train, y_pred_train),
            'mse': mean_squared_error(y_train, y_pred_train)
        },
        'test': {
            'r2': r2_score(y_test, y_pred_test),
            'mae': mean_absolute_error(y_test, y_pred_test),
            'mse': mean_squared_error(y_test, y_pred_test)
        }
    }
    
    # If using log-transform, calculate metrics in original scale
    if use_log:
        y_train_orig = np.expm1(y_train)
        y_test_orig = np.expm1(y_test)
        y_pred_train_orig = np.expm1(y_pred_train)
        y_pred_test_orig = np.expm1(y_pred_test)
        
        metrics['train_original_scale'] = {
            'mae': mean_absolute_error(y_train_orig, y_pred_train_orig),
            'mse': mean_squared_error(y_train_orig, y_pred_train_orig)
        }
        metrics['test_original_scale'] = {
            'mae': mean_absolute_error(y_test_orig, y_pred_test_orig),
            'mse': mean_squared_error(y_test_orig, y_pred_test_orig)
        }
    
    return lr_model, metrics

def visualize_metrics(metrics, use_log=True):
    """Visualize comparison between train and test metrics"""
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    
    # Log-scale metrics
    if use_log:
        ax[0].bar(['Train', 'Test'], [metrics['train']['mae'], metrics['test']['mae']], color=['blue', 'orange'])
        ax[0].set_title('MAE (log scale)')
        ax[1].bar(['Train', 'Test'], [metrics['train']['r2'], metrics['test']['r2']], color=['blue', 'orange'])
        ax[1].set_title('R² Score (log scale)')
    
    # Original-scale metrics
    if 'train_original_scale' in metrics:
        fig2, ax2 = plt.subplots(1, 2, figsize=(15, 6))
        ax2[0].bar(['Train', 'Test'], 
                  [metrics['train_original_scale']['mae'], 
                   metrics['test_original_scale']['mae']], 
                  color=['blue', 'orange'])
        ax2[0].set_title('MAE (original scale)')
        ax2[1].bar(['Train', 'Test'], 
                  [metrics['train_original_scale']['mse'], 
                   metrics['test_original_scale']['mse']], 
                  color=['blue', 'orange'])
        ax2[1].set_title('MSE (original scale)')
        
    plt.tight_layout()
    plt.show()
    plt.savefig('linear_regression_metrics_comparison.png')
    print("Metrics visualized and saved as linear_regression_metrics_comparison.png")

def main():
    # Configuration
    file_path = 'merged_nyc_grid_data.csv'
    use_log = True
    select_features = 25
    
    # Prepare data
    X_train, X_test, y_train, y_test, use_log = prepare_data_sklearn(
        file_path, use_log, select_features
    )
    
    # Train model
    model, metrics = train_linear_regression(X_train, X_test, y_train, y_test, use_log)
    
    # Print metrics
    print("\nLog-scale Metrics:")
    print(f"Train R²: {metrics['train']['r2']:.4f}, Test R²: {metrics['test']['r2']:.4f}")
    print(f"Train MAE: {metrics['train']['mae']:.4f}, Test MAE: {metrics['test']['mae']:.4f}")
    
    if use_log:
        print("\nOriginal-scale Metrics:")
        print(f"Train MAE: {metrics['train_original_scale']['mae']:.2f}, Test MAE: {metrics['test_original_scale']['mae']:.2f}")
        print(f"Train MSE: {metrics['train_original_scale']['mse']:.2f}, Test MSE: {metrics['test_original_scale']['mse']:.2f}")
    
    # Save model
    joblib.dump(model, 'linear_regression_model.pkl')
    print("\nModel saved to linear_regression_model.pkl")
    
    # Visualize metrics
    visualize_metrics(metrics, use_log)

if __name__ == "__main__":
    main()