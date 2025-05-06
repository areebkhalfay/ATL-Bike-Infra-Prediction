import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Set seeds for reproducibility
np.random.seed(42)


# Load the dataset
def load_and_split_data(file_path, use_log=True):
    """
    Load the NYC data from a csv file and split it into training and test sets
    using the same approach as in the original script.

    Parameters:
    - file_path: Path to CSV file containing the dataset
    - use_log: Whether to log-transform the target variable

    Returns:
    - X_train_raw: Features for training set
    - X_test_raw: Features for test set
    - y_train: Target for training set
    - y_test: Target for test set
    - feature_names: Names of features
    """

    print("Loading dataset from:", file_path)
    df = pd.read_csv(file_path)
    print('Dataset loaded')
    print('Dataset shape:', df.shape)

    # Extract features and target
    X = df.drop('bike_volume', axis=1)

    # Transform target if using log
    if use_log:
        y = np.log1p(df['bike_volume'])
        print("Using log-transformed target variable")
    else:
        y = df['bike_volume']

    # Store feature names
    feature_names = X.columns.tolist()

    # Split data using the same approach and random state
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    print(f"Train set: {X_train_raw.shape[0]} samples")
    print(f"Test set: {X_test_raw.shape[0]} samples")

    return X_train_raw, X_test_raw, y_train, y_test, feature_names


def analyze_split_balance(X_train, X_test, y_train, y_test, feature_names):
    """
    Analyze and visualize the balance between train and test splits
    for both features and target variable.
    """

    # 1. Target distribution
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    sns.histplot(y_train, kde=True, color='blue', alpha=0.5, label='Train')
    sns.histplot(y_test, kde=True, color='red', alpha=0.5, label='Test')
    plt.title('Target Distribution (bike_volume)')
    plt.xlabel('Log(bike_volume)' if use_log else 'bike_volume')
    plt.ylabel('Frequency')
    plt.legend()

    plt.subplot(1, 2, 2)
    sns.boxplot(data=[y_train, y_test], orient='h')
    plt.yticks([0, 1], ['Train', 'Test'])
    plt.title('Target Distribution - Boxplot')
    plt.xlabel('Log(bike_volume)' if use_log else 'bike_volume')

    plt.tight_layout()
    plt.savefig('target_distribution_comparison.png')
    plt.show()

    # 2. Feature distribution comparison

    # Calculate statistics for both sets
    train_means = X_train.mean()
    test_means = X_test.mean()
    train_std = X_train.std()
    test_std = X_test.std()

    # Create a dataframe for comparison
    comparison_df = pd.DataFrame({
        'Train_Mean': train_means,
        'Test_Mean': test_means,
        'Difference': train_means - test_means,
        'Percent_Diff': (train_means - test_means) / train_means * 100,
        'Train_Std': train_std,
        'Test_Std': test_std
    })

    # Find features with the largest differences
    largest_diff = comparison_df.abs().sort_values('Percent_Diff', ascending=False).head(10)
    print("\nFeatures with largest percentage differences between train and test:")
    print(largest_diff)

    # Plot the top 10 features with largest differences
    plt.figure(figsize=(14, 10))
    top_diff_features = largest_diff.index.tolist()

    for i, feature in enumerate(top_diff_features):
        plt.subplot(5, 2, i + 1)
        sns.histplot(X_train[feature], kde=True, color='blue', alpha=0.5, label='Train')
        sns.histplot(X_test[feature], kde=True, color='red', alpha=0.5, label='Test')
        plt.title(f"{feature}")
        plt.legend()

    plt.tight_layout()
    plt.savefig('top_diff_features_distribution.png')
    plt.show()

    # 3. PCA to visualize overall distribution
    # Standardize data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Apply PCA for visualization
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # Visualize
    plt.figure(figsize=(10, 8))
    plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], alpha=0.5, label='Train', color='blue')
    plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], alpha=0.5, label='Test', color='red')
    plt.title('PCA of Train and Test Sets')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('pca_train_test_comparison.png')
    plt.show()

    # 4. Feature importance analysis
    # Train a simple model to get feature importances
    from sklearn.ensemble import RandomForestRegressor

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Get feature importances
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Plot top 20 features
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importances')
    plt.bar(range(20), importances[indices[:20]], align='center')
    plt.xticks(range(20), [feature_names[i] for i in indices[:20]], rotation=90)
    plt.tight_layout()
    plt.savefig('feature_importances.png')
    plt.show()

    # Print top 20 features and their importances
    print("\nTop 20 features by importance:")
    for i in range(20):
        print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

    # 5. Distribution of target vs key features
    top_features = [feature_names[i] for i in indices[:5]]

    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(top_features):
        plt.subplot(2, 3, i + 1)
        plt.scatter(X_train[feature], y_train, alpha=0.5, label='Train', color='blue')
        plt.scatter(X_test[feature], y_test, alpha=0.5, label='Test', color='red')
        plt.title(f'{feature} vs Target')
        plt.xlabel(feature)
        plt.ylabel('Log(bike_volume)' if use_log else 'bike_volume')
        plt.legend()

    plt.tight_layout()
    plt.savefig('top_features_vs_target.png')
    plt.show()

    # 6. Summary statistics table
    print("\nSummary statistics for target variable:")
    train_stats = {
        'Mean': y_train.mean(),
        'Median': y_train.median(),
        'Std': y_train.std(),
        'Min': y_train.min(),
        'Max': y_train.max(),
    }

    test_stats = {
        'Mean': y_test.mean(),
        'Median': y_test.median(),
        'Std': y_test.std(),
        'Min': y_test.min(),
        'Max': y_test.max(),
    }

    stats_df = pd.DataFrame({
        'Train': train_stats,
        'Test': test_stats,
        'Difference': {k: train_stats[k] - test_stats[k] for k in train_stats},
        'Percent_Diff': {k: (train_stats[k] - test_stats[k]) / train_stats[k] * 100
        if train_stats[k] != 0 else float('nan') for k in train_stats}
    })

    print(stats_df)

    return comparison_df


# Main execution
if __name__ == "__main__":
    # Parameters
    file_path = 'merged_nyc_grid_data.csv'
    use_log = True  # Whether to use log(target) instead of target

    # Load and split data
    X_train, X_test, y_train, y_test, feature_names = load_and_split_data(file_path, use_log)

    # Analyze and visualize the balance
    comparison_results = analyze_split_balance(X_train, X_test, y_train, y_test, feature_names)

    # Save the comparison results
    comparison_results.to_csv('train_test_feature_comparison.csv')
    print("\nAnalysis complete. Check the generated plots and CSV files.")