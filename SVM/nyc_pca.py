import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib
import os
import matplotlib.pyplot as plt

"""
This script runs PCA on the NYC dataset and outputs the reduced dataset and the PCA parameters.
It then applies the same PCA parameters to the ATL dataset for deployment
NOTE: This is LLM-generated and needs review. 
"""

##########################################
###### PART 1: TRAINING DATA PCA ########
##########################################

def train_pca(training_csv_path, target_column=None, n_components=None, variance_threshold=0.95,
              output_dir='pca_output'):
    """
    Perform PCA on training data and save the PCA model for later use

    Parameters:
    -----------
    training_csv_path : str
        Path to the CSV file containing training data
    target_column : str or None
        Name of the target column. If None, assumes all columns are features
    n_components : int or None
        Number of PCA components to keep. If None, will use variance_threshold
    variance_threshold : float
        Amount of variance to retain when n_components is None (between 0 and 1)
    output_dir : str
        Directory to save PCA model and related files

    Returns:
    --------
    X_pca : numpy.ndarray
        PCA-transformed training data
    """
    print(f"Loading training data from {training_csv_path}...")
    df = pd.read_csv(training_csv_path)

    # Separate features and target if target_column is specified
    if target_column and target_column in df.columns:
        y = df[target_column].copy()
        X = df.drop(columns=[target_column])
    else:
        X = df.copy()
        y = None

    # Check for non-numeric columns and handle them
    non_numeric_cols = X.select_dtypes(exclude=np.number).columns.tolist()
    if non_numeric_cols:
        print(f"Warning: Dropping non-numeric columns: {non_numeric_cols}")
        X = X.select_dtypes(include=np.number)

    # Handle missing values
    if X.isnull().sum().sum() > 0:
        print("Warning: Dataset contains missing values. Filling with mean values.")
        X = X.fillna(X.mean())

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save column names for later use
    feature_names = X.columns.tolist()
    with open(os.path.join(output_dir, 'feature_names.txt'), 'w') as f:
        f.write('\n'.join(feature_names))

    # Standardize the features
    print("Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save the scaler
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))

    # Determine optimal number of components if n_components is not specified
    if n_components is None:
        # Start with a PCA that keeps all components
        temp_pca = PCA()
        temp_pca.fit(X_scaled)

        # Calculate cumulative explained variance
        cumulative_variance = np.cumsum(temp_pca.explained_variance_ratio_)

        # Find the number of components that explain the desired variance
        n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
        print(
            f"Determined optimal number of components: {n_components} (explaining {variance_threshold * 100:.1f}% of variance)")

        # Create a plot of explained variance
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
        plt.axhline(y=variance_threshold, color='r', linestyle='-')
        plt.axvline(x=n_components, color='g', linestyle='--')
        plt.title('Explained Variance by Components')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'explained_variance.png'))

    # Perform PCA with the selected number of components
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # Save the PCA model
    joblib.dump(pca, os.path.join(output_dir, 'pca_model.pkl'))

    # Save the transformed data with target if available
    if y is not None:
        pca_df = pd.DataFrame(X_pca, columns=[f'PC{i + 1}' for i in range(n_components)])
        pca_df[target_column] = y.values
        pca_df.to_csv(os.path.join(output_dir, 'pca_training_data.csv'), index=False)
    else:
        pca_df = pd.DataFrame(X_pca, columns=[f'PC{i + 1}' for i in range(n_components)])
        pca_df.to_csv(os.path.join(output_dir, 'pca_training_data.csv'), index=False)

    # Print component information
    print(f"PCA completed. Reduced dimensions from {X.shape[1]} to {n_components}.")
    print(f"Total explained variance: {sum(pca.explained_variance_ratio_) * 100:.2f}%")

    # Save feature importance/contribution to components
    component_df = pd.DataFrame(
        pca.components_,
        columns=feature_names,
        index=[f'PC{i + 1}' for i in range(n_components)]
    )
    component_df.to_csv(os.path.join(output_dir, 'pca_components.csv'))

    return X_pca, y


##########################################
###### PART 2: DEPLOYMENT DATA PCA ######
##########################################

def apply_pca_transform(deployment_csv_path, target_column=None, pca_dir='pca_output'):
    """
    Apply the pre-trained PCA transformation to new deployment data

    Parameters:
    -----------
    deployment_csv_path : str
        Path to the CSV file containing deployment data
    target_column : str or None
        Name of the target column to exclude from transformation
    pca_dir : str
        Directory containing the saved PCA model and related files

    Returns:
    --------
    X_pca : numpy.ndarray
        PCA-transformed deployment data
    """
    print(f"Loading deployment data from {deployment_csv_path}...")
    df = pd.read_csv(deployment_csv_path)

    # Load the original feature names
    with open(os.path.join(pca_dir, 'feature_names.txt'), 'r') as f:
        original_features = [line.strip() for line in f.readlines()]

    # Separate target if specified
    if target_column and target_column in df.columns:
        y = df[target_column].copy()
        df = df.drop(columns=[target_column])
    else:
        y = None

    # Check if all required features are present
    missing_features = set(original_features) - set(df.columns)
    if missing_features:
        raise ValueError(f"Deployment data is missing these features: {missing_features}")

    # Select only the features used during training
    X = df[original_features].copy()

    # Handle missing values
    if X.isnull().sum().sum() > 0:
        print("Warning: Deployment dataset contains missing values. Filling with mean values.")
        X = X.fillna(X.mean())

    # Load the scaler and PCA model
    scaler = joblib.load(os.path.join(pca_dir, 'scaler.pkl'))
    pca = joblib.load(os.path.join(pca_dir, 'pca_model.pkl'))

    # Apply the same transformations
    print("Applying standardization and PCA transformation...")
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)

    # Save the transformed data
    n_components = X_pca.shape[1]
    pca_df = pd.DataFrame(X_pca, columns=[f'PC{i + 1}' for i in range(n_components)])

    if y is not None:
        pca_df[target_column] = y.values

    output_file = os.path.join(pca_dir, 'pca_deployment_data.csv')
    pca_df.to_csv(output_file, index=False)
    print(f"Transformed deployment data saved to {output_file}")

    return X_pca, y


if __name__ == "__main__":
    # Example usage

    # 1. Training phase
    train_pca(
        training_csv_path='training_data.csv',  # Replace with your training data path
        target_column='target',  # Replace with your target column name or None
        n_components=None,  # Set to a number or None to use variance threshold
        variance_threshold=0.95,  # Adjust based on your needs
        output_dir='pca_output'  # Directory to save models and results
    )

    # 2. Deployment phase
    apply_pca_transform(
        deployment_csv_path='deployment_data.csv',  # Replace with your deployment data path
        target_column='target',  # Replace with your target column name or None
        pca_dir='pca_output'  # Directory with saved PCA model
    )