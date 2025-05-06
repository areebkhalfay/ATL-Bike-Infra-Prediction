import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

"""
Fixed BikeVolumeNN Interpretation Script
- Loads a trained BikeVolumeNN model
- Analyzes how feature importance flows through the network
- Works with PCA-transformed inputs to map back to original features
- Produces visualizations of the most influential features
"""


# BikeVolumeNN model definition
class BikeVolumeNN(torch.nn.Module):
    def __init__(self, input_dim, num_layers=2, dropout_percent=0.3):
        super(BikeVolumeNN, self).__init__()

        assert num_layers in [2, 3]
        assert dropout_percent <= 1.0

        self.num_layers = num_layers
        self.dropout_percent = dropout_percent

        if num_layers == 2:
            self.model = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 32),
                torch.nn.BatchNorm1d(32, momentum=0.9),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout_percent),

                torch.nn.Linear(32, 16),
                torch.nn.BatchNorm1d(16, momentum=0.9),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout_percent - 0.1),

                torch.nn.Linear(16, 1)
            )
        elif num_layers == 3:
            self.model = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 64),
                torch.nn.BatchNorm1d(64, momentum=0.9),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout_percent),

                torch.nn.Linear(64, 32),
                torch.nn.BatchNorm1d(32, momentum=0.9),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout_percent),

                torch.nn.Linear(32, 16),
                torch.nn.BatchNorm1d(16, momentum=0.9),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout_percent - 0.1),

                torch.nn.Linear(16, 1)
            )

    def forward(self, x):
        return self.model(x)


def find_model_file():
    """Find the most recent BikeVolumeNN model file in the current directory"""
    model_files = list(Path('.').glob('*.pth'))

    if not model_files:
        print("No model files found. Please place the model file in the current directory.")
        return None

    # Sort by modification time (most recent first)
    model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    model_path = str(model_files[0])
    print(f"Using model: {model_path}")

    return model_path


def load_model(model_path):
    """Load a BikeVolumeNN model from a checkpoint file"""
    try:
        checkpoint = torch.load(model_path, weights_only=False)

        input_dim = checkpoint['input_dim']
        network_layers = checkpoint.get('network_layers', 3)  # Default to 3 if not specified

        model = BikeVolumeNN(input_dim, num_layers=network_layers)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        print(f"Model loaded successfully:")
        print(f"- Architecture: {network_layers} layers")
        print(f"- Input dimensions: {input_dim}")
        print(f"- PCA components: {checkpoint.get('pca_components', 'N/A')}")
        print(f"- Test R²: {checkpoint.get('final_test_r2', 'N/A'):.4f}")

        return model, checkpoint

    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None


def extract_weights(model):
    """Extract weights from each layer of the model"""
    weights = []

    if model.num_layers == 2:
        # 2-layer architecture
        weights.append(model.model[0].weight.data.numpy())  # Input -> 32
        weights.append(model.model[4].weight.data.numpy())  # 32 -> 16
        weights.append(model.model[8].weight.data.numpy())  # 16 -> 1
    else:
        # 3-layer architecture
        weights.append(model.model[0].weight.data.numpy())  # Input -> 64
        weights.append(model.model[4].weight.data.numpy())  # 64 -> 32
        weights.append(model.model[8].weight.data.numpy())  # 32 -> 16
        weights.append(model.model[12].weight.data.numpy())  # 16 -> 1

    return weights


def analyze_input_importance(weights):
    """
    Calculate the importance of each input feature based on network weights

    This calculates how strongly each input feature influences the first layer
    of neurons, which is a good measure of overall feature importance.
    """
    # Get first layer weights
    first_layer_weights = weights[0]  # Shape: [hidden_neurons, input_features]

    # Calculate importance as the average absolute weight for each input feature
    # across all neurons in the first layer
    importance = np.abs(first_layer_weights).mean(axis=0)

    # Normalize to percentages
    importance = importance / importance.sum() * 100

    return importance


def load_pca_data():
    """Load PCA data if available"""
    # Load PCA parameters
    pca_params = None
    if os.path.exists('pca_parameters.csv'):
        pca_params = pd.read_csv('pca_parameters.csv')
        print(f"Loaded PCA parameters with {pca_params.shape[0]} features")
    else:
        print("PCA parameters file (pca_parameters.csv) not found")

    # Load PCA variance information
    pca_variance = None
    if os.path.exists('pca_variance.csv'):
        pca_variance = pd.read_csv('pca_variance.csv')
        print(f"Loaded PCA variance information for {pca_variance.shape[0]} components")
    else:
        print("PCA variance file (pca_variance.csv) not found")

    return pca_params, pca_variance


def map_pca_to_features(pca_importance, pca_params):
    """
    Map PCA component importance back to original features
    """
    if pca_params is None:
        print("No PCA parameters available - cannot map to original features")
        return None, None

    # Extract feature names
    feature_names = pca_params['Feature'].values

    # Initialize importance array for original features
    feature_importance = np.zeros(len(feature_names))

    # For each PCA component, distribute its importance to features based on loadings
    component_columns = [col for col in pca_params.columns if col.startswith('Component_')]

    for i, component_col in enumerate(component_columns):
        if i >= len(pca_importance):
            break

        # Get component importance
        importance_value = pca_importance[i]

        # Get absolute loadings for this component
        loadings = np.abs(pca_params[component_col].values)

        # Add weighted contribution to feature importance
        feature_importance += loadings * importance_value

    # Return both raw feature importance and feature names
    return feature_importance, feature_names


def plot_importance(values, labels, title, filename, top_n=15):
    """Plot and save feature importance visualization"""
    # Combine values and labels into tuples
    importance_data = list(zip(labels, values))

    # Sort by importance (descending)
    importance_data.sort(key=lambda x: x[1], reverse=True)

    # Take top N items
    importance_data = importance_data[:top_n]

    # Split back into labels and values
    top_labels = [item[0] for item in importance_data]
    top_values = [item[1] for item in importance_data]

    # Create horizontal bar chart
    plt.figure(figsize=(12, 8))
    y_pos = np.arange(len(top_labels))

    # Create bars
    bars = plt.barh(y_pos, top_values)

    # Add data labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.5, bar.get_y() + bar.get_height() / 2,
                 f'{top_values[i]:.2f}%', ha='left', va='center')

    # Add labels and title
    plt.yticks(y_pos, top_labels)
    plt.xlabel('Relative Importance (%)')
    plt.title(title)
    plt.tight_layout()

    # Save figure
    plt.savefig(filename)
    print(f"Saved visualization to {filename}")
    plt.close()

    return importance_data


def analyze_feature_groups(importance_data, feature_patterns):
    """Group features by patterns and analyze importance by group"""
    # Initialize dictionary to track group importance
    group_importance = {group: 0 for group in feature_patterns}
    feature_counts = {group: 0 for group in feature_patterns}

    # Analyze all features (not just top N)
    for feature, importance in importance_data:
        feature = str(feature).lower()
        # Check each pattern
        for group, patterns in feature_patterns.items():
            for pattern in patterns:
                if pattern.lower() in feature:
                    group_importance[group] += importance
                    feature_counts[group] += 1
                    break  # Count each feature only once per group

    # Create results dataframe
    results = []
    for group in feature_patterns:
        if feature_counts[group] > 0:
            results.append({
                'Group': group,
                'Total_Importance': group_importance[group],
                'Feature_Count': feature_counts[group],
                'Avg_Importance': group_importance[group] / feature_counts[group]
            })

    # Convert to DataFrame and sort
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values('Total_Importance', ascending=False)

    return results_df


def plot_groups(groups_df, filename):
    """Plot feature group importance"""
    if groups_df.empty:
        print("No feature groups to plot")
        return

    plt.figure(figsize=(12, 8))

    # Create horizontal bar chart of total importance by group
    plt.barh(groups_df['Group'], groups_df['Total_Importance'])

    plt.xlabel('Total Importance (%)')
    plt.title('Feature Group Importance')
    plt.tight_layout()

    # Save figure
    plt.savefig(filename)
    print(f"Saved group visualization to {filename}")
    plt.close()


def save_importance_csv(feature_names, importance_values, filename):
    """Save feature importance data to CSV"""
    # Combine data
    data = list(zip(feature_names, importance_values))

    # Sort by importance
    data.sort(key=lambda x: x[1], reverse=True)

    # Create DataFrame
    df = pd.DataFrame(data, columns=['Feature', 'Importance'])

    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Saved importance data to {filename}")


def main():
    # Define feature patterns for grouping
    feature_patterns = {
        'Age Demographics': ['age', 'under5', '5to9', '10to14', '15to19', '20to24',
                             '25to29', '30to34', '35to39', '40to44', '45to49',
                             '50to54', '55to59', '60to64', '65to69', '70to74',
                             '75to79', '80to84', 'over85'],
        'Income Demographics': ['income', 'under10k', '10to15k', '15to25k', '25to35k',
                                '35to50k', '50to75k', '75to100k', '100to150k',
                                '150to200k', 'over200k'],
        'Education': ['education', 'highschool', 'somecollege', 'bachelors', 'graduatedegree'],
        'Household': ['household', 'marriedhouseholds', 'cohabitingcouple', 'solomale',
                      'solofemale', 'housingunits', 'owneroccupied', 'renteroccupied'],
        'Transportation': ['vehicle', 'novehicles', '1vehicle', 'lessthan1vehicle'],
        'Population': ['population', 'density', 'male_percent']
    }

    # Find and load model
    model_path = find_model_file()
    if not model_path:
        return

    model, checkpoint = load_model(model_path)
    if model is None:
        return

    # Extract network weights
    print("\nExtracting neural network weights...")
    weights = extract_weights(model)

    # Analyze PCA component importance
    print("\nCalculating PCA component importance...")
    pca_importance = analyze_input_importance(weights)

    # Create component labels
    component_labels = [f"Component_{i + 1}" for i in range(len(pca_importance))]

    # Plot PCA component importance
    print("\nVisualizing PCA component importance...")
    pca_importance_data = plot_importance(
        pca_importance,
        component_labels,
        f"PCA Component Importance - BikeVolumeNN ({model.num_layers} layers)",
        "pca_importance.png",
        top_n=15
    )

    # Load PCA parameters
    print("\nLoading PCA parameters to map back to original features...")
    pca_params, pca_variance = load_pca_data()

    # Map PCA component importance to original features
    print("\nMapping PCA importance to original features...")
    feature_importance, feature_names = map_pca_to_features(pca_importance, pca_params)

    if feature_importance is not None:
        # Plot original feature importance
        print("\nVisualizing original feature importance...")
        feature_importance_data = plot_importance(
            feature_importance,
            feature_names,
            f"Feature Importance - BikeVolumeNN ({model.num_layers} layers)",
            "feature_importance.png",
            top_n=20
        )

        # Save full feature importance data
        importance_data = list(zip(feature_names, feature_importance))
        save_importance_csv(feature_names, feature_importance, "feature_importance.csv")

        # Analyze feature groups
        print("\nAnalyzing feature groups...")
        groups_df = analyze_feature_groups(importance_data, feature_patterns)

        # Print group importance
        print("\nFeature Group Importance:")
        print(groups_df)

        # Plot group importance
        plot_groups(groups_df, "feature_groups.png")

        # Save group importance to CSV
        if not groups_df.empty:
            groups_df.to_csv("feature_groups.csv", index=False)
            print("Saved feature group data to feature_groups.csv")

    # Print top 10 most important features
    if feature_importance is not None:
        print("\nTop 10 Most Important Features:")
        # Combine feature names and importance values
        combined = list(zip(feature_names, feature_importance))
        # Sort by importance (descending)
        combined.sort(key=lambda x: x[1], reverse=True)
        # Print top 10
        for i, (feature, importance) in enumerate(combined[:10]):
            print(f"{i + 1}. {feature}: {importance:.2f}%")

    # Print top 5 PCA components
    print("\nTop 5 Most Important PCA Components:")
    pca_combined = list(zip(component_labels, pca_importance))
    pca_combined.sort(key=lambda x: x[1], reverse=True)
    for i, (component, importance) in enumerate(pca_combined[:5]):
        if pca_variance is not None:
            var_row = pca_variance[pca_variance['Component'] == component]
            if not var_row.empty:
                var_value = var_row.iloc[0]['Explained_Variance_Ratio'] * 100
                print(f"{i + 1}. {component}: {importance:.2f}% importance, {var_value:.2f}% variance explained")
            else:
                print(f"{i + 1}. {component}: {importance:.2f}% importance")
        else:
            print(f"{i + 1}. {component}: {importance:.2f}% importance")

    print("\nNeural network interpretation completed!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error during execution: {e}")

# import torch
# import torch.nn as nn
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# import os
# from pathlib import Path
#
# """
# This script:
# 1. Loads a trained BikeVolumeNN model
# 2. Loads the PCA parameters used during training
# 3. Analyzes the network to determine feature importance
# 4. Visualizes which features influence the model output most strongly
# """
#
#
# class BikeVolumeNN(nn.Module):
#     """
#     BikeVolumeNN: Neural network that predicts bike volume.
#     This is the same model definition as in the training script.
#     """
#
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
# def load_model(model_path):
#     """
#     Load a trained BikeVolumeNN model
#     """
#     # Check if model exists
#     if not os.path.exists(model_path):
#         print(f"Model file not found: {model_path}")
#         return None, None
#
#     # Load the saved model
#     checkpoint = torch.load(model_path, weights_only=False)
#
#     # Extract model parameters
#     input_dim = checkpoint['input_dim']
#     network_layers = checkpoint.get('network_layers', 3)  # Default to 3 if not specified
#
#     # Create a new model with the same architecture
#     model = BikeVolumeNN(input_dim, num_layers=network_layers)
#
#     # Load the saved weights
#     model.load_state_dict(checkpoint['model_state_dict'])
#
#     # Set model to evaluation mode
#     model.eval()
#
#     return model, checkpoint
#
#
# def load_pca_parameters():
#     """
#     Load the PCA parameters and scaler parameters saved during training
#     """
#     pca_params = None
#     pca_variance = None
#     scaler_params = None
#
#     # Load PCA parameters if file exists
#     if os.path.exists('pca_parameters.csv'):
#         pca_params = pd.read_csv('pca_parameters.csv')
#     else:
#         print("PCA parameters file not found")
#
#     # Load PCA variance information if file exists
#     if os.path.exists('pca_variance.csv'):
#         pca_variance = pd.read_csv('pca_variance.csv')
#     else:
#         print("PCA variance file not found")
#
#     # Load scaler parameters if file exists
#     if os.path.exists('scaler_parameters.csv'):
#         scaler_params = pd.read_csv('scaler_parameters.csv')
#     else:
#         print("Scaler parameters file not found")
#
#     return pca_params, pca_variance, scaler_params
#
#
# def compute_feature_importance_through_pca(model, pca_params, pca_variance, top_n=10):
#     """
#     Compute feature importance by analyzing model weights and PCA components
#
#     This function:
#     1. Extracts weights from the first layer of the neural network
#     2. Combines these weights with PCA component loadings
#     3. Calculates the total impact of each original feature on the model output
#     """
#     # Get the first fully connected layer weights
#     if model.num_layers == 2:
#         first_layer_weights = model.model[0].weight.data.numpy()  # Shape: [32, input_dim]
#     else:  # 3 layers
#         first_layer_weights = model.model[0].weight.data.numpy()  # Shape: [64, input_dim]
#
#     # Calculate importance of each PCA component based on first layer weights
#     # Take sum of absolute values across neurons to get overall importance
#     pca_component_importance = np.abs(first_layer_weights).sum(axis=0)  # Shape: [input_dim]
#
#     # Normalize to get relative importance
#     pca_component_importance = pca_component_importance / np.sum(pca_component_importance)
#
#     # Create DataFrame with PCA component importance
#     component_importance_df = pd.DataFrame({
#         'Component': [f'Component_{i + 1}' for i in range(len(pca_component_importance))],
#         'Importance': pca_component_importance
#     })
#
#     # If we have variance information, add it
#     if pca_variance is not None:
#         component_importance_df = component_importance_df.merge(
#             pca_variance[['Component', 'Explained_Variance_Ratio']],
#             on='Component'
#         )
#
#     # Now calculate original feature importance through PCA components
#     if pca_params is not None:
#         # Get feature names and component loadings
#         features = pca_params['Feature'].values
#
#         # Initialize array to store feature importance
#         feature_importance = np.zeros(len(features))
#
#         # For each component, multiply its importance by its loadings and add to feature importance
#         for i, component in enumerate(component_importance_df['Component']):
#             # Get component index (extract from 'Component_X' format)
#             component_idx = int(component.split('_')[1]) - 1
#
#             # Get component importance
#             importance = component_importance_df.loc[
#                 component_importance_df['Component'] == component, 'Importance'
#             ].values[0]
#
#             # Get component loadings for all features
#             loadings = pca_params[component].values if component in pca_params.columns else pca_params[
#                 f'Component_{component_idx + 1}'].values
#
#             # Add weighted contribution to feature importance
#             feature_importance += np.abs(loadings) * importance
#
#         # Create DataFrame with original feature importance
#         feature_importance_df = pd.DataFrame({
#             'Feature': features,
#             'Importance': feature_importance
#         })
#
#         # Sort by importance
#         feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
#
#         return component_importance_df, feature_importance_df
#     else:
#         # If no PCA parameters, return only component importance
#         return component_importance_df, None
#
#
# def compute_saliency_map(model, sample_data, feature_names):
#     """
#     Compute saliency map (gradient-based feature importance) for a sample data point
#
#     This is an alternative method to analyze feature importance
#     """
#     # Convert data to tensor and require gradient
#     x = torch.tensor(sample_data, dtype=torch.float32, requires_grad=True)
#
#     # Forward pass
#     output = model(x)
#
#     # Backward pass (compute gradient of output with respect to input)
#     output.backward()
#
#     # Get gradients
#     gradients = x.grad.numpy()
#
#     # Take absolute value for importance
#     importance = np.abs(gradients)
#
#     # Create DataFrame with saliency scores
#     saliency_df = pd.DataFrame({
#         'Feature': feature_names,
#         'Saliency': importance.flatten()
#     })
#
#     # Sort by importance
#     saliency_df = saliency_df.sort_values('Saliency', ascending=False)
#
#     return saliency_df
#
#
# def plot_feature_importance(feature_importance_df, title, top_n=15):
#     """
#     Plot feature importance as a horizontal bar chart using matplotlib
#     """
#     plt.figure(figsize=(12, 8))
#
#     # Get top N features by importance
#     top_features = feature_importance_df.sort_values('Importance', ascending=False).head(top_n)
#
#     # Extract data for plotting
#     features = top_features['Feature'].values
#     importance = top_features['Importance'].values
#
#     # Plot horizontal bar chart
#     y_pos = np.arange(len(features))
#     plt.barh(y_pos, importance, align='center')
#     plt.yticks(y_pos, features)
#
#     plt.title(title, fontsize=16)
#     plt.xlabel('Importance Score')
#     plt.ylabel('Feature')
#     plt.tight_layout()
#
#     return plt
#
#
# def main():
#     # Find the model file
#     model_files = list(Path('.').glob('*.pth'))
#
#     if not model_files:
#         print("No model files found. Please place the model file in the current directory.")
#         return
#
#     # Select the latest model file
#     model_path = str(model_files[-1])
#     print(f"Using model: {model_path}")
#
#     # Load the model
#     model, checkpoint = load_model(model_path)
#
#     if model is None:
#         return
#
#     print(f"Model loaded successfully. Architecture: {model.num_layers} layers")
#     print(f"Input dimensions: {checkpoint['input_dim']}")
#     print(f"PCA Components used: {checkpoint.get('pca_components', 'N/A')}")
#     print(f"Final Test R²: {checkpoint.get('final_test_r2', 'N/A')}")
#
#     # Load PCA parameters
#     pca_params, pca_variance, scaler_params = load_pca_parameters()
#
#     # Compute feature importance through PCA analysis
#     component_importance, feature_importance = compute_feature_importance_through_pca(model, pca_params, pca_variance)
#
#     # Plot and save component importance
#     if component_importance is not None:
#         plt_components = plot_feature_importance(
#             component_importance,
#             f'PCA Component Importance for BikeVolumeNN ({model.num_layers} layers)',
#             top_n=min(15, len(component_importance))
#         )
#         component_plot_path = 'pca_component_importance.png'
#         plt_components.savefig(component_plot_path)
#         print(f"PCA component importance plot saved to {component_plot_path}")
#         plt_components.close()
#
#     # Plot and save feature importance
#     if feature_importance is not None:
#         plt_features = plot_feature_importance(
#             feature_importance,
#             f'Feature Importance for BikeVolumeNN ({model.num_layers} layers)',
#             top_n=min(20, len(feature_importance))
#         )
#         feature_plot_path = 'feature_importance.png'
#         plt_features.savefig(feature_plot_path)
#         print(f"Feature importance plot saved to {feature_plot_path}")
#         plt_features.close()
#
#         # Save feature importance to CSV
#         feature_importance.to_csv('feature_importance.csv', index=False)
#         print(f"Feature importance data saved to feature_importance.csv")
#
#     # Print top 10 most important features
#     if feature_importance is not None:
#         print("\nTop 10 most important features:")
#         for i, (feature, importance) in enumerate(
#                 zip(feature_importance['Feature'].head(10), feature_importance['Importance'].head(10))):
#             print(f"{i + 1}. {feature}: {importance:.4f}")
#
#     # Print top 5 PCA components and their explained variance
#     if component_importance is not None and 'Explained_Variance_Ratio' in component_importance.columns:
#         print("\nTop 5 most important PCA components:")
#         for i, row in component_importance.sort_values('Importance', ascending=False).head(5).iterrows():
#             print(
#                 f"{row['Component']}: Importance={row['Importance']:.4f}, Explained Variance={row['Explained_Variance_Ratio']:.4f}")
#
#
# if __name__ == "__main__":
#     main()