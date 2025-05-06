import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from pathlib import Path

"""Compares predictions by different models"""

def load_prediction_files(predictions_folder, pattern='atl_preds_*.csv'):
    """
    Load all prediction CSV files matching the pattern

    Parameters:
    - predictions_folder: Folder containing prediction CSV files
    - pattern: Pattern to match prediction files

    Returns:
    - List of tuples (file_name, dataframe)
    """
    prediction_files = glob.glob(os.path.join(predictions_folder, pattern))

    if not prediction_files:
        print(f"No prediction files found matching '{pattern}' in '{predictions_folder}'")
        return []

    print(f"Found {len(prediction_files)} prediction files:")

    results = []
    for file_path in prediction_files:
        file_name = Path(file_path).stem
        print(f"  - {file_name}")

        try:
            df = pd.read_csv(file_path)
            results.append((file_name, df))
        except Exception as e:
            print(f"Error loading {file_name}: {str(e)}")

    return results


def plot_predictions_separate(prediction_data, truth_file=None, output_folder='prediction_plots'):
    """
    Plot predictions from each model on a separate plot

    Parameters:
    - prediction_data: List of tuples (model_name, dataframe)
    - truth_file: Optional file containing ground truth data
    - output_folder: Folder to save the output plots
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Define colors for different models
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan']

    # Load ground truth if available
    ground_truth = None
    if truth_file and os.path.exists(truth_file):
        truth_df = pd.read_csv(truth_file)
        if 'bike_volume' in truth_df.columns:
            ground_truth = truth_df['bike_volume'].values
            print(f"Loaded ground truth data with {len(ground_truth)} values")

    # Create a list to store R² scores if ground truth is available
    r2_scores = []

    # Plot each model's predictions on a separate plot
    for i, (model_name, df) in enumerate(prediction_data):
        if 'bike_volume' not in df.columns:
            print(f"Warning: 'bike_volume' column not found in {model_name}")
            continue

        predictions = df['bike_volume'].values

        # Create a new figure for each model
        plt.figure(figsize=(10, 8))

        # Choose color
        color = colors[i % len(colors)]

        # If ground truth is available, plot actual vs predicted
        if ground_truth is not None and len(ground_truth) == len(predictions):
            plt.scatter(ground_truth, predictions,
                        color=color,
                        alpha=0.7)

            # Calculate R² score
            ss_tot = np.sum((ground_truth - np.mean(ground_truth)) ** 2)
            ss_res = np.sum((ground_truth - predictions) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            r2_scores.append((model_name, r2))

            # Add diagonal line (perfect predictions)
            min_val = min(min(ground_truth), min(predictions))
            max_val = max(max(ground_truth), max(predictions))
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, label='Perfect Prediction')

            plt.title(f'{model_name} - Actual vs Predicted Bike Volume\nR² = {r2:.4f}')
            plt.xlabel('Actual Bike Volume')
            plt.ylabel('Predicted Bike Volume')

        # If no ground truth, plot predictions against index
        else:
            plt.scatter(range(len(predictions)), predictions,
                        color=color,
                        alpha=0.7)

            plt.title(f'{model_name} - Predicted Bike Volume')
            plt.xlabel('Index')
            plt.ylabel('Predicted Bike Volume')

        # Add additional plot aesthetics
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Add stats in a text box if ground truth is available
        if ground_truth is not None:
            mae = np.mean(np.abs(ground_truth - predictions))
            rmse = np.sqrt(np.mean((ground_truth - predictions) ** 2))

            stats_text = (
                f"Statistics:\n"
                f"R² Score: {r2:.4f}\n"
                f"MAE: {mae:.4f}\n"
                f"RMSE: {rmse:.4f}\n"
                f"Mean Prediction: {np.mean(predictions):.4f}\n"
                f"Std Dev: {np.std(predictions):.4f}"
            )

            plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
                     fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Save the plot
        safe_filename = model_name.replace(" ", "_").replace("/", "_")
        output_file = os.path.join(output_folder, f"separate_{safe_filename}_plot.png")
        plt.savefig(output_file)
        print(f"Plot saved to '{output_file}'")

        # Close the figure to free memory
        plt.close()

    # If ground truth was available, create a summary plot of R² scores
    if r2_scores:
        plt.figure(figsize=(10, 6))

        # Sort models by R² score
        r2_scores.sort(key=lambda x: x[1], reverse=True)

        # Extract model names and scores
        model_names = [name for name, _ in r2_scores]
        scores = [score for _, score in r2_scores]

        # Create bar plot
        bars = plt.bar(model_names, scores, color='skyblue')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{height:.4f}', ha='center', va='bottom', fontsize=10)

        plt.title('Model Comparison - R² Score (higher is better)')
        plt.xlabel('Model')
        plt.ylabel('R² Score')
        plt.ylim(0, max(scores) * 1.1)  # Add some padding
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        # Save the comparison plot
        comparison_file = os.path.join(output_folder, "model_comparison_r2.png")
        plt.savefig(comparison_file)
        print(f"Comparison plot saved to '{comparison_file}'")
        plt.close()

        # Print R² scores in descending order
        print("\nModel Performance (R² Score):")
        for name, score in r2_scores:
            print(f"  {name}: R² = {score:.4f}")


def main():
    predictions_folder = 'atl_predictions'  # Current directory, change if needed
    truth_file = None  # Set to path of ground truth file if available
    output_folder = 'prediction_plots'  # Folder to save plots

    # Load prediction files
    prediction_data = load_prediction_files(predictions_folder)

    if not prediction_data:
        print("No prediction data to plot.")
        return

    # Plot predictions on separate plots
    plot_predictions_separate(prediction_data, truth_file, output_folder)


if __name__ == "__main__":
    main()

# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import glob
# import os
# from pathlib import Path
#
#
# def load_prediction_files(predictions_folder, pattern='atl_preds_*.csv'):
#     """
#     Load all prediction CSV files matching the pattern
#
#     Parameters:
#     - predictions_folder: Folder containing prediction CSV files
#     - pattern: Pattern to match prediction files
#
#     Returns:
#     - List of tuples (file_name, dataframe)
#     """
#     prediction_files = glob.glob(os.path.join(predictions_folder, pattern))
#
#     if not prediction_files:
#         print(f"No prediction files found matching '{pattern}' in '{predictions_folder}'")
#         return []
#
#     print(f"Found {len(prediction_files)} prediction files:")
#
#     results = []
#     for file_path in prediction_files:
#         file_name = Path(file_path).stem
#         print(f"  - {file_name}")
#
#         try:
#             df = pd.read_csv(file_path)
#             results.append((file_name, df))
#         except Exception as e:
#             print(f"Error loading {file_name}: {str(e)}")
#
#     return results
#
#
# def plot_predictions(prediction_data, truth_file=None, output_file='prediction_comparison.png'):
#     """
#     Plot predictions from multiple models on a single scatterplot
#
#     Parameters:
#     - prediction_data: List of tuples (model_name, dataframe)
#     - truth_file: Optional file containing ground truth data
#     - output_file: Path to save the output plot
#     """
#     plt.figure(figsize=(12, 10))
#
#     # Define colors for different models
#     colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta']
#     markers = ['o', 's', '^', 'D', 'x', '+', '*']
#
#     # Plot ground truth if available
#     ground_truth = None
#     if truth_file and os.path.exists(truth_file):
#         truth_df = pd.read_csv(truth_file)
#         if 'bike_volume' in truth_df.columns:
#             ground_truth = truth_df['bike_volume'].values
#             print(f"Loaded ground truth data with {len(ground_truth)} values")
#
#     # Create a list to store R² scores if ground truth is available
#     r2_scores = []
#
#     # Plot each model's predictions
#     for i, (model_name, df) in enumerate(prediction_data):
#         if 'bike_volume' not in df.columns:
#             print(f"Warning: 'bike_volume' column not found in {model_name}")
#             continue
#
#         predictions = df['bike_volume'].values
#
#         # If ground truth is available, plot actual vs predicted
#         if ground_truth is not None and len(ground_truth) == len(predictions):
#             color = colors[i % len(colors)]
#             marker = markers[i % len(markers)]
#
#             plt.scatter(ground_truth, predictions,
#                         color=color,
#                         marker=marker,
#                         alpha=0.6,
#                         label=model_name)
#
#             # Calculate R² score
#             ss_tot = np.sum((ground_truth - np.mean(ground_truth)) ** 2)
#             ss_res = np.sum((ground_truth - predictions) ** 2)
#             r2 = 1 - (ss_res / ss_tot)
#             r2_scores.append((model_name, r2))
#
#             # Add diagonal line (perfect predictions) only once
#             if i == 0:
#                 min_val = min(min(ground_truth), min(predictions))
#                 max_val = max(max(ground_truth), max(predictions))
#                 plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, label='Perfect Prediction')
#
#             plt.title('Actual vs Predicted Bike Volume')
#             plt.xlabel('Actual Bike Volume')
#             plt.ylabel('Predicted Bike Volume')
#
#         # If no ground truth, plot predictions against index
#         else:
#             color = colors[i % len(colors)]
#             plt.scatter(range(len(predictions)), predictions,
#                         color=color,
#                         alpha=0.6,
#                         label=model_name)
#
#             plt.title('Predicted Bike Volume by Model')
#             plt.xlabel('Index')
#             plt.ylabel('Predicted Bike Volume')
#
#     # Add R² scores to legend if available
#     if r2_scores:
#         handles, labels = plt.gca().get_legend_handles_labels()
#         new_labels = []
#         for label in labels:
#             if label == 'Perfect Prediction':
#                 new_labels.append(label)
#             else:
#                 r2 = next((score for name, score in r2_scores if name == label), None)
#                 if r2 is not None:
#                     new_labels.append(f"{label} (R² = {r2:.4f})")
#                 else:
#                     new_labels.append(label)
#
#         plt.legend(handles, new_labels)
#     else:
#         plt.legend()
#
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#
#     # Save the plot
#     plt.savefig(output_file)
#     print(f"Plot saved to '{output_file}'")
#
#     # Show the plot
#     plt.show()
#
#     # Print R² scores in descending order
#     if r2_scores:
#         print("\nModel Performance (R² Score):")
#         for name, score in sorted(r2_scores, key=lambda x: x[1], reverse=True):
#             print(f"  {name}: R² = {score:.4f}")
#
#
# def main():
#     predictions_folder = 'atl_predictions'  # Current directory, change if needed
#     truth_file = None  # Set to path of ground truth file if available
#
#     # Load prediction files
#     prediction_data = load_prediction_files(predictions_folder)
#
#     if not prediction_data:
#         print("No prediction data to plot.")
#         return
#
#     # Plot predictions
#     plot_predictions(prediction_data, truth_file)
#
#
# if __name__ == "__main__":
#     main()