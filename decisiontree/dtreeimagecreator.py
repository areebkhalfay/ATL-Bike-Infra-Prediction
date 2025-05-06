import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Load model
model = joblib.load("decision_tree_depth10_minsplit2.pkl")

# Load your original dataset to get the feature count
df = pd.read_csv("merged_nyc_grid_data.csv")
X = df.drop('bike_volume', axis=1)

# Use PCA component names for clarity
feature_names = [f"PC{i + 1}" for i in range(model.n_features_in_)]

# Plot the top 3 levels of the tree
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=feature_names, filled=True, max_depth=3, fontsize=10)
plt.title("Decision Tree Regressor (Top 3 Levels)")
plt.tight_layout()
plt.savefig("decision_tree_visualization.png")
plt.show()
