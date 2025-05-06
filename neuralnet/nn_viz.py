import torch
from torchviz import make_dot
import matplotlib.pyplot as plt
import numpy as np
from graphviz import Digraph


# Define your BikeVolumeNN class
class BikeVolumeNN(torch.nn.Module):
    def __init__(self, input_dim):
        super(BikeVolumeNN, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.BatchNorm1d(64, momentum=0.9),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(64, 32),
            torch.nn.BatchNorm1d(32, momentum=0.9),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),

            torch.nn.Linear(32, 16),
            torch.nn.BatchNorm1d(16, momentum=0.9),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),

            torch.nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.model(x)


# Function to visualize model architecture manually
def visualize_nn(model, input_dim):
    # Create a figure
    plt.figure(figsize=(12, 8))

    # Set up the layers in the network
    layers = [input_dim, 64, 32, 16, 1]
    layer_names = ['Input', 'Hidden 1', 'Hidden 2', 'Hidden 3', 'Output']
    n_layers = len(layers)

    # Maximum neurons for scaling
    max_neurons = max(layers)

    # Positions for neurons in each layer
    layer_positions = np.linspace(0, 10, n_layers)

    # Draw the neurons
    for i, (n_neurons, layer_name) in enumerate(zip(layers, layer_names)):
        # Calculate positions for drawing
        x = layer_positions[i]
        neuron_positions = np.linspace(0, 8, n_neurons)

        # Draw layer label
        plt.text(x, -1, layer_name, ha='center', va='center', fontsize=12, fontweight='bold')

        # Draw each neuron in this layer
        for j, y in enumerate(neuron_positions):
            # Only draw a sample of neurons if too many
            if n_neurons > 10 and j % max(1, n_neurons // 10) != 0 and j != n_neurons - 1:
                continue

            # Draw the neuron
            circle = plt.Circle((x, y), 0.2, fill=True, color='skyblue', ec='blue')
            plt.gca().add_patch(circle)

            # Draw connections to previous layer
            if i > 0:
                prev_n_neurons = layers[i - 1]
                prev_x = layer_positions[i - 1]
                prev_neuron_positions = np.linspace(0, 8, prev_n_neurons)

                # Connect to all neurons in previous layer
                for k, prev_y in enumerate(prev_neuron_positions):
                    # Only draw connections to a sample of neurons if too many
                    if prev_n_neurons > 10 and k % max(1, prev_n_neurons // 10) != 0 and k != prev_n_neurons - 1:
                        continue
                    plt.plot([prev_x, x], [prev_y, y], 'gray', alpha=0.3)

    # Add batch norm, ReLU, and dropout indicators
    for i in range(1, n_layers - 1):
        x = layer_positions[i]
        y = 9
        plt.text(x, y, f"BatchNorm\nReLU\nDropout({0.3 if i == 1 else 0.2})",
                 ha='center', va='center', fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gold", alpha=0.8))

    # Set labels and title
    plt.title('BikeVolumeNN Architecture', fontsize=16)
    plt.xlim(-1, 11)
    plt.ylim(-2, 10)
    plt.axis('off')

    # Save and show
    plt.tight_layout()
    plt.savefig('bike_nn_architecture.png', dpi=300, bbox_inches='tight')
    plt.show()


# Create model and generate visualization
input_dim = 25  # Using your selected_features value
model = BikeVolumeNN(input_dim)

# Generate visualization
visualize_nn(model, input_dim)

# # Optional - Use torchviz if available
# #try:
# # Create a sample input
# x = torch.randn(1, input_dim)
# y = model(x)
#
# # Generate dot graph
# dot = make_dot(y, params=dict(model.named_parameters()))
# dot.format = 'png'
# dot.render('bike_nn_torchviz', cleanup=True)
# print("Torchviz visualization generated as 'bike_nn_torchviz.png'")
# # except:
# #     print(
# #         "Torchviz visualization could not be generated. If you want to use torchviz, install it with 'pip install torchviz'")