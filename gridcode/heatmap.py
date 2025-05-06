import geopandas as gpd
import pandas as pd
import folium
from folium.plugins import HeatMap

# Load the updated grid data
grid_df = pd.read_csv("grid_data_updated.csv")

# Compute grid center points for visualization
grid_df["Center Lat"] = (grid_df["Min Lat"] + grid_df["Max Lat"]) / 2
grid_df["Center Lon"] = (grid_df["Min Long"] + grid_df["Max Long"]) / 2

# Create a base map centered on NYC
m = folium.Map(location=[40.7128, -74.0060], zoom_start=12, tiles="cartodbpositron")

# Prepare data for heatmap (only nonzero ride counts)
heat_data = grid_df[grid_df["Ride Count"] > 0][["Center Lat", "Center Lon", "Ride Count"]].values.tolist()

# Add heatmap layer
HeatMap(heat_data, radius=12, blur=8, max_zoom=14).add_to(m)

# Save and show map
m.save("nyc_ridership_heatmap.html")
print("Heatmap saved as nyc_ridership_heatmap.html")
