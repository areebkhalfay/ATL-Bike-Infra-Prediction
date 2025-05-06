import geopandas as gpd
import pandas as pd
import folium
from folium.plugins import HeatMap

modeltoplot = "Ride Count (Average, ATL Norm)"

# Load the updated grid data
grid_df = pd.read_csv("atl_grid_withpreds_allmodels.csv")

# Compute grid center points for visualization
grid_df["Center Lat"] = (grid_df["Min Lat"] + grid_df["Max Lat"]) / 2
grid_df["Center Lon"] = (grid_df["Min Long"] + grid_df["Max Long"]) / 2

# grid_df = grid_df[grid_df["total_population"] > 0]
grid_df = grid_df[grid_df["Max Long"] <= -84.34838196256136]

# Create a base map centered on ATL
m = folium.Map(location=[33.65, -84.38], zoom_start=11, tiles="OpenStreetMap")
# m = folium.Map(location=[33.65, -84.38], zoom_start=11, tiles="cartodbpositron")

# Prepare data for heatmap (only nonzero ride counts)
heat_data = grid_df[grid_df[modeltoplot] > 0][["Center Lat", "Center Lon", modeltoplot]].values.tolist()

# Add heatmap layer
# HeatMap(heat_data, radius=12, blur=8, max_zoom=14).add_to(m)
HeatMap(
    heat_data,
    radius=15,  # Reduced radius for more granularity
    blur=18,  # Reduced blur for sharper definition
    max_zoom=16,  # Increased max zoom for more detail
    min_opacity=0.1,  # Added minimum opacity
    max_opacity=0.2,
    # gradient={
    #     '0': 'transparent',
    #     '0.49': 'transparent',
    #     '0.5': 'blue',
    #     '0.7': 'yellow',
    #     '0.9': 'orange',
    #     '1.0': 'red'
    # },
    # overlay=True

).add_to(m)

folium.LayerControl().add_to(m)

# Save and show map
m.save("atl_ridership_heatmap_modelavg_nycnormalized.html")
print("Heatmap saved as atl_ridership_heatmap.html")
