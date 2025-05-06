import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
from shapely.geometry import Polygon
import pandas as pd


def feet_to_degrees(feet, latitude):
    feet_per_degree_lat = 364000
    feet_per_degree_lon = 280000
    lat_deg = feet / feet_per_degree_lat
    lon_deg = feet / feet_per_degree_lon
    return lat_deg, lon_deg


def create_grid(min_lat, max_lat, min_lon, max_lon, cell_size_ft):
    lat_step, lon_step = feet_to_degrees(cell_size_ft, (min_lat + max_lat) / 2)

    n_lat = int((max_lat - min_lat) / lat_step)
    n_lon = int((max_lon - min_lon) / lon_step)

    lat_steps = np.linspace(min_lat, max_lat, n_lat + 1)
    lon_steps = np.linspace(min_lon, max_lon, n_lon + 1)

    grid_cells = []
    for i in range(n_lat):
        for j in range(n_lon):
            cell = Polygon([
                (lon_steps[j], lat_steps[i]),
                (lon_steps[j + 1], lat_steps[i]),
                (lon_steps[j + 1], lat_steps[i + 1]),
                (lon_steps[j], lat_steps[i + 1]),
                (lon_steps[j], lat_steps[i])
            ])
            grid_cells.append(cell)

    grid = gpd.GeoDataFrame(geometry=grid_cells, crs="EPSG:4326")
    return grid, lat_steps, lon_steps


def plot_grid_on_atlanta(grid, min_lat, max_lat, min_lon, max_lon):
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot the grid
    grid.plot(ax=ax, edgecolor="blue", facecolor="none", linewidth=0.5)

    # Add OpenStreetMap background
    ctx.add_basemap(ax, crs=grid.crs, source=ctx.providers.OpenStreetMap.Mapnik)

    ax.set_xlim([min_lon, max_lon])
    ax.set_ylim([min_lat, max_lat])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("500ft x 500ft Grid Overlay on Atlanta Map")

    plt.show()


def save_grid_to_csv(grid, lat_steps, lon_steps, filename="atlanta_grid_data.csv"):
    grid_data = []
    grid_id = 1

    for i in range(len(lat_steps) - 1):
        for j in range(len(lon_steps) - 1):
            grid_data.append([
                grid_id,
                lat_steps[i], lat_steps[i + 1],
                lon_steps[j], lon_steps[j + 1]
            ])
            grid_id += 1

    df = pd.DataFrame(grid_data, columns=["Grid ID", "Min Lat", "Max Lat", "Min Long", "Max Long"])
    df.to_csv(filename, index=False)
    print(f"Grid data saved to {filename}")


# ATLANTA Bounding Box
atl_bounds = (33.70, 33.83, -84.44, -84.32)
cell_size_ft = 1000  # Change to 1000 if you want to match the NYC grid spacing

# Generate grid and steps
grid, lat_steps, lon_steps = create_grid(*atl_bounds, cell_size_ft)

# Plot the Atlanta grid
plot_grid_on_atlanta(grid, *atl_bounds)

# Save the grid to CSV
save_grid_to_csv(grid, lat_steps, lon_steps)