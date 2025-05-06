import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
from shapely.geometry import Polygon
from geodatasets import get_path
import pandas as pd


def feet_to_degrees(feet, latitude):
    """Converts feet to degrees for latitude and longitude."""
    feet_per_degree_lat = 364000  # Approximate feet per degree of latitude
    feet_per_degree_lon = 280000  # Approximate feet per degree of longitude at NYC's latitude

    lat_deg = feet / feet_per_degree_lat
    lon_deg = feet / feet_per_degree_lon
    return lat_deg, lon_deg


def create_grid(min_lat, max_lat, min_lon, max_lon, cell_size_ft):
    """Creates a grid with the given cell size in feet over the specified region."""
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
                (lon_steps[j], lat_steps[i])  # Close the polygon
            ])
            grid_cells.append(cell)

    grid = gpd.GeoDataFrame(geometry=grid_cells, crs="EPSG:4326")
    return grid


def plot_grid_on_nyc(grid, min_lat, max_lat, min_lon, max_lon):
    """Plots the NYC map with the overlaid grid."""
    fig, ax = plt.subplots(figsize=(10, 10))

    # Load NYC boundary (from OpenStreetMap)
    nyc_boundary = gpd.read_file(get_path('nybb')).to_crs(epsg=4326)
    nyc_boundary.plot(ax=ax, color="lightgrey", edgecolor="black", alpha=0.5)

    # Plot the grid
    grid.plot(ax=ax, edgecolor="blue", facecolor="none", linewidth=0.5)

    # Add OpenStreetMap background
    ctx.add_basemap(ax, crs=grid.crs, source=ctx.providers.OpenStreetMap.Mapnik)

    # Formatting
    ax.set_xlim([min_lon, max_lon])
    ax.set_ylim([min_lat, max_lat])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("500ft x 500ft Grid Overlay on NYC Map")

    plt.show()


# NYC Bounding Box
nyc_bounds = (40.626749, 40.900108, -74.045488, -73.845418)
cell_size_ft = 1000  # Each grid cell is 500 feet x 500 feet

# Generate grid
grid = create_grid(*nyc_bounds, cell_size_ft)

# Plot the NYC map with the grid overlay
plot_grid_on_nyc(grid, *nyc_bounds)


def save_grid_to_csv(grid, lat_steps, lon_steps, filename="grid_data.csv"):
    """Saves the grid data to a CSV file."""
    grid_data = []
    grid_id = 1

    for i in range(len(lat_steps) - 1):
        for j in range(len(lon_steps) - 1):
            grid_data.append([
                grid_id,
                lat_steps[i], lat_steps[i + 1],  # Min and max latitude
                lon_steps[j], lon_steps[j + 1]  # Min and max longitude
            ])
            grid_id += 1

    df = pd.DataFrame(grid_data, columns=["Grid ID", "Min Lat", "Max Lat", "Min Long", "Max Long"])
    df.to_csv(filename, index=False)
    print(f"Grid data saved to {filename}")


# Generate lat/lon steps again to pass to the function
lat_step, lon_step = feet_to_degrees(cell_size_ft, (nyc_bounds[0] + nyc_bounds[1]) / 2)
lat_steps = np.linspace(nyc_bounds[0], nyc_bounds[1], int((nyc_bounds[1] - nyc_bounds[0]) / lat_step) + 1)
lon_steps = np.linspace(nyc_bounds[2], nyc_bounds[3], int((nyc_bounds[3] - nyc_bounds[2]) / lon_step) + 1)

# Save the grid data to CSV
save_grid_to_csv(grid, lat_steps, lon_steps)
