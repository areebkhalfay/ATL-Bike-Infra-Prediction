import numpy as np


def feet_to_degrees(feet, latitude):
    """
    Converts a distance in feet to degrees for latitude and longitude.

    Parameters:
    - feet: The distance in feet.
    - latitude: The latitude where conversion is being applied (for longitude calculation).

    Returns:
    - (lat_deg, lon_deg): Tuple of latitude and longitude degree equivalents.
    """
    feet_per_degree_lat = 364000  # Approximate feet per degree of latitude
    feet_per_degree_lon = 280000  # Approximate feet per degree of longitude at the given latitude

    lat_deg = feet / feet_per_degree_lat
    lon_deg = feet / feet_per_degree_lon  # Adjusted for given latitude

    return lat_deg, lon_deg


def create_grid(min_lat, max_lat, min_lon, max_lon, cell_size_ft):
    """
    Creates a grid over a region defined by its bounding box with a given cell size in feet.

    Parameters:
    - min_lat, max_lat: float, min and max latitudes.
    - min_lon, max_lon: float, min and max longitudes.
    - cell_size_ft: int, size of each grid cell in feet.

    Returns:
    - List of dictionaries representing grid cells with min/max latitudes and longitudes.
    """
    # Compute degree equivalent of 500 feet
    lat_step, lon_step = feet_to_degrees(cell_size_ft, (min_lat + max_lat) / 2)

    # Determine number of grid cells in latitude and longitude directions
    n_lat = int((max_lat - min_lat) / lat_step)
    n_lon = int((max_lon - min_lon) / lon_step)

    # Create latitude and longitude intervals
    lat_steps = np.linspace(min_lat, max_lat, n_lat + 1)
    lon_steps = np.linspace(min_lon, max_lon, n_lon + 1)

    # Generate grid cells
    grid_cells = []
    for i in range(n_lat):
        for j in range(n_lon):
            cell = {
                'min_lat': lat_steps[i],
                'max_lat': lat_steps[i + 1],
                'min_lon': lon_steps[j],
                'max_lon': lon_steps[j + 1]
            }
            grid_cells.append(cell)

    return grid_cells, n_lat, n_lon


# Define bounding boxes for NYC and Atlanta
nyc_bounds = (40.477399, 40.917577, -74.259090, -73.700272)
atl_bounds = (33.640, 33.900, -84.570, -84.200)

# Define grid cell size in feet
cell_size_ft = 500

# Generate NYC grid
nyc_grid, nyc_n_lat, nyc_n_lon = create_grid(*nyc_bounds, cell_size_ft)
print(f"New York City Grid: {nyc_n_lat} rows × {nyc_n_lon} columns = {nyc_n_lat * nyc_n_lon} cells")

# Generate Atlanta grid
atl_grid, atl_n_lat, atl_n_lon = create_grid(*atl_bounds, cell_size_ft)
print(f"Atlanta Grid: {atl_n_lat} rows × {atl_n_lon} columns = {atl_n_lat * atl_n_lon} cells")
