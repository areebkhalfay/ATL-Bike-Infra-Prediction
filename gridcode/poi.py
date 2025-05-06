import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point


def get_pois_from_osm(bounds, tags):
    """Fetches Points of Interest (POIs) from OpenStreetMap based on specified tags."""
    min_lat, max_lat, min_lon, max_lon = bounds

    # Fetch POIs from OSM using specified tags
    pois = ox.features_from_bbox((max_lat, min_lat, max_lon, min_lon), tags)

    # Extract relevant columns
    pois = pois[['geometry']]
    pois = pois[pois.geometry.type == 'Point']  # Keep only point-based POIs

    return pois


def assign_pois_to_grid(grid, pois):
    """Assigns a POI indicator (1 if POI exists, 0 otherwise) to each grid cell."""
    grid['Points of Interest'] = 0

    for idx, cell in grid.iterrows():
        if pois.intersects(cell.geometry).any():
            grid.at[idx, 'Points of Interest'] = 1

    return grid


# Define NYC bounding box
nyc_bounds = (40.477399, 40.917577, -74.259090, -73.700272)

# Define OSM tags for POIs
poi_tags = {
    'amenity': ['hospital', 'library', 'police', 'fire_station'],
    'tourism': 'attraction',
    'shop': 'mall',
    'public_transport': True  # Fetch train stations, subway stations, etc.
}

# Fetch POIs from OSM
pois = get_pois_from_osm(nyc_bounds, poi_tags)

# Load existing grid CSV
grid = gpd.read_file("grid_data.csv")

# Assign POIs to grid cells
grid = assign_pois_to_grid(grid, pois)

# Save updated CSV
grid.to_csv("grid_data_with_pois.csv", index=False)
print("Updated grid saved with POI information.")
