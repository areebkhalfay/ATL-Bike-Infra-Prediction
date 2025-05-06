import pandas as pd
import bisect

rides_csvs = [ "201901-citibike-tripdata_1.csv", "201902-citibike-tripdata_1.csv", "201903-citibike-tripdata_1.csv",
    "201903-citibike-tripdata_2.csv", "201904-citibike-tripdata_1.csv", "201904-citibike-tripdata_2.csv",
    "201905-citibike-tripdata_1.csv", "201905-citibike-tripdata_2.csv", "201906-citibike-tripdata_1.csv",
    "201906-citibike-tripdata_2.csv", "201906-citibike-tripdata_3.csv", "201907-citibike-tripdata_1.csv",           
    "201907-citibike-tripdata_2.csv", "201907-citibike-tripdata_3.csv",
    "201908-citibike-tripdata_1.csv", "201908-citibike-tripdata_2.csv", "201908-citibike-tripdata_3.csv",
    "201909-citibike-tripdata_1.csv", "201909-citibike-tripdata_2.csv", "201909-citibike-tripdata_3.csv",
    "201910-citibike-tripdata_1.csv", "201910-citibike-tripdata_2.csv", "201910-citibike-tripdata_3.csv",
    "201911-citibike-tripdata_1.csv", "201911-citibike-tripdata_2.csv", "201912-citibike-tripdata_1.csv"
]

# Load the grid data
grid_df = pd.read_csv("grid_data.csv")

# Ensure "Ride Count" column exists
grid_df["Ride Count"] = grid_df.get("Ride Count", 0)

# Extract sorted lat/lon boundaries
lat_bounds = sorted(set(grid_df["Min Lat"]))
lon_bounds = sorted(set(grid_df["Min Long"]))

def find_grid(lat, lon):
    """Finds the grid cell index using binary search."""
    lat_idx = bisect.bisect_right(lat_bounds, lat) - 1
    lon_idx = bisect.bisect_right(lon_bounds, lon) - 1

    if lat_idx < 0 or lon_idx < 0:
        return None  # Out of bounds

    # Compute the corresponding Grid ID
    grid_id = lat_idx * len(lon_bounds) + lon_idx + 1

    # Ensure the found grid ID exists
    matching_rows = grid_df[grid_df["Grid ID"] == grid_id]
    return matching_rows.index[0] if not matching_rows.empty else None

# Process each ride dataset
for rides_csv in rides_csvs:
    print(f"Processing {rides_csv}...")
    rides_df = pd.read_csv(rides_csv)
    
    for _, ride in rides_df.iterrows():
        start_grid_idx = find_grid(ride["start station latitude"], ride["start station longitude"])
        end_grid_idx = find_grid(ride["end station latitude"], ride["end station longitude"])
        
        if start_grid_idx is not None:
            grid_df.at[start_grid_idx, "Ride Count"] += 1
        if end_grid_idx is not None:
            grid_df.at[end_grid_idx, "Ride Count"] += 1

# Save updated grid data
grid_df.to_csv("grid_data_updated.csv", index=False)
print("Updated grid data saved as grid_data_updated.csv")

