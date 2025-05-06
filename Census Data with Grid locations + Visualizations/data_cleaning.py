import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the initial datasets
nyc_data_path = 'nyc_data_normalized.csv'  # Replace with the correct path
grid_data_path = 'grid_data_updated.csv'  # Replace with the correct path
geoinfo_path = 'NYC_GEOINFO2023.GEOINFO-Data.csv'  # Replace with the correct path

nyc_data = pd.read_csv(nyc_data_path)
grid_data = pd.read_csv(grid_data_path)
geoinfo = pd.read_csv(geoinfo_path)

# Clean the GEOINFO data by removing the first row with descriptions
geoinfo_cleaned = geoinfo.iloc[1:].copy()  # Skip the header row

# Convert latitude and longitude columns to floats
geoinfo_cleaned['INTPTLAT'] = pd.to_numeric(geoinfo_cleaned['INTPTLAT'], errors='coerce')
geoinfo_cleaned['INTPTLON'] = pd.to_numeric(geoinfo_cleaned['INTPTLON'], errors='coerce')

# Function to map census tracts to grid IDs
def map_tract_to_grid(row, grid_df):
    lat, lon = row['INTPTLAT'], row['INTPTLON']
    match = grid_df[(grid_df['Min Lat'] <= lat) & (grid_df['Max Lat'] >= lat) &
                    (grid_df['Min Long'] <= lon) & (grid_df['Max Long'] >= lon)]
    if not match.empty:
        return match.iloc[0]['Grid ID']
    return None

# Map the GEOINFO data to grid IDs
geoinfo_cleaned['Grid ID'] = geoinfo_cleaned.apply(lambda row: map_tract_to_grid(row, grid_data), axis=1)

# Merge the mapped data with the original NYC data
nyc_data['GEO_ID'] = nyc_data['Geography'].astype(str)
merged_data = pd.merge(nyc_data, geoinfo_cleaned, on='GEO_ID', how='left')
merged_data = pd.merge(merged_data, grid_data, on='Grid ID', how='left')

# Remove redundant or unwanted columns
cleaned_data = merged_data.drop(['GEO_ID', 'Unnamed: 0'], axis=1, errors='ignore')

# Exclude lat/lon columns from normalization
exclude_columns = ['INTPTLAT', 'INTPTLON', 'Min Lat', 'Max Lat', 'Min Long', 'Max Long', 'Grid ID']
columns_to_normalize = [col for col in cleaned_data.select_dtypes(include=['float64', 'int64']).columns if col not in exclude_columns]

# Apply normalization only to the intended columns
scaler = MinMaxScaler()
cleaned_data[columns_to_normalize] = scaler.fit_transform(cleaned_data[columns_to_normalize])

# Remove the 'Unnamed: 8' column if it exists
cleaned_data_fixed = cleaned_data.drop(columns=['Unnamed: 8'], errors='ignore')

# Identify the location-related columns to move to the front
location_columns = ['Grid ID', 'INTPTLAT', 'INTPTLON', 'Min Lat', 'Max Lat', 'Min Long', 'Max Long']

# Ensure the location columns are present in the dataset
location_columns = [col for col in location_columns if col in cleaned_data_fixed.columns]

# Reorder the columns with location data at the front
remaining_columns = [col for col in cleaned_data_fixed.columns if col not in location_columns]
new_column_order = location_columns + remaining_columns
cleaned_data_reordered = cleaned_data_fixed[new_column_order]

cleaned_data_final = cleaned_data_reordered.drop(columns=['NAME'], errors='ignore')

# Drop rows where 'Grid ID' is missing again
cleaned_data_with_grid_id = cleaned_data_final.dropna(subset=['Grid ID'])

# Save the cleaned and normalized dataset
cleaned_normalized_csv_path = 'nyc_data_cleaned_normalized.csv'
cleaned_data_final.to_csv(cleaned_data_with_grid_id, index=False)