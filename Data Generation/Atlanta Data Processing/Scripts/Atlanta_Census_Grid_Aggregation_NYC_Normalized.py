#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from shapely import wkt


demo_df = pd.read_csv("atl_data_normalized_with_nyc.csv").drop('X', axis=1).rename(columns={'Geography': 'GEOID'})

grid_df = pd.read_csv("atlanta_grid_data.csv")

tract_df = pd.read_csv("atl_census_tracts_with_data.csv")
tract_df['geometry'] = tract_df['geometry'].apply(wkt.loads)
tracts_gdf = gpd.GeoDataFrame(tract_df, geometry='geometry', crs="EPSG:4326")

def make_grid_poly(row):
    return box(row['Min Long'], row['Min Lat'], row['Max Long'], row['Max Lat'])

grid_df['geometry'] = grid_df.apply(make_grid_poly, axis=1)
grid_gdf = gpd.GeoDataFrame(grid_df, geometry='geometry', crs="EPSG:4326")


# In[12]:


intersections = gpd.overlay(tracts_gdf, grid_gdf, how='intersection')

intersections['intersection_area'] = intersections.geometry.area
tracts_gdf['tract_area'] = tracts_gdf.geometry.area

intersections = intersections.merge(
    tracts_gdf[['GEOID', 'tract_area']],
    on='GEOID'
)
intersections['area_ratio'] = intersections['intersection_area'] / intersections['tract_area']


# In[13]:


demo_df = demo_df.drop(columns=['Geography_name']).set_index('GEOID').add_prefix(f"{"demographics"}_")

# demo_merged = (
#     merge_on_geo(income_df, 'income')
#     .join(merge_on_geo(education_df, 'edu'), how='outer')
#     .join(merge_on_geo(transport_df, 'trans'), how='outer')
#     .join(merge_on_geo(population_df, 'pop'), how='outer')
# )

intersections = intersections.set_index('GEOID').join(demo_df, how='left').reset_index()


# In[14]:


for col in intersections.columns:
    if col not in ['GEOID', 'Grid ID', 'intersection_area', 'tract_area', 'area_ratio', 'geometry']:
        intersections[col] = pd.to_numeric(intersections[col], errors='coerce')

feature_cols = [
    col for col in intersections.columns
    if col not in ['GEOID', 'Grid ID', 'intersection_area', 'tract_area', 'area_ratio', 'geometry']
]

for col in feature_cols:
    intersections[col] = intersections[col] * intersections['area_ratio']

cleaned_data = intersections.groupby('Grid ID')[feature_cols].sum().reset_index()


# In[15]:


cleaned_data = cleaned_data.drop(['GEO_ID', 'Unnamed: 0'], axis=1, errors='ignore')
cleaned_data = cleaned_data.drop(['Min Lat', 'Max Lat', 'Min Long', 'Max Long', 'Grid ID'], axis=1).dropna().reset_index().drop('index', axis=1)
# cleaned_data = cleaned_data[cleaned_data['Ride Count'] != 0]
cleaned_data = cleaned_data.dropna()
cleaned_data = cleaned_data[cleaned_data['demographics_total_population'] != 0]
cleaned_data.to_csv("atl_grid_demographics_nyc_normalized.csv", index=False)


# In[16]:


cleaned_data


# In[ ]:




