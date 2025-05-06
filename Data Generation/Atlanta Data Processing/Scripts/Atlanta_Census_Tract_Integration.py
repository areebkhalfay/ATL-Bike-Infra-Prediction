#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import geopandas as gpd


# In[2]:


csv_path = "ATL_GEOINFO.csv"
df = pd.read_csv(csv_path).drop(index=0)


# In[5]:


df.head()


# In[6]:


gdf = gpd.read_file("tl_2023_13_tract/tl_2023_13_tract.shp")


# In[7]:


gdf.head()


# In[9]:


df['GEOID'] = df['GEO_ID'].astype(str)
gdf['GEOID'] = gdf['GEOIDFQ'].astype(str)


# In[10]:


df = df.drop('GEO_ID', axis=1)
gdf = gdf.drop('GEOIDFQ', axis=1)


# In[13]:


merged_gdf = gdf.merge(df, on='GEOID')
atl_geo_census_polygons = merged_gdf[['GEOID', 'NAMELSAD', 'AREALAND', 'AREAWATR', 'AREALAND_SQMI', 'AREAWATR_SQMI', 'geometry']]


# In[14]:


atl_geo_census_polygons.head()


# In[16]:


atl_geo_census_polygons.to_csv("atl_census_tracts_with_data.csv", index=False)


# In[ ]:




