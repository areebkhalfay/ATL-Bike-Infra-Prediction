1. `cd` into this folder. Run `pip install -r requirements.txt` to download all necessary dependencies.
2. Adjust Atlanta Bounds and cell size in `mapOverlayAtlanta.py` as necessary.
3. Run `python3 mapOverlayAtlanta.py` to generate `atlanta_grid_data.csv` and view how the gridding is being done over Atlanta. When satisfied, close the grid image window.
4. Run `python3 Atlanta_Census_Tract_Integration.py` to generate `atl_census_tracts_with_data.csv`.
5. Run `python3 Atlanta_Census_Grid_Aggregation_Numeric.py` to generate `atl_grid_demographics_numeric.csv`. This is the final gridded dataset done with the numeric Atlanta demographic data.
6. Run `python3 Atlanta_Census_Grid_Aggregation_NYC_Normalized.py` to generate `atl_grid_demographics_nyc_normzalized.csv`. This is the final gridded dataset done with the NYC Normalized Atlanta demographic data.
7. Your final datasets are ready!