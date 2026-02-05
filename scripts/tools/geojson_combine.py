# Combine origins and destinations into one single geojson
import geopandas as gpd
import time

def combine_residential_employment(file1_path, file2_path, output_path):
    gdf1 = gpd.read_file(file1_path)
    gdf2 = gpd.read_file(file2_path)

    cols_to_keep1 = [col for col in ['cell_id', 'geometry', 'residential_intensity'] if col in gdf1.columns]
    gdf1 = gdf1[cols_to_keep1]
    
    col_2 = ['employment_intensity']
    cols_to_keep2 = [col for col in col_2 if col in gdf2.columns]
    df2_subset = gdf2[cols_to_keep2]

    # Join by index
    combined_gdf = gdf1.join(df2_subset)

    combined_gdf.to_file(output_path, driver='GeoJSON')
    print(f"File saved with columns: {combined_gdf.columns.tolist()}")

def combine_re_amenity(file1_path, file2_path, output_path):
    gdf1 = gpd.read_file(file1_path)
    gdf2 = gpd.read_file(file2_path)

    cols_to_keep1 = [col for col in ['cell_id', 'geometry', 'residential_intensity', 'employment_intensity'] if col in gdf1.columns]
    gdf1 = gdf1[cols_to_keep1]
    
    col_2 = ['amenity_nhb_intensity', 'amenity_hbnw_intensity']
    cols_to_keep2 = [col for col in col_2 if col in gdf2.columns]
    df2_subset = gdf2[cols_to_keep2]

    # Join by index
    combined_gdf = gdf1.join(df2_subset)

    combined_gdf.to_file(output_path, driver='GeoJSON')
    print(f"File saved with columns: {combined_gdf.columns.tolist()}")

combine_residential_employment('data/raw/residential_1000m.geojson', 'data/raw/employment_1000m.geojson', 'data/raw/re_1000m_v2.geojson')
time.sleep(5)
combine_re_amenity('data/raw/re_1000m_v2.geojson', 'data/raw/services_amenities_1000m.geojson', 'data/raw/rea_1000m_v2.geojson')
