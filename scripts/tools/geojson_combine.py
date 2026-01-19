# Combine origins and destinations into one single geojson
import geopandas as gpd

def combine_geojsons(file1_path, file2_path, output_path):
    gdf1 = gpd.read_file(file1_path)
    gdf2 = gpd.read_file(file2_path)

    cols_to_keep1 = [col for col in ['cell_id', 'geometry', 'origin_score'] if col in gdf1.columns]
    gdf1 = gdf1[cols_to_keep1]

    cols_to_keep2 = [col for col in ['peak_destination_score'] if col in gdf2.columns]
    df2_subset = gdf2[cols_to_keep2]

    # Join by index
    combined_gdf = gdf1.join(df2_subset)

    combined_gdf.to_file(output_path, driver='GeoJSON')
    print(f"File saved with columns: {combined_gdf.columns.tolist()}")

combine_geojsons('data/raw/origin_v2_1_1000m.geojson', 'data/raw/destination_v4_peak_1000m.geojson', 'data/raw/combined_v4_peak_1000m.geojson')