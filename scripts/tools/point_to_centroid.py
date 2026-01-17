import geopandas as gpd

def convert_to_centroids(input_file, output_file):
    gdf = gpd.read_file(input_file)
    gibran = gdf.copy()

    gibran['geometry'] = gibran.geometry.centroid
    
    gibran.to_file(output_file, driver='GeoJSON')
    print(f"Successfully saved centroids to {output_file}")
    print(type(gibran['geometry']))

convert_to_centroids('data/combined_1000m.geojson', 'data/combined_1000m_centroids.geojson')