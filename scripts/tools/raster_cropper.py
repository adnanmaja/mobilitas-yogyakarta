# Crop the Indonesia raster to only keep Yogyakarta (DIY) raster
import rasterio
from rasterio.mask import mask
import geopandas as gpd
import json


gdf = gpd.read_file("Yogyakarta.geojson")

with rasterio.open("idn_pop_2025_CN_100m_R2025A_v1.tif") as src:
    gdf = gdf.to_crs(src.crs)
    
    shapes = [json.loads(gdf.to_json())['features'][0]['geometry']]

    # crop=True shaves off the empty space outside Yogyakarta
    out_image, out_transform = mask(src, shapes, crop=True)
    out_meta = src.meta.copy()

out_meta.update({
    "driver": "GTiff",
    "height": out_image.shape[1],
    "width": out_image.shape[2],
    "transform": out_transform
})

with rasterio.open("v2_clipped_2025_yogyakarta_100m.tif", "w", **out_meta) as dest:
    dest.write(out_image)

print("Crop complete! Your file size just dropped significantly.")