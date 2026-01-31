# Crop rasters to DIY boundaries
import rasterio
from rasterio.mask import mask
import geopandas as gpd
import json


gdf = gpd.read_file("data/raw/Yogyakarta.geojson")

with rasterio.open("data/raw/VIIRS/VNL_npp_2024_global_vcmslcfg_v2_c202502261200.average_masked.dat.tif") as src:
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

with rasterio.open("data/raw/VIIRS/Cropped_VNL_npp_2024_global_vcmslcfg_v2_c202502261200.average_masked.dat.tif", "w", **out_meta) as dest:
    dest.write(out_image)

print("Completed")