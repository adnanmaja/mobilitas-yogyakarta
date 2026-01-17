from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
import rasterio

worldpop_path = "clipped_2025_yogyakarta_100m.tif"
dst_crs = "EPSG:32749"

with rasterio.open(worldpop_path) as src:
    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds
    )
    
    kwargs = src.meta.copy()
    kwargs.update({
        "crs": dst_crs,
        "transform": transform,
        "width": width,
        "height": height
    })
    
    reprojected_path = "clipped_utm49s_2025_yogyakarta_100m.tif"
    
    with rasterio.open(reprojected_path, "w", **kwargs) as dst:
        reproject(
            source=rasterio.band(src, 1),
            destination=rasterio.band(dst, 1),
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.sum  
        )
