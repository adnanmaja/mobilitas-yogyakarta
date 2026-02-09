import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon
import rasterio
from rasterio.mask import mask
import yaml
from types import SimpleNamespace

# Configurations
with open("config.yaml") as f:
    config_dict = yaml.safe_load(f)
    cfg = SimpleNamespace(**{k: SimpleNamespace(**v) if isinstance(v, dict) else v 
                           for k, v in config_dict.items()})
    
def residential_analysis():
    print("Starting Yogyakarta enhanced origin analysis...")
    
    # DIY Boundary
    boundary_gdf = gpd.read_file(cfg.data_paths.boundary)  

    if boundary_gdf.crs != 'EPSG:32749':
        boundary_gdf = boundary_gdf.to_crs(epsg=32749)

    boundary = boundary_gdf
    west, south, east, north = boundary.total_bounds
    
    cell_size = cfg.cell_size
    cols = np.arange(west, east, cell_size)
    rows = np.arange(south, north, cell_size)
    
    polygons = []
    for x in cols:
        for y in rows:
            polygons.append(Polygon([
                (x, y), (x + cell_size, y),
                (x + cell_size, y + cell_size), (x, y + cell_size)
            ]))
    
    # Create grid GeoDataFrame
    grid = gpd.GeoDataFrame({'geometry': polygons}, crs=boundary.crs)
    grid = grid.to_crs(epsg=32749)  # 49S 
    grid['cell_id'] = range(len(grid))
    
    # Initialize intensity scores
    grid['residential_intensity'] = 0  
    
    print(f"Created {len(grid)} grid cells")
    
    # Residential intensity, inferred from WorldPop
    print("Getting residential intensity from WorldPop...")
    try:
        with rasterio.open(cfg.data_paths.worldpop) as src:
            residential_scores = []
            
            for idx, row in grid.iterrows():
                geom = [row.geometry]
                
                try:
                    out_img, _ = mask(src, geom, crop=True)
                    pop = out_img[0]
                    pop = np.where(pop < 0, 0, pop)
                    residential_scores.append(pop.sum())
                except:
                    residential_scores.append(0)

        grid['residential_intensity'] = residential_scores

        # Normalize to 0–100
        if grid['residential_intensity'].max() > 0:
            grid['residential_intensity'] = (
                grid['residential_intensity'] / grid['residential_intensity'].max()
            ) * 100

        print("Residential intensity assigned from WorldPop")
        
    except Exception as e:
        print(f"Error getting population data: {e}")
    

    print("Complete!")

    return grid, boundary

grid, boundary = residential_analysis()

# Plot results
fig, ax = plt.subplots(1, 1, figsize=(18, 12))

boundary.boundary.plot(ax=ax, color='black', linewidth=1, alpha=0.5) # Boundary outline

grid.plot(column='residential_intensity', cmap='Reds', legend=True, ax=ax)
ax.set_title('Residential Intensity', fontsize=16)

plt.tight_layout()
plt.savefig(cfg.figure_paths.resdidential, dpi=300, bbox_inches='tight')
plt.show()

# Save data to geojson
export_gdf = grid.copy()
export_gdf['geometry'] = export_gdf.geometry.centroid
export_gdf = export_gdf.to_crs(epsg=4326)  # To match mapbox's system
export_gdf.to_file(cfg.export_paths.residential, driver='GeoJSON')
print(cfg.export_paths.residential)

# Print some statistics
print("\n=== STATISTICS ===")
print(f"Total grid cells: {len(grid)}")
print(f"Residential intensity range: {grid['residential_intensity'].min():.1f} - {grid['residential_intensity'].max():.1f}")
print(f"Cells with residential intensity > 0: {(grid['residential_intensity'] > 0).sum()}")

categories = ['residential_intensity']
for category in categories:
    top_cells = grid.nlargest(3, category)[['cell_id', category]]
    print(f"\nTop 3 {category}:")
    print(top_cells.to_string(index=False))

print(f"Average residential intensity: {grid['residential_intensity'].mean():.2f}")

