import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon, Point
import pandas as pd
import rasterio
from rasterio.mask import mask
from shapely.geometry import box
import warnings
warnings.filterwarnings('ignore')

place = "Yogyakarta, Indonesia"

def employment_analysis():
    print("Starting Yogyakarta enhanced origin analysis...")
    
    # Get boundary of DIY
    boundary = ox.geocode_to_gdf(place)
    boundary = boundary.to_crs(epsg=32749)
    west, south, east, north = boundary.total_bounds
    
    # Create grid
    cell_size = 1000  # meter
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
    grid['employment_intensity'] = 0  # From GHSL
    grid['viirs_intensity'] = 0      # From VIIRS
    grid['combined_intensity'] = 0   # Combined score
    
    print(f"Created {len(grid)} grid cells")
    
    # RESIDENTIAL INTENSITY (from GHSL)
    print("Getting employment intensity from GHSL...")
    try:
        reprojected_path = "data/raw/GHSL/Cropped_GHS_BUILT_V_NRES_E2025_GLOBE_R2023A_32749_3ss_V1_0_R10_C30.tif"
        with rasterio.open(reprojected_path) as src:
            employment_scores = []
            
            for idx, row in grid.iterrows():
                geom = [row.geometry]
                
                try:
                    out_img, _ = mask(src, geom, crop=True)
                    pop = out_img[0]
                    pop = np.where(pop < 0, 0, pop)
                    employment_scores.append(pop.sum())
                except:
                    employment_scores.append(0)

        grid['employment_intensity'] = employment_scores

        # Square root normalization
        grid['employment_intensity'] = np.sqrt(grid['employment_intensity'])

        # Normalize to 0-100 scale
        sqrt_max = grid['employment_intensity'].max()
        sqrt_min = grid['employment_intensity'].min()

        if sqrt_max > sqrt_min:
            grid['employment_intensity'] = (
                (grid['employment_intensity'] - sqrt_min) / (sqrt_max - sqrt_min)
            ) * 100
        else:
            grid['employment_intensity'] = 0

        print("Employment intensity assigned from GHSL")
        
    except Exception as e:
        print(f"Error getting population data: {e}")
    

    grid = load_viirs_data(grid, boundary)
    
    # Combine GHSL and VIIRS data
    print("Combining GHSL and VIIRS data...")
    
    # Take sqrt of normalized values and multiply
    ghsl_norm = grid['employment_intensity'].fillna(0) / 100.0
    viirs_norm = grid['viirs_intensity'].fillna(0) / 100.0
    
    # Combined score: sqrt(GHSL) * sqrt(VIIRS)
    combined = ghsl_norm * viirs_norm
    
    # Log1p transformation on combined data
    grid['combined_intensity'] = np.log1p(combined * 100)  
    
    # Normalize combined score to 0-100
    combined_max = grid['combined_intensity'].max()
    combined_min = grid['combined_intensity'].min()
    
    if combined_max > combined_min:
        grid['combined_intensity'] = (
            (grid['combined_intensity'] - combined_min) / (combined_max - combined_min)
        ) * 100
    else:
        grid['combined_intensity'] = 0
    
    print("Combined intensity calculated")
    print("Analysis complete!")

    return grid, boundary

def load_viirs_data(grid, boundary):
    print("Getting VIIRS nighttime lights data...")
    
    try:
        viirs_path = "data/raw/VIIRS/Cropped_reproj_VNL_npp_2024_global_vcmslcfg_v2_c202502261200.average_masked.dat.tif"  
        
        with rasterio.open(viirs_path) as src:
            viirs_scores = []
            
            for idx, row in grid.iterrows():
                geom = [row.geometry]
                
                try:
                    out_img, _ = mask(src, geom, crop=True)
                    lights = out_img[0]
                    lights = np.where(lights < 0, 0, lights)
                    
                    # Gentle capping: Use 95th percentile to prevent extreme outliers
                    cap_value = np.percentile(lights[lights > 0], 95) if np.any(lights > 0) else 0
                    if cap_value > 0:
                        lights = np.where(lights > cap_value, cap_value, lights)
                    
                    viirs_scores.append(lights.sum())
                except:
                    viirs_scores.append(0)

        grid['viirs_intensity'] = viirs_scores
        
        # Square root normalization
        # grid['viirs_intensity'] = np.sqrt(grid['viirs_intensity'])
        
        # Normalize to 0-100 scale
        viirs_max = grid['viirs_intensity'].max()
        viirs_min = grid['viirs_intensity'].min()
        
        if viirs_max > viirs_min:
            grid['viirs_intensity'] = (
                (grid['viirs_intensity'] - viirs_min) / (viirs_max - viirs_min)
            ) * 100
        else:
            grid['viirs_intensity'] = 0
            
        print("VIIRS data processed and normalized")
        return grid
        
    except Exception as e:
        print(f"Error loading VIIRS data: {e}")
        grid['viirs_intensity'] = 0
        return grid


grid, boundary = employment_analysis()

# Plot results - 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(24, 8))

# Plot 1: GHSL Employment Intensity
boundary.boundary.plot(ax=axes[0], color='black', linewidth=1, alpha=0.5)
grid.plot(column='employment_intensity', cmap='Reds', legend=True, ax=axes[0])
axes[0].set_title('GHSL Employment Intensity', fontsize=14)

# Plot 2: VIIRS Nighttime Lights
boundary.boundary.plot(ax=axes[1], color='black', linewidth=1, alpha=0.5)
grid.plot(column='viirs_intensity', cmap='Reds', legend=True, ax=axes[1])
axes[1].set_title('VIIRS Nighttime Lights', fontsize=14)

# Plot 3: Combined Intensity
boundary.boundary.plot(ax=axes[2], color='black', linewidth=1, alpha=0.5)
grid.plot(column='combined_intensity', cmap='viridis', legend=True, ax=axes[2])
axes[2].set_title('Combined Intensity (sqrt(GHSL) * sqrt(VIIRS))', fontsize=14)

plt.tight_layout()
plt.savefig('data/figures/employment_combined_1000m.png', dpi=300, bbox_inches='tight')
plt.show()

# Save data to geojson
export_gdf = grid.copy()
export_gdf['geometry'] = export_gdf.geometry.centroid
export_gdf = export_gdf.to_crs(epsg=4326)  # To match mapbox's system
export_gdf.to_file('data/raw/employment_combined_1000m.geojson', driver='GeoJSON')
print("Data saved to 'data/raw/employment_combined_1000m.geojson'")

# Print statistics
print("\n=== STATISTICS ===")
print(f"Total grid cells: {len(grid)}")
print(f"GHSL intensity range: {grid['employment_intensity'].min():.1f} - {grid['employment_intensity'].max():.1f}")
print(f"VIIRS intensity range: {grid['viirs_intensity'].min():.1f} - {grid['viirs_intensity'].max():.1f}")
print(f"Combined intensity range: {grid['combined_intensity'].min():.1f} - {grid['combined_intensity'].max():.1f}")
print(f"Cells with GHSL intensity > 0: {(grid['employment_intensity'] > 0).sum()}")
print(f"Cells with VIIRS intensity > 0: {(grid['viirs_intensity'] > 0).sum()}")

# Additional analysis
categories = ['employment_intensity', 'viirs_intensity', 'combined_intensity']
for category in categories:
    top_cells = grid.nlargest(3, category)[['cell_id', category]]
    print(f"\nTop 3 {category}:")
    print(top_cells.to_string(index=False))

print(f"\nAverage GHSL intensity: {grid['employment_intensity'].mean():.2f}")
print(f"Average VIIRS intensity: {grid['viirs_intensity'].mean():.2f}")
print(f"Average combined intensity: {grid['combined_intensity'].mean():.2f}")