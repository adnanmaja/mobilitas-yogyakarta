import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon
import rasterio
from rasterio.mask import mask
import warnings
from scripts.grids.grid_config import Config
warnings.filterwarnings('ignore')

# Configurations
config = Config.from_yaml()

def employment_analysis():
    print("Starting the employment analysis...")
    
    boundary_gdf = gpd.read_file(config.data_paths['boundary'])  

    if boundary_gdf.crs != 'EPSG:32749':
        boundary_gdf = boundary_gdf.to_crs(epsg=32749)

    boundary = boundary_gdf
    west, south, east, north = boundary.total_bounds
    
    cell_size = config.cell_size
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
    grid['ghsl_intensity'] = 0  
    grid['viirs_intensity'] = 0      
    grid['employment_intensity'] = 0   # Combined score
    
    print(f"Created {len(grid)} grid cells")
    
    # Employment analysis, inferred from GHSL and VIIRS
    print("Getting employment intensity from GHSL...")
    try:
        with rasterio.open(config.data_paths['ghsl']) as src:
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

        grid['ghsl_intensity'] = employment_scores

        # Square root normalization
        grid['ghsl_intensity'] = np.sqrt(grid['ghsl_intensity'])

        # Normalize to 0-100 scale
        sqrt_max = grid['ghsl_intensity'].max()
        sqrt_min = grid['ghsl_intensity'].min()

        if sqrt_max > sqrt_min:
            grid['ghsl_intensity'] = (
                (grid['ghsl_intensity'] - sqrt_min) / (sqrt_max - sqrt_min)
            ) * 100
        else:
            grid['ghsl_intensity'] = 0

        print("Employment intensity assigned from GHSL")
        
    except Exception as e:
        print(f"Error getting population data: {e}")
    

    grid = load_viirs_data(grid, boundary)
    
    # Combine GHSL and VIIRS data
    print("Combining GHSL and VIIRS data...")
    
    # Take sqrt of normalized values and multiply
    ghsl_norm = grid['ghsl_intensity'].fillna(0) / 100.0
    viirs_norm = grid['viirs_intensity'].fillna(0) / 100.0
    
    # Combined score using euclidean norm
    # combined = np.sqrt(np.square(ghsl_norm) + np.square(viirs_norm))

    # Combined score using fuzzy logic / harmonic mean
    # combined = 2 * ((ghsl_norm * viirs_norm/ghsl_norm + viirs_norm + 0.00001))

    # Combined score using weighted linear
    combined = (ghsl_norm * config.grid_weights['ghsl']) + (viirs_norm * config.grid_weights['viirs'])
    
    # Log1p transformation on combined data
    grid['employment_intensity'] = combined 
    
    # Normalize combined score to 0-100
    combined_max = grid['employment_intensity'].max()
    combined_min = grid['employment_intensity'].min()
    
    if combined_max > combined_min:
        grid['employment_intensity'] = (
            (grid['employment_intensity'] - combined_min) / (combined_max - combined_min)
        ) * 100
    else:
        grid['employment_intensity'] = 0
    
    print("Combined intensity calculated")
    print("Analysis complete!")

    return grid, boundary

def load_viirs_data(grid, boundary):
    print("Getting VIIRS nighttime lights data...")
    
    try:
        with rasterio.open(config.data_paths['viirs']) as src:
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

# Plot results 
fig, axes = plt.subplots(1, 3, figsize=(24, 8))

# Plot 1: GHSL Employment Intensity
boundary.boundary.plot(ax=axes[0], color='black', linewidth=1, alpha=0.5)
grid.plot(column='ghsl_intensity', cmap='Reds', legend=True, ax=axes[0])
axes[0].set_title('GHSL Employment Intensity', fontsize=14)

# Plot 2: VIIRS Nighttime Lights
boundary.boundary.plot(ax=axes[1], color='black', linewidth=1, alpha=0.5)
grid.plot(column='viirs_intensity', cmap='Reds', legend=True, ax=axes[1])
axes[1].set_title('VIIRS Nighttime Lights', fontsize=14)

# Plot 3: Combined Intensity
boundary.boundary.plot(ax=axes[2], color='black', linewidth=1, alpha=0.5)
grid.plot(column='employment_intensity', cmap='viridis', legend=True, ax=axes[2])
axes[2].set_title('Combined Intensity (sqrt(GHSL) * sqrt(VIIRS))', fontsize=14)

plt.tight_layout()
plt.savefig(config.figure_paths['employment'], dpi=300, bbox_inches='tight')
plt.show()

# Save data to geojson
export_gdf = grid.copy()
export_gdf['geometry'] = export_gdf.geometry.centroid
export_gdf = export_gdf.to_crs(epsg=4326)  # To match mapbox's system
export_gdf.to_file(config.export_paths['employment'], driver='GeoJSON')
print(config.export_paths['employment'])

# Print statistics
print("\n=== STATISTICS ===")
print(f"Total grid cells: {len(grid)}")
print(f"GHSL intensity range: {grid['ghsl_intensity'].min():.1f} - {grid['ghsl_intensity'].max():.1f}")
print(f"VIIRS intensity range: {grid['viirs_intensity'].min():.1f} - {grid['viirs_intensity'].max():.1f}")
print(f"Combined intensity range: {grid['employment_intensity'].min():.1f} - {grid['employment_intensity'].max():.1f}")
print(f"Cells with Combined intensity > 0: {(grid['employment_intensity'] > 0).sum()}")
print(f"Cells with GHSL intensity > 0: {(grid['ghsl_intensity'] > 0).sum()}")
print(f"Cells with VIIRS intensity > 0: {(grid['viirs_intensity'] > 0).sum()}")

# Additional analysis
categories = ['ghsl_intensity', 'viirs_intensity', 'employment_intensity']
for category in categories:
    top_cells = grid.nlargest(3, category)[['cell_id', category]]
    print(f"\nTop 3 {category}:")
    print(top_cells.to_string(index=False))

print(f"\nAverage GHSL intensity: {grid['ghsl_intensity'].mean():.2f}")
print(f"Average VIIRS intensity: {grid['viirs_intensity'].mean():.2f}")
print(f"Average combined intensity: {grid['employment_intensity'].mean():.2f}")