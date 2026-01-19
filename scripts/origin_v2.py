import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon, Point
import pandas as pd
import rasterio
from rasterio.mask import mask

place = "Yogyakarta, Indonesia"

def simple_origin_analysis():
    print("Starting Yogyakarta origin analysis...")
    
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
    grid['origin_score'] = 0
    grid['population_score'] = 0
    grid['road_score'] = 0
    
    print(f"Created {len(grid)} grid cells")
    
    # WorldPop population data
    print("Getting population data...")
    try:
        print("Processing WorldPop raster...")

        reprojected_path = "data/raw/clipped_utm49s_2025_yogyakarta_100m.tif"
        with rasterio.open(reprojected_path) as src:
            population_scores = []
            
            for idx, row in grid.iterrows():
                geom = [row.geometry]
                
                try:
                    out_img, _ = mask(src, geom, crop=True)
                    pop = out_img[0]
                    pop = np.where(pop < 0, 0, pop)
                    population_scores.append(pop.sum())
                except:
                    population_scores.append(0)

        grid['population_score'] = population_scores

        # Normalize to 0–100
        if grid['population_score'].max() > 0:
            grid['population_score'] = (
                grid['population_score'] / grid['population_score'].max()
            ) * 100

        print("Population scores assigned from WorldPop")
        
        print("Generated population scores based on land use")
        
    except Exception as e:
        print(f"Error getting population data: {e}")
    
    print("Getting road network...")
    try:
        # Get all roads in DIY
        G = ox.graph_from_place(place, network_type='all', simplify=True)
        nodes, edges = ox.graph_to_gdfs(G)
        edges = edges.to_crs(grid.crs)
        

        # Check different possible column names
        highway_col = None
        for col in ['highway', 'highway_1', 'highway_2']:
            if col in edges.columns:
                highway_col = col
                break
        
        if highway_col:
            filtered_edges = edges.copy()
            
            edges['highway_norm'] = edges['highway'].apply(
                lambda x: x if isinstance(x, list) else [x]
            )

            # Filter out big roads (nobody live right next to it (even if there is, its negligible i think))
            excluded = {'trunk', 'primary', 'secondary', 'service', 'trunk_link', 'primary_link'}

            filtered_edges = edges[
                ~edges['highway_norm'].apply(
                    lambda lst: any(h in excluded for h in lst)
                )
            ]

            
            print(f"Filtered road network: {len(edges)} → {len(filtered_edges)} edges")
            
            # Calculate road density per cell
            road_density = []
            
            for idx, row in grid.iterrows():
                cell_geom = row['geometry']
                
                # Find roads that intersect this cell
                intersecting_roads = filtered_edges[filtered_edges.intersects(cell_geom)]
                
                if not intersecting_roads.empty:
                    # Calculate total road length in this cell
                    total_length = 0
                    for _, road in intersecting_roads.iterrows():
                        try:
                            # Intersect road with cell
                            intersection = road.geometry.intersection(cell_geom)
                            if intersection.length > 0:
                                total_length += intersection.length
                        except:
                            continue
                    
                    road_density.append(total_length / (cell_size * cell_size))  # meters per sq km
                else:
                    road_density.append(0)
            
            grid['road_density'] = road_density
            
            # Normalize road density to 0-100 scale
            if grid['road_density'].max() > 0:
                grid['road_score'] = (grid['road_density'] / grid['road_density'].max()) * 100
            else:
                grid['road_score'] = 0
                
            print(f"Calculated road density for {len(filtered_edges)} road segments")
            
        else:
            print("Could not find highway column in edges data")
            grid['road_score'] = 0
            
    except Exception as e:
        print(f"Error processing road network: {e}")
        grid['road_score'] = 0
    
    #  Combine scores for origin attractiveness (70% population, 30% roads)
    print("Calculating combined origin scores...")
    
    # Normalize scores (0-1 range)
    if grid['population_score'].max() > 0:
        grid['pop_norm'] = grid['population_score'] / grid['population_score'].max()
    else:
        grid['pop_norm'] = 0
    
    if grid['road_score'].max() > 0:
        grid['road_norm'] = grid['road_score'] / grid['road_score'].max()
    else:
        grid['road_norm'] = 0
    
    # Combine: 70% population, 30% road density
    grid['origin_score'] = (grid['pop_norm'] * 0.7 + grid['road_norm'] * 0.3) * 100
    
    print("Analysis complete!")
    print(f"Origin scores range: {grid['origin_score'].min():.1f} - {grid['origin_score'].max():.1f}")
    
    return grid, boundary, filtered_edges if 'filtered_edges' in locals() else None

grid, boundary, roads = simple_origin_analysis()

# Plot results
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Boundary outline for all plots
for i in range(6):
    boundary.boundary.plot(ax=axes.flatten()[i], color='black', linewidth=1, alpha=0.5)

# Plot 1: Population Score
grid.plot(column='population_score', cmap='Reds', legend=True, ax=axes[0, 0])
axes[0, 0].set_title('Population Density (Simulated)')

# Plot 2: Road Density Score
grid.plot(column='road_score', cmap='Greens', legend=True, ax=axes[0, 1])
axes[0, 1].set_title('Road Density (Excl. Primary/Service)')

# Plot 3: Combined Origin Score
grid.plot(column='origin_score', cmap='Oranges', legend=True, ax=axes[0, 2])
axes[0, 2].set_title('Combined Origin Score')

# Plot 4: Road Network
if roads is not None and len(roads) > 0:
    roads.plot(ax=axes[1, 0], linewidth=0.5, color='blue', alpha=0.7)
axes[1, 0].set_title('Local Road Network')

# Plot 5: Grid cells with high origin scores
# Highlight top 20% of cells
threshold = grid['origin_score'].quantile(0.8)
high_origin = grid[grid['origin_score'] >= threshold]
high_origin.plot(ax=axes[1, 1], color='gold', alpha=0.7)
axes[1, 1].set_title(f'Top 20% Origin Cells (Score ≥ {threshold:.1f})')

# Plot 6: Boundary with grid overlay
grid.boundary.plot(ax=axes[1, 2], linewidth=0.2, color='gray', alpha=0.3)
grid.plot(column='origin_score', cmap='Oranges', alpha=0.5, legend=True, ax=axes[1, 2])
axes[1, 2].set_title('Grid Overlay with Origin Scores')

# Adjust layout
plt.tight_layout()
plt.savefig('data/figures/origin_v2_1_700m.png', dpi=300, bbox_inches='tight')
plt.show()

# Save data to geojson
export_gdf = grid.copy()
export_gdf['geometry'] = export_gdf.geometry.centroid
export_gdf = export_gdf.to_crs(epsg=4326) # To match mapbox's system
export_gdf.to_file('data/raw/origin_v2_1_700m.geojson', driver='geoJSON')
print("Data saved to 'data/raw/origin_v2_1_700m.geojson'")

# Print some statistics
print("\n=== STATISTICS ===")
print(f"Total grid cells: {len(grid)}")
print(f"Cells with population score > 0: {(grid['population_score'] > 0).sum()}")
print(f"Cells with road score > 0: {(grid['road_score'] > 0).sum()}")
print(f"Cells with origin score > 0: {(grid['origin_score'] > 0).sum()}")
print(f"Average origin score: {grid['origin_score'].mean():.2f}")

# Find top 5 origin cells
top_origins = grid.nlargest(5, 'origin_score')[['cell_id', 'origin_score', 'population_score', 'road_score']]
print("\nTop 5 origin cells:")
print(top_origins.to_string(index=False))

# Additional analysis
print("\n=== ADDITIONAL ANALYSIS ===")
print(f"Maximum road density: {grid['road_density'].max():.2f} m/km²")
print(f"Average road density: {grid['road_density'].mean():.2f} m/km²")
print(f"Maximum population score: {grid['population_score'].max():.0f}")
print(f"Cells in top 20%: {len(high_origin)}")