import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon, Point
import pandas as pd

place = "Yogyakarta, Indonesia"

def simple_grid_analysis():

    print("Starting Yogyakarta grid analysis...")
    
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
    grid = grid.to_crs(epsg=32749)  # UTM 49S 
    grid['cell_id'] = range(len(grid))
    grid['destination_score'] = 0
    grid['node_score'] = 0
    grid['poi_score'] = 0
    grid['road_score'] = 0
    grid['special_score'] = 0
    
    print(f"Created {len(grid)} grid cells")
    
    # Get street network
    print("Getting street network...")
    G = ox.graph_from_place(place, network_type='all', simplify=True)
    nodes, edges = ox.graph_to_gdfs(G)
    nodes['geometry'] = nodes.geometry.centroid
    nodes = nodes.to_crs(grid.crs)
    edges = edges.to_crs(grid.crs)
    
    # Spatial join to count nodes per cell
    joined_nodes = gpd.sjoin(grid, nodes, how='left', predicate='contains')
    node_counts = joined_nodes.groupby('cell_id').size()
    
    # Store node scores
    for cell_id, count in node_counts.items():
        grid.loc[grid['cell_id'] == cell_id, 'node_score'] = count
    
    print(f"Found {len(nodes)} street nodes")
    
    # Score primary and secondary roads (shop-lined streets, a common sight here in Yogyakarta)
    print("Scoring primary and secondary roads...")
    
    # Filter for primary and secondary roads
    commercial_roads = edges[edges['highway'].isin([
        'primary', 'secondary', 
        'primary_link', 'secondary_link'
    ])].copy()
    
    if not commercial_roads.empty:
        # Calculate road length per cell
        for idx, row in grid.iterrows():
            cell_geom = row.geometry
            # Find roads that intersect this cell
            intersecting = commercial_roads[commercial_roads.intersects(cell_geom)]
            
            if len(intersecting) > 0:
                # Sum up the length of road segments in this cell
                total_length = 0
                for _, road in intersecting.iterrows():
                    intersection = road.geometry.intersection(cell_geom)
                    total_length += intersection.length
                
                grid.loc[idx, 'road_score'] = total_length
        
        print(f"Found {len(commercial_roads)} primary/secondary road segments")
    
    # Add Malioboro as a special landmark bcs it isnt tagged in OSM (Might wanna add other landmarks later on)
    print("Adding special landmarks...")
    
    special_locations = [
        {'name': 'Malioboro', 'lat': -7.795203, 'lon': 110.365659, 'weight': 10, 'radius': 500},
    ]
    
    # Create GeoDataFrame for the special landmark
    special_gdf = gpd.GeoDataFrame(
        special_locations,
        geometry=[Point(loc['lon'], loc['lat']) for loc in special_locations],
        crs='EPSG:4326'
    ).to_crs(grid.crs)
    
    # Apply weighted distance decay to grid cells
    for idx, landmark in special_gdf.iterrows():
        center = landmark.geometry
        weight = landmark['weight']
        radius = landmark['radius']
        
        for cell_idx, cell in grid.iterrows():
            cell_center = cell.geometry.centroid
            distance = center.distance(cell_center)
            
            # Distance decay function: score decreases with distance
            if distance < radius:
                # Linear decay from weight to 0 within radius
                score = weight * (1 - distance / radius)
                grid.loc[cell_idx, 'special_score'] += score
    
    print(f"Added {len(special_locations)} special landmarks")
    
    # Get Point of Interests using OSM tags
    print("Getting POIs...")
    
    tags_list = [
        {'amenity': ['university', 'college', 'school']},  # Education
        # Commercial
        {'amenity': ['restaurant', 'cafe', 'fast_food', 'marketplace']}, 
        {'shop': ['mall', 'supermarket', 'convenience']},  
        # Tourism
        {'tourism': ['hotel', 'guest_house', 'attraction', 'museum', 'zoo', 'theme_park']},  
        {'aeroway': ['aerodome']},
        {'railway': ['station']},
        {'leisure': ['pitch', 'park', 'golf_course']},
        {'natural': ['beach']},
        {'amenity': ['hospital', 'clinic']},  # Healthcare
        {'amenity': ['place_of_worship']},  # Mosques, temples
        {'office': ['company', 'government']},  # Offices
        {'landuse': ['industrial']} 
    ]
    
    all_pois = gpd.GeoDataFrame()
    
    for tags in tags_list:
        try:
            pois = ox.features_from_place(place, tags)
            # Keep only point geometries
            pois = pois[pois.geometry.type == 'Point']
            all_pois = pd.concat([all_pois, pois], ignore_index=True)
            print(f"Found {len(pois)} POIs for tags: {tags}")
        except Exception as e:
            print(f"Couldn't get POIs for {tags}: {e}")
    
    # Remove duplicates and convert CRS
    if not all_pois.empty:
        # Remove exact duplicates
        all_pois = all_pois.drop_duplicates(subset=['geometry'])
        all_pois = all_pois.to_crs(grid.crs)
        
        # Count POIs per cell
        joined_pois = gpd.sjoin(grid, all_pois, how='left', predicate='contains')
        poi_counts = joined_pois.groupby('cell_id').size()
        
        # Store POI scores
        for cell_id, count in poi_counts.items():
            grid.loc[grid['cell_id'] == cell_id, 'poi_score'] = count
        
        print(f"Found {len(all_pois)} unique POIs")
    else:
        print("Warning: No POIs found")
        grid['poi_score'] = 0
    
    # 6. Combine scores for destination attractiveness
    print("Calculating combined scores...")
    
    # Apply non-linear scaling before normalization
    # Square root for POIs (diminishing returns, many small POIs don't equal one major destination)
    grid['poi_scaled'] = np.sqrt(grid['poi_score'])
    
    # Square root for nodes too (intersection density has diminishing returns)
    grid['node_scaled'] = np.sqrt(grid['node_score'])
    
    # Linear for roads (road length is pretty directly proportional)
    grid['road_scaled'] = grid['road_score']
    
    # Power 1.2 for special landmarks (amplify importance of major destinations)
    grid['special_scaled'] = grid['special_score'] ** 1.2
    
    # NOW normalize the scaled scores (0-1 range)
    if grid['node_scaled'].max() > 0:
        grid['node_norm'] = grid['node_scaled'] / grid['node_scaled'].max()
    else:
        grid['node_norm'] = 0
    
    if grid['poi_scaled'].max() > 0:
        grid['poi_norm'] = grid['poi_scaled'] / grid['poi_scaled'].max()
    else:
        grid['poi_norm'] = 0
    
    if grid['road_scaled'].max() > 0:
        grid['road_norm'] = grid['road_scaled'] / grid['road_scaled'].max()
    else:
        grid['road_norm'] = 0
    
    if grid['special_scaled'].max() > 0:
        grid['special_norm'] = grid['special_scaled'] / grid['special_scaled'].max()
    else:
        grid['special_norm'] = 0
    
    # Combine with adjusted weights:
    # 15% street network nodes (√scaled)
    # 45% POIs (√scaled - diminishing returns)
    # 20% commercial roads (linear)
    # 20% special landmarks (^1.2 scaled - amplified importance)
    grid['destination_score'] = (
        grid['node_norm'] * 0.15 + 
        grid['poi_norm'] * 0.45 + 
        grid['road_norm'] * 0.20 +
        grid['special_norm'] * 0.20
    ) * 100
    
    # Also create a linear version for comparison
    grid['node_norm_linear'] = grid['node_score'] / grid['node_score'].max() if grid['node_score'].max() > 0 else 0
    grid['poi_norm_linear'] = grid['poi_score'] / grid['poi_score'].max() if grid['poi_score'].max() > 0 else 0
    grid['road_norm_linear'] = grid['road_score'] / grid['road_score'].max() if grid['road_score'].max() > 0 else 0
    grid['special_norm_linear'] = grid['special_score'] / grid['special_score'].max() if grid['special_score'].max() > 0 else 0
    
    grid['destination_score_linear'] = (
        grid['node_norm_linear'] * 0.15 + 
        grid['poi_norm_linear'] * 0.45 + 
        grid['road_norm_linear'] * 0.20 +
        grid['special_norm_linear'] * 0.20
    ) * 100
    
    print("Analysis complete!")
    print(f"Non-linear destination scores range: {grid['destination_score'].min():.1f} - {grid['destination_score'].max():.1f}")
    print(f"Linear destination scores range: {grid['destination_score_linear'].min():.1f} - {grid['destination_score_linear'].max():.1f}")
    
    return grid, boundary, nodes, all_pois if not all_pois.empty else None, special_gdf

# Run the analysis
grid, boundary, nodes, pois, special_locs = simple_grid_analysis()

# Plot results - comparing linear vs non-linear
fig, axes = plt.subplots(2, 4, figsize=(24, 12))

# Boundary outline for all plots
for i in range(8):
    boundary.boundary.plot(ax=axes.flatten()[i], color='black', linewidth=1, alpha=0.5)

# Plot 1: Street Network Score
grid.plot(column='node_score', cmap='Oranges', legend=True, ax=axes[0, 0])
axes[0, 0].set_title('Street Network Density (raw)')

# Plot 2: POI Score
grid.plot(column='poi_score', cmap='Purples', legend=True, ax=axes[0, 1])
axes[0, 1].set_title('POI Density (raw)')

# Plot 3: Road Score
grid.plot(column='road_score', cmap='Greens', legend=True, ax=axes[0, 2])
axes[0, 2].set_title('Primary/Secondary Road Score')

# Plot 4: Special Landmarks
grid.plot(column='special_score', cmap='Reds', legend=True, ax=axes[0, 3])
special_locs.plot(ax=axes[0, 3], color='yellow', markersize=100, marker='*', edgecolor='black')
axes[0, 3].set_title('Special Landmarks Score')

# Plot 5: Non-Linear Combined Score
grid.plot(column='destination_score', cmap='Blues', legend=True, ax=axes[1, 0])
axes[1, 0].set_title('Non-Linear Combined Score\n(√POI, √nodes, ^1.2 special)')

# Plot 6: Linear Combined Score (for comparison)
grid.plot(column='destination_score_linear', cmap='Blues', legend=True, ax=axes[1, 1])
axes[1, 1].set_title('Linear Combined Score\n(for comparison)')

# Plot 7: Difference between methods
grid['score_diff'] = grid['destination_score'] - grid['destination_score_linear']
grid.plot(column='score_diff', cmap='RdYlGn', legend=True, ax=axes[1, 2])
axes[1, 2].set_title('Difference (Non-linear - Linear)\nGreen = Non-linear favors\nRed = Linear favors')

# Plot 8: All POIs
if pois is not None and not pois.empty:
    pois.plot(ax=axes[1, 3], markersize=2, color='green', alpha=0.5)
axes[1, 3].set_title('POI Locations')

# Adjust layout
plt.tight_layout()
plt.savefig('data/figures/destination_v3_1000.png', dpi=300, bbox_inches='tight')
plt.show()

# Save data to geojson
export_gdf = grid.copy() 
export_gdf['geometry'] = export_gdf.geometry.centroid
export_gdf = export_gdf.to_crs(epsg=4326) # To match mapbox's system
export_gdf.to_file('data/destination_v3_1000m.geojson', driver='GeoJSON')
print("Data saved to 'data/destination_v3_1000m.geojson'")

# Print some statistics
print("\n=== STATISTICS ===")
print(f"Total grid cells: {len(grid)}")
print(f"Cells with destination score > 0: {(grid['destination_score'] > 0).sum()}")
print(f"Average destination score (non-linear): {grid['destination_score'].mean():.2f}")
print(f"Average destination score (linear): {grid['destination_score_linear'].mean():.2f}")

# Compare top cells between methods
print("\n=== TOP 10 CELLS - NON-LINEAR METHOD ===")
top_dest = grid.nlargest(10, 'destination_score')[
    ['cell_id', 'destination_score', 'node_score', 'poi_score', 'road_score', 'special_score']
]
print(top_dest.to_string(index=False))

print("\n=== TOP 10 CELLS - LINEAR METHOD ===")
top_dest_linear = grid.nlargest(10, 'destination_score_linear')[
    ['cell_id', 'destination_score_linear', 'node_score', 'poi_score', 'road_score', 'special_score']
]
print(top_dest_linear.to_string(index=False))

# Show which cells moved the most
print("\n=== BIGGEST SCORE CHANGES (Non-linear vs Linear) ===")
grid['rank_nonlinear'] = grid['destination_score'].rank(ascending=False)
grid['rank_linear'] = grid['destination_score_linear'].rank(ascending=False)
grid['rank_change'] = grid['rank_linear'] - grid['rank_nonlinear']

print("\nCells that improved most with non-linear (likely major destinations):")
improved = grid.nlargest(5, 'rank_change')[['cell_id', 'destination_score', 'poi_score', 'special_score', 'rank_change']]
print(improved.to_string(index=False))

print("\nCells that dropped with non-linear (likely many minor destinations):")
dropped = grid.nsmallest(5, 'rank_change')[['cell_id', 'destination_score', 'poi_score', 'special_score', 'rank_change']]
print(dropped.to_string(index=False))