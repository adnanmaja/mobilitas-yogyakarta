import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon, Point
import pandas as pd

place = "Yogyakarta, Indonesia"


POI_TEMPORAL_PROFILES = {
    # different destinations attract traffic at different time
    # (peak_weight, off_peak_weight, weekend_weight)

    'university': (1.0, 0.3, 0.1),
    'school': (1.0, 0.2, 0.0),
    'office': (1.0, 0.3, 0.2),
    'government': (0.9, 0.4, 0.1),

    'restaurant': (0.9, 0.7, 1.0),
    'cafe': (0.8, 0.6, 1.0),
    'mall': (0.8, 0.6, 1.2),
    'supermarket': (0.7, 0.5, 0.9),
    'marketplace': (1.0, 0.6, 1.1),

    'museum': (0.6, 0.8, 1.2),
    'attraction': (0.7, 0.9, 1.3),
    'hotel': (0.5, 0.5, 0.8),
    'theme_park': (0.7, 0.8, 1.5),
    
    'hospital': (1.0, 1.0, 1.0),
    'place_of_worship': (0.5, 0.3, 1.2),
    
    'default': (0.7, 0.5, 0.8)
}

# Getting POIs using OSM tags
def categorize_poi(poi_row):
    if 'amenity' in poi_row and pd.notna(poi_row['amenity']):
        amenity = poi_row['amenity']
        if amenity in ['university', 'college']:
            return 'university'
        elif amenity == 'school':
            return 'school'
        elif amenity in ['restaurant', 'fast_food']:
            return 'restaurant'
        elif amenity == 'cafe':
            return 'cafe'
        elif amenity in ['hospital', 'clinic']:
            return 'hospital'
        elif amenity == 'place_of_worship':
            return 'place_of_worship'
        elif amenity == 'marketplace':
            return 'marketplace'
    
    if 'shop' in poi_row and pd.notna(poi_row['shop']):
        shop = poi_row['shop']
        if shop == 'mall':
            return 'mall'
        elif shop in ['supermarket', 'convenience']:
            return 'supermarket'
    
    if 'tourism' in poi_row and pd.notna(poi_row['tourism']):
        tourism = poi_row['tourism']
        if tourism == 'museum':
            return 'museum'
        elif tourism in ['attraction', 'zoo', 'theme_park']:
            return 'attraction'
        elif tourism in ['hotel', 'guest_house']:
            return 'hotel'
    
    if 'office' in poi_row and pd.notna(poi_row['office']):
        if poi_row['office'] == 'government':
            return 'government'
        return 'office'
    
    return 'default'

# Calculate POI scores for different time categories
def calculate_temporal_poi_scores(grid, all_pois, time_category='peak'):

    # Map time category to profile index
    profile_idx = {'peak': 0, 'off_peak': 1, 'weekend': 2}[time_category]
    
    # Initialize temporal score column
    score_col = f'poi_score_{time_category}'
    grid[score_col] = 0.0
    
    # Score each POI with temporal weight
    for idx, poi in all_pois.iterrows():
        poi_type = categorize_poi(poi)
        temporal_profile = POI_TEMPORAL_PROFILES.get(poi_type, POI_TEMPORAL_PROFILES['default'])
        weight = temporal_profile[profile_idx]
        
        # Find which cell contains this POI
        cell_idx = grid[grid.contains(poi.geometry)].index
        if len(cell_idx) > 0:
            grid.loc[cell_idx[0], score_col] += weight
    
    return grid

def simple_grid_analysis():

    print("Starting grid analysis...")
    
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
    grid['node_score'] = 0
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
    
    # Score primary and secondary roads
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
            intersecting = commercial_roads[commercial_roads.intersects(cell_geom)]
            
            if len(intersecting) > 0:
                total_length = 0
                for _, road in intersecting.iterrows():
                    intersection = road.geometry.intersection(cell_geom)
                    total_length += intersection.length
                
                grid.loc[idx, 'road_score'] = total_length
        
        print(f"Found {len(commercial_roads)} primary/secondary road segments")
    
    # Add special landmarks (OSM tags doesnt recognize malioboro area)
    print("Adding special landmarks...")
    special_locations = [
        {'name': 'Malioboro', 'lat': -7.795203, 'lon': 110.365659, 'weight': 10, 'radius': 500},
    ]
    
    special_gdf = gpd.GeoDataFrame(
        special_locations,
        geometry=[Point(loc['lon'], loc['lat']) for loc in special_locations],
        crs='EPSG:4326'
    ).to_crs(grid.crs)
    
    # Apply weighted distance decay
    for idx, landmark in special_gdf.iterrows():
        center = landmark.geometry
        weight = landmark['weight']
        radius = landmark['radius']
        
        for cell_idx, cell in grid.iterrows():
            cell_center = cell.geometry.centroid
            distance = center.distance(cell_center)
            
            if distance < radius:
                score = weight * (1 - distance / radius)
                grid.loc[cell_idx, 'special_score'] += score
    
    print(f"Added {len(special_locations)} special landmarks")
    
    # Get POIs
    print("Getting POIs...")
    
    tags_list = [
        {'amenity': ['university', 'college', 'school']},
        {'amenity': ['restaurant', 'cafe', 'fast_food', 'marketplace']}, 
        {'shop': ['mall', 'supermarket', 'convenience']},  
        {'tourism': ['hotel', 'guest_house', 'attraction', 'museum', 'zoo', 'theme_park']},  
        {'aeroway': ['aerodome']},
        {'railway': ['station']},
        {'leisure': ['pitch', 'park', 'golf_course']},
        {'natural': ['beach']},
        {'amenity': ['hospital', 'clinic']},
        {'amenity': ['place_of_worship']},
        {'office': ['company', 'government']},
        {'landuse': ['industrial']} 
    ]
    
    all_pois = gpd.GeoDataFrame()
    
    for tags in tags_list:
        try:
            pois = ox.features_from_place(place, tags)
            pois = pois[pois.geometry.type == 'Point']
            all_pois = pd.concat([all_pois, pois], ignore_index=True)
            print(f"Found {len(pois)} POIs for tags: {tags}")
        except Exception as e:
            print(f"Couldn't get POIs for {tags}: {e}")
    
    # Process POIs
    if not all_pois.empty:
        all_pois = all_pois.drop_duplicates(subset=['geometry'])
        all_pois = all_pois.to_crs(grid.crs)
        
        # Calculate temporal scores for each time period
        for time_cat in ['peak', 'off_peak', 'weekend']:
            grid = calculate_temporal_poi_scores(grid, all_pois, time_cat)
            print(f"Calculated {time_cat} POI scores")
        
        # Also calculate total POI count (for reference)
        joined_pois = gpd.sjoin(grid, all_pois, how='left', predicate='contains')
        poi_counts = joined_pois.groupby('cell_id').size()
        grid['poi_score'] = 0
        for cell_id, count in poi_counts.items():
            grid.loc[grid['cell_id'] == cell_id, 'poi_score'] = count
        
        print(f"Found {len(all_pois)} unique POIs")
    else:
        print("Warning: No POIs found")
        for time_cat in ['peak', 'off_peak', 'weekend']:
            grid[f'poi_score_{time_cat}'] = 0
        grid['poi_score'] = 0
    
    # Calculate combined scores
    print("Calculating combined scores...")
    
    # Apply non-linear scaling for base scores (used by all time periods)
    grid['node_scaled'] = np.sqrt(grid['node_score'])
    grid['road_scaled'] = grid['road_score']
    grid['special_scaled'] = grid['special_score'] ** 1.2
    
    # Normalize base scores
    if grid['node_scaled'].max() > 0:
        grid['node_norm'] = grid['node_scaled'] / grid['node_scaled'].max()
    else:
        grid['node_norm'] = 0
    
    if grid['road_scaled'].max() > 0:
        grid['road_norm'] = grid['road_scaled'] / grid['road_scaled'].max()
    else:
        grid['road_norm'] = 0
    
    if grid['special_scaled'].max() > 0:
        grid['special_norm'] = grid['special_scaled'] / grid['special_scaled'].max()
    else:
        grid['special_norm'] = 0
    
    # Calculate destination scores for each time period
    for time_cat in ['peak', 'off_peak', 'weekend']:
        poi_col = f'poi_score_{time_cat}'
        dest_col = f'destination_score_{time_cat}'
        
        # Apply non-linear scaling to temporal POI scores
        grid[f'poi_scaled_{time_cat}'] = np.sqrt(grid[poi_col])
        
        # Normalize temporal POI scores
        if grid[f'poi_scaled_{time_cat}'].max() > 0:
            grid[f'poi_norm_{time_cat}'] = grid[f'poi_scaled_{time_cat}'] / grid[f'poi_scaled_{time_cat}'].max()
        else:
            grid[f'poi_norm_{time_cat}'] = 0
        
        # Combine scores
        grid[dest_col] = (
            grid['node_norm'] * 0.15 + 
            grid[f'poi_norm_{time_cat}'] * 0.45 + 
            grid['road_norm'] * 0.20 +
            grid['special_norm'] * 0.20
        ) * 100
    
    print("Analysis complete!")
    print(f"Peak hours scores range: {grid['destination_score_peak'].min():.1f} - {grid['destination_score_peak'].max():.1f}")
    print(f"Off-peak hours scores range: {grid['destination_score_off_peak'].min():.1f} - {grid['destination_score_off_peak'].max():.1f}")
    print(f"Weekend scores range: {grid['destination_score_weekend'].min():.1f} - {grid['destination_score_weekend'].max():.1f}")
    
    return grid, boundary, nodes, all_pois if not all_pois.empty else None, special_gdf

# Run the analysis
grid, boundary, nodes, pois, special_locs = simple_grid_analysis()

# Plot results - temporal comparison
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Boundary outline for all plots
for ax in axes.flatten():
    boundary.boundary.plot(ax=ax, color='black', linewidth=1, alpha=0.5)

# 1: Individual time periods
time_categories = ['peak', 'off_peak', 'weekend']
titles = ['Peak Hours\n(Weekday Rush)', 'Off-Peak Hours\n(Weekday Midday)', 'Weekend']

for idx, (time_cat, title) in enumerate(zip(time_categories, titles)):
    grid.plot(column=f'destination_score_{time_cat}', cmap='YlOrRd', legend=True, ax=axes[0, idx])
    axes[0, idx].set_title(title)
    if pois is not None:
        pois_sample = pois.sample(min(100, len(pois)))
        pois_sample.plot(ax=axes[0, idx], markersize=1, color='blue', alpha=0.3)

# 2: Comparison of peak vs off-peak
grid['diff_peak_offpeak'] = grid['destination_score_peak'] - grid['destination_score_off_peak']
grid.plot(column='diff_peak_offpeak', cmap='RdBu_r', legend=True, ax=axes[1, 0])
axes[1, 0].set_title('Peak vs Off-Peak\n(Red = Peak higher, Blue = Off-peak higher)')

# 3: Comparison of peak vs weekend
grid['diff_peak_weekend'] = grid['destination_score_peak'] - grid['destination_score_weekend']
grid.plot(column='diff_peak_weekend', cmap='RdBu_r', legend=True, ax=axes[1, 1])
axes[1, 1].set_title('Peak vs Weekend\n(Red = Peak higher, Blue = Weekend higher)')

# 4: Comparison of off-peak vs weekend
grid['diff_offpeak_weekend'] = grid['destination_score_off_peak'] - grid['destination_score_weekend']
grid.plot(column='diff_offpeak_weekend', cmap='RdBu_r', legend=True, ax=axes[1, 2])
axes[1, 2].set_title('Off-Peak vs Weekend\n(Red = Off-peak higher, Blue = Weekend higher)')

plt.tight_layout()
plt.savefig('data/figures/destination_v4_1000m.png', dpi=300, bbox_inches='tight')
plt.show()

# Save data for each time period
for time_cat in ['peak', 'off_peak', 'weekend']:
    export_gdf = grid.copy()
    
    # Keep relevant columns
    cols_to_keep = ['cell_id', f'destination_score_{time_cat}', 'node_score', 
                    f'poi_score_{time_cat}', 'road_score', 'special_score', 'geometry']
    export_gdf = export_gdf[cols_to_keep]
    
    # Rename for consistency
    export_gdf = export_gdf.rename(columns={
        f'destination_score_{time_cat}': 'destination_score',
        f'poi_score_{time_cat}': 'poi_score'
    })
    
    export_gdf['geometry'] = export_gdf.geometry.centroid
    export_gdf = export_gdf.to_crs(epsg=4326)
    export_gdf.to_file(f'data/raw/destination_v4_{time_cat}_1000m.geojson', driver='GeoJSON')
    print(f"Saved {time_cat} data to 'data/raw/destination_v4_{time_cat}_1000m.geojson'")

# Print some statistics
print("\n=== STATISTICS ===")
print(f"Total grid cells: {len(grid)}")

for time_cat in ['peak', 'off_peak', 'weekend']:
    dest_col = f'destination_score_{time_cat}'
    print(f"\n{time_cat.upper().replace('_', ' ')}:")
    print(f"  Cells with score > 0: {(grid[dest_col] > 0).sum()}")
    print(f"  Average score: {grid[dest_col].mean():.2f}")
    print(f"  Max score: {grid[dest_col].max():.2f}")

# Top destinations for each time period
for time_cat in ['peak', 'off_peak', 'weekend']:
    print(f"\n=== TOP 10 DESTINATIONS - {time_cat.upper().replace('_', ' ')} ===")
    dest_col = f'destination_score_{time_cat}'
    poi_col = f'poi_score_{time_cat}'
    top_dest = grid.nlargest(10, dest_col)[
        ['cell_id', dest_col, 'node_score', poi_col, 'road_score', 'special_score']
    ]
    print(top_dest.to_string(index=False))

print("\n=== TEMPORAL SHIFTS ===")
print("Cells that are much busier during peak hours (likely work/education areas):")
peak_dominant = grid.nlargest(5, 'diff_peak_offpeak')[['cell_id', 'destination_score_peak', 'poi_score_peak', 'diff_peak_offpeak']]
print(peak_dominant.to_string(index=False))

print("\nCells that are busier on weekends (likely tourism/leisure areas):")
weekend_dominant = grid.nsmallest(5, 'diff_peak_weekend')[['cell_id', 'destination_score_weekend', 'poi_score_weekend', 'diff_peak_weekend']]
print(weekend_dominant.to_string(index=False))