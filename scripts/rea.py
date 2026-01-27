import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon, Point
import pandas as pd
import rasterio
from rasterio.mask import mask
from shapely.geometry import box

place = "Yogyakarta, Indonesia"

def enhanced_origin_analysis():
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
    grid['residential_intensity'] = 0  # From WorldPop
    grid['employment_edu_intensity'] = 0  # From OSM tags
    grid['amenity_intensity'] = 0  # From OSM tags + manual additions
    grid['combined_intensity'] = 0  # Combined score
    
    print(f"Created {len(grid)} grid cells")
    
    # 1. RESIDENTIAL INTENSITY (from WorldPop)
    print("Getting residential intensity from WorldPop...")
    try:
        reprojected_path = "data/raw/clipped_utm49s_2025_yogyakarta_100m.tif"
        with rasterio.open(reprojected_path) as src:
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
    
    # 2. EMPLOYMENT/EDUCATION INTENSITY (from OSM)
    print("Getting employment/education intensity from OSM...")
    try:
        # Define employment/education related tags
        employment_edu_tags = {
            # Employment
            'office': True,
            'industrial': True,
            'landuse': ['industrial', 'retail'],
            'commercial': True,
            'building': ['commercial', 'university', 'hospital', 'college'],
            'retail': True,
            'craft': True,
            'healthcare': ['hospital', 'clinic', 'doctors'],
            'man_made': ['works'],
            
            # Education
            'amenity': [
                'university', 'college', 'school', 
                'research_institute', 'kindergarten', 'restaurant'
            ],
            
            # Government
            'government': True,
            'public_building': True,
        }
        
        # Get employment/education features from OSM
        gdf_employment = ox.features_from_place(place, employment_edu_tags)
        gdf_employment = gdf_employment.to_crs(grid.crs)
        
        # Calculate employment/education intensity per grid cell
        employment_scores = np.zeros(len(grid))
        
        for idx, feature in gdf_employment.iterrows():
            if feature.geometry:
                # Find which grid cells intersect with this feature
                intersecting_cells = grid[grid.intersects(feature.geometry)]
                
                if len(intersecting_cells) > 0:
                    # Add score based on area (larger features have more impact)
                    if hasattr(feature.geometry, 'area'):
                        score = min(feature.geometry.area / 1000, 10)  # Cap at 10 per feature
                    else:
                        score = 1
                    
                    for cell_idx in intersecting_cells.index:
                        # Calculate intersection area
                        try:
                            intersection = grid.loc[cell_idx, 'geometry'].intersection(feature.geometry)
                            if hasattr(intersection, 'area') and intersection.area > 0:
                                # Weight by intersection area
                                weight = intersection.area / grid.loc[cell_idx, 'geometry'].area
                                employment_scores[cell_idx] += score * weight
                        except:
                            employment_scores[cell_idx] += score / len(intersecting_cells)
        
        grid['employment_edu_intensity'] = employment_scores
        
        # Normalize to 0-100
        if grid['employment_edu_intensity'].max() > 0:
            grid['employment_edu_intensity'] = (
                grid['employment_edu_intensity'] / grid['employment_edu_intensity'].max()
            ) * 100
        
        print(f"Found {len(gdf_employment)} employment/education features")
        
    except Exception as e:
        print(f"Error getting employment/education data: {e}")
        print(f"Error details: {str(e)}")
    
    # 3. AMENITY/SERVICES INTENSITY (from OSM + manual additions)
    print("Getting amenity/services intensity from OSM...")
    try:
        # Define amenity/services related tags
        amenity_tags = {
            'amenity': [
                'restaurant', 'cafe', 'fast_food', 'bar', 'pub',
                'marketplace', 'supermarket', 'convenience',
                'bank', 'atm', 'pharmacy',
                'place_of_worship',  # Includes mosques, churches, temples
                'police', 'fire_station', 'post_office',
                'library', 'theatre', 'cinema', 'arts_centre',
                'sports_centre', 'swimming_pool', 'stadium',
                'wetland', 'conference_centre'
            ],
            'building': ['commercial'],
            'shop': True,  # All shops
            'leisure': ['park', 'garden', 'playground', 'sports_centre', 'golf_course', 'stadium'],
            'tourism': ['hotel', 'guest_house', 'hostel', 'museum', 'attraction'],
            'natural': ['beach']
        }
        
        # Get amenity features from OSM
        gdf_amenity = ox.features_from_place(place, amenity_tags)
        gdf_amenity = gdf_amenity.to_crs(grid.crs)

        incorrect_park_bounds = box(410295.29, 9148048.26, 412765.03, 9149655.77) # daerah tinalah di kulonprogo, gtw kok ada 'park' gede bat
        gdf_amenity = gdf_amenity[~gdf_amenity.geometry.intersects(incorrect_park_bounds)]
        
        # Calculate amenity intensity per grid cell
        amenity_scores = np.zeros(len(grid))
        
        for idx, feature in gdf_amenity.iterrows():
            if feature.geometry:
                # Find which grid cells intersect with this feature
                intersecting_cells = grid[grid.intersects(feature.geometry)]
                
                if len(intersecting_cells) > 0:
                    # Add score (amenities are generally smaller, so lower base score)
                    score = 2  # Base score for amenities
                    
                    # Higher score for important amenities
                    if 'amenity' in feature:
                        if feature['amenity'] in ['hospital', 'university', 'college']:
                            score = 5
                        elif feature['amenity'] in ['supermarket', 'marketplace']:
                            score = 3
                    
                    for cell_idx in intersecting_cells.index:
                        try:
                            intersection = grid.loc[cell_idx, 'geometry'].intersection(feature.geometry)
                            if hasattr(intersection, 'area') and intersection.area > 0:
                                weight = intersection.area / grid.loc[cell_idx, 'geometry'].area
                                amenity_scores[cell_idx] += score * weight
                        except:
                            amenity_scores[cell_idx] += score / len(intersecting_cells)
        
        # MANUAL ADDITION: Malioboro Street area (famous shopping/tourist area in Yogyakarta)
        # Approximate coordinates for Malioboro area
        malioboro_bounds = box(429798, 9137207, 430309, 9139673)  # Adjust based on your CRS
        malioboro_bounds = malioboro_bounds  # Ensure same CRS
        
        # Find cells in Malioboro area
        malioboro_cells = grid[grid.intersects(malioboro_bounds)]
        
        for cell_idx in malioboro_cells.index:
            # Add significant bonus for Malioboro area
            intersection = grid.loc[cell_idx, 'geometry'].intersection(malioboro_bounds)
            if hasattr(intersection, 'area') and intersection.area > 0:
                weight = intersection.area / grid.loc[cell_idx, 'geometry'].area
                amenity_scores[cell_idx] += 7 * weight  # Significant bonus
        
        grid['amenity_intensity'] = amenity_scores
        
        # Normalize to 0-100
        if grid['amenity_intensity'].max() > 0:
            grid['amenity_intensity'] = (
                grid['amenity_intensity'] / grid['amenity_intensity'].max()
            ) * 100
        
        print(f"Found {len(gdf_amenity)} amenity features + added Malioboro area")
        
    except Exception as e:
        print(f"Error getting amenity data: {e}")
        print(f"Error details: {str(e)}")
    
    # 4. COMBINE INTENSITIES
    print("Calculating combined intensity scores...")
    
    # Normalize each intensity to 0-1 range
    for intensity in ['residential_intensity', 'employment_edu_intensity', 'amenity_intensity']:
        if grid[intensity].max() > 0:
            grid[f'{intensity}_norm'] = grid[intensity] / grid[intensity].max()
        else:
            grid[f'{intensity}_norm'] = 0
    
    # Weighted combination: 40% residential, 35% employment/edu, 25% amenity
    # These weights can be adjusted based on your specific needs
    grid['combined_intensity'] = (
        grid['residential_intensity_norm'] * 0.4 +
        grid['employment_edu_intensity_norm'] * 0.35 +
        grid['amenity_intensity_norm'] * 0.25
    ) * 100
    
    print("Enhanced analysis complete!")
    print(f"Residential intensity range: {grid['residential_intensity'].min():.1f} - {grid['residential_intensity'].max():.1f}")
    print(f"Employment/Education intensity range: {grid['employment_edu_intensity'].min():.1f} - {grid['employment_edu_intensity'].max():.1f}")
    print(f"Amenity intensity range: {grid['amenity_intensity'].min():.1f} - {grid['amenity_intensity'].max():.1f}")
    print(f"Combined intensity range: {grid['combined_intensity'].min():.1f} - {grid['combined_intensity'].max():.1f}")
    
    return grid, boundary

grid, boundary = enhanced_origin_analysis()

# Plot results
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Boundary outline for all plots
for i in range(6):
    boundary.boundary.plot(ax=axes.flatten()[i], color='black', linewidth=1, alpha=0.5)

# Plot 1: Residential Intensity
grid.plot(column='residential_intensity', cmap='Reds', legend=True, ax=axes[0, 0])
axes[0, 0].set_title('Residential Intensity (WorldPop)')

# Plot 2: Employment/Education Intensity
grid.plot(column='employment_edu_intensity', cmap='Blues', legend=True, ax=axes[0, 1])
axes[0, 1].set_title('Employment/Education Intensity (OSM)')

# Plot 3: Amenity Intensity
grid.plot(column='amenity_intensity', cmap='Greens', legend=True, ax=axes[0, 2])
axes[0, 2].set_title('Amenity/Services Intensity (OSM + Malioboro)')

# Plot 4: Combined Intensity
grid.plot(column='combined_intensity', cmap='Oranges', legend=True, ax=axes[1, 0])
axes[1, 0].set_title('Combined Intensity (Weighted)')

# Plot 5: Top 20% intensity cells
threshold = grid['combined_intensity'].quantile(0.8)
high_intensity = grid[grid['combined_intensity'] >= threshold]
high_intensity.plot(ax=axes[1, 1], color='purple', alpha=0.7)
axes[1, 1].set_title(f'Top 20% Intensity Cells (≥ {threshold:.1f})')

# Plot 6: Comparison of all intensities
# Create a small multiple plot showing normalized intensities
for idx, intensity in enumerate(['residential_intensity_norm', 'employment_edu_intensity_norm', 'amenity_intensity_norm']):
    grid.plot(column=intensity, cmap=['Reds', 'Blues', 'Greens'][idx], 
              alpha=0.6, ax=axes[1, 2], legend=False)
axes[1, 2].set_title('All Intensities Overlay')

# Adjust layout
plt.tight_layout()
plt.savefig('data/figures/rea_1000m_2.png', dpi=300, bbox_inches='tight')
plt.show()

# Save data to geojson
export_gdf = grid.copy()
export_gdf['geometry'] = export_gdf.geometry.centroid
export_gdf = export_gdf.to_crs(epsg=4326)  # To match mapbox's system
export_gdf.to_file('data/raw/rea_1000m.geojson', driver='GeoJSON')
print("Data saved to 'data/raw/rea_1000m.geojson'")

# Print statistics
print("\n=== STATISTICS ===")
print(f"Total grid cells: {len(grid)}")
print(f"Cells with residential intensity > 0: {(grid['residential_intensity'] > 0).sum()}")
print(f"Cells with employment/edu intensity > 0: {(grid['employment_edu_intensity'] > 0).sum()}")
print(f"Cells with amenity intensity > 0: {(grid['amenity_intensity'] > 0).sum()}")
print(f"Cells with combined intensity > 0: {(grid['combined_intensity'] > 0).sum()}")

# Calculate and display correlation between intensities
print("\n=== CORRELATIONS ===")
corr_matrix = grid[['residential_intensity', 'employment_edu_intensity', 'amenity_intensity']].corr()
print("Correlation matrix:")
print(corr_matrix)

# Find top intensity cells for each category
print("\n=== TOP CELLS BY CATEGORY ===")
categories = ['residential_intensity', 'employment_edu_intensity', 'amenity_intensity', 'combined_intensity']
for category in categories:
    top_cells = grid.nlargest(3, category)[['cell_id', category]]
    print(f"\nTop 3 {category}:")
    print(top_cells.to_string(index=False))

# Additional analysis
print("\n=== ADDITIONAL ANALYSIS ===")
print(f"Average residential intensity: {grid['residential_intensity'].mean():.2f}")
print(f"Average employment/edu intensity: {grid['employment_edu_intensity'].mean():.2f}")
print(f"Average amenity intensity: {grid['amenity_intensity'].mean():.2f}")
print(f"Average combined intensity: {grid['combined_intensity'].mean():.2f}")
print(f"Cells in top 20% combined intensity: {len(high_intensity)}")

# Create summary table of cells with high scores in multiple categories
print("\n=== MULTI-CATEGORY HOTSPOTS ===")
# Cells in top 25% of all three categories
top_residential = grid['residential_intensity'] >= grid['residential_intensity'].quantile(0.75)
top_employment = grid['employment_edu_intensity'] >= grid['employment_edu_intensity'].quantile(0.75)
top_amenity = grid['amenity_intensity'] >= grid['amenity_intensity'].quantile(0.75)

multi_hotspots = grid[top_residential & top_employment & top_amenity]
print(f"Cells in top 25% of ALL three categories: {len(multi_hotspots)}")
if len(multi_hotspots) > 0:
    print("These are likely major urban centers")