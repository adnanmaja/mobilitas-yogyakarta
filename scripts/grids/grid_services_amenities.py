import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon
import warnings
import yaml
from types import SimpleNamespace
warnings.filterwarnings('ignore', message='Could not parse column')

# Configurations
with open("config.yaml") as f:
    config_dict = yaml.safe_load(f)
    cfg = SimpleNamespace(**{k: SimpleNamespace(**v) if isinstance(v, dict) else v 
                           for k, v in config_dict.items()})

CRS_PROJECTED = 32749  # UTM 49S
CRS_GEOGRAPHIC = 4326  # WGS84

def create_grid(boundary_gdf, cell_size=1000):
    if boundary_gdf.crs != 'EPSG:32749':
        boundary_gdf = boundary_gdf.to_crs(epsg=32749)

    boundary = boundary_gdf
    west, south, east, north = boundary.total_bounds
    
    cols = np.arange(west, east, cell_size)
    rows = np.arange(south, north, cell_size)
    
    polygons = []
    for x in cols:
        for y in rows:
            polygons.append(Polygon([
                (x, y), 
                (x + cell_size, y),
                (x + cell_size, y + cell_size), 
                (x, y + cell_size)
            ]))
    
    grid = gpd.GeoDataFrame({'geometry': polygons}, crs=f'EPSG:{CRS_PROJECTED}')
    grid['cell_id'] = range(len(grid))
    
    print(f"Created {len(grid)} grid cells")
    return grid


def calculate_intensity(grid, places_gdf, intensity_column, scale: str):
    # Ensure same CRS
    places_projected = places_gdf.to_crs(epsg=CRS_PROJECTED)
    
    # Add temporary count column
    places_projected['count'] = 1
    
    # Spatial join
    joined = gpd.sjoin(grid, places_projected, how='left', predicate='intersects')
    
    # Count places per grid cell
    counts = joined.groupby('cell_id')['count'].sum().reset_index()
    
    # Merge back to grid
    grid = grid.merge(counts, on='cell_id', how='left')
    grid[intensity_column] = grid['count'].fillna(0)
    grid = grid.drop(columns=['count'])

    scale = scale.lower()
    if scale == 'sqrt':
        grid[intensity_column] = np.sqrt(grid[intensity_column])
    elif scale == 'log1p': 
        grid[intensity_column] = np.log1p(grid[intensity_column])
    else:
        grid[intensity_column] = grid[intensity_column]

    # Normalize to 0-100 scale
    max_count = grid[intensity_column].max()
    if max_count > 0:
        grid[intensity_column] = (grid[intensity_column] / max_count) * 100
    
    min_val = grid[intensity_column].min()
    max_val = grid[intensity_column].max()
    print(f"{intensity_column} range: {min_val:.1f} to {max_val:.1f}")
    
    return grid


def load_overture_data(categories, category_name):
    try:
        overture_gdf = gpd.read_file(cfg.data_paths.overture)
        filtered = overture_gdf[overture_gdf['basic_category'].isin(categories)].copy()
        print(f"Found {len(filtered)} {category_name}")
        return filtered
    except Exception as e:
        print(f"Error loading Overture Maps data: {e}")
        return None

def analyze_place_of_worship(grid):
    print("\n=== Place of Worship Analysis ===")
    
    worship_categories = [
        'place_of_worship', 
        'muslim_place_of_worship', 
        'christian_place_of_worshop', # not a typo, its actually is 'worshop' for some reason
        'jewish_place_of_worship', 
        'hindu_place_of_worship', 
        'buddhist_place_of_worship'
    ]
    
    worship_places = load_overture_data(worship_categories, "places of worship")
    
    if worship_places is not None:
        grid = calculate_intensity(grid, worship_places, 'place_of_worship_intensity', scale=cfg.scales.worship)
    else:
        grid['place_of_worship_intensity'] = 0
    
    return grid

def analyze_commercial_nhb(grid):
    print("\n=== Commercial Analysis ===")
    
    commercial_categories = [
        'beverage_shop', 
        'coffee_shop', 
        'sandwich_shop',  
        'shopping_mall', 
        'food_or_beverage_store', 
        'convenience_store', 
        'grocery_store', 
        'bookstore',
        'cafe',
        'office_supply_store'
        'restaurant',
        'food_court',
        'casual_eatery',
        'pub',
        'gym',
        'gas_station',
        'atm'
    ]
    
    commercial_places = load_overture_data(commercial_categories, "commercial places (Non Home Based)")
    
    if commercial_places is not None:
        grid = calculate_intensity(grid, commercial_places, 'commercial_intensity_nhb', scale=cfg.scales.commercial_nhb) 
    else:
        grid['commercial_intensity_nhb'] = 0
    
    return grid

def analyze_commercial_hbnw(grid):
    print("\n=== Commercial Analysis (Home Based Non Work) ===")
    
    commercial_categories = [
        'auto_body_shop', 
        'antique_shop', 
        'barber_shop',  
        'shopping_mall', 
        'butcher_shop', 
        'gift_shop', 
        'grocery_store', 
        'bookstore',
        'second_hand_shop',
        'office_supply_store'
        'restaurant',
        'machine_shop',
        'clothing_store',
        'toy_store',
        'produce_store',
        'music_store',
        'shoe_store',
        'specialty_store',
        'sporting_goods_store',
        'jewelry_store',
        'fashion_or_apparel_store',
        'hardware_store',
        'eyewear_store',
        'pet_store',
        'electronics_store',
        'art_craft_hobby_store',
        'department_store',
        'movie_theater',
        'bank',
        'beauty_salon',
        'motorcycle_dealer',
        'nail_salon',
        'animal_shelter',
        'music_store',
        'massage_therapy'
    ]
    
    commercial_places = load_overture_data(commercial_categories, "commercial places (Home Based Non Work)")
    
    if commercial_places is not None:
        grid = calculate_intensity(grid, commercial_places, 'commercial_intensity_hbnw', scale=cfg.scales.commercial_hbnw)
    else:
        grid['commercial_intensity_hbnw'] = 0
    
    return grid

def analyze_leisure(grid):
    print("\n=== Leisure Analysis ===")
    
    leisure_categories = [
        'amusement_park', 
        'dog_park', 
        'equestrian_facility',  
        'beach',
        'community_center',
        'recreational_vehicle_dealer'
        'race_track',
        'tour_operator',
        'monument',
        'history_museum',
        'shooting_range',
        'skate_park',
        'playground',
        'swimming_pool',
        'nature_outdoors',
        'sport_field',
        'golf_course',
        'campground',
        'sport_court',
        'public_fountain',
        'science_museum',
        'hotel',
        'zoo',
        'museum',
        'art_museum'
        'forest',
        'mountain',
        'dance_club',
        'aquarium',
        'nature_reserve',
        'national_park',
        'entertainment_location',
        'gaming_venue', 
        'music_venue',
        'childrens_museum',
        'historic_site'
    ]
    
    leisure_places = load_overture_data(leisure_categories, "leisure places")
    
    if leisure_places is not None:
        grid = calculate_intensity(grid, leisure_places, 'leisure_intensity', scale=cfg.scales.leisure)
    else:
        grid['leisure_intensity'] = 0
    
    return grid

def analyze_services(grid):
    print("\n=== Services Analysis (Essential) ===")
    
    service_categories = [
        'mental_health', 
        'psychology', 
        'taxi_service',  
        'hospital', 
        'fire_station', 
        'food_bank', 
        'emergency_room', 
        'police_station',
        'clinic_or_treatment_center'
        ]
    
    service_places = load_overture_data(service_categories, "service places")
    
    if service_places is not None:
        grid = calculate_intensity(grid, service_places, 'service_intensity', scale=cfg.scales.service)
    else:
        grid['service_intensity'] = 0
    
    return grid

def combine_intensities(grid):
    print("\n=== Combining Intensities ===")
    
    grid['amenity_nhb_intensity'] = (
        (grid['commercial_intensity_nhb'] * cfg.grid_weights.commercial_nhb) + 
        (grid['place_of_worship_intensity'] * cfg.grid_weights.worship)
    )
    
    grid['amenity_hbnw_intensity'] = (
        (grid['commercial_intensity_hbnw'] * cfg.grid_weights.commercial_hbnw) + 
        (grid['leisure_intensity'] * cfg.grid_weights.leisure) + 
        (grid['service_intensity'] * cfg.grid_weights.service)
    )
    
    # Normalize each to 0-100 scale
    for col in ['amenity_nhb_intensity', 'amenity_hbnw_intensity']:
        max_val = grid[col].max()
        if max_val > 0:
            grid[col] = (grid[col] / max_val) * 100
    
    print(f"amenity_nhb_intensity range: {grid['amenity_nhb_intensity'].min():.1f} to {grid['amenity_nhb_intensity'].max():.1f}")
    print(f"amenity_hbnw_intensity range: {grid['amenity_hbnw_intensity'].min():.1f} to {grid['amenity_hbnw_intensity'].max():.1f}")
    
    return grid


def plot_results(grid, boundary):
    boundary = boundary.to_crs(epsg=CRS_PROJECTED)

    fig, axes = plt.subplots(2, 4, figsize=(22, 12)) 
    
    # List of all intensity columns to plot
    intensity_columns = [
        ('place_of_worship_intensity', 'Place of Worship'),
        ('commercial_intensity_nhb', 'Commercial (NHB)'),
        ('commercial_intensity_hbnw', 'Commercial (HBNW)'),
        ('leisure_intensity', 'Leisure'),
        ('service_intensity', 'Services'),
        ('amenity_nhb_intensity', 'Amenity NHB'),
        ('amenity_hbnw_intensity', 'Amenity HBNW')
    ]
    
    # Plot each intensity
    for idx, (col_name, title) in enumerate(intensity_columns):
        row = idx // 4
        col = idx % 4
        
        # Plot boundary
        boundary.boundary.plot(ax=axes[row, col], color='black', linewidth=1, alpha=0.5)
        
        # Plot intensity
        grid.plot(column=col_name, cmap='Reds', 
                  legend=True, ax=axes[row, col])
        
        axes[row, col].set_title(title, fontsize=14)
        axes[row, col].axis('off')
    
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(cfg.figure_paths.service_amenity, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\nFigure saved to '{cfg.figure_paths.service_amenity}'")

def print_statistics(grid):
    print("\n=== STATISTICS ===")
    print(f"Total grid cells: {len(grid)}")
    print(f"Cells with place of worship intensity > 0: {(grid['place_of_worship_intensity'] > 0).sum()}")
    print(f"Cells with commercial (NHB) intensity > 0: {(grid['commercial_intensity_nhb'] > 0).sum()}")
    print(f"Cells with commercial (HBNW) intensity > 0: {(grid['commercial_intensity_hbnw'] > 0).sum()}")
    print(f"Cells with leisure intensity > 0: {(grid['leisure_intensity'] > 0).sum()}")
    print(f"Cells with service intensity > 0: {(grid['service_intensity'] > 0).sum()}")
    print(f"Cells with amenity_nhb_intensity > 0: {(grid['amenity_nhb_intensity'] > 0).sum()}")
    print(f"Cells with amenity_hbnw_intensity > 0: {(grid['amenity_hbnw_intensity'] > 0).sum()}")
    
    # Top intensity cells
    top_worship = grid.nlargest(5, 'place_of_worship_intensity')[['cell_id', 'place_of_worship_intensity']]
    print("\nTop 5 place of worship intensity cells:")
    print(top_worship.to_string(index=False))
    
    top_commercial_nhb = grid.nlargest(5, 'commercial_intensity_nhb')[['cell_id', 'commercial_intensity_nhb']]
    print("\nTop 5 commercial (NHB) intensity cells:")
    print(top_commercial_nhb.to_string(index=False))
    
    top_commercial_hbnw = grid.nlargest(5, 'commercial_intensity_hbnw')[['cell_id', 'commercial_intensity_hbnw']]
    print("\nTop 5 commercial (HBNW) intensity cells:")
    print(top_commercial_hbnw.to_string(index=False))

    top_leisure = grid.nlargest(5, 'leisure_intensity')[['cell_id', 'leisure_intensity']]
    print("\nTop 5 leisure intensity cells:")
    print(top_leisure.to_string(index=False))

    top_service = grid.nlargest(5, 'service_intensity')[['cell_id', 'service_intensity']]
    print("\nTop 5 service intensity cells:")
    print(top_service.to_string(index=False))

    top_nhb = grid.nlargest(5, 'amenity_nhb_intensity')[['cell_id', 'amenity_nhb_intensity']]
    print("\nTop 5 amenity_nhb_intensity cells:")
    print(top_nhb.to_string(index=False))
    
    top_hbnw = grid.nlargest(5, 'amenity_hbnw_intensity')[['cell_id', 'amenity_hbnw_intensity']]
    print("\nTop 5 amenity_hbnw_intensity cells:")
    print(top_hbnw.to_string(index=False))

    # Additional analysis for non-zero cells
    worship_nonzero = grid[grid['place_of_worship_intensity'] > 0]['place_of_worship_intensity']
    if len(worship_nonzero) > 0:
        print(f"\nAverage place of worship intensity (non-zero cells): {worship_nonzero.mean():.2f}")
    
    commercial_nhb_nonzero = grid[grid['commercial_intensity_nhb'] > 0]['commercial_intensity_nhb']
    if len(commercial_nhb_nonzero) > 0:
        print(f"Average commercial (NHB) intensity (non-zero cells): {commercial_nhb_nonzero.mean():.2f}")

    commercial_hbnw_nonzero = grid[grid['commercial_intensity_hbnw'] > 0]['commercial_intensity_hbnw']
    if len(commercial_hbnw_nonzero) > 0:
        print(f"Average commercial (HBNW) intensity (non-zero cells): {commercial_hbnw_nonzero.mean():.2f}")

    leisure_nonzero = grid[grid['leisure_intensity'] > 0]['leisure_intensity']
    if len(leisure_nonzero) > 0:
        print(f"Average leisure intensity (non-zero cells): {leisure_nonzero.mean():.2f}")

    service_nonzero = grid[grid['service_intensity'] > 0]['service_intensity']
    if len(service_nonzero) > 0:
        print(f"Average service intensity (non-zero cells): {service_nonzero.mean():.2f}")

    # Some more
    print(f"Place of worship scale: {cfg.scales.worship}")
    print(f"Commercial (NHB) scale: {cfg.scales.commercial_nhb}")
    print(f"Commercial (HBNW) scale: {cfg.scales.commercial_hbnw}")
    print(f"leisure scale: {cfg.scales.leisure}")
    print(f"Service scale: {cfg.scales.service}")

def export_to_geojson(grid, output_path=cfg.export_paths.service_amenity):
    export_gdf = grid.copy()
    export_gdf['geometry'] = export_gdf.geometry.centroid
    export_gdf = export_gdf.to_crs(epsg=CRS_GEOGRAPHIC)
    export_gdf.to_file(output_path, driver='GeoJSON')
    print(f"\nData saved to '{output_path}'")


def main():
    print(f"Starting analysis...")
    
    boundary = gpd.read_file(cfg.data_paths.boundary)  
    
    grid = create_grid(boundary, cfg.cell_size)
    
    grid = analyze_place_of_worship(grid)
    grid = analyze_commercial_nhb(grid)
    grid = analyze_commercial_hbnw(grid)
    grid = analyze_leisure(grid)
    grid = analyze_services(grid)
    grid = combine_intensities(grid)
    
    plot_results(grid, boundary)
    
    print_statistics(grid)
    
    export_to_geojson(grid)
    

if __name__ == "__main__":
    main()