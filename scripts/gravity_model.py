# Idk what's going on here

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import time
from typing import Tuple, List, Dict
import geopandas as gpd
import warnings
warnings.filterwarnings('ignore')


class GravityModelOD:
    """
    Gravity Model for Origin-Destination Matrix calculation
    
    The gravity model formula: T_ij = k * (O_i^α * D_j^β) / (d_ij^γ)
    where:
    - T_ij: trips from origin i to destination j
    - O_i: origin mass at grid i
    - D_j: destination mass at grid j
    - d_ij: distance between i and j
    - α, β, γ: parameters (typically α=1, β=1, γ=2 for basic model)
    - k: scaling constant
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 1.0, gamma: float = 2.0):
        """
        Initialize gravity model with parameters
        
        Parameters:
        -----------
        alpha : float, exponent for origin mass
        beta : float, exponent for destination mass
        gamma : float, exponent for distance (friction parameter)
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.k = None  # Scaling constant (calculated during fitting)
        
    def calculate_distances(self, coordinates: np.ndarray, 
                           distance_type: str = 'euclidean') -> np.ndarray:
        """
        Calculate distance matrix between all grid points
        
        Parameters:
        -----------
        coordinates : np.ndarray, shape (n_grids, 2)
            Coordinates of grid centers
        distance_type : str, type of distance metric
        
        Returns:
        --------
        distance_matrix : np.ndarray, shape (n_grids, n_grids)
        """
        print(f"Calculating {distance_type} distance matrix for {len(coordinates)} grids...")
        start_time = time.time()
        
        distance_matrix = cdist(coordinates, coordinates, metric=distance_type)
        
        # Avoid division by zero for self-distances
        np.fill_diagonal(distance_matrix, 1.0)
        
        elapsed_time = time.time() - start_time
        print(f"Distance matrix calculated in {elapsed_time:.2f} seconds")
        
        return distance_matrix
    
    def calculate_od_matrix(self, origins: np.ndarray, destinations: np.ndarray, 
                           coordinates: np.ndarray, total_trips: float = None,
                           distance_matrix: np.ndarray = None,
                           distance_type: str = 'euclidean') -> Tuple[np.ndarray, Dict]:
        """
        Calculate OD matrix using gravity model
        
        Parameters:
        -----------
        origins : np.ndarray, shape (n_grids,)
            Origin points per grid
        destinations : np.ndarray, shape (n_grids,)
            Destination points per grid
        coordinates : np.ndarray, shape (n_grids, 2)
            Grid coordinates (x, y)
        total_trips : float, optional
            Total number of trips to scale to (if None, calculated from data)
        distance_matrix : np.ndarray, optional
            Pre-computed distance matrix
        distance_type : str, distance metric
        
        Returns:
        --------
        od_matrix : np.ndarray, shape (n_grids, n_grids)
            Origin-Destination matrix
        info_dict : Dict with metadata
        """
        print("=" * 60)
        print("CALCULATING ORIGIN-DESTINATION MATRIX USING GRAVITY MODEL")
        print("=" * 60)
        
        n_grids = len(origins)
        print(f"Number of grids: {n_grids:,}")
        print(f"Total origins: {origins.sum():,.0f}")
        print(f"Total destinations: {destinations.sum():,.0f}")
        
        # Calculate or use provided distance matrix
        if distance_matrix is None:
            distance_matrix = self.calculate_distances(coordinates, distance_type)
        
        # Apply gravity model formula
        print("\nApplying gravity model...")
        start_time = time.time()
        
        # Reshape for broadcasting
        O = origins.reshape(-1, 1)  # Column vector
        D = destinations.reshape(1, -1)  # Row vector
        
        # Calculate unnormalized gravity model
        # T_ij = (O_i^α * D_j^β) / (d_ij^γ)
        with np.errstate(divide='ignore', invalid='ignore'):
            od_unnormalized = (O ** self.alpha) * (D ** self.beta) / (distance_matrix ** self.gamma)
        
        # Replace infinities with zeros
        od_unnormalized = np.nan_to_num(od_unnormalized, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Calculate scaling constant k
        if total_trips is None:
            total_trips = min(origins.sum(), destinations.sum())
        
        total_unnormalized = od_unnormalized.sum()
        self.k = total_trips / total_unnormalized if total_unnormalized > 0 else 1.0
        
        # Apply scaling
        od_matrix = od_unnormalized * self.k
        
        # Round small values to zero to save memory
        od_matrix[od_matrix < 1e-10] = 0
        
        elapsed_time = time.time() - start_time
        print(f"OD matrix calculated in {elapsed_time:.2f} seconds")
        
        # Calculate statistics
        stats = self._calculate_statistics(od_matrix, origins, destinations)
        
        # Create info dictionary
        info_dict = {
            'n_grids': n_grids,
            'total_trips': total_trips,
            'scaling_constant_k': self.k,
            'model_parameters': {
                'alpha': self.alpha,
                'beta': self.beta,
                'gamma': self.gamma
            },
            'statistics': stats,
            'distance_type': distance_type
        }
        
        return od_matrix, info_dict
    
    def _calculate_statistics(self, od_matrix: np.ndarray, 
                             origins: np.ndarray, 
                             destinations: np.ndarray) -> Dict:
        """Calculate statistics about the OD matrix"""
        stats = {}
        
        # Basic statistics
        stats['od_matrix_shape'] = od_matrix.shape
        stats['total_estimated_trips'] = od_matrix.sum()
        stats['non_zero_elements'] = np.count_nonzero(od_matrix)
        stats['sparsity'] = 1 - (stats['non_zero_elements'] / od_matrix.size)
        
        # Row and column sums
        row_sums = od_matrix.sum(axis=1)
        col_sums = od_matrix.sum(axis=0)
        
        # Correlation with input data
        stats['origin_correlation'] = np.corrcoef(origins, row_sums)[0, 1]
        stats['destination_correlation'] = np.corrcoef(destinations, col_sums)[0, 1]
        
        # Error metrics
        origin_error = np.abs(row_sums - origins).sum() / origins.sum()
        destination_error = np.abs(col_sums - destinations).sum() / destinations.sum()
        
        stats['origin_mean_absolute_error'] = origin_error
        stats['destination_mean_absolute_error'] = destination_error
        
        return stats
    
    def export_as_vectors(self, od_matrix: np.ndarray, grid_ids: np.ndarray = None) -> List[Dict]:
        """
        Export OD matrix as list of vectors for efficient storage
        
        Parameters:
        -----------
        od_matrix : np.ndarray, OD matrix
        grid_ids : np.ndarray, optional grid identifiers
        
        Returns:
        --------
        vectors_list : List of dictionaries with vector format
        """
        print("\nConverting OD matrix to vector format...")
        start_time = time.time()
        
        n_grids = od_matrix.shape[0]
        if grid_ids is None:
            grid_ids = np.arange(n_grids)
        
        vectors_list = []
        
        # For large matrices, process in chunks
        chunk_size = 1000
        
        for i in range(0, n_grids, chunk_size):
            i_end = min(i + chunk_size, n_grids)
            
            for origin_idx in range(i, i_end):
                origin_id = grid_ids[origin_idx]
                
                # Get non-zero destinations for this origin
                dest_indices = np.where(od_matrix[origin_idx] > 0)[0]
                
                if len(dest_indices) > 0:
                    vector_dict = {
                        'origin_id': int(origin_id),
                        'destinations': [
                            {
                                'destination_id': int(grid_ids[dest_idx]),
                                'trips': float(od_matrix[origin_idx, dest_idx])
                            }
                            for dest_idx in dest_indices
                        ],
                        'total_trips': float(od_matrix[origin_idx].sum())
                    }
                    vectors_list.append(vector_dict)
        
        elapsed_time = time.time() - start_time
        print(f"Converted to vector format in {elapsed_time:.2f} seconds")
        print(f"Number of vectors (non-zero origins): {len(vectors_list)}")
        
        return vectors_list
    
    def save_vectors(self, vectors_list: List[Dict], filename: str):
        """Save vectors to JSON file"""
        import json
        
        # Convert numpy types to Python native types
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            else:
                return obj
        
        vectors_converted = convert_numpy_types(vectors_list)
        
        with open(filename, 'w') as f:
            json.dump(vectors_converted, f, indent=2)
        
        print(f"\nVectors saved to {filename}")


def generate_sample_data(n_grids: int = 10088) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate sample data for testing
    
    Parameters:
    -----------
    n_grids : int, number of grids
    
    Returns:
    --------
    origins : np.ndarray, origin points
    destinations : np.ndarray, destination points
    coordinates : np.ndarray, grid coordinates
    """
    print(f"Generating sample data for {n_grids} grids...")
    
    np.random.seed(42)  # For reproducibility
    
    # Generate random coordinates (simulating city layout)
    x_coords = np.random.uniform(0, 100, n_grids)
    y_coords = np.random.uniform(0, 100, n_grids)
    coordinates = np.column_stack([x_coords, y_coords])
    
    # Generate origin and destination points (scale 1-100)
    # Using Poisson distribution to simulate real-world clustering
    origins = np.random.poisson(lam=20, size=n_grids) + 1
    destinations = np.random.poisson(lam=15, size=n_grids) + 1
    
    # Scale to 1-100 range
    origins = np.clip(origins, 1, 100).astype(float)
    destinations = np.clip(destinations, 1, 100).astype(float)
    
    # Add some correlation between origins and destinations
    destinations = 0.7 * destinations + 0.3 * origins + np.random.normal(0, 5, n_grids)
    destinations = np.clip(destinations, 1, 100)
    
    print(f"Sample data generated:")
    print(f"  Origins range: {origins.min():.1f} - {origins.max():.1f}")
    print(f"  Destinations range: {destinations.min():.1f} - {destinations.max():.1f}")
    
    return origins, destinations, coordinates


def load_data(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load data from GeoJSON file with Point geometry in EPSG:4326
    
    Expected GeoJSON structure:
    - Geometry: Point (in EPSG:4326 - lat/lon)
    - Properties should include:
        - origin_points (or origins, origin_count, etc.)
        - destination_points (or destinations, destination_count, etc.)
        - Optional: grid_id
    
    Parameters:
    -----------
    filepath : str
        Path to GeoJSON file
    
    Returns:
    --------
    origins : np.ndarray
        Origin points per grid
    destinations : np.ndarray
        Destination points per grid
    coordinates : np.ndarray
        Grid coordinates in projected system (meters)
    grid_ids : np.ndarray
        Grid IDs (if available, otherwise sequential)
    """
    print(f"Loading data from {filepath}...")
    
    # Read GeoJSON
    gdf = gpd.read_file(filepath)
    
    print(f"Loaded {len(gdf)} features")
    print(f"Current CRS: {gdf.crs}")
    print(f"Columns: {list(gdf.columns)}")
    
    # Check geometry type
    if not all(gdf.geometry.geom_type == 'Point'):
        raise ValueError("All geometries must be Point type")
    
    # Reproject to a projected CRS for distance calculations
    # Using UTM zone based on centroid, or Web Mercator as fallback
    if gdf.crs.to_epsg() == 4326:
        print("Reprojecting from EPSG:4326 to EPSG:3857 (Web Mercator)...")
        gdf = gdf.to_crs(epsg=3857)  # Web Mercator in meters
        print(f"Reprojected to: {gdf.crs}")
    
    # Extract coordinates (now in meters)
    coordinates = np.column_stack([gdf.geometry.x.values, gdf.geometry.y.values])
    
    # Try to find origin and destination columns (flexible column names)
    origin_cols = [col for col in gdf.columns if 'origin' in col.lower()]
    dest_cols = [col for col in gdf.columns if 'destination' in col.lower() or 'dest' in col.lower()]
    
    if not origin_cols:
        raise ValueError(f"Could not find origin column. Available columns: {list(gdf.columns)}")
    if not dest_cols:
        raise ValueError(f"Could not find destination column. Available columns: {list(gdf.columns)}")
    
    origin_col = origin_cols[0]
    dest_col = dest_cols[0]
    
    print(f"Using origin column: '{origin_col}'")
    print(f"Using destination column: '{dest_col}'")
    
    origins = gdf[origin_col].values.astype(float)
    destinations = gdf[dest_col].values.astype(float)
    
    # Try to get grid IDs
    id_cols = [col for col in gdf.columns if 'id' in col.lower() or 'grid' in col.lower()]
    if id_cols and id_cols[0] != 'geometry':
        grid_ids = gdf[id_cols[0]].values
        print(f"Using grid ID column: '{id_cols[0]}'")
    else:
        grid_ids = np.arange(len(gdf))
        print("No grid ID column found, using sequential IDs")
    
    print(f"\nData summary:")
    print(f"  Number of grids: {len(gdf)}")
    print(f"  Origins range: {origins.min():.1f} - {origins.max():.1f}")
    print(f"  Destinations range: {destinations.min():.1f} - {destinations.max():.1f}")
    print(f"  Coordinate range X: {coordinates[:, 0].min():.1f} - {coordinates[:, 0].max():.1f}")
    print(f"  Coordinate range Y: {coordinates[:, 1].min():.1f} - {coordinates[:, 1].max():.1f}")
    
    return origins, destinations, coordinates, grid_ids


def main():
    """
    Main execution function
    """
    print("GRAVITY MODEL OD MATRIX CALCULATOR")
    print("=" * 50)
    
    # ============================================
    # 1. LOAD THE DATA HERE
    # ============================================
    # Load your GeoJSON data:
    origins, destinations, coordinates, grid_ids = load_data('data/combined_1000m.geojson')
    
    # Or for demonstration, use sample data:
    # origins, destinations, coordinates = generate_sample_data(n_grids=10088)
    # grid_ids = np.arange(len(origins))
    
    # ============================================
    # 2. INITIALIZE AND RUN GRAVITY MODEL
    # ============================================
    
    from gravity_model import GravityModelOD  # Import your existing class
    
    model = GravityModelOD(alpha=1.0, beta=1.0, gamma=2.0)
    
    # Calculate OD matrix
    od_matrix, info_dict = model.calculate_od_matrix(
        origins=origins,
        destinations=destinations,
        coordinates=coordinates,
        total_trips=None,
        distance_type='euclidean'
    )
    
    # ============================================
    # 3. EXPORT WITH ACTUAL GRID IDs
    # ============================================
    
    vectors_list = model.export_as_vectors(od_matrix, grid_ids)
    model.save_vectors(vectors_list, 'od_vectors.json')
    
    print("\n" + "=" * 60)
    print("PROCESS COMPLETED SUCCESSFULLY!")
    print("=" * 60)


def parameter_sensitivity_analysis(origins, destinations, coordinates):
    """
    Example function to test different parameter combinations
    """
    print("\n" + "=" * 60)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 60)
    
    gamma_values = [1.5, 2.0, 2.5]
    
    for gamma in gamma_values:
        print(f"\nTesting gamma = {gamma}")
        model = GravityModelOD(alpha=1.0, beta=1.0, gamma=gamma)
        
        od_matrix, info_dict = model.calculate_od_matrix(
            origins=origins,
            destinations=destinations,
            coordinates=coordinates
        )
        
        stats = info_dict['statistics']
        print(f"  Total trips: {stats['total_estimated_trips']:,.0f}")
        print(f"  Sparsity: {stats['sparsity']:.2%}")
        print(f"  Origin correlation: {stats['origin_correlation']:.4f}")


if __name__ == "__main__":
    # Run main function
    main()
    
    # For parameter sensitivity analysis (optional)
    # origins, destinations, coordinates = generate_sample_data(n_grids=5000)
    # parameter_sensitivity_analysis(origins, destinations, coordinates)