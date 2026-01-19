# Idk what's going on here. Essentially, make an OD matrix and apply the gravity model

# Memory-efficient gravity model with chunked calculation

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
    Memory-efficient Gravity Model for Origin-Destination Matrix calculation
    
    The gravity model formula: T_ij = k * (O_i^α * D_j^β) / (d_ij^γ)
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 1.0, gamma: float = 2.0):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.k = None
        
    def calculate_distances(self, coordinates: np.ndarray, 
                           distance_type: str = 'euclidean') -> np.ndarray:
        """Calculate distance matrix between all grid points"""
        print(f"Calculating {distance_type} distance matrix for {len(coordinates)} grids...")
        start_time = time.time()
        
        distance_matrix = cdist(coordinates, coordinates, metric=distance_type)
        np.fill_diagonal(distance_matrix, 1.0)
        
        elapsed_time = time.time() - start_time
        print(f"Distance matrix calculated in {elapsed_time:.2f} seconds")
        
        return distance_matrix
    
    def calculate_od_matrix(self, origins: np.ndarray, destinations: np.ndarray, 
                           coordinates: np.ndarray, total_trips: float = None,
                           distance_matrix: np.ndarray = None,
                           distance_type: str = 'euclidean',
                           chunk_size: int = 500) -> Tuple[np.ndarray, Dict]:
        """
        Calculate OD matrix using gravity model with memory-efficient chunking
        
        Parameters:
        -----------
        chunk_size : int
            Number of origin rows to process at once (reduce if still running out of memory)
        """
        print("=" * 60)
        print("CALCULATING OD MATRIX (MEMORY-EFFICIENT MODE)")
        print("=" * 60)
        
        n_grids = len(origins)
        print(f"Number of grids: {n_grids:,}")
        print(f"Chunk size: {chunk_size} rows at a time")
        print(f"Total origins: {origins.sum():,.0f}")
        print(f"Total destinations: {destinations.sum():,.0f}")
        
        # Calculate or use provided distance matrix
        if distance_matrix is None:
            distance_matrix = self.calculate_distances(coordinates, distance_type)
        
        print("\nApplying gravity model in chunks...")
        start_time = time.time()
        
        # First pass: calculate sum of unnormalized values for scaling
        print("Pass 1/2: Calculating scaling constant...")
        total_unnormalized = 0.0
        
        D_beta = destinations ** self.beta  # Pre-calculate once
        
        for i in range(0, n_grids, chunk_size):
            end_idx = min(i + chunk_size, n_grids)
            
            # Get chunk of origins
            O_chunk = origins[i:end_idx].reshape(-1, 1)
            O_alpha_chunk = O_chunk ** self.alpha
            
            # Get chunk of distances
            dist_chunk = distance_matrix[i:end_idx, :]
            
            # Calculate unnormalized gravity for this chunk
            with np.errstate(divide='ignore', invalid='ignore'):
                chunk_unnormalized = (O_alpha_chunk * D_beta) / (dist_chunk ** self.gamma)
            
            chunk_unnormalized = np.nan_to_num(chunk_unnormalized, nan=0.0, posinf=0.0, neginf=0.0)
            total_unnormalized += chunk_unnormalized.sum()
            
            if (i // chunk_size) % 5 == 0:
                progress = (end_idx / n_grids) * 100
                print(f"  Progress: {progress:.1f}% ({end_idx}/{n_grids})")
        
        # Calculate scaling constant
        if total_trips is None:
            total_trips = min(origins.sum(), destinations.sum())
        
        self.k = total_trips / total_unnormalized if total_unnormalized > 0 else 1.0
        print(f"Scaling constant k = {self.k:.6e}")
        
        # Second pass: calculate and store actual OD values
        print("\nPass 2/2: Generating final OD matrix...")
        
        # Initialize sparse storage
        od_vectors = []
        
        for i in range(0, n_grids, chunk_size):
            end_idx = min(i + chunk_size, n_grids)
            
            # Get chunk of origins
            O_chunk = origins[i:end_idx].reshape(-1, 1)
            O_alpha_chunk = O_chunk ** self.alpha
            
            # Get chunk of distances
            dist_chunk = distance_matrix[i:end_idx, :]
            
            # Calculate gravity for this chunk
            with np.errstate(divide='ignore', invalid='ignore'):
                chunk_od = self.k * (O_alpha_chunk * D_beta) / (dist_chunk ** self.gamma)
            
            chunk_od = np.nan_to_num(chunk_od, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Store non-zero values for each origin in chunk
            for local_idx, global_idx in enumerate(range(i, end_idx)):
                row = chunk_od[local_idx]
                non_zero_mask = row > 1e-10  # Threshold for sparsity
                
                if non_zero_mask.any():
                    od_vectors.append({
                        'origin_idx': global_idx,
                        'dest_indices': np.where(non_zero_mask)[0],
                        'values': row[non_zero_mask]
                    })
            
            if (i // chunk_size) % 5 == 0:
                progress = (end_idx / n_grids) * 100
                print(f"  Progress: {progress:.1f}% ({end_idx}/{n_grids})")
        
        elapsed_time = time.time() - start_time
        print(f"\nOD matrix calculated in {elapsed_time:.2f} seconds")
        
        # Calculate statistics from sparse representation
        stats = self._calculate_statistics_sparse(od_vectors, origins, destinations, n_grids)
        
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
        
        return od_vectors, info_dict
    
    def _calculate_statistics_sparse(self, od_vectors: List[Dict], 
                                     origins: np.ndarray, 
                                     destinations: np.ndarray,
                                     n_grids: int) -> Dict:
        """Calculate statistics from sparse OD representation"""
        stats = {}
        
        # Calculate row and column sums efficiently
        row_sums = np.zeros(n_grids)
        col_sums = np.zeros(n_grids)
        total_trips = 0.0
        non_zero_count = 0
        
        for vector in od_vectors:
            origin_idx = vector['origin_idx']
            dest_indices = vector['dest_indices']
            values = vector['values']
            
            row_sum = values.sum()
            row_sums[origin_idx] = row_sum
            
            for dest_idx, value in zip(dest_indices, values):
                col_sums[dest_idx] += value
            
            total_trips += row_sum
            non_zero_count += len(values)
        
        stats['total_estimated_trips'] = total_trips
        stats['non_zero_elements'] = non_zero_count
        stats['sparsity'] = 1 - (non_zero_count / (n_grids * n_grids))
        
        # Correlations
        stats['origin_correlation'] = np.corrcoef(origins, row_sums)[0, 1]
        stats['destination_correlation'] = np.corrcoef(destinations, col_sums)[0, 1]
        
        # Error metrics
        origin_error = np.abs(row_sums - origins).sum() / origins.sum()
        destination_error = np.abs(col_sums - destinations).sum() / destinations.sum()
        
        stats['origin_mean_absolute_error'] = origin_error
        stats['destination_mean_absolute_error'] = destination_error
        
        return stats
    
    def save_vectors_streaming(self, od_vectors: List[Dict], grid_ids: np.ndarray, 
                               filename: str, chunk_size: int = 1000):
        """
        Stream vectors directly to JSON file without loading all into memory
        
        Parameters:
        -----------
        od_vectors : List[Dict]
            Sparse OD representation from calculate_od_matrix
        grid_ids : np.ndarray
            Grid IDs for mapping
        filename : str
            Output JSON filename
        chunk_size : int
            Number of vectors to process at once
        """
        import json
        
        print("\nStreaming vectors to file...")
        start_time = time.time()
        
        total_vectors = len(od_vectors)
        
        with open(filename, 'w') as f:
            # Write opening bracket
            f.write('[\n')
            
            # Process in chunks
            for i in range(0, total_vectors, chunk_size):
                end_idx = min(i + chunk_size, total_vectors)
                chunk = od_vectors[i:end_idx]
                
                # Convert chunk to export format
                for j, vector in enumerate(chunk):
                    origin_idx = vector['origin_idx']
                    origin_id = grid_ids[origin_idx]
                    
                    destinations_list = [
                        {
                            'destination_id': int(grid_ids[dest_idx]),
                            'trips': float(value)
                        }
                        for dest_idx, value in zip(vector['dest_indices'], vector['values'])
                    ]
                    
                    export_vector = {
                        'origin_id': int(origin_id),
                        'destinations': destinations_list,
                        'total_trips': float(vector['values'].sum())
                    }
                    
                    # Write to file
                    json_str = json.dumps(export_vector, indent=2)
                    
                    # Add proper JSON array formatting
                    if i + j < total_vectors - 1:
                        f.write('  ' + json_str.replace('\n', '\n  ') + ',\n')
                    else:
                        f.write('  ' + json_str.replace('\n', '\n  ') + '\n')
                
                # Progress update
                if (i // chunk_size) % 5 == 0:
                    progress = (end_idx / total_vectors) * 100
                    print(f"  Progress: {progress:.1f}% ({end_idx}/{total_vectors})")
            
            # Write closing bracket
            f.write(']\n')
        
        elapsed_time = time.time() - start_time
        print(f"Vectors saved to {filename} in {elapsed_time:.2f} seconds")
        print(f"Total vectors written: {total_vectors}")
    
    def save_vectors(self, vectors_list: List[Dict], filename: str):
        """Save vectors to JSON file"""
        import json
        
        with open(filename, 'w') as f:
            json.dump(vectors_list, f, indent=2)
        
        print(f"\nVectors saved to {filename}")


def load_data(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load data from GeoJSON file"""
    print(f"Loading data from {filepath}...")
    
    gdf = gpd.read_file(filepath)
    
    print(f"Loaded {len(gdf)} features")
    print(f"Current CRS: {gdf.crs}")
    
    if not all(gdf.geometry.geom_type == 'Point'):
        raise ValueError("All geometries must be Point type")
    
    if gdf.crs.to_epsg() == 4326:
        print("Reprojecting to EPSG:3857 (Web Mercator)...")
        gdf = gdf.to_crs(epsg=3857)
    
    coordinates = np.column_stack([gdf.geometry.x.values, gdf.geometry.y.values])
    
    # Find origin and destination columns
    origin_cols = [col for col in gdf.columns if 'origin' in col.lower()]
    dest_cols = [col for col in gdf.columns if 'destination' in col.lower() or 'dest' in col.lower()]
    
    if not origin_cols or not dest_cols:
        raise ValueError(f"Could not find origin/destination columns in: {list(gdf.columns)}")
    
    origins = gdf[origin_cols[0]].values.astype(float)
    destinations = gdf[dest_cols[0]].values.astype(float)
    
    # Get grid IDs
    id_cols = [col for col in gdf.columns if 'id' in col.lower()]
    grid_ids = gdf[id_cols[0]].values if id_cols else np.arange(len(gdf))
    
    print(f"\nData summary:")
    print(f"  Number of grids: {len(gdf)}")
    print(f"  Origins: {origins.min():.1f} - {origins.max():.1f}")
    print(f"  Destinations: {destinations.min():.1f} - {destinations.max():.1f}")
    
    return origins, destinations, coordinates, grid_ids


def main():
    """Main execution function"""
    print("MEMORY-EFFICIENT GRAVITY MODEL")
    print("=" * 50)
    
    # Load data
    origins, destinations, coordinates, grid_ids = load_data('data/raw/combined_v4_weekend_1000m.geojson')
    
    # Initialize model
    model = GravityModelOD(alpha=1.0, beta=1.0, gamma=2.0)
    
    # Calculate OD matrix with chunking (adjust chunk_size if needed)
    # Reduce chunk_size (e.g., 200, 100) if still running out of memory
    od_vectors, info_dict = model.calculate_od_matrix(
        origins=origins,
        destinations=destinations,
        coordinates=coordinates,
        total_trips=None,
        distance_type='euclidean',
        chunk_size=500  # Adjust this value based on your available memory
    )
    
    # Export
    model.save_vectors_streaming(
        od_vectors=od_vectors,
        grid_ids=grid_ids,
        filename='data/raw/weekend_od_vectors.json',
        chunk_size=1000  # Process 1000 vectors at a time
    )
    
    print("\n" + "=" * 60)
    print("PROCESS COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()