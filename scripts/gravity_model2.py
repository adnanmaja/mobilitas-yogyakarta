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
    
    def combine_od_vectors(
        self,
        vectors1: List[Dict], 
        vectors2: List[Dict], 
        vectors3: List[Dict], 
        weights: Tuple[float, float, float],
        n_grids: int
    ) -> List[Dict]:
        """
        Combine multiple OD vectors with weights
        """
        w1, w2, w3 = weights
        
        # Convert to dictionaries for easier combination
        od_dict = {i: {} for i in range(n_grids)}
        
        # Helper function to add vectors to dictionary
        def add_to_dict(vectors, weight):
            for vector in vectors:
                origin_idx = vector['origin_idx']
                for dest_idx, value in zip(vector['dest_indices'], vector['values']):
                    key = (origin_idx, dest_idx)
                    if key in od_dict[origin_idx]:
                        od_dict[origin_idx][key] += value * weight
                    else:
                        od_dict[origin_idx][key] = value * weight
        
        # Add all vectors
        add_to_dict(vectors1, w1)
        add_to_dict(vectors2, w2)
        add_to_dict(vectors3, w3)
        
        # Convert back to vector format
        combined_vectors = []
        
        for origin_idx in range(n_grids):
            if od_dict[origin_idx]:
                dest_indices = []
                values = []
                
                for (_, dest_idx), value in od_dict[origin_idx].items():
                    if value > 1e-10:  # Apply threshold
                        dest_indices.append(dest_idx)
                        values.append(value)
                
                if dest_indices:  # Only add if there are non-zero destinations
                    combined_vectors.append({
                        'origin_idx': origin_idx,
                        'dest_indices': np.array(dest_indices),
                        'values': np.array(values)
                    })
        
        return combined_vectors
    
    def calculate_combined_od(
        self,
        residential: np.ndarray,
        employment: np.ndarray,
        amenity: np.ndarray,
        coordinates: np.ndarray,
        grid_ids: np.ndarray,
        weights: Tuple[float, float, float] = (0.5, 0.3, 0.2),  # w1, w2, w3
        model_params: Dict[str, Dict] = None
    ) -> Tuple[List[Dict], Dict]:
        """
        Calculate combined OD matrix from multiple trip purposes
        
        Parameters:
        -----------
        weights : Tuple[float, float, float]
            Weights for HBW, HBNW, NHB trip purposes (w1, w2, w3)
        model_params : Dict[str, Dict]
            Optional dictionary with specific parameters for each model
            Example: {
                'HBW': {'alpha': 1.0, 'beta': 1.0, 'gamma': 2.0},
                'HBNW': {'alpha': 1.0, 'beta': 1.0, 'gamma': 1.8},
                'NHB': {'alpha': 1.0, 'beta': 1.0, 'gamma': 2.2}
            }
        """
        
        # Default parameters if not provided
        if model_params is None:
            model_params = {
                'HBW': {'alpha': 1.0, 'beta': 1.0, 'gamma': 2.0},
                'HBNW': {'alpha': 1.0, 'beta': 1.0, 'gamma': 2.0},
                'NHB': {'alpha': 1.0, 'beta': 1.0, 'gamma': 2.0}
            }
        
        w1, w2, w3 = weights
        
        # Calculate distance matrix once (reused for all models)
        print("Calculating distance matrix...")
        distance_matrix = cdist(coordinates, coordinates, metric='euclidean')
        np.fill_diagonal(distance_matrix, 1.0)
        
        # 1. HBW: Residential → Employment
        print("\n" + "=" * 50)
        print("Calculating HBW trips (Residential → Employment)")
        print("=" * 50)
        model_hbw = GravityModelOD(**model_params['HBW'])
        od_hbw, info_hbw = model_hbw.calculate_od_matrix(
            origins=residential,
            destinations=employment,
            coordinates=coordinates,
            distance_matrix=distance_matrix,
            chunk_size=500
        )
        
        # 2. HBNW: Residential → Amenity
        print("\n" + "=" * 50)
        print("Calculating HBNW trips (Residential → Amenity)")
        print("=" * 50)
        model_hbnw = GravityModelOD(**model_params['HBNW'])
        od_hbnw, info_hbnw = model_hbnw.calculate_od_matrix(
            origins=residential,
            destinations=amenity,
            coordinates=coordinates,
            distance_matrix=distance_matrix,
            chunk_size=500
        )
        
        # 3. NHB: Employment → Amenity
        print("\n" + "=" * 50)
        print("Calculating NHB trips (Employment → Amenity)")
        print("=" * 50)
        model_nhb = GravityModelOD(**model_params['NHB'])
        od_nhb, info_nhb = model_nhb.calculate_od_matrix(
            origins=employment,
            destinations=amenity,
            coordinates=coordinates,
            distance_matrix=distance_matrix,
            chunk_size=500
        )
        
        # Combine the OD matrices with weights
        print("\n" + "=" * 50)
        print("Combining trip purposes with weights")
        print(f"Weights: HBW={w1:.2f}, HBNW={w2:.2f}, NHB={w3:.2f}")
        print("=" * 50)
        
        combined_vectors = self.combine_od_vectors(
            od_hbw, od_hbnw, od_nhb, 
            weights=(w1, w2, w3), 
            n_grids=len(residential)
        )
        
        # Create combined info dictionary
        combined_info = {
            'weights': {
                'HBW': w1,
                'HBNW': w2,
                'NHB': w3
            },
            'model_parameters': model_params,
            'HBW_info': info_hbw,
            'HBNW_info': info_hbnw,
            'NHB_info': info_nhb
        }
        
        return combined_vectors, combined_info
    
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
    residential_cols = [col for col in gdf.columns if 'residential_intensity_norm' in col.lower()]
    employment_cols = [col for col in gdf.columns if 'employment_edu_intensity_norm' in col.lower()]
    amenity_cols = [col for col in gdf.columns if 'amenity_intensity_norm' in col.lower()]
    
    if not residential_cols or not employment_cols or not amenity_cols:
        raise ValueError(f"Could not find origin/destination columns in: {list(gdf.columns)}")
    
    residential = gdf[residential_cols[0]].values.astype(float)
    employment = gdf[employment_cols[0]].values.astype(float)
    amenity = gdf[amenity_cols[0]].values.astype(float)
    
    # Get grid IDs
    id_cols = [col for col in gdf.columns if 'id' in col.lower()]
    grid_ids = gdf[id_cols[0]].values if id_cols else np.arange(len(gdf))
    
    print(f"\nData summary:")
    print(f"  Number of grids: {len(gdf)}")
    print(f"  Residential intensity: {residential.min():.1f} - {residential.max():.1f}")
    print(f"  Employment intensity: {employment.min():.1f} - {employment.max():.1f}")
    print(f"  Amenity intensity: {amenity.min():.1f} - {amenity.max():.1f}")
    
    return residential, employment, amenity, coordinates, grid_ids, gdf


def main():
    """Main execution function for combined trip purposes"""
    print("COMBINED GRAVITY MODEL FOR MULTIPLE TRIP PURPOSES")
    print("=" * 60)
    
    # Load data with multiple intensities
    residential, employment, amenity, coordinates, grid_ids, gdf = load_data(
        'data/raw/rea_1000m.geojson'
    )
    
    # Define weights for time of day (adjust these based on your analysis)
    # Example: Morning peak might have higher HBW weight
    weights = (0.5, 0.3, 0.2)  # w1 (HBW), w2 (HBNW), w3 (NHB)
    
    # Optional: Define different parameters for each trip purpose
    model_params = {
        'HBW': {'alpha': 1.0, 'beta': 1.0, 'gamma': 2.0},   # Work trips are more distance-sensitive
        'HBNW': {'alpha': 0.9, 'beta': 0.9, 'gamma': 1.8},  # Non-work trips are less distance-sensitive
        'NHB': {'alpha': 1.0, 'beta': 1.0, 'gamma': 2.2}    # Non-home-based trips might be different
    }
    
    # Calculate combined OD matrix
    model = GravityModelOD()
    combined_vectors, combined_info = model.calculate_combined_od(
        residential=residential,
        employment=employment,
        amenity=amenity,
        coordinates=coordinates,
        grid_ids=grid_ids,
        weights=weights,
        model_params=model_params
    )
    
    # Export combined results
    # Create instance for saving
    model.save_vectors_streaming(
        od_vectors=combined_vectors,
        grid_ids=grid_ids,
        filename='data/raw/rea_1000m_vectors.json',
        chunk_size=1000
    )
    
    # Save individual trip purposes for analysis
    print("\nSaving individual trip purposes...")
    
    print("\n" + "=" * 60)
    print("PROCESS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Total combined vectors: {len(combined_vectors)}")


if __name__ == "__main__":
    main()