import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import time
from typing import Tuple, List, Dict, Optional
import geopandas as gpd
import warnings
warnings.filterwarnings('ignore')

class ImprovedGravityModel:
    """
    Enhanced Gravity Model with IPF balancing and realistic trip purpose modeling
    
    Step-by-step implementation:
    1. Set total trips per purpose
    2. Define productions & attractions
    3. Gravity model with different gammas
    4. IPF balancing (Furness)
    5. Combine trip purposes
    """
    
    def __init__(self, chunk_size: int = 500):
        self.chunk_size = chunk_size
        self.total_trips = None
        
    def calculate_distances(self, coordinates: np.ndarray) -> np.ndarray:
        """Calculate Euclidean distance matrix"""
        print("Calculating distance matrix...")
        start_time = time.time()
        
        distance_matrix = cdist(coordinates, coordinates, metric='euclidean')
        np.fill_diagonal(distance_matrix, 1.0)  # Avoid division by zero
        
        elapsed_time = time.time() - start_time
        print(f"Distance matrix calculated in {elapsed_time:.2f} seconds")
        
        return distance_matrix
    
    def gravity_model_chunked(self,
                            productions: np.ndarray,
                            attractions: np.ndarray,
                            distance_matrix: np.ndarray,
                            gamma: float,
                            total_trips: float = None,
                            alpha: float = 1.0,
                            beta: float = 1.0) -> np.ndarray:
        """
        Calculate gravity model T_ij = (P_i^α * A_j^β) / (d_ij^γ) with chunking
        
        Returns unnormalized gravity matrix
        """
        n = len(productions)
        
        # Calculate unnormalized gravity matrix in chunks
        od_matrix = np.zeros((n, n))
        
        # Pre-calculate productions^alpha and attractions^beta
        P_alpha = productions ** alpha
        A_beta = attractions ** beta
        
        for i in range(0, n, self.chunk_size):
            end_i = min(i + self.chunk_size, n)
            
            # Get chunk of production values
            P_chunk = P_alpha[i:end_i].reshape(-1, 1)
            
            # Get chunk of distance matrix
            dist_chunk = distance_matrix[i:end_i, :]
            
            # Calculate gravity for this chunk
            with np.errstate(divide='ignore', invalid='ignore'):
                gravity_chunk = (P_chunk * A_beta) / (dist_chunk ** gamma)
            
            # Handle invalid values
            gravity_chunk = np.nan_to_num(gravity_chunk, nan=0.0, posinf=0.0, neginf=0.0)
            od_matrix[i:end_i, :] = gravity_chunk
            
            if (i // self.chunk_size) % 10 == 0:
                progress = (end_i / n) * 100
                print(f"  Gravity calculation: {progress:.1f}%")
        
        return od_matrix
    
    def ipf_balancing(self,
                     od_matrix: np.ndarray,
                     productions: np.ndarray,
                     attractions: np.ndarray,
                     max_iterations: int = 15,
                     tolerance: float = 1e-6) -> np.ndarray:
        """
        Iterative Proportional Fitting (Furness method) to balance OD matrix
        
        Ensures: sum_j T_ij = P_i and sum_i T_ij = A_j
        """
        print(f"  IPF balancing (max {max_iterations} iterations)...")
        
        n = len(productions)
        T = od_matrix.copy()
        
        # Initialize scaling factors
        row_factors = np.ones(n)
        col_factors = np.ones(n)
        
        for iteration in range(max_iterations):
            # Row balancing (match productions)
            row_sums = T.sum(axis=1)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            row_factors_new = productions / row_sums
            
            T = T * row_factors_new[:, np.newaxis]
            
            # Column balancing (match attractions)
            col_sums = T.sum(axis=0)
            col_sums[col_sums == 0] = 1
            col_factors_new = attractions / col_sums
            
            T = T * col_factors_new[np.newaxis, :]
            
            # Calculate convergence
            row_error = np.abs(T.sum(axis=1) - productions).sum() / productions.sum()
            col_error = np.abs(T.sum(axis=0) - attractions).sum() / attractions.sum()
            total_error = max(row_error, col_error)
            
            if iteration % 5 == 0:
                print(f"    Iteration {iteration}: error = {total_error:.6f}")
            
            if total_error < tolerance:
                print(f"  IPF converged after {iteration + 1} iterations")
                break
        
        # Final scaling to ensure exact match (optional but recommended)
        T = self._exact_scaling(T, productions, attractions)
        
        return T
    
    def _exact_scaling(self, T: np.ndarray, P: np.ndarray, A: np.ndarray) -> np.ndarray:
        """Ensure exact match to productions and attractions"""
        # First scale rows to match productions exactly
        row_sums = T.sum(axis=1)
        row_sums[row_sums == 0] = 1
        T = T * (P / row_sums)[:, np.newaxis]
        
        # Then scale columns to match attractions exactly
        col_sums = T.sum(axis=0)
        col_sums[col_sums == 0] = 1
        T = T * (A / col_sums)[np.newaxis, :]
        
        return T
    
    def calculate_trip_purpose(self,
                              productions: np.ndarray,
                              attractions: np.ndarray,
                              distance_matrix: np.ndarray,
                              purpose_name: str,
                              purpose_params: Dict) -> np.ndarray:
        """
        Calculate OD matrix for a single trip purpose with IPF balancing
        
        purpose_params should contain:
            - gamma: distance decay parameter
            - total_trips: total trips for this purpose (optional)
            - alpha, beta: production/attraction elasticities (default 1.0)
        """
        print(f"\n{'='*60}")
        print(f"Calculating {purpose_name} trips")
        print(f"{'='*60}")
        
        # Extract parameters
        gamma = purpose_params.get('gamma', 2.0)
        total_trips = purpose_params.get('total_trips')
        alpha = purpose_params.get('alpha', 1.0)
        beta = purpose_params.get('beta', 1.0)
        
        print(f"  Gamma (distance decay): {gamma}")
        print(f"  Alpha (production elasticity): {alpha}")
        print(f"  Beta (attraction elasticity): {beta}")
        print(f"  Total trips target: {total_trips if total_trips else 'Not specified'}")
        
        # Step 1: Calculate unnormalized gravity matrix
        print("  Step 1: Calculating gravity model...")
        start_time = time.time()
        
        T0 = self.gravity_model_chunked(
            productions=productions,
            attractions=attractions,
            distance_matrix=distance_matrix,
            gamma=gamma,
            alpha=alpha,
            beta=beta
        )
        
        gravity_time = time.time() - start_time
        print(f"    Gravity model calculated in {gravity_time:.2f} seconds")
        print(f"    Unnormalized total: {T0.sum():.2f}")
        
        # Step 2: Scale to total trips if specified
        if total_trips is not None:
            scale_factor = total_trips / T0.sum()
            T0 = T0 * scale_factor
            print(f"  Step 2: Scaled to {total_trips:.0f} trips (scale factor: {scale_factor:.6f})")
        
        # Step 3: IPF balancing
        print("  Step 3: Applying IPF balancing...")
        start_time = time.time()
        
        T_balanced = self.ipf_balancing(
            od_matrix=T0,
            productions=productions,
            attractions=attractions
        )
        
        ipf_time = time.time() - start_time
        print(f"    IPF completed in {ipf_time:.2f} seconds")
        
        # Calculate statistics
        self._calculate_statistics(T_balanced, productions, attractions, purpose_name)
        
        return T_balanced
    
    def combine_trip_purposes(self,
                             od_matrices: Dict[str, np.ndarray],
                             weights: Dict[str, float]) -> np.ndarray:
        """
        Combine multiple OD matrices with weights
        
        Parameters:
        -----------
        od_matrices: Dictionary with purpose_name -> OD matrix
        weights: Dictionary with purpose_name -> weight (should sum to 1.0)
        """
        print(f"\n{'='*60}")
        print("Combining trip purposes")
        print(f"{'='*60}")
        
        # Validate weights
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            print(f"Warning: Weights sum to {total_weight:.3f}, normalizing to 1.0")
            for key in weights:
                weights[key] /= total_weight
        
        # Combine matrices
        combined = None
        for purpose_name, od_matrix in od_matrices.items():
            weight = weights.get(purpose_name, 0.0)
            print(f"  {purpose_name}: weight = {weight:.3f}, trips = {od_matrix.sum():.0f}")
            
            if combined is None:
                combined = od_matrix * weight
            else:
                combined += od_matrix * weight
        
        print(f"\n  Combined total trips: {combined.sum():.0f}")
        
        return combined
    
    def _calculate_statistics(self, od_matrix: np.ndarray,
                            productions: np.ndarray,
                            attractions: np.ndarray,
                            purpose_name: str):
        """Calculate and display statistics for OD matrix"""
        row_sums = od_matrix.sum(axis=1)
        col_sums = od_matrix.sum(axis=0)
        
        print(f"\n  Statistics for {purpose_name}:")
        print(f"    Total trips: {od_matrix.sum():.0f}")
        print(f"    Non-zero cells: {(od_matrix > 0).sum():,} ({100 * (od_matrix > 0).sum() / od_matrix.size:.1f}%)")
        print(f"    Avg trips per origin: {row_sums.mean():.2f}")
        print(f"    Max trips per origin: {row_sums.max():.2f}")
        
        # Production-Attraction correlations
        prod_corr = np.corrcoef(productions, row_sums)[0, 1]
        attr_corr = np.corrcoef(attractions, col_sums)[0, 1]
        
        print(f"    Production correlation: {prod_corr:.3f}")
        print(f"    Attraction correlation: {attr_corr:.3f}")
        
        # Errors
        prod_error = np.abs(row_sums - productions).sum() / productions.sum()
        attr_error = np.abs(col_sums - attractions).sum() / attractions.sum()
        
        print(f"    Production error (MAE): {prod_error:.3%}")
        print(f"    Attraction error (MAE): {attr_error:.3%}")
    
    def save_sparse_vectors(self, od_matrix: np.ndarray,
                           grid_ids: np.ndarray,
                           filename: str,
                           threshold: float = 1e-6):
        """
        Save OD matrix as sparse vectors to JSON file
        
        Each vector: {origin_id: X, destinations: [{dest_id: Y, trips: Z}, ...]}
        """
        import json
        
        print(f"\nSaving sparse vectors to {filename}...")
        start_time = time.time()
        
        n = len(od_matrix)
        vectors = []
        
        for i in range(n):
            # Get non-zero destinations for this origin
            row = od_matrix[i]
            non_zero_mask = row > threshold
            
            if non_zero_mask.any():
                dest_indices = np.where(non_zero_mask)[0]
                values = row[non_zero_mask]
                
                # Create destinations list
                destinations = []
                for dest_idx, value in zip(dest_indices, values):
                    destinations.append({
                        'destination_id': int(grid_ids[dest_idx]),
                        'trips': float(value)
                    })
                
                vectors.append({
                    'origin_id': int(grid_ids[i]),
                    'destinations': destinations,
                    'total_trips': float(values.sum())
                })
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(vectors, f, indent=2)
        
        elapsed_time = time.time() - start_time
        print(f"  Saved {len(vectors)} vectors in {elapsed_time:.2f} seconds")
        print(f"  Sparsity: {(od_matrix > threshold).sum() / od_matrix.size:.1%}")

# Main execution
def main():
    print("IMPROVED GRAVITY MODEL WITH IPF BALANCING")
    print("=" * 60)
    
    # Load data
    def load_data(filepath: str):
        print(f"\nLoading data from {filepath}...")
        gdf = gpd.read_file(filepath)
        
        # Reproject if needed
        if gdf.crs.to_epsg() == 4326:
            print("Reprojecting to Web Mercator (EPSG:3857)...")
            gdf = gdf.to_crs(epsg=3857)
        
        # Extract coordinates
        coordinates = np.column_stack([gdf.geometry.x.values, gdf.geometry.y.values])
        
        # Find intensity columns
        residential_cols = [col for col in gdf.columns if 'residential_intensity_norm' in col.lower()]
        employment_cols = [col for col in gdf.columns if 'employment_edu_intensity_norm' in col.lower()]
        amenity_cols = [col for col in gdf.columns if 'amenity_intensity_norm' in col.lower()]
        id_cols = [col for col in gdf.columns if 'id' in col.lower()]
        
        residential = gdf[residential_cols[0]].values if residential_cols else np.ones(len(gdf))
        employment = gdf[employment_cols[0]].values if employment_cols else np.ones(len(gdf))
        amenity = gdf[amenity_cols[0]].values if amenity_cols else np.ones(len(gdf))
        grid_ids = gdf[id_cols[0]].values if id_cols else np.arange(len(gdf))
        
        # Normalize to sum to 1 for IPF
        residential = residential / residential.sum() * len(gdf)
        employment = employment / employment.sum() * len(gdf)
        amenity = amenity / amenity.sum() * len(gdf)
        
        print(f"  Loaded {len(gdf)} grid cells")
        print(f"  Residential sum: {residential.sum():.1f}")
        print(f"  Employment sum: {employment.sum():.1f}")
        print(f"  Amenity sum: {amenity.sum():.1f}")
        
        return residential, employment, amenity, coordinates, grid_ids, gdf
    
    # Load your data
    residential, employment, amenity, coordinates, grid_ids, gdf = load_data(
        'data/raw/rea_1000m.geojson'
    )
    
    # Initialize model
    model = ImprovedGravityModel(chunk_size=500)
    
    # Calculate distance matrix (once, reused for all purposes)
    print("\nCalculating distance matrix...")
    distance_matrix = model.calculate_distances(coordinates)
    
    # Step 1: Define total trips per purpose
    # You can adjust these based on your study area
    total_trips_hbw = 10000.0   # T_HBW = 1.0
    total_trips_hbnw = 6000.0   # T_HBNW = 0.6
    total_trips_nhb = 4000.0    # T_NHB = 0.4
    
    # Step 2-4: Calculate each trip purpose with IPF
    od_matrices = {}
    
    # HBW: Home-Based Work (Residential -> Employment)
    od_hbw = model.calculate_trip_purpose(
        productions=residential,
        attractions=employment,
        distance_matrix=distance_matrix,
        purpose_name="HBW",
        purpose_params={
            'gamma': 1.4,        # Work trips are more distance-sensitive
            'total_trips': total_trips_hbw,
            'alpha': 1.0,
            'beta': 1.0
        }
    )
    od_matrices['HBW'] = od_hbw
    
    # HBNW: Home-Based Non-Work (Residential -> Amenity)
    od_hbnw = model.calculate_trip_purpose(
        productions=residential,
        attractions=amenity,
        distance_matrix=distance_matrix,
        purpose_name="HBNW",
        purpose_params={
            'gamma': 2.5,        # Non-work trips are less distance-sensitive
            'total_trips': total_trips_hbnw,
            'alpha': 1.0,
            'beta': 1.0
        }
    )
    od_matrices['HBNW'] = od_hbnw
    
    # NHB: Non-Home-Based (Employment -> Amenity)
    od_nhb = model.calculate_trip_purpose(
        productions=employment,
        attractions=amenity,
        distance_matrix=distance_matrix,
        purpose_name="NHB",
        purpose_params={
            'gamma': 2.0,        # Intermediate distance sensitivity
            'total_trips': total_trips_nhb,
            'alpha': 1.0,
            'beta': 1.0
        }
    )
    od_matrices['NHB'] = od_nhb
    
    # Step 5: Combine trip purposes
    # You can adjust these weights based on time of day or other factors
    weights = {
        'HBW': 0.5,   # 50% of total trips
        'HBNW': 0.3,  # 30% of total trips
        'NHB': 0.2    # 20% of total trips
    }
    
    combined_od = model.combine_trip_purposes(od_matrices, weights)
    
    # Save results
    print(f"\n{'='*60}")
    print("Saving results...")
    
    # Save combined OD matrix
    model.save_sparse_vectors(
        od_matrix=combined_od,
        grid_ids=grid_ids,
        filename='data/raw/rea_1000m_vectors_v2.json'
    )
        
    print(f"\n{'='*60}")
    print("PROCESS COMPLETED!")
    print(f"{'='*60}")
    print(f"Total combined trips: {combined_od.sum():.0f}")
    print(f"Number of origins: {len(combined_od)}")
    print(f"Number of OD pairs: {(combined_od > 0).sum():,}")

if __name__ == "__main__":
    main()