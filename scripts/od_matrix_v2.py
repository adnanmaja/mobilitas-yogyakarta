import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import time
from typing import Tuple, List, Dict, Optional
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm
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
        target_total = total_trips
        print(f"  Total trips target: {total_trips if total_trips else 'Not specified'}")
        
        if total_trips is not None:
            productions = productions * total_trips
            attractions = attractions * total_trips
        else:
            # If no total_trips specified, use original values
            productions = productions.copy()
            attractions = attractions.copy()

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
        
        
        # Step 2: IPF balancing
        print("  Step 2: Applying IPF balancing...")
        start_time = time.time()
        
        T_balanced = self.ipf_balancing(
            od_matrix=T0,
            productions=productions,
            attractions=attractions
        )

        ipf_time = time.time() - start_time
        print(f"    IPF completed in {ipf_time:.2f} seconds")
        
        if target_total is not None:
            actual_total = T_balanced.sum()
            difference = abs(actual_total - target_total)
            relative_error = difference / target_total if target_total > 0 else 0
            
            print(f"Trip total verification:")
            print(f"Target total: {target_total:.1f}")
            print(f"Actual total: {actual_total:.1f}")
            print(f"Difference: {difference:.1f} ({relative_error:.2%})")
            
            if relative_error > 0.01:  # More than 1% error
                print(f"Warning: Significant deviation from target!")

        # Calculate statistics
        self._calculate_statistics(T_balanced, productions, attractions, purpose_name)
        
        return T_balanced
    
    def combine_trip_purposes(self,
                             od_matrices: Dict[str, np.ndarray],
                             weights: Dict[str, float]) -> np.ndarray:
        """
        Combine multiple OD matrices with weights
        (Uncalled)
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

    def calculate_average_distance(self, od_matrix: np.ndarray, 
                                distance_matrix: np.ndarray,
                                purpose_name: str) -> float:
        """
        Calculate weighted average trip distance for a given OD matrix
        
        Parameters:
        -----------
        od_matrix: OD matrix with trip counts
        distance_matrix: Distance matrix (same shape as od_matrix)
        purpose_name: Name of trip purpose for printing
        
        Returns:
        --------
        Weighted average distance in same units as distance_matrix
        """
        # Create masks for non-zero trips
        non_zero_mask = od_matrix > 0
        
        if not non_zero_mask.any():
            return 0.0
        
        # Get non-zero trips and corresponding distances
        trips = od_matrix[non_zero_mask]
        distances = distance_matrix[non_zero_mask]
        
        # Calculate weighted average
        weighted_sum = np.sum(trips * distances)
        total_trips = np.sum(trips)
        
        average_distance = weighted_sum / total_trips if total_trips > 0 else 0.0
        
        print(f"\n  Average {purpose_name} trip distance: {average_distance:.2f} meters")
        
        return average_distance
    
    def apply_mode_choice_chunked(self, od_matrix: np.ndarray, 
                                distance_matrix: np.ndarray,
                                purpose: str,
                                chunk_size: int = 1000) -> Dict[str, np.ndarray]:
        """
        Apply rule-based mode choice splitting with chunked processing
        
        Parameters:
        -----------
        od_matrix: OD matrix for a specific purpose
        distance_matrix: Distance matrix
        purpose: Trip purpose (HBW, HBNW, NHB)
        chunk_size: Number of rows to process at once
        
        Returns:
        --------
        Dictionary with mode -> OD matrix
        """
        print(f"  Applying chunked mode choice for {purpose} trips...")
        
        n = len(od_matrix)
        
        # Initialize output matrices with zeros (will fill them chunk by chunk)
        mode_od = {
            # 'walk': np.zeros_like(od_matrix),
            'motorbike': np.zeros_like(od_matrix),
            'car': np.zeros_like(od_matrix)
        }
        
        # Process in chunks to reduce memory usage
        for start_idx in range(0, n, chunk_size):
            end_idx = min(start_idx + chunk_size, n)
            
            # Get chunk of OD matrix and distance matrix
            od_chunk = od_matrix[start_idx:end_idx, :]
            dist_chunk = distance_matrix[start_idx:end_idx, :]
            
            # Calculate mode shares for this chunk
            mode_shares_chunk = self._calculate_mode_shares_chunk(
                distance_matrix=dist_chunk,
                purpose=purpose
            )
            
            # Apply shares to OD chunk and store results
            for mode in ['motorbike', 'car']:
                mode_od[mode][start_idx:end_idx, :] = od_chunk * mode_shares_chunk[mode]
            
            # Progress reporting
            if (start_idx // chunk_size) % 10 == 0:
                progress = (end_idx / n) * 100
                print(f"    Processed {progress:.1f}% of rows")
        
        # Print mode statistics
        total_trips = od_matrix.sum()
        print(f"  Mode statistics for {purpose}:")
        for mode, matrix in mode_od.items():
            mode_trips = matrix.sum()
            print(f"    {mode}: {mode_trips:.0f} trips ({mode_trips/total_trips*100:.1f}%)")
        
        return mode_od

    def _calculate_mode_shares_chunk(self, distance_matrix: np.ndarray, 
                                purpose: str) -> Dict[str, np.ndarray]:
        """
        Calculate mode shares for a chunk of the distance matrix
        
        Returns dictionary with mode -> share matrix (same shape as input)
        """
        d_km = distance_matrix / 1000  # Convert meters to kilometers
        
        # Initialize mode shares
        mode_shares = {
            # 'walk': np.zeros_like(distance_matrix),
            'motorbike': np.zeros_like(distance_matrix),
            'car': np.zeros_like(distance_matrix)
        }
        
        # Calculate base shares by distance
        mask_short = d_km < 3
        # mode_shares['walk'][mask_short] = 0.20
        mode_shares['motorbike'][mask_short] = 0.90
        mode_shares['car'][mask_short] = 0.10
        
        mask_medium = (d_km >= 3) & (d_km < 10)
        mode_shares['motorbike'][mask_medium] = 0.60
        mode_shares['car'][mask_medium] = 0.40
        
        mask_long = (d_km >= 10) & (d_km < 15)
        mode_shares['motorbike'][mask_long] = 0.50
        mode_shares['car'][mask_long] = 0.50
        
        mask_very_long = d_km >= 15
        mode_shares['motorbike'][mask_very_long] = 0.40
        mode_shares['car'][mask_very_long] = 0.60
        
        # Apply purpose-based adjustments
        self._apply_purpose_adjustments(mode_shares, purpose)
        
        # Normalize each cell to sum to 1
        total_shares =  mode_shares['motorbike'] + mode_shares['car'] # + mode_shares['walk']
        
        # Avoid division by zero
        valid_mask = total_shares > 0
        for mode in mode_shares:
            mode_shares[mode][valid_mask] = mode_shares[mode][valid_mask] / total_shares[valid_mask]
            mode_shares[mode][~valid_mask] = 0
        
        return mode_shares

    def _apply_purpose_adjustments(self, mode_shares: Dict[str, np.ndarray], purpose: str):
        """Apply purpose-specific adjustments to mode shares"""
        if purpose == 'HBW':
            # More car for work trips
            # mode_shares['walk'] *= 0.9
            mode_shares['motorbike'] *= 0.8
            mode_shares['car'] *= 1.2
        elif purpose == 'HBNW':
            # More motorbike for non-work trips
            # mode_shares['walk'] *= 1.0
            mode_shares['motorbike'] *= 1.2
            mode_shares['car'] *= 0.9
        elif purpose == 'NHB':
            # More walk + motorbike for non-home-based
            # mode_shares['walk'] *= 1.3
            mode_shares['motorbike'] *= 1.1
            mode_shares['car'] *= 0.8
            
    
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



    def plot_od_heatmap(self, gdf, od_matrix, title, output_file):
        """
        Plot heatmap of trips starting from each cell
        
        Parameters:
        -----------
        gdf : GeoDataFrame
            The original grid data
        od_matrix : np.ndarray
            OD matrix (n x n)
        title : str
            Plot title
        output_file : str
            Output filename
        """
        print(f"\nPlotting OD heatmap...")
        
        # Calculate total trips starting from each cell (row sums)
        trips_per_cell = od_matrix.sum(axis=1)
        
        # Create figure
        fig, ax = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot 1: Spatial distribution of trip origins
        gdf = gdf.copy()
        gdf['trips_from'] = trips_per_cell
        
        # Create bins for coloring
        non_zero_trips = trips_per_cell[trips_per_cell > 0]
        if len(non_zero_trips) > 0:
            # Use log scale for coloring if there's wide variation
            vmin = non_zero_trips.min()
            vmax = non_zero_trips.max()
            
            # Create log norm
            norm = LogNorm(vmin=vmin, vmax=vmax)
            
            # Plot
            gdf.plot(column='trips_from', 
                    cmap='YlOrRd', 
                    ax=ax[0], 
                    legend=True,
                    norm=norm,
                    edgecolor='black',
                    linewidth=0.2)
        else:
            gdf.plot(ax=ax[0], color='lightgray', edgecolor='black', linewidth=0.2)
        
        ax[0].set_title(f'Trip Origins (Total: {trips_per_cell.sum():.0f})')
        ax[0].set_xlabel('Longitude')
        ax[0].set_ylabel('Latitude')
        
        # Add stats
        n_cells_with_trips = (trips_per_cell > 0).sum()
        ax[0].text(0.02, 0.98, 
                f'Cells with trips: {n_cells_with_trips}/{len(gdf)}\n'
                f'({100*n_cells_with_trips/len(gdf):.1f}%)',
                transform=ax[0].transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Plot 2: Histogram of trip counts
        if len(non_zero_trips) > 0:
            ax[1].hist(non_zero_trips, bins=50, edgecolor='black', alpha=0.7)
            ax[1].set_xlabel('Trips per cell (log scale)')
            ax[1].set_ylabel('Frequency')
            ax[1].set_yscale('log')
            ax[1].set_title('Distribution of Trips per Cell')
            ax[1].grid(True, alpha=0.3)
            
            # Add statistics
            stats_text = (f'Mean: {non_zero_trips.mean():.2f}\n'
                        f'Median: {np.median(non_zero_trips):.2f}\n'
                        f'Max: {non_zero_trips.max():.2f}\n'
                        f'Min: {non_zero_trips.min():.2f}')
            ax[1].text(0.02, 0.98, stats_text,
                    transform=ax[1].transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"  Saved plot to {output_file}")
        print(f"  Cells with trips: {n_cells_with_trips}/{len(gdf)} ({100*n_cells_with_trips/len(gdf):.1f}%)")
        
        return gdf

    def plot_od_matrix_detailed(gdf, od_matrix, grid_ids, title="Detailed OD Matrix", 
                            max_origins=20, output_file="od_matrix_detailed.png"):
        """
        Plot detailed view of OD matrix for first N origins
        """
        print(f"\nPlotting detailed OD matrix view...")
        
        # Convert to DataFrame for easier handling
        n = min(max_origins, len(od_matrix))
        
        fig, axes = plt.subplots(n, 1, figsize=(15, 3*n))
        if n == 1:
            axes = [axes]
        
        for i in range(n):
            # Get trips from this origin
            origin_trips = od_matrix[i]
            
            # Create a temporary GeoDataFrame with trip destinations
            temp_gdf = gdf.copy()
            temp_gdf['trips_to'] = origin_trips
            
            # Plot
            non_zero_dest = origin_trips[origin_trips > 0]
            if len(non_zero_dest) > 0:
                vmin = non_zero_dest.min()
                vmax = non_zero_dest.max()
                norm = LogNorm(vmin=vmin, vmax=vmax)
                
                temp_gdf.plot(column='trips_to', 
                            cmap='Blues', 
                            ax=axes[i], 
                            legend=False,
                            norm=norm,
                            edgecolor='black',
                            linewidth=0.1)
            else:
                temp_gdf.plot(ax=axes[i], color='lightgray', 
                            edgecolor='black', linewidth=0.1)
            
            # Mark the origin cell
            origin_geom = gdf.iloc[i].geometry
            if hasattr(origin_geom, 'centroid'):
                centroid = origin_geom.centroid
                axes[i].plot(centroid.x, centroid.y, 'ro', markersize=8, 
                            markeredgecolor='black', markeredgewidth=1)
            
            axes[i].set_title(f'Origin Cell {grid_ids[i]}: {origin_trips.sum():.1f} trips to {(origin_trips>0).sum()} destinations')
            axes[i].set_xlabel('')
            axes[i].set_ylabel('')
        
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"  Saved detailed plot to {output_file}")    

# Main execution
def main():
    print("Gravity model n IPF balancing")
    print("=" * 60)
    
    def load_data(filepath: str):
        print(f"\nLoading data from {filepath}...")
        gdf = gpd.read_file(filepath)
        
        if gdf.crs.to_epsg() == 4326:
            gdf = gdf.to_crs(epsg=3857)
        
        # Extract coordinates
        coordinates = np.column_stack([gdf.geometry.x.values, gdf.geometry.y.values])
        
        # Find intensity columns
        residential_cols = [col for col in gdf.columns if 'residential_intensity' in col.lower()]
        employment_cols = [col for col in gdf.columns if 'employment_intensity' in col.lower()]
        amenity_hbnw_cols = [col for col in gdf.columns if 'amenity_hbnw_intensity' in col.lower()]
        amenity_nhb_cols = [col for col in gdf.columns if 'amenity_nhb_intensity' in col.lower()]
        id_cols = [col for col in gdf.columns if 'id' in col.lower()]
        
        residential = gdf[residential_cols[0]].values if residential_cols else np.ones(len(gdf))
        employment = gdf[employment_cols[0]].values if employment_cols else np.ones(len(gdf))
        amenity_hbnw = gdf[amenity_hbnw_cols[0]].values if amenity_hbnw_cols else np.ones(len(gdf))
        amenity_nhb = gdf[amenity_nhb_cols[0]].values if amenity_nhb_cols else np.ones(len(gdf))
        grid_ids = gdf[id_cols[0]].values if id_cols else np.arange(len(gdf))
        
        # Normalize to sum to 1 for IPF
        residential = residential / residential.sum() 
        employment = employment / employment.sum() 
        amenity_hbnw = amenity_hbnw / amenity_hbnw.sum() 
        amenity_nhb = amenity_nhb/ amenity_nhb.sum() 

        print(f"  Loaded {len(gdf)} grid cells")
        print(f"  Residential sum: {residential.sum():.1f}")
        print(f"  Employment sum: {employment.sum():.1f}")
        print(f"  Amenity (HBNW) sum: {amenity_hbnw.sum():.1f}")
        print(f"  Amenity (NHB) sum: {amenity_nhb.sum():.1f}")

        print("\n=== DATA VALIDITY CHECKS ===")
        print(f"Number of grid cells: {len(gdf)}")
        print(f"Residential zeros: {(residential == 0).sum()} / {len(residential)}")
        print(f"Employment zeros: {(employment == 0).sum()} / {len(employment)}")
        print(f"Amenity HBNW zeros: {(amenity_hbnw == 0).sum()} / {len(amenity_hbnw)}")
        print(f"Amenity NHB zeros: {(amenity_nhb == 0).sum()} / {len(amenity_nhb)}")

        # Check coordinate ranges
        print(f"\nCoordinate ranges:")
        print(f"X min/max: {coordinates[:, 0].min():.0f} / {coordinates[:, 0].max():.0f}")
        print(f"Y min/max: {coordinates[:, 1].min():.0f} / {coordinates[:, 1].max():.0f}")
        
        return residential, employment, amenity_hbnw, amenity_nhb, coordinates, grid_ids, gdf
    
    residential, employment,  amenity_hbnw, amenity_nhb, coordinates, grid_ids, gdf = load_data(
        'data/raw/rea_1000m_v2.geojson'
    )
    
    # Initialize model
    model = ImprovedGravityModel(chunk_size=500)
    
    # Calculate distance matrix (once, reused for all purposes)
    print("\nCalculating distance matrix...")
    distance_matrix = model.calculate_distances(coordinates)
    
    # Define total trip ratios per purpose
    # (Devi et al., 2019)
    total_trips_hbw = 623.8   
    total_trips_hbnw = 277.7   
    total_trips_nhb = 98.6      
    
    # Calculate each trip purpose with IPF
    od_matrices = {}
    
    # HBW: Home-Based Work (Residential -> Employment)
    od_hbw = model.calculate_trip_purpose(
        productions=residential,
        attractions=employment,
        distance_matrix=distance_matrix,
        purpose_name="HBW",
        purpose_params={
            'gamma': 1.2,        # Fine with longer trip
            'total_trips': total_trips_hbw,
            'alpha': 1.0,
            'beta': 1.0
        }
    )
    od_matrices['HBW'] = od_hbw
    avg_hbw_distance = model.calculate_average_distance(od_hbw, distance_matrix, "HBW")
    model.plot_od_heatmap(gdf, od_hbw, title="HBW Trip Origins", output_file="hbw_trip_origins.png")
    
    # HBNW: Home-Based Non-Work (Residential -> Amenity)
    od_hbnw = model.calculate_trip_purpose(
        productions=residential,
        attractions=amenity_hbnw,
        distance_matrix=distance_matrix,
        purpose_name="HBNW",
        purpose_params={
            'gamma': 3.5,        # Prefer spots that's closer to home
            'total_trips': total_trips_hbnw,
            'alpha': 1.0,
            'beta': 1.0
        }
    )
    od_matrices['HBNW'] = od_hbnw
    avg_hbnw_distance = model.calculate_average_distance(od_hbnw, distance_matrix, "HBNW")
    model.plot_od_heatmap(gdf, od_hbnw, title="HBNW Trip Origins", output_file="hbnw_trip_origins.png")
    
    # NHB: Non-Home-Based (Employment -> Amenity)
    od_nhb = model.calculate_trip_purpose(
        productions=employment,
        attractions=amenity_nhb,
        distance_matrix=distance_matrix,
        purpose_name="NHB",
        purpose_params={
            'gamma': 3.5,        # Much prefer shorter destinations
            'total_trips': total_trips_nhb,
            'alpha': 1.0,
            'beta': 1.0
        }
    )
    od_matrices['NHB'] = od_nhb
    avg_nhb_distance = model.calculate_average_distance(od_nhb, distance_matrix, "NHB")
    model.plot_od_heatmap(gdf, od_nhb, title="NHB Trip Origins", output_file="nhb_trip_origins.png")

    print(f"\n{'='*60}")
    print("Applying Time-of-Day Weights")
    print(f"{'='*60}")

    # Time-of-day weights (AM peak)
    time_of_day_factors = {
        'HBW': 0.25,    # 25% of HBW trips in AM peak
        'HBNW': 0.15,   # 15% of HBNW trips in AM peak
        'NHB': 0.10,    # 10% of NHB trips in AM peak
    }
        
    # Apply mode choice
    mode_od_matrices = {}

    for purpose_name, od_matrix in od_matrices.items():
        print(f"Time-of-day weighting for {purpose_name}")
        
        # Get time-of-day factor
        tod_factor = time_of_day_factors.get(purpose_name, 1.0)
        print(f"  Time-of-day factor (AM peak): {tod_factor:.2%}")
        
        # Apply time-of-day weighting
        od_matrix_tod = od_matrix * tod_factor
        print(f"  Before weighting: {od_matrix.sum():.0f} trips")
        print(f"  After weighting: {od_matrix_tod.sum():.0f} trips")
        print(f"  Reduction: {1 - tod_factor:.1%}")
        
        print(f"\n{'='*60}")
        print(f"Mode choice for {purpose_name}")
        print(f"{'='*60}")

        purpose_modes = model.apply_mode_choice_chunked(
            od_matrix=od_matrix_tod,  # Use time-weighted matrix
            distance_matrix=distance_matrix,
            purpose=purpose_name
        )

        for mode, mode_matrix in purpose_modes.items():
            key = f"{purpose_name}_{mode}"
            mode_od_matrices[key] = mode_matrix

    print(f"\n{'='*60}")
    print("Combining modes across all purposes")
    print(f"{'='*60}")

    # combined_car = np.zeros_like(od_hbw)
    # combined_motorbike = np.zeros_like(od_hbw)
    # combined_walk = np.zeros_like(od_hbw)

    modes = ['car', 'motorbike']
    combined_modes = {mode: np.zeros_like(od_hbw) for mode in modes}

    for purpose in ['HBW', 'HBNW', 'NHB']:
        for mode in modes:
            key = f"{purpose}_{mode}"
            if key in mode_od_matrices:
                combined_modes[mode] += mode_od_matrices[key]

    print("\nCombined mode statistics:")
    total_all_modes = sum(m.sum() for m in combined_modes.values())
    for mode, matrix in combined_modes.items():
        mode_trips = matrix.sum()
        print(f"  {mode}: {mode_trips:.0f} trips ({mode_trips/total_all_modes*100:.1f}%)")       
        model.plot_od_heatmap(gdf, matrix, 
                    title=f"{mode.upper()} Trip Origins", 
                    output_file=f"{mode}_trip_origins.png")     

    
    # Save combined OD matrix
    print(f"\n{'='*60}")
    print("Saving results...")
    print(f"{'='*60}")

    for mode, matrix in combined_modes.items():
        filename = f'data/raw/rea_1000m_{mode}_vectors_v2.json'
        print(f"\nSaving {mode} OD matrix...")
        model.save_sparse_vectors(
            od_matrix=matrix,
            grid_ids=grid_ids,
            filename=filename
        )

        avg_dist = model.calculate_average_distance(matrix, distance_matrix, f"{mode} trips")
        print(f"  Average {mode} trip distance: {avg_dist/1000:.2f} km")

    total_avg_distance = (
        avg_hbw_distance  + 
        avg_hbnw_distance  + 
        avg_nhb_distance 
    )
        
    print(f"\n{'='*60}")
    print("PROCESS COMPLETED!")
    print(f"{'='*60}")
    # print(f"Total combined trips: {combined_od.sum():.0f}")
    # print(f"Number of origins: {len(combined_od)}")
    # print(f"Number of OD pairs: {(combined_od > 0).sum():,}")

    print("DISTANCES ANALYTICS")
    print(f"Weighted average trip distance: {total_avg_distance/1000:.2f} km")
    print(f"HBW average: {avg_hbw_distance/1000:.2f} km")
    print(f"HBNW average: {avg_hbnw_distance/1000:.2f} km")
    print(f"NHB average: {avg_nhb_distance/1000:.2f} km")

if __name__ == "__main__":
    main()