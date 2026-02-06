import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import time
from typing import Tuple, List, Dict, Optional
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tools.to_parquet import JsonToParquet
import gc
import warnings
warnings.filterwarnings('ignore')

# Note: also check out
# Ansusanto, J. Dwijoko. “Karakteristik Pola Perjalanan Di Perkotaan (Studi Kasus Kota Yogyakarta).” Simposium FSTPT XVI, UMS.
# Risdiyanto, Risdiyanto & Munawar, Ahmad & Irawan, Muhammad & Nugraha, A. (2020). Model selection of online motorcycle taxi and motorcycle modes on work trips. IOP Conference Series: Materials Science and Engineering. 1007. 012059. 10.1088/1757-899X/1007/1/012059. 
# Dharmowijoyo, Dimas & Susilo, Yusak & Karlström, Anders. (2015). Day-to-day variability in travellers’ activity-travel patterns in the Jakarta metropolitan area. Transportation. 43. 10.1007/s11116-015-9591-4. 
# Ansusanto, Dwijoko. (2016). POLA PERJALANAN DI PERKOTAAN YOGYAKARTA. Jurnal Teknik Sipil. 12. 10.24002/jts.v12i4.633. 

# Mode Shares Per Distance, uncalibrated 
#  - Short (<3km)
SHORT_DISTANCE_THRES = 3 
SHORT_MOTORBIKE_SHARES = 0.90
SHORT_CAR_SHARES = 0.10
#  - Medium (3km - 10km)
MEDIUM_DISTANCE_THERS = 10
MEDIUM_MOTORBIKE_SHARES = 0.60
MEDIUM_CAR_SHARES = 0.40
#  - Long (10km - 15km)
LONG_DISTANCE_THRES = 15
LONG_MOTORBIKE_SHARES = 0.50
LONG_CAR_SHARES = 0.50
#  - Very long (>15km)
VERY_LONG_MOTORBIKE_SHARES = 0.40
VERY_LONG_CAR_SHARES = 0.60

# Mode Shares per Trip Purposes
# BPS "Provinsi Daerah Istimewa Yogyakarta Dalam Angka 2025"
# Total motor di DIY = 2,929,766 (87%); Mobil = 449,913 (13)%

#  - Work trips
HBW_MOTORBIKE_SHARES = 1.2
HBW_CAR_SHARES = 0.8
# - More motorbike for non-work trips
HBNW_MOTORBIKE_SHARES = 1.4
HBNW_CAR_SHARES = 0.6
# - Alot more motorbike for non-home-based
NHB_MOTORBIKE_SHARES = 1.8
NHB_CAR_SHARES = 0.2

# Total Trip Ratio Per Purposes, (Devi et al., 2019)
# 1156 trips as baseline 
HBW_TOTAL_TRIPS = 623.8       #7110.556 # 1156 * (46.11 + 15.40)/100 = 711.0556
HBNW_TOTAL_TRIPS = 277.7     #3029.876 # 1156 * (26.21)/100 = 302.9876
NHB_TOTAL_TRIPS = 98.6      #1320.152 # 1156 * (6.06 + 1.21 + 1.47 + 1.12 + 1.56)/100 = 132.0152

# Gravity Model Params 
# gamma is the distance decay factor
GAMMA_HBW = 0.9   # 10 - 13km avg, fine with longer trip
ALPHA_HBW = 1.0
BETA_HBW = 1.0

GAMMA_HBNW = 3.5  # 6 - 9km avg, prefer closer spots
ALPHA_HBNW = 1.0
BETA_HBNW = 1.0

GAMMA_NHB = 3.5   # 1 - 4km avg, much prefer shorter destinations
ALPHA_NHB = 1.0
BETA_NHB = 1.0

# Time of Day factors (Right now its AM peak)
TOD_HBW = 1  
TOD_HBNW = 0.15
TOD_NHB = 0.05

# Scaling factor (To populate v/c if needed)
OD_SCALING_FACTOR = 1

# IPF
IPF_ITERATIONS = 15
IPF_TOLERANCE = 1e-6

# Boundary leakage (to simulate traffic going out of DIY)
BOUNDARY_BUFFER = 3.0 # km
LEAKAGE_HBW = 0.30
LEAKAGE_HBNW = 0.10
LEAKAGE_NHB = 0.05

# Data files
DIY_BOUNDARY = 'data/raw/Yogyakarta.geojson'
GRID_DATA = 'data/raw/rea_1000m_v2.geojson'

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
                     attractions: np.ndarray) -> np.ndarray:
        """
        Iterative Proportional Fitting (Furness method) to balance OD matrix
        
        Ensures: sum_j T_ij = P_i and sum_i T_ij = A_j
        """
        print(f"  IPF balancing (max {IPF_ITERATIONS} iterations)...")
        
        n = len(productions)
        T = od_matrix.copy()
        
        # Initialize scaling factors
        row_factors = np.ones(n)
        col_factors = np.ones(n)
        
        for iteration in range(IPF_ITERATIONS):
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
            
            if total_error < IPF_TOLERANCE:
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
        Calculate weighted average and median trip distance for a given OD matrix
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
        
        # Calculate weighted median using numpy's percentile with weights
        normalized_trips = trips / total_trips
        
        # Sort both arrays by distance
        sorted_indices = np.argsort(distances)
        sorted_distances = distances[sorted_indices]
        sorted_weights = normalized_trips[sorted_indices]
        
        # Calculate cumulative weights
        cumulative_weights = np.cumsum(sorted_weights)
        
        # Find the median using linear interpolation
        # Find index where cumulative weight crosses 0.5
        idx = np.searchsorted(cumulative_weights, 0.5)
        
        if idx == 0:
            median_distance = sorted_distances[0]
        elif idx >= len(sorted_distances):
            median_distance = sorted_distances[-1]
        else:
            # Linear interpolation between adjacent values
            weight_before = cumulative_weights[idx - 1]
            weight_at = cumulative_weights[idx]
            
            # Interpolate between the two distances
            if weight_at - weight_before > 1e-10:  # Avoid division by zero
                fraction = (0.5 - weight_before) / (weight_at - weight_before)
                median_distance = sorted_distances[idx - 1] + fraction * (
                    sorted_distances[idx] - sorted_distances[idx - 1])
            else:
                median_distance = sorted_distances[idx]
        
        print(f"\n  {purpose_name} trip distances:")
        print(f"    Average: {average_distance:.2f} meters")
        print(f"    Median:  {median_distance:.2f} meters")
        
        return average_distance
    
    def apply_mode_choice_chunked(self, od_matrix: np.ndarray, 
                                distance_matrix: np.ndarray,
                                purpose: str,
                                chunk_size: int = 500) -> Dict[str, np.ndarray]:
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
            gc.collect()
            
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
        mask_short = d_km < SHORT_DISTANCE_THRES
        mode_shares['motorbike'][mask_short] = SHORT_MOTORBIKE_SHARES
        mode_shares['car'][mask_short] = SHORT_CAR_SHARES
        
        mask_medium = (d_km >= SHORT_DISTANCE_THRES) & (d_km < MEDIUM_DISTANCE_THERS)
        mode_shares['motorbike'][mask_medium] = MEDIUM_MOTORBIKE_SHARES
        mode_shares['car'][mask_medium] = MEDIUM_CAR_SHARES
        
        mask_long = (d_km >= MEDIUM_DISTANCE_THERS) & (d_km < LONG_DISTANCE_THRES)
        mode_shares['motorbike'][mask_long] = LONG_MOTORBIKE_SHARES
        mode_shares['car'][mask_long] = LONG_CAR_SHARES
        
        mask_very_long = d_km >= LONG_DISTANCE_THRES
        mode_shares['motorbike'][mask_very_long] = VERY_LONG_MOTORBIKE_SHARES
        mode_shares['car'][mask_very_long] = VERY_LONG_CAR_SHARES
        
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
            mode_shares['motorbike'] *= HBW_MOTORBIKE_SHARES
            mode_shares['car'] *= HBW_CAR_SHARES
        elif purpose == 'HBNW':
            # More motorbike for non-work trips
            mode_shares['motorbike'] *= HBNW_MOTORBIKE_SHARES
            mode_shares['car'] *= HBNW_CAR_SHARES
        elif purpose == 'NHB':
            # More walk + motorbike for non-home-based
            mode_shares['motorbike'] *= NHB_MOTORBIKE_SHARES
            mode_shares['car'] *= NHB_CAR_SHARES
            
    
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

def scale_matrix_in_chunks(matrix, scaling_factor, chunk_size=1000):
    """Scale a matrix in chunks to reduce memory usage"""
    n = matrix.shape[0]
    for i in range(0, n, chunk_size):
        end_i = min(i + chunk_size, n)
        matrix[i:end_i, :] = matrix[i:end_i, :] * scaling_factor
    return matrix

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

# Add this function after the ImprovedGravityModel class but before main()
def apply_boundary_leakage(gdf, od_matrices, distance_matrix, boundary_buffer_km=2.0, 
                          leakage_factors=None):
    """
    Apply boundary leakage to OD matrices
    
    Parameters:
    -----------
    gdf : GeoDataFrame
        Grid data with geometry
    od_matrices : dict
        Dictionary of OD matrices by purpose
    distance_matrix : np.ndarray
        Distance matrix in meters
    boundary_buffer_km : float
        Buffer distance from boundary in kilometers
    leakage_factors : dict
        Dictionary of leakage factors by purpose
        
    Returns:
    --------
    dict : Modified OD matrices with leakage applied
    float : Total leaked trips
    """

    try:
        diy_boundary = gpd.read_file(DIY_BOUNDARY)
        diy_boundary = diy_boundary.to_crs(gdf.crs)
        
        # Create buffer around boundary
        buffer_distance = boundary_buffer_km * 1000  # Convert to meters
        boundary_buffer = diy_boundary.buffer(-buffer_distance)  # Negative buffer = inside
        
        # Identify zones near boundary
        gdf_copy = gdf.copy()
        near_boundary = gdf_copy.geometry.intersects(boundary_buffer.unary_union)
        boundary_zone_indices = np.where(near_boundary)[0]
        
        print(f"\n{'='*60}")
        print("BOUNDARY LEAKAGE ANALYSIS")
        print(f"{'='*60}")
        print(f"Boundary buffer: {boundary_buffer_km} km")
        print(f"Zones near boundary: {len(boundary_zone_indices)}/{len(gdf)}")
        
        total_leaked_trips = 0
        leaked_by_purpose = {}
        
        # Apply leakage to each purpose
        for purpose, od_matrix in od_matrices.items():
            if purpose not in leakage_factors:
                continue
                
            leakage_factor = leakage_factors[purpose]
            purpose_trips_before = od_matrix.sum()
            
            # Create copy to modify
            od_matrix_leaked = od_matrix.copy()
            
            # Reduce trips from boundary zones
            for zone_idx in boundary_zone_indices:
                # Reduce all outgoing trips from this zone
                row_sum = od_matrix_leaked[zone_idx, :].sum()
                if row_sum > 0:
                    reduction = row_sum * leakage_factor
                    # Scale down all destinations proportionally
                    scale_factor = (row_sum - reduction) / row_sum
                    od_matrix_leaked[zone_idx, :] *= scale_factor
            
            # Calculate leaked trips
            purpose_trips_after = od_matrix_leaked.sum()
            leaked_trips = purpose_trips_before - purpose_trips_after
            leaked_percentage = (leaked_trips / purpose_trips_before * 100) if purpose_trips_before > 0 else 0
            
            leaked_by_purpose[purpose] = {
                'leaked_trips': leaked_trips,
                'percentage': leaked_percentage,
                'before': purpose_trips_before,
                'after': purpose_trips_after
            }
            
            total_leaked_trips += leaked_trips
            od_matrices[purpose] = od_matrix_leaked
            
            print(f"\n{purpose}:")
            print(f"  Leakage factor: {leakage_factor:.1%}")
            print(f"  Trips before leakage: {purpose_trips_before:.0f}")
            print(f"  Trips after leakage: {purpose_trips_after:.0f}")
            print(f"  Leaked trips: {leaked_trips:.0f} ({leaked_percentage:.1f}%)")
        
        print(f"\nTotal leaked trips: {total_leaked_trips:.0f}")
        print(f"Leakage as % of total: {total_leaked_trips/sum(m.sum() for m in od_matrices.values())*100:.1f}%")
        
        return od_matrices, total_leaked_trips
        
    except FileNotFoundError:
        print("Warning: DIY boundary file not found")
        return od_matrices, 0

# Main execution
def main():
    print("Gravity model n IPF balancing")
    print("=" * 60)
    
    residential, employment,  amenity_hbnw, amenity_nhb, coordinates, grid_ids, gdf = load_data(GRID_DATA)
    
    # Initialize model
    model = ImprovedGravityModel(chunk_size=500)
    
    # Calculate distance matrix (once, reused for all purposes)
    print("\nCalculating distance matrix...")
    distance_matrix = model.calculate_distances(coordinates)
    
    # Calculate each trip purpose with IPF
    od_matrices = {}
    
    # HBW: Home-Based Work (Residential -> Employment)
    od_hbw = model.calculate_trip_purpose(
        productions=residential,
        attractions=employment,
        distance_matrix=distance_matrix,
        purpose_name="HBW",
        purpose_params={
            'gamma': GAMMA_HBW,        
            'total_trips': HBNW_TOTAL_TRIPS,
            'alpha': ALPHA_HBW,
            'beta': BETA_HBW
        }
    )
    od_matrices['HBW'] = od_hbw
    avg_hbw_distance = model.calculate_average_distance(od_hbw, distance_matrix, "HBW")
    model.plot_od_heatmap(gdf, od_hbw, "OD HBW", 'Tesst.png')
    gc.collect()
    
    # HBNW: Home-Based Non-Work (Residential -> Amenity)
    od_hbnw = model.calculate_trip_purpose(
        productions=residential,
        attractions=amenity_hbnw,
        distance_matrix=distance_matrix,
        purpose_name="HBNW",
        purpose_params={
            'gamma': GAMMA_HBNW,        # Prefer spots that's closer to home
            'total_trips': HBNW_TOTAL_TRIPS,
            'alpha': ALPHA_HBNW,
            'beta': BETA_HBNW
        }
    )
    od_matrices['HBNW'] = od_hbnw
    avg_hbnw_distance = model.calculate_average_distance(od_hbnw, distance_matrix, "HBNW")
    model.plot_od_heatmap(gdf, od_hbnw, "OD HBNW", 'Tesst2.png')
    del od_hbnw
    gc.collect()
    
    
    # NHB: Non-Home-Based (Employment -> Amenity)
    od_nhb = model.calculate_trip_purpose(
        productions=employment,
        attractions=amenity_nhb,
        distance_matrix=distance_matrix,
        purpose_name="NHB",
        purpose_params={
            'gamma': GAMMA_HBNW,        # Much prefer shorter destinations
            'total_trips': NHB_TOTAL_TRIPS,
            'alpha': ALPHA_NHB,
            'beta': BETA_NHB
        }
    )
    od_matrices['NHB'] = od_nhb
    avg_nhb_distance = model.calculate_average_distance(od_nhb, distance_matrix, "NHB")
    model.plot_od_heatmap(gdf, od_nhb, "OD NHB", 'Tesst3.png')
    del od_nhb
    gc.collect()

    # Apply boundary leakage (to simulate traffic going out of DIY)
    print("\nApplying Boundary Leakage...")
    
    od_matrices, total_leaked = apply_boundary_leakage(
        gdf=gdf,
        od_matrices=od_matrices,  
        distance_matrix=distance_matrix,
        boundary_buffer_km=BOUNDARY_BUFFER,  
        leakage_factors={
            'HBW': LEAKAGE_HBW,    
            'HBNW': LEAKAGE_HBNW,  
            'NHB': LEAKAGE_NHB     
        }
    )

    if 'od_hbnw' in locals(): del od_hbnw
    if 'od_nhb' in locals(): del od_nhb
    gc.collect()

    print("\nApplying Time-of-Day Weights...")

    # Time-of-day weights (AM peak)
    time_of_day_factors = {
        'HBW': TOD_HBW,    
        'HBNW': TOD_HBNW,  
        'NHB': TOD_NHB,    
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
            od_matrix=od_matrix_tod,  
            distance_matrix=distance_matrix,
            purpose=purpose_name
        )

        for mode, mode_matrix in purpose_modes.items():
            key = f"{purpose_name}_{mode}"
            mode_od_matrices[key] = mode_matrix
            model.plot_od_heatmap(gdf, mode_matrix, f"{purpose_name} {mode}", f'Tesst{purpose_name}_{mode}.png')

            del mode_matrix
            gc.collect()

        del od_matrix
        gc.collect()    

    print(f"\n{'='*60}")
    print("Combining modes across all purposes")
    print(f"{'='*60}")

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
    
    for purpose in list(od_matrices.keys()):
        del od_matrices[purpose]
    del od_matrices
    gc.collect()
        
    # Scale OD to bump up the v/c ratios later
    print("Applying scaling factor...")
    
    for mode in combined_modes:
        before = combined_modes[mode].sum()
        combined_modes[mode] = scale_matrix_in_chunks(combined_modes[mode], OD_SCALING_FACTOR)
        after = combined_modes[mode].sum()
        print(f"  {mode}: {before:.0f} → {after:.0f} trips (×{OD_SCALING_FACTOR})")    

    
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

        model.calculate_average_distance(matrix, distance_matrix, f"{mode} trips")

    total_avg_distance = (
        avg_hbw_distance  + 
        avg_hbnw_distance  + 
        avg_nhb_distance 
    )
        
    print(f"\n{'='*60}")
    print("PROCESS COMPLETED!")
    print(f"{'='*60}")

    print("DISTANCES ANALYTICS")
    print(f"Weighted average trip distance: {(total_avg_distance/3)/1000:.2f} km")
    print(f"HBW average: {avg_hbw_distance/1000:.2f} km")
    print(f"HBNW average: {avg_hbnw_distance/1000:.2f} km")
    print(f"NHB average: {avg_nhb_distance/1000:.2f} km")

    # Save to parquet
    for mode in modes:
        toparquet = JsonToParquet()
        toparquet.convert(mode)

if __name__ == "__main__":
    main()