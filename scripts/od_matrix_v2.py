import numpy as np
from scipy.spatial.distance import cdist
import time
from typing import Tuple, List, Dict, Optional, Any
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import gc
import os
import pyarrow as pa
import pyarrow.parquet as pq        
import yaml
from dataclasses import dataclass, fields

import warnings
warnings.filterwarnings('ignore')

# Configuration
@dataclass
class Config:
    distance_thresh: Dict[str, int]
    distance_shares: Dict[str, Dict[str, float]]
    purpose_shares: Dict[str, Dict[str, float]]
    total_trips: Dict[str, float]
    gravity: Dict[str, Dict[str, float]]
    time_weight: Dict[str, float]
    od_scale: float
    ipf: Dict[str, float]
    boundary_buffer: float
    leakage: Dict[str, float]
    data_paths: Dict[str, str]
    export_paths: Dict[str, str]

    @classmethod
    def from_yaml(cls, config_path: str = "config.yaml"):
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        class_fields = {f.name for f in fields(cls)}
        filtered_dict = {
            k: v for k, v in config_dict.items() 
            if k in class_fields
        }
        return cls(**filtered_dict)


class DataHandler:
    """Handles data loading and saving operations"""

    def __init__(self, config: Config):
        self.config = config
        self.residential: Optional[np.ndarray] = None
        self.employment: Optional[np.ndarray] = None
        self.amenity_hbnw: Optional[np.ndarray] = None
        self.amenity_nhb: Optional[np.ndarray] = None
        self.grid_ids: Optional[np.ndarray] = None
        self.gdf: Optional[gpd.GeoDataFrame] = None
        self.coordinates: Optional[np.ndarray] = None

    def load_data(self, grid_data_path: str) -> None:
        """Load and preprocess grid data"""
        print(f"\nLoading data from {grid_data_path}...")
        gdf = gpd.read_file(grid_data_path)
        self.gdf = gdf
        
        if gdf.crs.to_epsg() == 4326:
            gdf = gdf.to_crs(epsg=3857)
        
        # Extract coordinates
        self.coordinates = np.column_stack([gdf.geometry.x.values, gdf.geometry.y.values])
        
        # Find intensity columns
        residential_cols = [col for col in gdf.columns if 'residential_intensity' in col.lower()]
        employment_cols = [col for col in gdf.columns if 'employment_intensity' in col.lower()]
        amenity_hbnw_cols = [col for col in gdf.columns if 'amenity_hbnw_intensity' in col.lower()]
        amenity_nhb_cols = [col for col in gdf.columns if 'amenity_nhb_intensity' in col.lower()]
        id_cols = [col for col in gdf.columns if 'id' in col.lower()]
        
        self.residential = gdf[residential_cols[0]].values if residential_cols else np.ones(len(gdf))
        self.employment = gdf[employment_cols[0]].values if employment_cols else np.ones(len(gdf))
        self.amenity_hbnw = gdf[amenity_hbnw_cols[0]].values if amenity_hbnw_cols else np.ones(len(gdf))
        self.amenity_nhb = gdf[amenity_nhb_cols[0]].values if amenity_nhb_cols else np.ones(len(gdf))
        self.grid_ids = gdf[id_cols[0]].values if id_cols else np.arange(len(gdf))
        
        print(f"  Loaded {len(gdf)} grid cells")
        print(f"  Residential sum: {self.residential.sum():.1f}")
        print(f"  Employment sum: {self.employment.sum():.1f}")
        print(f"  Amenity (HBNW) sum: {self.amenity_hbnw.sum():.1f}")
        print(f"  Amenity (NHB) sum: {self.amenity_nhb.sum():.1f}")

        self._validate_data()

    def _validate_data(self) -> None:
        """Perform data validity checks"""
        print("\n=== DATA VALIDITY CHECKS ===")
        print(f"Number of grid cells: {len(self.gdf)}")
        print(f"Residential zeros: {(self.residential == 0).sum()} / {len(self.residential)}")
        print(f"Employment zeros: {(self.employment == 0).sum()} / {len(self.employment)}")
        print(f"Amenity HBNW zeros: {(self.amenity_hbnw == 0).sum()} / {len(self.amenity_hbnw)}")
        print(f"Amenity NHB zeros: {(self.amenity_nhb == 0).sum()} / {len(self.amenity_nhb)}")

        if self.coordinates is not None:
            print(f"\nCoordinate ranges:")
            print(f"X min/max: {self.coordinates[:, 0].min():.0f} / {self.coordinates[:, 0].max():.0f}")
            print(f"Y min/max: {self.coordinates[:, 1].min():.0f} / {self.coordinates[:, 1].max():.0f}")

    def save_sparse_vectors(self, od_matrix: np.ndarray,
                           grid_ids: np.ndarray,
                           filename: str,
                           threshold: float = 1e-6) -> int:
        """Save OD matrix as sparse vectors to Parquet file"""
        start_time = time.time()
        
        n = len(od_matrix)
        
        # Collect data in lists for efficiency
        origin_ids_list = []
        dest_ids_list = []
        trips_list = []
        
        # Collect sparse matrix data
        for i in range(n):
            # Get non-zero destinations for this origin
            row = od_matrix[i]
            non_zero_mask = row > threshold
            
            if non_zero_mask.any():
                dest_indices = np.where(non_zero_mask)[0]
                values = row[non_zero_mask]
                
                for dest_idx, value in zip(dest_indices, values):
                    origin_ids_list.append(int(grid_ids[i]))
                    dest_ids_list.append(int(grid_ids[dest_idx]))
                    trips_list.append(float(value))
        
        if origin_ids_list:
            # Create PyArrow arrays
            origin_arr = pa.array(origin_ids_list, type=pa.int64())
            dest_arr = pa.array(dest_ids_list, type=pa.int64())
            trips_arr = pa.array(trips_list, type=pa.float64())
            
            # Create table with explicit schema
            table = pa.table({
                'origin_id': origin_arr,
                'destination_id': dest_arr,
                'trips': trips_arr
            })
            
            # Save to Parquet with compression
            pq.write_table(table, filename, compression='snappy')
            
            elapsed_time = time.time() - start_time
            print(f"  Saved {len(origin_ids_list)} OD pairs in {elapsed_time:.2f} seconds")
            print(f"  Sparsity: {(od_matrix > threshold).sum() / od_matrix.size:.1%}")
            
            # Check file size if file exists
            if os.path.exists(filename):
                print(f"  File size: {os.path.getsize(filename) / 1024 / 1024:.2f} MB")
            return len(origin_ids_list)
        else:
            print("  No data to save (all values below threshold)")
            return 0


class SpatialEngine:
    """Handles spatial operations like distance calculations"""
    
    def __init__(self, config: Config):
        self.config = config
        self.distance_matrix: Optional[np.ndarray] = None

    def calculate_distance_matrix(self, coordinates: np.ndarray) -> np.ndarray:
        """Calculate Euclidean distance matrix"""
        start_time = time.time()
        
        self.distance_matrix = cdist(coordinates, coordinates, metric='euclidean')
        np.fill_diagonal(self.distance_matrix, 1.0)  # Avoid division by zero
        
        elapsed_time = time.time() - start_time
        print(f"Distance matrix calculated in {elapsed_time:.2f} seconds")
        
        return self.distance_matrix

    @staticmethod
    def calculate_average_distance(od_matrix: np.ndarray, 
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


class DemandModel:
    """Contains core gravity model and IPF balancing methods"""
    
    def __init__(self, config: Config, chunk_size: int = 500):
        self.config = config
        self.chunk_size = chunk_size
    
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
                print(f"    Gravity calculation: {progress:.1f}%")
        
        return od_matrix
    
    def ipf_balancing(self,
                     od_matrix: np.ndarray,
                     productions: np.ndarray,
                     attractions: np.ndarray) -> np.ndarray:
        """
        Iterative Proportional Fitting (Furness method) to balance OD matrix
        """
        print(f"  IPF balancing (max {self.config.ipf['iterations']} iterations)...")
        
        n = len(productions)
        T = od_matrix.copy()
        
        for iteration in range(self.config.ipf['iterations']):
            # Row balancing (match productions)
            row_sums = T.sum(axis=1)
            row_sums[row_sums == 0] = 1
            row_factors = productions / row_sums
            T = T * row_factors[:, np.newaxis]
            
            # Column balancing (match attractions)
            col_sums = T.sum(axis=0)
            col_sums[col_sums == 0] = 1
            col_factors = attractions / col_sums
            T = T * col_factors[np.newaxis, :]
            
            # Calculate convergence
            row_error = np.abs(T.sum(axis=1) - productions).sum() / productions.sum()
            col_error = np.abs(T.sum(axis=0) - attractions).sum() / attractions.sum()
            total_error = max(row_error, col_error)
            
            if iteration % 5 == 0:
                print(f"    Iteration {iteration}: error = {total_error:.6f}")

            if total_error < self.config.ipf['tolerance']:
                print(f"  IPF converged after {iteration + 1} iterations")
                break
        
        # Final scaling to ensure exact match
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


class TransportAnalyst:
    """Handles transport analysis including mode choice"""
    
    def __init__(self, config: Config, chunk_size: int = 500):
        self.config = config
        self.chunk_size = chunk_size
    
    def calculate_trip_purpose(self,
                              demand_model: DemandModel,
                              productions: np.ndarray,
                              attractions: np.ndarray,
                              distance_matrix: np.ndarray,
                              purpose_name: str,
                              purpose_params: Dict) -> np.ndarray:
        """
        Calculate OD matrix for a single trip purpose with IPF balancing
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
        
        if total_trips is not None:
            scaled_productions = productions * total_trips
            scaled_attractions = attractions * total_trips
            print(f"  Total trips target: {total_trips}")
        else:
            scaled_productions = productions.copy()
            scaled_attractions = attractions.copy()
            print(f"  Total trips target: Not specified")

        # Step 1: Calculate unnormalized gravity matrix
        print("  Step 1: Calculating gravity model...")
        start_time = time.time()
        
        T0 = demand_model.gravity_model_chunked(
            productions=scaled_productions,
            attractions=scaled_attractions,
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
        
        T_balanced = demand_model.ipf_balancing(
            od_matrix=T0,
            productions=scaled_productions,
            attractions=scaled_attractions
        )

        ipf_time = time.time() - start_time
        print(f"    IPF completed in {ipf_time:.2f} seconds")
        
        if total_trips is not None:
            actual_total = T_balanced.sum()
            difference = abs(actual_total - total_trips)
            relative_error = difference / total_trips if total_trips > 0 else 0
            
            print(f"  Trip total verification:")
            print(f"    Target total: {total_trips:.1f}")
            print(f"    Actual total: {actual_total:.1f}")
            print(f"    Difference: {difference:.1f} ({relative_error:.2%})")

        # Calculate statistics
        self._calculate_statistics(T_balanced, scaled_productions, scaled_attractions, purpose_name)
        
        return T_balanced
    
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

    def apply_mode_choice_chunked(self, od_matrix: np.ndarray, 
                                distance_matrix: np.ndarray,
                                purpose: str) -> Dict[str, np.ndarray]:
        """
        Apply rule-based mode choice splitting with chunked processing
        """
        print(f"  Applying chunked mode choice for {purpose} trips...")
        
        n = len(od_matrix)
        
        # Initialize output matrices with zeros
        mode_od = {
            'motorbike': np.zeros_like(od_matrix),
            'car': np.zeros_like(od_matrix)
        }
        
        # Process in chunks to reduce memory usage
        for start_idx in range(0, n, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, n)
            
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
            if (start_idx // self.chunk_size) % 10 == 0:
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
        """
        d_km = distance_matrix / 1000  # Convert meters to kilometers
        
        # Initialize mode shares
        mode_shares = {
            'motorbike': np.zeros_like(distance_matrix),
            'car': np.zeros_like(distance_matrix)
        }
        
        # Calculate base shares by distance
        mask_short = d_km < self.config.distance_thresh['short']
        mode_shares['motorbike'][mask_short] = self.config.distance_shares['short']['motorbike']
        mode_shares['car'][mask_short] = self.config.distance_shares['short']['car']
        
        mask_medium = (d_km >= self.config.distance_thresh['short']) & (d_km < self.config.distance_thresh['medium'])
        mode_shares['motorbike'][mask_medium] = self.config.distance_shares['medium']['motorbike']
        mode_shares['car'][mask_medium] = self.config.distance_shares['medium']['car']
        
        mask_long = (d_km >= self.config.distance_thresh['medium']) & (d_km < self.config.distance_thresh['long'])
        mode_shares['motorbike'][mask_long] = self.config.distance_shares['long']['motorbike']
        mode_shares['car'][mask_long] = self.config.distance_shares['long']['car']
        
        mask_very_long = d_km >= self.config.distance_thresh['long']
        mode_shares['motorbike'][mask_very_long] = self.config.distance_shares['vlong']['motorbike']
        mode_shares['car'][mask_very_long] = self.config.distance_shares['vlong']['car']
        
        # Apply purpose-based adjustments
        self._apply_purpose_adjustments(mode_shares, purpose)
        
        # Normalize each cell to sum to 1
        total_shares = mode_shares['motorbike'] + mode_shares['car']
        
        # Avoid division by zero
        valid_mask = total_shares > 0
        for mode in mode_shares:
            mode_shares[mode][valid_mask] = mode_shares[mode][valid_mask] / total_shares[valid_mask]
            mode_shares[mode][~valid_mask] = 0
        
        return mode_shares
    
    def _apply_purpose_adjustments(self, mode_shares: Dict[str, np.ndarray], purpose: str):
        """Apply purpose-specific adjustments to mode shares"""
        if purpose == 'HBW':
            mode_shares['motorbike'] *= self.config.purpose_shares['hbw']['motorbike']
            mode_shares['car'] *= self.config.purpose_shares['hbw']['car']
        elif purpose == 'HBNW':
            mode_shares['motorbike'] *= self.config.purpose_shares['hbnw']['motorbike']
            mode_shares['car'] *= self.config.purpose_shares['hbnw']['car']
        elif purpose == 'NHB':
            mode_shares['motorbike'] *= self.config.purpose_shares['nhb']['motorbike']
            mode_shares['car'] *= self.config.purpose_shares['nhb']['car']


class BoundaryLeakage:
    """Handles boundary leakage calculations"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def apply_boundary_leakage(self, gdf: gpd.GeoDataFrame, 
                             od_matrices: Dict[str, np.ndarray],
                             distance_matrix: np.ndarray,
                             boundary_buffer_km: float = None,
                             leakage_factors: Dict[str, float] = None) -> Tuple[Dict[str, np.ndarray], float]:
        """
        Apply boundary leakage to OD matrices
        """
        try:
            if boundary_buffer_km is None:
                boundary_buffer_km = self.config.boundary_buffer
            if leakage_factors is None:
                leakage_factors = self.config.leakage
            
            diy_boundary = gpd.read_file(self.config.data_paths['boundary'])
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
            total_all_trips = sum(m.sum() for m in od_matrices.values())
            print(f"Leakage as % of total: {total_leaked_trips/total_all_trips*100:.1f}%")
            
            return od_matrices, total_leaked_trips
            
        except FileNotFoundError:
            print("Warning: DIY boundary file not found")
            return od_matrices, 0


class Visualization:
    """Handles visualization of results"""
    
    @staticmethod
    def plot_od_heatmap(gdf: gpd.GeoDataFrame, 
                       od_matrix: np.ndarray, 
                       title: str, 
                       output_file: str) -> gpd.GeoDataFrame:
        """
        Plot heatmap of trips starting from each cell
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


class ImprovedGravityModel:
    """
    Main class that orchestrates the improved gravity model workflow
    """
    
    def __init__(self, chunk_size: int = 500, config = Config):
        self.chunk_size = chunk_size
        self.config = Config
        
        # Initialize components
        self.data_handler = DataHandler(self.config)
        self.spatial_engine = SpatialEngine(self.config)
        self.demand_model = DemandModel(self.config, chunk_size)
        self.transport_analyst = TransportAnalyst(self.config, chunk_size)
        self.boundary_leakage = BoundaryLeakage(self.config)
        self.visualization = Visualization()
        
        # Results storage
        self.od_matrices: Dict[str, np.ndarray] = {}
        self.average_distances: Dict[str, float] = {}
        self.mode_od_matrices: Dict[str, np.ndarray] = {}
        
    def run(self):
        """Execute the complete gravity model workflow"""
        print("Gravity model with IPF balancing")
        print("=" * 60)
        
        # Load data
        self.data_handler.load_data(self.config.data_paths['grid'])
        
        # Calculate distance matrix
        print("\nCalculating distance matrix...")
        distance_matrix = self.spatial_engine.calculate_distance_matrix(
            self.data_handler.coordinates
        )
        
        # Calculate each trip purpose
        self._calculate_all_trip_purposes(distance_matrix)
        
        # Apply boundary leakage
        print("\nApplying Boundary Leakage...")
        self.od_matrices, total_leaked = self.boundary_leakage.apply_boundary_leakage(
            gdf=self.data_handler.gdf,
            od_matrices=self.od_matrices,
            distance_matrix=distance_matrix
        )
        
        # Apply time-of-day weights and mode choice
        self._apply_time_of_day_and_mode_choice(distance_matrix)
        
        # Combine modes and scale
        combined_modes = self._combine_and_scale_modes()
        
        # Save results
        self._save_results(combined_modes, distance_matrix)
        
        # Print summary
        self._print_summary()
    
    def _calculate_all_trip_purposes(self, distance_matrix: np.ndarray):
        """Calculate all trip purposes"""
        # HBW: Home-Based Work (Residential -> Employment)
        od_hbw = self.transport_analyst.calculate_trip_purpose(
            demand_model=self.demand_model,
            productions=self.data_handler.residential,
            attractions=self.data_handler.employment,
            distance_matrix=distance_matrix,
            purpose_name="HBW",
            purpose_params={
                'gamma': self.config.gravity['hbw']['gamma'],
                'total_trips': self.config.total_trips['hbw'],
                'alpha': self.config.gravity['hbw']['alpha'],
                'beta': self.config.gravity['hbw']['beta']
            }
        )
        self.od_matrices['HBW'] = od_hbw
        self.average_distances['HBW'] = SpatialEngine.calculate_average_distance(
            od_hbw, distance_matrix, "HBW"
        )
        gc.collect()
        
        # HBNW: Home-Based Non-Work (Residential -> Amenity)
        od_hbnw = self.transport_analyst.calculate_trip_purpose(
            demand_model=self.demand_model,
            productions=self.data_handler.residential,
            attractions=self.data_handler.amenity_hbnw,
            distance_matrix=distance_matrix,
            purpose_name="HBNW",
            purpose_params={
                'gamma': self.config.gravity['hbnw']['gamma'],
                'total_trips': self.config.total_trips['hbnw'],
                'alpha': self.config.gravity['hbnw']['alpha'],
                'beta': self.config.gravity['hbnw']['beta']
            }
        )
        self.od_matrices['HBNW'] = od_hbnw
        self.average_distances['HBNW'] = SpatialEngine.calculate_average_distance(
            od_hbnw, distance_matrix, "HBNW"
        )
        gc.collect()
        
        # NHB: Non-Home-Based (Employment -> Amenity)
        od_nhb = self.transport_analyst.calculate_trip_purpose(
            demand_model=self.demand_model,
            productions=self.data_handler.employment,
            attractions=self.data_handler.amenity_nhb,
            distance_matrix=distance_matrix,
            purpose_name="NHB",
            purpose_params={
                'gamma': self.config.gravity['nhb']['gamma'],
                'total_trips': self.config.total_trips['nhb'],
                'alpha': self.config.gravity['nhb']['alpha'],
                'beta': self.config.gravity['nhb']['beta']
            }
        )
        self.od_matrices['NHB'] = od_nhb
        self.average_distances['NHB'] = SpatialEngine.calculate_average_distance(
            od_nhb, distance_matrix, "NHB"
        )
        gc.collect()
    
    def _apply_time_of_day_and_mode_choice(self, distance_matrix: np.ndarray):
        """Apply time-of-day weights and mode choice"""
        print("\nApplying Time-of-Day Weights and Mode Choice...")
        
        time_of_day_factors = {
            'HBW': self.config.time_weight['hbw'],
            'HBNW': self.config.time_weight['hbnw'],
            'NHB': self.config.time_weight['nhb']
        }
        
        for purpose_name, od_matrix in self.od_matrices.items():
            print(f"\nProcessing {purpose_name}...")
            
            # Apply time-of-day factor
            tod_factor = time_of_day_factors.get(purpose_name, 1.0)
            print(f"  Time-of-day factor (AM peak): {tod_factor:.2%}")
            
            od_matrix_tod = od_matrix * tod_factor
            print(f"  Before weighting: {od_matrix.sum():.0f} trips")
            print(f"  After weighting: {od_matrix_tod.sum():.0f} trips")
            
            # Apply mode choice
            purpose_modes = self.transport_analyst.apply_mode_choice_chunked(
                od_matrix=od_matrix_tod,
                distance_matrix=distance_matrix,
                purpose=purpose_name
            )
            
            # Store results
            for mode, mode_matrix in purpose_modes.items():
                key = f"{purpose_name}_{mode}"
                self.mode_od_matrices[key] = mode_matrix
            
            # Clean up
            del od_matrix
            gc.collect()
    
    def _combine_and_scale_modes(self) -> Dict[str, np.ndarray]:
        """Combine modes across all purposes and apply scaling"""
        print(f"\n{'='*60}")
        print("Combining modes across all purposes")
        print(f"{'='*60}")
        
        modes = ['car', 'motorbike']
        combined_modes = {}
        
        # Initialize combined matrices
        if 'HBW_car' in self.mode_od_matrices:
            shape = self.mode_od_matrices['HBW_car'].shape
            for mode in modes:
                combined_modes[mode] = np.zeros(shape)
        
        # Sum matrices across purposes
        for purpose in ['HBW', 'HBNW', 'NHB']:
            for mode in modes:
                key = f"{purpose}_{mode}"
                if key in self.mode_od_matrices:
                    combined_modes[mode] += self.mode_od_matrices[key]
        
        # Print statistics
        print("\nCombined mode statistics:")
        total_all_modes = sum(m.sum() for m in combined_modes.values())
        for mode, matrix in combined_modes.items():
            mode_trips = matrix.sum()
            print(f"  {mode}: {mode_trips:.0f} trips ({mode_trips/total_all_modes*100:.1f}%)")
        
        # Apply scaling factor
        print("\nApplying scaling factor...")
        for mode in combined_modes:
            before = combined_modes[mode].sum()
            combined_modes[mode] = self._scale_matrix_in_chunks(
                combined_modes[mode], 
                self.config.od_scale
            )
            after = combined_modes[mode].sum()
            print(f"  {mode}: {before:.0f} → {after:.0f} trips (×{self.config.od_scale})")
        
        return combined_modes
    
    def _scale_matrix_in_chunks(self, matrix: np.ndarray, scaling_factor: float, 
                               chunk_size: int = 1000) -> np.ndarray:
        """Scale a matrix in chunks to reduce memory usage"""
        n = matrix.shape[0]
        for i in range(0, n, chunk_size):
            end_i = min(i + chunk_size, n)
            matrix[i:end_i, :] = matrix[i:end_i, :] * scaling_factor
        return matrix
    
    def _save_results(self, combined_modes: Dict[str, np.ndarray], 
                     distance_matrix: np.ndarray):
        """Save all results to files"""
        print(f"\n{'='*60}")
        print("Saving results...")
        print(f"{'='*60}")
        
        for mode, matrix in combined_modes.items():
            filename = f'data/raw/Parquet/rea_1000m_{mode}_vectors_v2.parquet'
            print(f"\nSaving {mode} OD matrix...")
            self.data_handler.save_sparse_vectors(
                od_matrix=matrix,
                grid_ids=self.data_handler.grid_ids,
                filename=filename
            )
            
            # Calculate and display average distance
            avg_dist = SpatialEngine.calculate_average_distance(
                matrix, distance_matrix, f"{mode} trips"
            )
            self.average_distances[f"{mode}_avg"] = avg_dist
    
    def _print_summary(self):
        """Print final summary of results"""
        print(f"\n{'='*60}")
        print("PROCESS COMPLETED!")
        print(f"{'='*60}")
        
        print("DISTANCES ANALYTICS")
        total_avg_distance = (
            self.average_distances.get('HBW', 0) +
            self.average_distances.get('HBNW', 0) +
            self.average_distances.get('NHB', 0)
        )
        
        if len(self.average_distances) >= 3:
            print(f"Weighted average trip distance: {(total_avg_distance/3)/1000:.2f} km")
        
        for purpose in ['HBW', 'HBNW', 'NHB']:
            if purpose in self.average_distances:
                print(f"{purpose} average: {self.average_distances[purpose]/1000:.2f} km")


def main():
    """Main entry point"""
    model = ImprovedGravityModel(chunk_size=500)
    model.run()


if __name__ == "__main__":
    main()