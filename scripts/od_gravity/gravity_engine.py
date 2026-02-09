import numpy as np
import time
from typing import Dict
from config import Config

import warnings
warnings.filterwarnings('ignore')

class DemandModel:
    """Contains core gravity model and IPF balancing methods"""
    
    def __init__(self, config, chunk_size: int = 500):
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