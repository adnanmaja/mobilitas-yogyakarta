import numpy as np
from typing import Tuple, List, Dict, Optional, Any
import geopandas as gpd


class BoundaryLeakage:
    """Handles boundary leakage calculations"""
    
    def __init__(self, config):
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
        
