import numpy as np
from scipy.spatial.distance import cdist
import time
from typing import Tuple, List, Dict, Optional, Any   

import warnings
warnings.filterwarnings('ignore')


class SpatialEngine:
    """Handles spatial operations like distance calculations"""
    
    def __init__(self):
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