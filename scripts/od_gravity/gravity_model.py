import numpy as np
import time
from typing import Tuple, List, Dict, Optional, Any
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import gc
from scripts.od_gravity.data_handler import DataHandler
from scripts.od_gravity.spatial_engine import SpatialEngine
from scripts.od_gravity.gravity_engine import DemandModel, TransportAnalyst
from scripts.od_gravity.config import Config
from scripts.od_gravity.boundary_leakage import BoundaryLeakage

import warnings
warnings.filterwarnings('ignore')

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
        self.config = config.from_yaml()
        
        self.data_handler = DataHandler(self.config)
        self.spatial_engine = SpatialEngine()
        self.demand_model = DemandModel(self.config, chunk_size)
        self.transport_analyst = TransportAnalyst(self.config, chunk_size)
        self.boundary_leakage = BoundaryLeakage(self.config)
        self.visualization = Visualization()
        
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