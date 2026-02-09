import numpy as np
import time
from typing import Tuple, List, Dict, Optional, Any
import geopandas as gpd
import os
import pyarrow as pa
import pyarrow.parquet as pq        

import warnings
warnings.filterwarnings('ignore')

class DataHandler:
    """Handles data loading and saving operations"""

    def __init__(self, config):
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