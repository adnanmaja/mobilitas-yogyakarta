import os
import json
import logging
import numpy as np
from shapely.geometry import Point, LineString
import pyarrow.parquet as pq
from typing import Dict, List, Tuple, Optional, Any
from scripts.vector_routing.config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    """Handles loading of points and OD vector data"""
    
    def __init__(self, config: Config):
        self.config = config.from_yaml()
        self.points = {}
        self.car_vectors_by_origin = {}
        self.motorbike_vectors_by_origin = {}
        
    def load_points(self, points_file: str) -> None:
        logger.info(f"Loading points from {points_file}")
        with open(points_file, 'r') as f:
            data = json.load(f)
        
        self.points = {
            feat['properties']['cell_id']: {
                'lat': feat['geometry']['coordinates'][1],
                'lon': feat['geometry']['coordinates'][0]
            } for feat in data['features']
        }
        logger.info(f"Loaded {len(self.points)} points")
    
    def load_vectors(self, car_vectors_file: str, motorbike_vectors_file: str) -> None:
        logger.info(f"Loading car vectors from {car_vectors_file}")
        self.car_vectors_by_origin = self._load_vector_file(car_vectors_file, "car")
        
        logger.info(f"Loading motorbike vectors from {motorbike_vectors_file}")
        self.motorbike_vectors_by_origin = self._load_vector_file(motorbike_vectors_file, "motorbike")
        
        logger.info(
            f"Loaded {sum(len(v) for v in self.car_vectors_by_origin.values())} car OD pairs, "
            f"{sum(len(v) for v in self.motorbike_vectors_by_origin.values())} motorbike OD pairs"
        )
    
    def _load_vector_file(self, vectors_file: str, vehicle_type: str = "car") -> Dict:
        """Helper to load and filter vector files with distance band filtering"""
        
        # Distance band thresholds (km)
        if vehicle_type == "car":
            NEAR_THRESH = self.config.near_thresh['car']
            MED_THRESH = self.config.medium_thresh['car']
        else:  # motorbike
            NEAR_THRESH = self.config.near_thresh['motorbike']
            MED_THRESH = self.config.medium_thresh['motorbike']
        
        vectors_by_origin = {}
        
        # Precompute all point coordinates once
        point_coords = {
            pid: (self.points[pid]['lat'], self.points[pid]['lon'])
            for pid in self.points
        }
        
        # Read Parquet file
        table = pq.read_table(vectors_file)
        df = table.to_pandas()
        
        # Group by origin_id
        for origin_id, group in df.groupby('origin_id'):
            origin_lat, origin_lon = point_coords[origin_id]
            
            # Filter out zero/negative trips
            group = group[group['trips'] > 0]
            
            if len(group) == 0:
                continue
            
            # Extract data
            dest_ids = group['destination_id'].values
            trips_arr = group['trips'].values.astype(np.float32)
            
            # Vectorized distance calculation
            dest_lats = np.array([point_coords[dest_id][0] for dest_id in dest_ids])
            dest_lons = np.array([point_coords[dest_id][1] for dest_id in dest_ids])
            distances_arr = haversine_distance_vectorized(origin_lat, origin_lon, dest_lats, dest_lons)
            
            # Create masks for each band
            near_mask = distances_arr <= NEAR_THRESH
            med_mask = (distances_arr > NEAR_THRESH) & (distances_arr <= MED_THRESH)
            far_mask = distances_arr > MED_THRESH
            
            # Sample from each band
            sampled_dests = []
            
            for mask, k in [(near_mask, self.config.k_near), (med_mask, self.config.k_med), (far_mask, self.config.k_far)]:
                band_dest_ids = dest_ids[mask]
                band_trips = trips_arr[mask]
                
                if len(band_dest_ids) == 0 or k <= 0:
                    continue
                
                # Limit k
                k = min(k, len(band_dest_ids))
                
                # Probabilistic sampling
                if len(band_dest_ids) > 0:
                    probs = band_trips / band_trips.sum()
                    try:
                        sampled_indices = np.random.choice(
                            len(band_dest_ids),
                            size=k,
                            replace=False,
                            p=probs
                        )
                        
                        for idx in sampled_indices:
                            sampled_dests.append({
                                'destination_id': band_dest_ids[idx],
                                'trips': float(band_trips[idx])
                            })
                    except ValueError:
                        # Fallback to uniform sampling if probabilities fail
                        sampled_indices = np.random.choice(
                            len(band_dest_ids),
                            size=min(k, len(band_dest_ids)),
                            replace=False
                        )
                        for idx in sampled_indices:
                            sampled_dests.append({
                                'destination_id': band_dest_ids[idx],
                                'trips': float(band_trips[idx])
                            })
            
            if sampled_dests:
                vectors_by_origin[origin_id] = sampled_dests
        
        return vectors_by_origin

class FlowExporter:
    """Handles export of edge flows to GeoJSON format"""
    
    @staticmethod
    def save_edge_flows(edge_flows: Dict, 
                       graph: Any, 
                       filename: str = "edge_flows.geojson",
                       config: Optional[Config] = None) -> None:
        """Save aggregated edge flows as GeoJSON"""
        features = []

        for (u, v, key), flow_dict in edge_flows.items():
            car_flow = flow_dict.get("car_flow", 0)
            motorbike_flow = flow_dict.get("motorbike_flow", 0)
            total_flow = car_flow + motorbike_flow
            
            if total_flow <= 0:
                continue

            edge = graph[u][v][key]

            geom = edge.get("geometry")
            if geom is None:
                geom = LineString([
                    (graph.nodes[u]["x"], graph.nodes[u]["y"]),
                    (graph.nodes[v]["x"], graph.nodes[v]["y"])
                ])

            features.append({
                "type": "Feature",
                "properties": {
                    "u": u,
                    "v": v,
                    "car_flow": car_flow,
                    "motorbike_flow": motorbike_flow,
                    "total_flow": total_flow,
                    "length_m": edge.get("length", 0),
                    "highway": edge.get("highway", None),
                    "name": edge.get("name", None)
                },
                "geometry": geom.__geo_interface__
            })

        with open(filename, "w") as f:
            json.dump({
                "type": "FeatureCollection",
                "features": features
            }, f, indent=2)

        logger.info(f"Saved edge flows to {filename}")

        # Debugging
        print(f"\nEdge flow statistics:")
        print(f"  Total edges with flow: {len(features)}")
        print(f"  Max car flow: {max([f['properties']['car_flow'] for f in features]) if features else 0:.2f}")
        print(f"  Max motorbike flow: {max([f['properties']['motorbike_flow'] for f in features]) if features else 0:.2f}")

        # Check geographic extent
        lats = []
        lons = []
        for feature in features:
            geom = feature['geometry']['coordinates']
            for coord in geom:
                lons.append(coord[0])
                lats.append(coord[1])

        if lats:
            print(f"  Geographic extent: Lat {min(lats):.4f} to {max(lats):.4f}, Lon {min(lons):.4f} to {max(lons):.4f}")


def haversine_distance_vectorized(lat1, lon1, lat2_arr, lon2_arr):
    """
    Vectorized version for calculating distances from one point to many points.
    Uses numpy for speed when calculating many distances at once.
    
    Args:
        lat1, lon1: Single origin point (degrees)
        lat2_arr, lon2_arr: Arrays of destination coordinates (degrees)
    
    Returns:
        Array of distances in kilometers
    """
    R = 6371.0
    
    # Convert to radians
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2_arr)
    delta_lat = np.radians(lat2_arr - lat1)
    delta_lon = np.radians(lon2_arr - lon1)
    
    # Haversine formula (vectorized)
    a = (np.sin(delta_lat / 2) ** 2 + 
        np.cos(lat1_rad) * np.cos(lat2_rad) * 
        np.sin(delta_lon / 2) ** 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c