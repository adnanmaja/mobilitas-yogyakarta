import os
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import osmnx as ox
from tqdm import tqdm
from shapely.geometry import Point, LineString
import pickle
from collections import defaultdict
from threading import Lock
import gc
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import yaml
from dataclasses import dataclass, fields
from typing import Dict, List, Tuple, Optional, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Configuration
@dataclass
class Config:
    pcu: Dict[str, float]
    road_penalties: Dict[str, Dict[str, float]]
    noise_delta: Dict[str, float]
    turn_penalty: Dict[str, float]
    base_speeds: Dict[str, float]
    default_speed: float
    default_penalty: float
    near_thresh: Dict[str, float]
    medium_thresh: Dict[str, float]
    k_near: int
    k_med: int
    k_far: int
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


# Graph related classes
class GraphManager:
    """Manages OSM graph loading, caching, and basic operations"""
    
    def __init__(self, place_name: str = "Yogyakarta, Indonesia", cache_dir: str = "./cache"):
        self.place_name = place_name
        self.cache_dir = cache_dir
        self.graph = None
        self.graph_proj = None
        self.node_coords = None
        self.node_ids = None
        self.kdtree = None
        
        os.makedirs(cache_dir, exist_ok=True)

    def load_network(self, force_download: bool = False) -> None:
        cache_file = os.path.join(self.cache_dir, f"graph_{self.place_name.replace(', ', '_')}.pkl")
        
        if not force_download and os.path.exists(cache_file):
            logger.info(f"Loading cached graph from {cache_file}")
            with open(cache_file, 'rb') as f:
                self.graph = pickle.load(f)
            self._build_kdtree()
        else:
            logger.info(f"Downloading OSM data for {self.place_name}...")
            self.graph = ox.graph_from_place(self.place_name, network_type='drive', simplify=True)
            self._build_kdtree()
            self.graph = ox.add_edge_speeds(self.graph)
            self.graph = ox.add_edge_travel_times(self.graph)

            with open(cache_file, 'wb') as f:
                pickle.dump(self.graph, f)
            logger.info(f"Cached to {cache_file}")

        self.graph_proj = ox.project_graph(self.graph)
        logger.info(f"Loaded: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")

    def _build_kdtree(self) -> None:
        """Build KDTree for fast nearest node queries"""
        coords = []
        node_ids = []
        for node_id, data in self.graph.nodes(data=True):
            coords.append([data['y'], data['x']])  # lat, lon
            node_ids.append(node_id)
        
        self.node_coords = np.array(coords)
        self.node_ids = np.array(node_ids)
        self.kdtree = cKDTree(self.node_coords)
        logger.info(f"Built KDTree with {len(node_ids)} nodes")

    def find_nearest_node(self, lat: float, lon: float) -> Tuple[Optional[int], float]:
        try:
            dist, idx = self.kdtree.query([lat, lon])
            node_id = self.node_ids[idx]
            dist_meters = dist * 111139  # rough conversion
            return node_id, dist_meters
        except Exception as e:
            logger.warning(f"Error finding node for ({lat}, {lon}): {e}")
            return None, float('inf')


class ImpedanceCalculator:
    """Calculates and manages edge impedances for different vehicle types"""
    
    def __init__(self, config: Config):
        self.config = config
        
    def add_impedance(self, graph: Any, vehicle_type: str = "car") -> None:
        vehicle_weights = self.config.road_penalties.get(vehicle_type, self.config.road_penalties['car'])
        delta = self.config.noise_delta.get(vehicle_type, 0.01)
        turn_multiplier = self.config.turn_penalty.get(vehicle_type, 1.0)
        base_speed = self.config.base_speeds.get(vehicle_type, self.config.default_speed)
        
        for u, v, data in graph.edges(data=True):
            # Get base travel time from OSMnx (includes turn penalties)
            base_time = data.get('travel_time', 0)

            # Adjust turn penalty component
            length = data.get('length', 0)
            straight_time = length / base_speed
            
            # If actual time is longer, it includes turn penalties
            turn_penalty = max(0, base_time - straight_time)
            
            # Apply vehicle-specific adjustment to turn penalty only
            adjusted_turn_penalty = turn_penalty * turn_multiplier
            adjusted_time = straight_time + adjusted_turn_penalty
            
            # Apply road type multiplier
            highway = data.get('highway', 'residential')
            if isinstance(highway, list):
                highway = highway[0]
            
            penalty = vehicle_weights.get(highway, self.config.default_penalty)

            # Add uniform noise: ε ~ Uniform(-δ, +δ)
            epsilon = np.random.uniform(-delta, delta)
            
            impedance_noisy = adjusted_time * penalty * (1 + epsilon)
            impedance_noisy = max(impedance_noisy, adjusted_time * penalty * 0.001)

            # Store different impedance values for different vehicles
            if 'impedance' not in data or not isinstance(data['impedance'], dict):
                data['impedance'] = {}
            
            # Store both noisy and clean impedance for comparison/debugging
            data['impedance'][vehicle_type] = impedance_noisy
            data['impedance'][f'{vehicle_type}_clean'] = base_time * penalty
            data['impedance'][f'{vehicle_type}_base_time'] = base_time

    def update_impedances_from_congestion(self, graph: Any, congestion_geojson_path: str) -> None:
        """Update edge impedances based on congestion-calculated travel times."""
        with open(congestion_geojson_path, 'r') as f:
            congestion_data = json.load(f)
        
        # Create a mapping from (u, v) to travel times
        congestion_map = {}
        for feature in congestion_data['features']:
            props = feature['properties']
            u = props.get('u')
            v = props.get('v')
            if u is not None and v is not None:
                congestion_map[(u, v)] = {
                    'car_travel_time': props.get('car_travel_time'),
                    'motorbike_travel_time': props.get('motorbike_travel_time')
                }
        
        # Update graph edge impedances
        for u, v, data in graph.edges(data=True):
            if (u, v) in congestion_map:
                travel_times = congestion_map[(u, v)]
                
                if 'impedance' not in data:
                    data['impedance'] = {}
                
                if travel_times['car_travel_time']:
                    data['impedance']['car'] = travel_times['car_travel_time']
                
                if travel_times['motorbike_travel_time']:
                    data['impedance']['motorbike'] = travel_times['motorbike_travel_time']
        
        logger.info(f"Updated impedances for {len(congestion_map)} edges from congestion data")


class SparseGraphBuilder:
    """Builds and manages sparse graph representations for routing"""
    
    def __init__(self):
        self.sparse_graph_car = None
        self.sparse_graph_motorbike = None
        self.node_to_idx = None
        self.idx_to_node = None
    
    def build_sparse_graph(self, graph: Any, vehicle_type: str = "car") -> csr_matrix:
        nodes = list(graph.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        idx_to_node = {idx: node for node, idx in node_to_idx.items()}
        
        n = len(nodes)
        row, col, data = [], [], []
        
        for u, v, edge_data in graph.edges(data=True):
            u_idx = node_to_idx[u]
            v_idx = node_to_idx[v]
            
            # Get impedance for the specific vehicle type
            impedance = edge_data.get('impedance', edge_data.get('length', 0))
            if isinstance(impedance, dict):
                impedance = impedance.get(vehicle_type, edge_data.get('length', 0))
            
            row.append(u_idx)
            col.append(v_idx)
            data.append(impedance)
        
        sparse_graph = csr_matrix((data, (row, col)), shape=(n, n))
        
        # Store the mapping
        self.node_to_idx = node_to_idx
        self.idx_to_node = idx_to_node
        
        if vehicle_type == "car":
            self.sparse_graph_car = sparse_graph
        else:
            self.sparse_graph_motorbike = sparse_graph
        
        logger.info(f"Built sparse graph for {vehicle_type}: {n} nodes, {len(data)} edges")
        return sparse_graph
    
    def get_sparse_graph(self, vehicle_type: str = "car") -> csr_matrix:
        if vehicle_type == "car":
            return self.sparse_graph_car
        else:
            return self.sparse_graph_motorbike


class DataLoader:
    """Handles loading of points and OD vector data"""
    
    def __init__(self, config: Config):
        self.config = config
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


class PointSnapper:
    """Handles snapping points to nearest graph nodes"""
    
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = cache_dir
        self.point_to_node = {}
        os.makedirs(cache_dir, exist_ok=True)
    
    def snap_points(self, points: Dict, graph_manager: GraphManager, force: bool = False) -> bool:
        if not force and self.load_point_to_node():
            return True

        self.point_to_node = {}

        for pid, p in tqdm(points.items(), desc="Snapping points"):
            node, _ = graph_manager.find_nearest_node(p["lat"], p["lon"])
            if node is not None:
                self.point_to_node[pid] = node

        self.save_point_to_node()

        logger.info(
            f"Snapped {len(self.point_to_node)} / {len(points)} points to OSM nodes"
        )
        return True
    
    def save_point_to_node(self) -> None:
        path = os.path.join(self.cache_dir, "point_to_node.pkl")
        with open(path, "wb") as f:
            pickle.dump(self.point_to_node, f)
        logger.info(f"Saved point_to_node cache to {path}")
    
    def load_point_to_node(self) -> bool:
        path = os.path.join(self.cache_dir, "point_to_node.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.point_to_node = pickle.load(f)
            logger.info(f"Loaded point_to_node cache from {path}")
            return True
        return False


class Router:
    """Handles routing computations for different vehicle types"""
    
    def __init__(self, config: Config):
        self.config = config
        self.edge_flows = defaultdict(lambda: {"car_flow": 0.0, "motorbike_flow": 0.0})
        self.flow_lock = Lock()
    
    def route_vehicle_type(self, 
                          vehicle_type: str, 
                          vectors_by_origin: Dict, 
                          point_snapper: PointSnapper,
                          sparse_graph_builder: SparseGraphBuilder,
                          graph: Any,
                          chunk_size: int = 1000) -> None:
        """Compute routes using chunked Dijkstra to save memory"""
        
        # Get all unique origin node indices with their IDs
        origin_mapping = {}  # origin_idx -> [origin_ids that map to it]
        for origin_id in vectors_by_origin.keys():
            node_id = point_snapper.point_to_node.get(origin_id)
            if node_id and node_id in sparse_graph_builder.node_to_idx:
                origin_idx = sparse_graph_builder.node_to_idx[node_id]
                if origin_idx not in origin_mapping:
                    origin_mapping[origin_idx] = []
                origin_mapping[origin_idx].append(origin_id)
        
        origin_indices = list(origin_mapping.keys())
        total_origins = len(origin_indices)
        logger.info(f"Processing {total_origins} unique origins for {vehicle_type} in chunks of {chunk_size}")
        
        sparse_graph = sparse_graph_builder.get_sparse_graph(vehicle_type)
        
        # Process in chunks
        for chunk_start in tqdm(range(0, total_origins, chunk_size), desc=f"{vehicle_type} routing"):
            chunk_end = min(chunk_start + chunk_size, total_origins)
            chunk_indices = origin_indices[chunk_start:chunk_end]
            
            # Compute Dijkstra for this chunk
            dist_matrix, predecessors = dijkstra(
                sparse_graph,
                directed=True,
                indices=chunk_indices,
                return_predecessors=True
            )
            
            # Process each origin in the chunk
            for i, origin_idx in enumerate(chunk_indices):
                distances = dist_matrix[i]
                preds = predecessors[i]
                
                # Process all origin_ids that map to this node
                for origin_id in origin_mapping[origin_idx]:
                    destination_flows = vectors_by_origin[origin_id]
                    
                    # Route to each destination
                    for dest in destination_flows:
                        dest_id = dest["destination_id"]
                        flow = dest["trips"]

                        dest_node = point_snapper.point_to_node.get(dest_id)
                        if not dest_node or dest_node not in sparse_graph_builder.node_to_idx:
                            continue

                        dest_idx = sparse_graph_builder.node_to_idx[dest_node]

                        if distances[dest_idx] == np.inf:
                            continue

                        path_indices = self._reconstruct_path(preds, dest_idx)
                        if len(path_indices) < 2:
                            continue

                        self._accumulate_flow(path_indices, flow, vehicle_type, sparse_graph_builder, graph)
                                
            # Free memory after each chunk
            del dist_matrix, predecessors
            gc.collect()
    
    def _reconstruct_path(self, predecessors: np.ndarray, dest_idx: int) -> List[int]:
        """Reconstruct path from predecessors array"""
        path = []
        current = dest_idx
        
        while current != -9999:  # -9999 is scipy's sentinel for "no predecessor"
            path.append(current)
            pred = predecessors[current]
            if pred == -9999 or pred == current:
                break
            current = pred
        
        return path[::-1]
    
    def _accumulate_flow(self, 
                        path_indices: List[int], 
                        flow: float, 
                        vehicle_type: str,
                        sparse_graph_builder: SparseGraphBuilder,
                        graph: Any) -> None:
        """Accumulate flow along a path"""
        flow_key = f"{vehicle_type}_flow"
        
        for i in range(len(path_indices) - 1):
            u_idx = path_indices[i]
            v_idx = path_indices[i + 1]
            
            u = sparse_graph_builder.idx_to_node[u_idx]
            v = sparse_graph_builder.idx_to_node[v_idx]
            
            # Handle multi-edges
            if v in graph[u]:
                edges = graph[u][v]
                if isinstance(edges, dict):
                    key = min(edges.keys())
                else:
                    key = 0
                
                with self.flow_lock:
                    edge_key = (u, v, key)
                    self.edge_flows[edge_key][flow_key] += float(flow)


class FlowAnalyzer:
    """Analyzes traffic flow distributions and generates statistics"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def calculate_pcu_km(self, edge_flows: Dict, graph: Any) -> Tuple[Dict, Dict]:
        """Calculate PCU-km for each edge"""
        pcu_km_by_edge = {}
        pcu_km_by_road_class = defaultdict(float)
        
        for (u, v, key), flow_dict in edge_flows.items():
            car_flow = flow_dict.get("car_flow", 0)
            motorbike_flow = flow_dict.get("motorbike_flow", 0)
            
            # Convert to PCU
            pcu_flow = (car_flow * self.config.pcu['car'] + 
                        motorbike_flow * self.config.pcu['motorbike'])
            
            # Get edge length in km
            edge = graph[u][v][key]
            length_km = edge.get("length", 0) / 1000.0
            
            # Calculate PCU-km
            pcu_km = pcu_flow * length_km
            
            if pcu_km > 0:
                pcu_km_by_edge[(u, v, key)] = pcu_km
                
                # Group by road class
                highway = edge.get("highway", "unclassified")
                if isinstance(highway, list):
                    highway = highway[0]
                
                # Categorize
                if highway in ['motorway', 'trunk']:
                    road_class = 'trunk'
                elif highway == 'primary':
                    road_class = 'primary'
                elif highway == 'secondary':
                    road_class = 'secondary'
                elif highway == 'tertiary':
                    road_class = 'tertiary'
                else:
                    road_class = 'other'
                
                pcu_km_by_road_class[road_class] += pcu_km
        
        return pcu_km_by_edge, pcu_km_by_road_class
    
    def calculate_gini_coefficient(self, values: List[float]) -> float:
        """Calculate Gini coefficient for a list of values"""
        if not values:
            return 0.0
        
        # Sort values
        sorted_values = np.sort(np.array(values))
        n = len(sorted_values)
        
        # Gini formula
        index = np.arange(1, n + 1)
        gini = (np.sum((2 * index - n - 1) * sorted_values)) / (n * np.sum(sorted_values))
        
        return gini
    
    def generate_lorenz_curve_data(self, values: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Generate data points for Lorenz curve"""
        if not values:
            return np.array([]), np.array([])
        
        sorted_values = np.sort(np.array(values))
        cumulative_values = np.cumsum(sorted_values)
        total = cumulative_values[-1]
        
        # Normalize
        cumulative_percentage = cumulative_values / total * 100
        population_percentage = np.arange(1, len(values) + 1) / len(values) * 100
        
        return population_percentage, cumulative_percentage
    
    def analyze_flow_distribution(self, 
                                 edge_flows: Dict, 
                                 graph: Any, 
                                 output_path: str = "analysis", 
                                 plot_lorenz: bool = True) -> Dict:
        """Analyze flow distribution and generate statistics"""
        
        # 1. Calculate PCU-km
        pcu_km_by_edge, pcu_km_by_road_class = self.calculate_pcu_km(edge_flows, graph)
        
        # 2. Calculate Gini coefficient for link flows (PCU-km)
        pcu_km_values = list(pcu_km_by_edge.values())
        gini = self.calculate_gini_coefficient(pcu_km_values)
        
        # 3. Generate Lorenz curve data
        pop_percent, cum_percent = self.generate_lorenz_curve_data(pcu_km_values)
        
        # 4. Calculate percentage by road class
        total_pcu_km = sum(pcu_km_by_road_class.values())
        percentages_by_class = {}
        for road_class, value in pcu_km_by_road_class.items():
            percentages_by_class[road_class] = (value / total_pcu_km * 100) if total_pcu_km > 0 else 0
        
        # 5. Plot Lorenz curve if requested
        if plot_lorenz and len(pop_percent) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(pop_percent, cum_percent, 'b-', linewidth=2, label='Lorenz Curve')
            plt.plot([0, 100], [0, 100], 'r--', linewidth=1, label='Perfect Equality')
            plt.fill_between(pop_percent, cum_percent, pop_percent, alpha=0.3)
            
            # Add Gini coefficient annotation
            plt.annotate(f'Gini = {gini:.3f}', 
                        xy=(0.6, 0.2), 
                        xycoords='axes fraction',
                        fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            plt.xlabel('Cumulative Percentage of Road Links (%)', fontsize=12)
            plt.ylabel('Cumulative Percentage of PCU-km (%)', fontsize=12)
            plt.title('Lorenz Curve of Traffic Flow Distribution', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.legend(loc='upper left')
            plt.axis('equal')
            
            # Save the plot
            plot_file = output_path.replace('.json', '_lorenz.png') if output_path.endswith('.json') else f"{output_path}_lorenz.png"
            plt.tight_layout()
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved Lorenz curve plot to {plot_file}")
        
        # 6. Save results to JSON
        results = {
            'gini_coefficient': float(gini),
            'total_pcu_km': float(total_pcu_km),
            'pcu_km_by_road_class': dict(percentages_by_class),
            'lorenz_curve': {
                'population_percentage': pop_percent.tolist(),
                'cumulative_percentage': cum_percent.tolist()
            },
            'edge_level_pcu_km': [
                {
                    'u': key[0],
                    'v': key[1],
                    'pcu_km': value,
                    'road_class': self._get_road_class(graph, key)
                }
                for key, value in pcu_km_by_edge.items()
            ]
        }
        
        # Save to JSON
        json_file = output_path if output_path.endswith('.json') else f"{output_path}.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved distribution analysis to {json_file}")
        
        # Print summary
        print(f"\n=== Flow Distribution Analysis ===")
        print(f"Gini Coefficient: {gini:.4f}")
        print(f"Total PCU-km: {total_pcu_km:.2f}")
        print("\nPCU-km by Road Class (%):")
        for road_class, percentage in percentages_by_class.items():
            print(f"  {road_class}: {percentage:.2f}%")
        
        return results
    
    def _get_road_class(self, graph: Any, edge_key: Tuple) -> str:
        """Helper to get road class for an edge"""
        u, v, key = edge_key
        edge = graph[u][v][key]
        highway = edge.get("highway", "unclassified")
        if isinstance(highway, list):
            highway = highway[0]
        
        if highway in ['motorway', 'trunk']:
            return 'trunk'
        elif highway == 'primary':
            return 'primary'
        elif highway == 'secondary':
            return 'secondary'
        elif highway == 'tertiary':
            return 'tertiary'
        elif highway == 'residential':
            return 'residential'
        else:
            return 'other'


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


class VectorRouter:
    """Main orchestrator class that coordinates all components"""
    
    def __init__(self, place_name: str = "Yogyakarta, Indonesia", cache_dir: str = "./cache", config = Config):
        self.place_name = place_name
        self.cache_dir = cache_dir
        self.config = config
        
        # Initialize components
        self.graph_manager = GraphManager(place_name, cache_dir)
        self.impedance_calc = ImpedanceCalculator(self.config)
        self.sparse_graph_builder = SparseGraphBuilder()
        self.data_loader = DataLoader(self.config)
        self.point_snapper = PointSnapper(cache_dir)
        self.router = Router(self.config)
        self.flow_analyzer = FlowAnalyzer(self.config)
        
        np.random.seed(67)
    
    def load_network(self, force_download: bool = False) -> None:
        self.graph_manager.load_network(force_download)
    
    def load_data(self) -> None:
        self.data_loader.load_points(self.config.data_paths['grid'])
        self.data_loader.load_vectors(
            self.config.data_paths['car_vector'],
            self.config.data_paths['motorbike_vector']
        )
    
    def precompute_nearest_nodes(self, force: bool = False) -> None:
        self.point_snapper.snap_points(self.data_loader.points, self.graph_manager, force)
    
    def process_vehicle_type(self, vehicle_type: str) -> None:
        logger.info(f"Processing {vehicle_type} routes...")
        
        # Add impedance
        self.impedance_calc.add_impedance(self.graph_manager.graph, vehicle_type)
        
        # Build sparse graph
        self.sparse_graph_builder.build_sparse_graph(self.graph_manager.graph, vehicle_type)
        
        # Get vectors
        if vehicle_type == "car":
            vectors = self.data_loader.car_vectors_by_origin
        else:
            vectors = self.data_loader.motorbike_vectors_by_origin
        
        # Route
        self.router.route_vehicle_type(
            vehicle_type,
            vectors,
            self.point_snapper,
            self.sparse_graph_builder,
            self.graph_manager.graph,
            chunk_size=100
        )
    
    def process_all(self, output_file: str = 'routes.geojson') -> None:
        # Process car routes
        self.process_vehicle_type("car")
        
        # Process motorbike routes
        self.process_vehicle_type("motorbike")
        
        logger.info(f"Accumulated flows on {len(self.router.edge_flows)} edges")
        
        # Save edge flows
        FlowExporter.save_edge_flows(
            self.router.edge_flows,
            self.graph_manager.graph,
            output_file,
            self.config
        )
        
        # Analyze flow distribution
        analysis_path = "data/analysis/distribution_analysis.json".replace('.geojson', '')
        self.flow_analyzer.analyze_flow_distribution(
            self.router.edge_flows,
            self.graph_manager.graph,
            output_path=analysis_path,
            plot_lorenz=True
        )


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

def main():
    router = VectorRouter("Yogyakarta, Indonesia", cache_dir="./osm_cache")
    
    router.load_network(force_download=False)
    
    cfg = Config.from_yaml()

    router.load_data()

    router.precompute_nearest_nodes()
    
    router.process_all(output_file=cfg.export_paths['edge_flow'])
    
    logger.info("Complete!")


if __name__ == "__main__":
    main()