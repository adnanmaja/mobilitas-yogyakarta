import json
import logging
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from threading import Lock
import gc
from scipy.sparse.csgraph import dijkstra
from typing import Dict, List, Tuple, Optional, Any
from scripts.vector_routing.config import Config
from scripts.vector_routing.graph_engine import SparseGraphBuilder, PointSnapper

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Router:
    """Handles routing computations for different vehicle types"""
    
    def __init__(self, config: Config):
        self.config = config.from_yaml()
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