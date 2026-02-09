import os
import pickle
import logging
import numpy as np
from scipy.spatial import cKDTree
import tqdm
import osmnx as ox
from scipy.sparse import csr_matrix
from typing import Dict, List, Tuple, Optional, Any
from scripts.vector_routing.config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Graph related classes
class GraphManager:
    """Manages OSM graph loading, caching, and basic operations"""
    
    def __init__(self, config = Config):
        self.config = config.from_yaml()
        self.graph = None
        self.graph_proj = None
        self.node_coords = None
        self.node_ids = None
        self.kdtree = None
        

    def load_network(self, force_download: bool = False) -> None:
        
        if not force_download and os.path.exists(self.config.cache_paths['graph']):
            logger.info(f"Loading cached graph from {self.config.cache_paths['graph']}")
            with open(self.config.cache_paths['graph'], 'rb') as f:
                self.graph = pickle.load(f)
            self._build_kdtree()
        else:
            logger.info(f"Downloading OSM data for {self.config.city}...")
            self.graph = ox.graph_from_place(self.config.city, network_type='drive', simplify=True)
            self._build_kdtree()
            self.graph = ox.add_edge_speeds(self.graph)
            self.graph = ox.add_edge_travel_times(self.graph)

            with open(self.config.cache_paths['graph'], 'wb') as f:
                pickle.dump(self.graph, f)
            logger.info(f"Cached to {self.config.cache_paths['graph']}")

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