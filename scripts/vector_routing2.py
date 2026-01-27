import os
import json
import logging
import pandas as pd
import networkx as nx
import osmnx as ox
from tqdm import tqdm
from shapely.geometry import Point, LineString
import pickle
from collections import defaultdict
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
import ijson
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VectorRouter:
    # Convert origin-destination vectors to real paths using OSM
    
    def __init__(self, place_name: str = "Yogyakarta, Indonesia", cache_dir: str = "./cache"):
        self.place_name = place_name
        self.cache_dir = cache_dir
        self.graph = None
        self.edge_flows = defaultdict(float)
        self.flow_lock = Lock()
        os.makedirs(cache_dir, exist_ok=True)

    def add_impedance(self):
        # Add travel cost based on road hierarchy. Lower = better (preferable)
        ROAD_WEIGHTS = {
            'motorway': 1.0,      
            'trunk': 1.0,
            'primary': 1.1,
            'secondary': 1.2,
            'tertiary': 1.3,
            'residential': 2.0,  
            'service': 3.0,       
            'unclassified': 2.5,
        }
        
        for u, v, data in self.graph.edges(data=True):
            length = data.get('length', 0)
            highway = data.get('highway', 'residential')
            
            # Handle lists (OSM sometimes has multiple types)
            if isinstance(highway, list):
                highway = highway[0]
            
            # Get penalty multiplier
            penalty = ROAD_WEIGHTS.get(highway, 2.0)
            
            # Effective length = actual length × penalty
            data['impedance'] = length * penalty

    def build_sparse_graph(self):
        # Convert NetworkX graph to scipy sparse matrix (scipy is much much faster)
        nodes = list(self.graph.nodes())
        self.node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        self.idx_to_node = {idx: node for node, idx in self.node_to_idx.items()}
        
        n = len(nodes)
        row, col, data = [], [], []
        
        for u, v, edge_data in self.graph.edges(data=True):
            u_idx = self.node_to_idx[u]
            v_idx = self.node_to_idx[v]
            impedance = edge_data.get('impedance', edge_data.get('length', 0))
            
            row.append(u_idx)
            col.append(v_idx)
            data.append(impedance)
        
        self.sparse_graph = csr_matrix((data, (row, col)), shape=(n, n))
        logger.info(f"Built sparse graph: {n} nodes, {len(data)} edges")

    # Load OSM street network
    def load_network(self, force_download: bool = False):
        cache_file = os.path.join(self.cache_dir, f"graph_{self.place_name.replace(', ', '_')}.pkl")
        
        if not force_download and os.path.exists(cache_file):
            logger.info(f"Loading cached graph from {cache_file}")
            with open(cache_file, 'rb') as f:
                self.graph = pickle.load(f)
        else:
            logger.info(f"Downloading OSM data for {self.place_name}...")
            self.graph = ox.graph_from_place(self.place_name, network_type='drive', simplify=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(self.graph, f)
            logger.info(f"Cached to {cache_file}")
        
        self.graph_proj = ox.project_graph(self.graph)
        self.add_impedance()
        self.build_sparse_graph()
        logger.info(f"Loaded: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")
        
    # Load point coordinates and OD vectors
    def load_data(self, points_file: str, vectors_file: str):
        logger.info(f"Loading points from {points_file}")
        with open(points_file, 'r') as f:
            data = json.load(f)
        
        # GeoJSON
        self.points = {
                feat['properties']['cell_id']: {
                    'lat': feat['geometry']['coordinates'][1],
                    'lon': feat['geometry']['coordinates'][0]
                } for feat in data['features']
            }
        
        logger.info(f"Loading vectors from {vectors_file}")
 
        # Filter vectors (faster compute time)
        FLOW_EPS = 0 #1e-6
        TOP_K = 30

        self.vectors_by_origin = {}

        with open(vectors_file, 'rb') as f:
            # Stream the array items one by one
            for item in ijson.items(f, 'item'):
                origin_id = item['origin_id']
                
                valid_dests = [
                    d for d in item['destinations']
                    if d['trips'] > FLOW_EPS
                ]
                
                valid_dests = sorted(
                    valid_dests,
                    key=lambda d: d['trips'],
                    reverse=True
                )[:TOP_K]
                
                if valid_dests:
                    self.vectors_by_origin[origin_id] = valid_dests
        
        logger.info(
            f"Loaded {len(self.points)} points and "
            f"{sum(len(v) for v in self.vectors_by_origin.values())} OD pairs after filtering"
        )

    def precompute_nearest_nodes(self, force=False):
        if not force and self.load_point_to_node():
            return

        self.point_to_node = {}

        for pid, p in tqdm(self.points.items(), desc="Snapping points"):
            node, _ = self.find_nearest_node(p["lat"], p["lon"])
            if node is not None:
                self.point_to_node[pid] = node

        self.save_point_to_node()

        logger.info(
            f"Snapped {len(self.point_to_node)} / {len(self.points)} points to OSM nodes"
        )


    # Route all destinations from one origin using Dijkstra
    def route_from_origin(self, origin_id: int):
       
        src_node = self.point_to_node.get(origin_id)
        if not src_node:
            return

        src_idx = self.node_to_idx[src_node]

        try:
            distances, predecessors = dijkstra(
                self.sparse_graph,
                indices=src_idx,
                return_predecessors=True
            )
        except Exception as e:
            logger.warning(f"Routing failed for origin {origin_id}: {e}")
            return

        for dest in self.vectors_by_origin.get(origin_id, []):
            dest_id = dest['destination_id']
            flow = dest['trips']

            dest_node = self.point_to_node.get(dest_id)
            if dest_node is None:
                continue
            
            dest_idx = self.node_to_idx[dest_node]
            
            if predecessors[dest_idx] == -9999:
                continue

            # Reconstruct path
            route = []
            current = dest_idx
            while current != src_idx:
                route.append(self.idx_to_node[current])
                current = predecessors[current]
                if current == -9999:
                    break
            if current == src_idx:
                route.append(self.idx_to_node[src_idx])
                route.reverse()
            else:
                continue

            # Accumulate edge flows
            with self.flow_lock:
                for u, v in zip(route[:-1], route[1:]):
                    # Get the edge with minimum length
                    edges = self.graph[u][v]
                    key = min(edges.keys(), key=lambda k: edges[k].get('length', float('inf')))
                    self.edge_flows[(u, v, key)] += float(flow)


    def find_nearest_node(self, lat: float, lon: float):
        try:
            node_id, dist = ox.distance.nearest_nodes(
                self.graph, X=lon, Y=lat, return_dist=True
            )
            return node_id, dist
        except Exception as e:
            logger.warning(f"Error finding node for ({lat}, {lon}): {e}")
            return None, float('inf')

    
    def process_all(self, output_file: str = 'routes.geojson', force=False):
        if not force and self.load_edge_flows_cache():
            self.save_edge_flows(output_file)
            return

        origin_ids = list(self.vectors_by_origin.keys())

        with ThreadPoolExecutor(max_workers=4) as executor:
            list(tqdm(
                executor.map(self.route_from_origin, origin_ids),
                total=len(origin_ids),
                desc="Routing from origins"
            ))

        logger.info(
            f"Accumulated flows on {len(self.edge_flows)} edges "
            f"from {len(origin_ids)} origins"
        )

        self.save_edge_flows_cache()
        self.save_edge_flows(output_file)


    # Save aggregated edge flows as GeoJSON
    def save_edge_flows(self, filename: str = "edge_flows.geojson"):

        features = []

        for (u, v, key), flow in self.edge_flows.items():
            if flow <= 0:
                continue

            edge = self.graph[u][v][key]

            geom = edge.get("geometry")
            if geom is None:
                # fallback: straight line
                geom = LineString([
                    (self.graph.nodes[u]["x"], self.graph.nodes[u]["y"]),
                    (self.graph.nodes[v]["x"], self.graph.nodes[v]["y"])
                ])

            features.append({
                "type": "Feature",
                "properties": {
                    "u": u,
                    "v": v,
                    "flow": flow,
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

    # Caching of precompute_nearest_nodes()
    def save_point_to_node(self):
        path = os.path.join(self.cache_dir, "point_to_node.pkl")
        with open(path, "wb") as f:
            pickle.dump(self.point_to_node, f)
        logger.info(f"Saved point_to_node cache to {path}")

    def load_point_to_node(self):
        path = os.path.join(self.cache_dir, "point_to_node.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.point_to_node = pickle.load(f)
            logger.info(f"Loaded point_to_node cache from {path}")
            return True
        return False
    
    # Caching of process_all()
    def save_edge_flows_cache(self):
        path = os.path.join(self.cache_dir, "edge_flows.pkl")
        with open(path, "wb") as f:
            pickle.dump(dict(self.edge_flows), f)
        logger.info(f"Saved edge_flows cache to {path}")

    def load_edge_flows_cache(self):
        path = os.path.join(self.cache_dir, "edge_flows.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.edge_flows = defaultdict(float, pickle.load(f))
            logger.info(f"Loaded edge_flows cache from {path}")
            return True
        return False


def main():
    router = VectorRouter("Yogyakarta, Indonesia", cache_dir="./osm_cache")
    
    # Load OSM network
    router.load_network(force_download=False)
    
    # Load data
    router.load_data(
        points_file="data/raw/rea_1000m.geojson",
        vectors_file="data/raw/rea_1000m_vectors_v2.json"
    )

    # Pre-snap all points to nearest nodes
    router.precompute_nearest_nodes()
    
    # Process all vectors
    router.process_all(output_file='data/raw/rea_1000m_edge_flows_v2.geojson')
    
    logger.info("Complete!")


if __name__ == "__main__":
    main()