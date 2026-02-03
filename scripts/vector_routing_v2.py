import os
import json
import logging
import pandas as pd
import math
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

    # Debugging
    def debug_edge_matching(self, geojson_path: str):
        """
        Debug why edges aren't matching between GeoJSON and OSM graph.
        """
        with open(geojson_path, 'r') as f:
            geojson_data = json.load(f)
        
        print(f"\n{'='*60}")
        print("EDGE MATCHING DEBUG")
        print('='*60)
        
        # Count edges in GeoJSON
        geojson_edges = []
        for feature in geojson_data['features']:
            props = feature['properties']
            u = props.get('u')
            v = props.get('v')
            geojson_edges.append((u, v))
        
        print(f"GeoJSON edges: {len(geojson_edges)}")
        print(f"OSM graph edges: {len(self.graph.edges)}")
        
        # Check if (u, v) properties exist
        edges_with_uv = sum(1 for (u, v) in geojson_edges if u is not None and v is not None)
        print(f"GeoJSON edges with (u,v): {edges_with_uv}/{len(geojson_edges)}")
        
        # Check a few sample edges
        print("\nSample GeoJSON edges (first 5):")
        for u, v in geojson_edges[:5]:
            print(f"  ({u}, {v})")
        
        # Check if these nodes exist in OSM graph
        print("\nChecking if sample nodes exist in OSM graph:")
        for u, v in geojson_edges[:5]:
            u_in_graph = u in self.graph.nodes
            v_in_graph = v in self.graph.nodes
            print(f"  ({u}, {v}) -> u exists: {u_in_graph}, v exists: {v_in_graph}")
            
            if u_in_graph and v_in_graph:
                # Check if edge exists
                edge_exists = self.graph.has_edge(u, v)
                print(f"    Edge exists in graph: {edge_exists}")
                if edge_exists:
                    print(f"    Edge data keys: {list(self.graph[u][v].keys())}")
        
        # Check projection
        print(f"\nGraph CRS: {self.graph.graph.get('crs')}")
        print(f"Graph projected: {'projected' in self.graph.graph}")
        
        return geojson_edges

    def add_impedance(self, vehicle_type="car"):
        # Road weights for different vehicle types
        ROAD_WEIGHTS = {
            # Car weights (as before)
            'car': {
                'motorway': 1.0,
                'trunk': 1.0,
                'primary': 1.1,
                'secondary': 1.2,
                'tertiary': 1.3,
                'residential': 2.0,
                'service': 3.0,
                'unclassified': 2.5,
                'living_street': 4.0,  # Very slow for cars
                'track': 5.0,          # Bad for cars
                'path': 10.0,          # Nearly impossible for cars
                'pedestrian': 50.0,    # Not for cars
            },
            # Motorbike weights - can use smaller roads more easily
            'motorbike': {
                'motorway': 1.0,
                'trunk': 1.0,
                'primary': 1.05,       # Slightly better than cars
                'secondary': 1.1,
                'tertiary': 1.15,
                'residential': 1.2,    # Much better than cars
                'service': 1.3,        # Much better than cars
                'unclassified': 1.4,
                'living_street': 1.5,  # Motorbikes can handle these
                'track': 2.0,          # Can use dirt tracks
                'path': 3.0,           # Can use paths
                'pedestrian': 10.0,    # Avoid but possible
            }
        }
        
        vehicle_weights = ROAD_WEIGHTS.get(vehicle_type, ROAD_WEIGHTS['car'])
        
        for u, v, data in self.graph.edges(data=True):
            length = data.get('length', 0)
            highway = data.get('highway', 'residential')
            
            if isinstance(highway, list):
                highway = highway[0]
            
            # Get penalty multiplier for this vehicle type
            penalty = vehicle_weights.get(highway, 2.0)
            
            # Store different impedance values for different vehicles
            if 'impedance' not in data or not isinstance(data['impedance'], dict):
                data['impedance'] = {}
            data['impedance'][vehicle_type] = length * penalty

    def update_impedances_from_congestion(self, congestion_geojson_path: str):
        """
        Update edge impedances based on congestion-calculated travel times.
        
        Args:
            congestion_geojson_path: Path to GeoJSON with updated travel times
        """
        with open(congestion_geojson_path, 'r') as f:
            congestion_data = json.load(f)
        
        # Create a mapping from (u, v) to travel times
        congestion_map = {}
        for feature in congestion_data['features']:
            props = feature['properties']
            u = props.get('u')
            v = props.get('v')
            if u is not None and v is not None:
                # Use car_travel_time for car impedance
                # Use motorbike_travel_time for motorbike impedance
                congestion_map[(u, v)] = {
                    'car_travel_time': props.get('car_travel_time'),
                    'motorbike_travel_time': props.get('motorbike_travel_time')
                }
        
        # Update graph edge impedances
        for u, v, data in self.graph.edges(data=True):
            if (u, v) in congestion_map:
                travel_times = congestion_map[(u, v)]
                
                # Update car impedance
                if 'impedance' not in data:
                    data['impedance'] = {}
                
                if travel_times['car_travel_time']:
                    # Convert travel time to impedance (seconds instead of meters)
                    data['impedance']['car'] = travel_times['car_travel_time']
                
                if travel_times['motorbike_travel_time']:
                    data['impedance']['motorbike'] = travel_times['motorbike_travel_time']
        
        logger.info(f"Updated impedances for {len(congestion_map)} edges from congestion data")

    def build_sparse_graph(self, vehicle_type="car"):
        nodes = list(self.graph.nodes())
        self.node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        self.idx_to_node = {idx: node for node, idx in self.node_to_idx.items()}
        
        n = len(nodes)
        row, col, data = [], [], []
        
        for u, v, edge_data in self.graph.edges(data=True):
            u_idx = self.node_to_idx[u]
            v_idx = self.node_to_idx[v]
            
            # Get impedance for the specific vehicle type
            impedance = edge_data.get('impedance', edge_data.get('length', 0))
            if isinstance(impedance, dict):
                impedance = impedance.get(vehicle_type, edge_data.get('length', 0))
            
            row.append(u_idx)
            col.append(v_idx)
            data.append(impedance)
        
        self.sparse_graph = csr_matrix((data, (row, col)), shape=(n, n))
        logger.info(f"Built sparse graph for {vehicle_type}: {n} nodes, {len(data)} edges")

    def rebuild_sparse_graphs(self):
        """Rebuild sparse graphs after impedance updates"""
        self.build_sparse_graph("car")
        self.build_sparse_graph("motorbike")
        logger.info("Rebuilt sparse graphs with updated impedances")

    # Load OSM street network
    def load_network(self, force_download: bool = False):
        cache_file = os.path.join(self.cache_dir, f"graph_{self.place_name.replace(', ', '_')}.pkl")
        
        if not force_download and os.path.exists(cache_file):
            logger.info(f"Loading cached graph from {cache_file}")
            with open(cache_file, 'rb') as f:
                self.graph = pickle.load(f)
        else:
            logger.info(f"Downloading OSM data for {self.place_name}...")
            self.graph = ox.graph_from_place(self.place_name, network_type='drive', simplify=False)
            with open(cache_file, 'wb') as f:
                pickle.dump(self.graph, f)
            logger.info(f"Cached to {cache_file}")
        
        self.graph_proj = ox.project_graph(self.graph)
        self.add_impedance()
        self.build_sparse_graph()
        logger.info(f"Loaded: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")
        
    # Load point coordinates and OD vectors
    def load_data(self, points_file: str, car_vectors_file: str, motorbike_vectors_file: str):
        logger.info(f"Loading points from {points_file}")
        with open(points_file, 'r') as f:
            data = json.load(f)
        
        self.points = {
            feat['properties']['cell_id']: {
                'lat': feat['geometry']['coordinates'][1],
                'lon': feat['geometry']['coordinates'][0]
            } for feat in data['features']
        }
        
        logger.info(f"Loading car vectors from {car_vectors_file}")
        self.car_vectors_by_origin = self._load_vector_file(car_vectors_file, "car")
        
        logger.info(f"Loading motorbike vectors from {motorbike_vectors_file}")
        self.motorbike_vectors_by_origin = self._load_vector_file(motorbike_vectors_file, "motorbike")
        
        logger.info(
            f"Loaded {len(self.points)} points, "
            f"{sum(len(v) for v in self.car_vectors_by_origin.values())} car OD pairs, "
            f"{sum(len(v) for v in self.motorbike_vectors_by_origin.values())} motorbike OD pairs"
        )

    def _load_vector_file(self, vectors_file: str, vehicle_type: str = "car"):
        """Helper to load and filter vector files with distance band filtering"""
        
        # Distance band thresholds (km)
        if vehicle_type == "car":
            NEAR_THRESH = 5.0   # km
            MED_THRESH = 13.0   # km
        else:  # motorbike
            NEAR_THRESH = 3.0   # km
            MED_THRESH = 10.0   # km
        
        # Top k per band
        K_NEAR = 10
        K_MED = 10
        K_FAR = 10
        
        # You might need access to points for distance calculation
        # If points aren't available here, you'll need to pass them
        # or restructure the loading logic
        
        vectors_by_origin = {}
        
        with open(vectors_file, 'rb') as f:
            for item in ijson.items(f, 'item'):
                origin_id = item['origin_id']
                origin_lat = self.points[origin_id]['lat']
                origin_lon = self.points[origin_id]['lon']
                
                # We need to calculate distances and group by bands
                near_dests = []
                med_dests = []
                far_dests = []
                
                for dest in item['destinations']:
                    dest_id = dest['destination_id']
                    trips = dest['trips']
                    
                    # Skip if no trips
                    if trips <= 0:
                        continue
                    
                    # Calculate distance (Haversine or simple Euclidean approximation)
                    dest_lat = self.points[dest_id]['lat']
                    dest_lon = self.points[dest_id]['lon']
                    
                    # Simple Euclidean approximation (for small distances)
                    # For more accurate distances, use geopy or calculate haversine
                    dlat = dest_lat - origin_lat
                    dlon = dest_lon - origin_lon
                    distance_km = math.sqrt(dlat**2 + dlon**2) * 111  # approx km per degree
                    
                    # Categorize by distance band
                    if distance_km <= NEAR_THRESH:
                        near_dests.append({'destination_id': dest_id, 'trips': trips, 'distance': distance_km})
                    elif distance_km <= MED_THRESH:
                        med_dests.append({'destination_id': dest_id, 'trips': trips, 'distance': distance_km})
                    else:
                        far_dests.append({'destination_id': dest_id, 'trips': trips, 'distance': distance_km})
                
                # Sort each band by trips (descending) and take top k
                near_dests = sorted(near_dests, key=lambda d: d['trips'], reverse=True)[:K_NEAR]
                med_dests = sorted(med_dests, key=lambda d: d['trips'], reverse=True)[:K_MED]
                far_dests = sorted(far_dests, key=lambda d: d['trips'], reverse=True)[:K_FAR]
                
                # Union all bands
                all_dests = near_dests + med_dests + far_dests
                
                if all_dests:
                    vectors_by_origin[origin_id] = all_dests
        
        return vectors_by_origin

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
    def route_from_origin(self, origin_id: int, vehicle_type="car"):
        src_node = self.point_to_node.get(origin_id)
        if not src_node:
            logger.warning(f"No OSM node for point {origin_id}")
            return

        src_idx = self.node_to_idx.get(src_node)
        if src_idx is None:
            logger.warning(f"Node {src_node} not in sparse graph")
            return
        
        # Get the right vector set based on vehicle type
        if vehicle_type == "car":
            vectors = self.car_vectors_by_origin.get(origin_id, [])
            flow_key = "car_flow"
        else:
            vectors = self.motorbike_vectors_by_origin.get(origin_id, [])
            flow_key = "motorbike_flow"

        if not vectors:
            return

        try:
            distances, predecessors = dijkstra(
                self.sparse_graph,
                indices=src_idx,
                return_predecessors=True
            )
        except Exception as e:
            logger.warning(f"Routing failed for origin {origin_id} ({vehicle_type}): {e}")
            return

        for dest in vectors:
            dest_id = dest['destination_id']
            flow = dest['trips']

            dest_node = self.idx_to_node.get(dest_id)
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

            # Accumulate edge flows with vehicle type distinction
            with self.flow_lock:
                for u, v in zip(route[:-1], route[1:]):
                    edges = self.graph[u][v]
                    key = min(edges.keys(), key=lambda k: edges[k].get('length', float('inf')))
                    
                    # Store separate flow values or combine
                    edge_key = (u, v, key)
                    if edge_key not in self.edge_flows:
                        self.edge_flows[edge_key] = {"car_flow": 0.0, "motorbike_flow": 0.0}
                    
                    self.edge_flows[edge_key][flow_key] += float(flow)

    def find_nearest_node(self, lat: float, lon: float):
        try:
            node_id, dist = ox.distance.nearest_nodes(
                self.graph, X=lon, Y=lat, return_dist=True
            )
            return node_id, dist
        except Exception as e:
            logger.warning(f"Error finding node for ({lat}, {lon}): {e}")
            return None, float('inf')
        
    def process_car(self):
        logger.info("Processing car routes...")
        self.add_impedance("car")
        self.build_sparse_graph("car")
        
        car_origin_ids = list(self.car_vectors_by_origin.keys())
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(tqdm(
                executor.map(lambda x: self.route_from_origin(x, "car"), car_origin_ids),
                total=len(car_origin_ids),
                desc="Car routing"
            ))

    def process_motorbike(self):
        logger.info("Processing motorbike routes...")
        self.add_impedance("motorbike")
        self.build_sparse_graph("motorbike")
        
        motorbike_origin_ids = list(self.motorbike_vectors_by_origin.keys())
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(tqdm(
                executor.map(lambda x: self.route_from_origin(x, "motorbike"), motorbike_origin_ids),
                total=len(motorbike_origin_ids),
                desc="Motorbike routing"
            ))        

    
    def process_all(self, output_file: str = 'routes.geojson', force=False):
        if not force and self.load_edge_flows_cache():
            self.save_edge_flows(output_file)
            return
        
        # Process car routes
        self.process_car()
        
        # Process motorbike routes
        self.process_motorbike()
        
        logger.info(
            f"Accumulated flows on {len(self.edge_flows)} edges"
        )
        
        self.save_edge_flows(output_file)


    # Save aggregated edge flows as GeoJSON
    def save_edge_flows(self, filename: str = "edge_flows.geojson"):
        features = []

        for (u, v, key), flow_dict in self.edge_flows.items():
            car_flow = flow_dict.get("car_flow", 0)
            motorbike_flow = flow_dict.get("motorbike_flow", 0)
            total_flow = car_flow + motorbike_flow
            
            if total_flow <= 0:
                continue

            edge = self.graph[u][v][key]

            geom = edge.get("geometry")
            if geom is None:
                geom = LineString([
                    (self.graph.nodes[u]["x"], self.graph.nodes[u]["y"]),
                    (self.graph.nodes[v]["x"], self.graph.nodes[v]["y"])
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
    
    router.load_network(force_download=False)
    
    router.load_data(
        points_file="data/raw/rea_1000m_v2.geojson",
        car_vectors_file="data/raw/rea_1000m_car_vectors_v2.json",
        motorbike_vectors_file="data/raw/rea_1000m_motorbike_vectors_v2.json"
    )

    router.precompute_nearest_nodes()
    
    router.process_all(output_file='data/raw/rea_1000m_edge_flows_v3.geojson')
    
    logger.info("Complete!")


if __name__ == "__main__":
    main()