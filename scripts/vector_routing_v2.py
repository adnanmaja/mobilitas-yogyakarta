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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


PCU_FACTORS = {
    'car': 1.0,
    'motorbike': 0.25  
}

# Road penalties for different vehicle types
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

# Noise parameters: Uniform(-δ, +δ)
NOISE_DELTA = {
    'car': 0.05,      # ±5% noise for cars
    'motorbike': 0.15 # ±15% noise for motorbikes
}

TURN_PENALTY_MULTIPLIERS = {
    'car': 1.0,      # Standard turn penalties
    'motorbike': 0.6  # Motorbikes handle turns better
}

BASE_SPEEDS = {
    'car': 13.89,      # 50 km/h
    'motorbike': 11.11  # 40 km/h (more conservative for safety)
}

class VectorRouter:
    # Convert origin-destination vectors to real paths using OSM
    
    def __init__(self, place_name: str = "Yogyakarta, Indonesia", cache_dir: str = "./cache"):
        self.place_name = place_name
        self.cache_dir = cache_dir
        self.graph = None
        self.edge_flows = defaultdict(float)
        self.flow_lock = Lock()
        os.makedirs(cache_dir, exist_ok=True)

        self.PCU_FACTORS = PCU_FACTORS

        np.random.seed(67)

    def build_kdtree(self):
        """Build KDTree for fast nearest node queries"""
        # Extract all node coordinates
        coords = []
        node_ids = []
        for node_id, data in self.graph.nodes(data=True):
            coords.append([data['y'], data['x']])  # lat, lon
            node_ids.append(node_id)
        
        self.node_coords = np.array(coords)
        self.node_ids = np.array(node_ids)
        self.kdtree = cKDTree(self.node_coords)
        logger.info(f"Built KDTree with {len(node_ids)} nodes")

    def add_impedance(self, vehicle_type="car"):
        # Road penalties for different vehicle types
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

        # Noise parameters: Uniform(-δ, +δ)
        NOISE_DELTA = {
            'car': 0.05,      # ±5% noise for cars
            'motorbike': 0.15 # ±15% noise for motorbikes
        }

        TURN_PENALTY_MULTIPLIERS = {
            'car': 1.0,      # Standard turn penalties
            'motorbike': 0.6  # Motorbikes handle turns better
        }

        BASE_SPEEDS = {
            'car': 13.89,      # 50 km/h
            'motorbike': 11.11  # 40 km/h (more conservative for safety)
        }
            
        vehicle_weights = ROAD_WEIGHTS.get(vehicle_type, ROAD_WEIGHTS['car'])
        delta = NOISE_DELTA.get(vehicle_type, 0.01)
        turn_multiplier = TURN_PENALTY_MULTIPLIERS.get(vehicle_type, 1.0)
        base_speed = BASE_SPEEDS.get(vehicle_type, 13.89)
        
        for u, v, data in self.graph.edges(data=True):
            # Get base travel time from OSMnx (includes turn penalties)
            base_time = data.get('travel_time', 0)

            # Adjust turn penalty component
            # We can estimate it by comparing to straight-line time
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
            
            penalty = vehicle_weights.get(highway, 2.0)

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
            self.build_kdtree()
        else:
            logger.info(f"Downloading OSM data for {self.place_name}...")
            self.graph = ox.graph_from_place(self.place_name, network_type='drive', simplify=True)
            self.build_kdtree()
            self.graph = ox.add_edge_speeds(self.graph)

            # Add travel times with turn penalties
            self.graph = ox.add_edge_travel_times(self.graph)

            with open(cache_file, 'wb') as f:
                pickle.dump(self.graph, f)
            logger.info(f"Cached to {cache_file}")

        self.graph_proj = ox.project_graph(self.graph)
        self.add_impedance()
        # self.build_sparse_graph()
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
            NEAR_THRESH = 5.0
            MED_THRESH = 13.0
        else:  # motorbike
            NEAR_THRESH = 3.0
            MED_THRESH = 10.0   
        
        # Top k per band
        K_NEAR = 13
        K_MED = 9
        K_FAR = 5
        
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
            
            for mask, k in [(near_mask, K_NEAR), (med_mask, K_MED), (far_mask, K_FAR)]:
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
    def route_from_origin(self, vehicle_type="car", chunk_size=1000):
        """Compute routes using chunked Dijkstra to save memory"""
    
        # Get the appropriate vectors
        if vehicle_type == "car":
            vectors_by_origin = self.car_vectors_by_origin
        else:
            vectors_by_origin = self.motorbike_vectors_by_origin
        
        # Get all unique origin node indices with their IDs
        origin_mapping = {}  # origin_idx -> [origin_ids that map to it]
        for origin_id in vectors_by_origin.keys():
            node_id = self.point_to_node.get(origin_id)
            if node_id and node_id in self.node_to_idx:
                origin_idx = self.node_to_idx[node_id]
                if origin_idx not in origin_mapping:
                    origin_mapping[origin_idx] = []
                origin_mapping[origin_idx].append(origin_id)
        
        origin_indices = list(origin_mapping.keys())
        total_origins = len(origin_indices)
        logger.info(f"Processing {total_origins} unique origins for {vehicle_type} in chunks of {chunk_size}")
        
        # Process in chunks
        for chunk_start in tqdm(range(0, total_origins, chunk_size), desc=f"{vehicle_type} routing"):
            chunk_end = min(chunk_start + chunk_size, total_origins)
            chunk_indices = origin_indices[chunk_start:chunk_end]
            
            # Compute Dijkstra for this chunk
            dist_matrix, predecessors = dijkstra(
                self.sparse_graph,
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

                        dest_node = self.point_to_node.get(dest_id)
                        if not dest_node or dest_node not in self.node_to_idx:
                            continue

                        dest_idx = self.node_to_idx[dest_node]

                        if distances[dest_idx] == np.inf:
                            continue

                        path_indices = self._reconstruct_path(preds, dest_idx)
                        if len(path_indices) < 2:
                            continue

                        self._accumulate_flow(path_indices, flow, vehicle_type)
                                
            # Free memory after each chunk
            del dist_matrix, predecessors
            gc.collect()
    
    def _reconstruct_path(self, predecessors, dest_idx):
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
    
    def _accumulate_flow(self, path_indices, flow, vehicle_type):
        """Accumulate flow along a path"""
        flow_key = f"{vehicle_type}_flow"
        
        for i in range(len(path_indices) - 1):
            u_idx = path_indices[i]
            v_idx = path_indices[i + 1]
            
            u = self.idx_to_node[u_idx]
            v = self.idx_to_node[v_idx]
            
            # Handle multi-edges
            if v in self.graph[u]:
                edges = self.graph[u][v]
                if isinstance(edges, dict):
                    key = min(edges.keys())
                else:
                    key = 0
                
                with self.flow_lock:
                    edge_key = (u, v, key)
                    if edge_key not in self.edge_flows:
                        self.edge_flows[edge_key] = {"car_flow": 0.0, "motorbike_flow": 0.0}
                    
                    self.edge_flows[edge_key][flow_key] += float(flow)

    def find_nearest_node(self, lat: float, lon: float):
        try:
            dist, idx = self.kdtree.query([lat, lon])
            node_id = self.node_ids[idx]
            dist_meters = dist * 111139 # rough conversion
            return node_id, dist_meters
        except Exception as e:
            logger.warning(f"Error finding node for ({lat}, {lon}): {e}")
            return None, float('inf')
        
    def process_car(self):
        logger.info("Processing car routes...")
        self.add_impedance("car")
        self.build_sparse_graph("car")
        self.route_from_origin("car", chunk_size=100)

    def process_motorbike(self):
        logger.info("Processing motorbike routes...")
        self.add_impedance("motorbike")
        self.build_sparse_graph("motorbike")
        self.route_from_origin("motorbike", chunk_size=100) 
    
    def process_all(self, output_file: str = 'routes.geojson', force=False): 
        # Process car routes
        self.process_car()
        
        # Process motorbike routes
        self.process_motorbike()
        
        logger.info(
            f"Accumulated flows on {len(self.edge_flows)} edges"
        )
        
        self.save_edge_flows(output_file)

        self.analyze_flow_distribution(output_path="data/analysis/distribution_analysis.json".replace('.geojson', ''), plot_lorenz=True)


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
    
    def calculate_pcu_km(self):
        """Calculate PCU-km for each edge"""
        pcu_km_by_edge = {}
        pcu_km_by_road_class = defaultdict(float)
        
        for (u, v, key), flow_dict in self.edge_flows.items():
            car_flow = flow_dict.get("car_flow", 0)
            motorbike_flow = flow_dict.get("motorbike_flow", 0)
            
            # Convert to PCU
            pcu_flow = (car_flow * self.PCU_FACTORS['car'] + 
                        motorbike_flow * self.PCU_FACTORS['motorbike'])
            
            # Get edge length in km
            edge = self.graph[u][v][key]
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
    
    def calculate_gini_coefficient(self, values):
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
    
    def generate_lorenz_curve_data(self, values):
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
    
    def analyze_flow_distribution(self, output_path="analysis", plot_lorenz=True):
        """Analyze flow distribution and generate statistics"""
    
        # 1. Calculate PCU-km
        pcu_km_by_edge, pcu_km_by_road_class = self.calculate_pcu_km()
        
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
                    'road_class': self._get_road_class(key)
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

    def _get_road_class(self, edge_key):
        """Helper to get road class for an edge"""
        u, v, key = edge_key
        edge = self.graph[u][v][key]
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
    
    router.load_data(
        points_file="data/raw/rea_1000m_v2.geojson",
        car_vectors_file="data/raw/Parquet/rea_1000m_car_vectors_v2.parquet",
        motorbike_vectors_file="data/raw/Parquet/rea_1000m_motorbike_vectors_v2.parquet"
    )

    router.precompute_nearest_nodes()
    
    router.process_all(output_file='data/raw/rea_1000m_edge_flows_v3.geojson')
    
    logger.info("Complete!")


if __name__ == "__main__":
    main()