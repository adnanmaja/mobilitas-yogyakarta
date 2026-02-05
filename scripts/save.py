import os
import json
import logging
import numpy as np
import pandas as pd
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
import matplotlib.pyplot as plt
from scipy import integrate

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Road type penalties for different vehicle types
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

# Perception noise parameters: Uniform(-δ, +δ)
NOISE_DELTA = {
    'car': 0.01,      # ±1% noise for cars
    'motorbike': 0.03 # ±3% noise for motorbikes
}

# Turn penalty, assumed
TURN_PENALTY_MULTIPLIERS = {
    'car': 1.0,      
    'motorbike': 0.6 
}

# Base speed, to calculate turn penalty, uncalibrated
BASE_SPEEDS = {
    'car': 13.89,      # 50 km/h
    'motorbike': 11.11  # 40 km/h (more conservative for safety)
}

# Distance band thresholds (km), assumed
NEAR_THRESH_CAR = 5.0
MED_THRESH_CAR = 13.0
NEAR_THRESH_MOTORBIKE = 3.0
MED_THRESH_MOTORBIKE = 10.0

# Top k per band
K_NEAR = 13
K_MED = 9
K_FAR = 5

# PCU, also check the ones on od_matrix_v2.py
CAR_PCU = 1.0
MOTORBIKE_PCU = 0.25


class VectorRouter:
    # Convert origin-destination vectors to real paths using OSM
    
    def __init__(self, place_name: str = "Yogyakarta, Indonesia", cache_dir: str = "./cache"):
        self.place_name = place_name
        self.cache_dir = cache_dir
        self.graph = None
        self.edge_flows = defaultdict(float)
        self.flow_lock = Lock()
        os.makedirs(cache_dir, exist_ok=True)

        np.random.seed(67)

    def add_impedance(self, vehicle_type="car"):
        vehicle_weights = ROAD_WEIGHTS.get(vehicle_type, ROAD_WEIGHTS['car'])
        delta = NOISE_DELTA.get(vehicle_type, 0.01)
        turn_multiplier = TURN_PENALTY_MULTIPLIERS.get(vehicle_type, 1.0)
        base_speed = BASE_SPEEDS.get(vehicle_type, 13.89)
        
        for u, v, data in self.graph.edges(data=True):
            # Get base travel time from OSMnx (includes turn penalties)
            base_time = data.get('travel_time', 0)

            # Adjust turn penalty component, estimated by comparing to straight-line time
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
        else:
            logger.info(f"Downloading OSM data for {self.place_name}...")
            self.graph = ox.graph_from_place(self.place_name, network_type='drive', simplify=False)

            # Add travel times with turn penalties
            self.graph = ox.add_edge_travel_times(self.graph)

            with open(cache_file, 'wb') as f:
                pickle.dump(self.graph, f)
            logger.info(f"Cached to {cache_file}")
        
        # self.visualize_coverage()
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
            NEAR_THRESH = NEAR_THRESH_CAR
            MED_THRESH = MED_THRESH_CAR
        else: 
            NEAR_THRESH = NEAR_THRESH_MOTORBIKE
            MED_THRESH = MED_THRESH_MOTORBIKE
        
        vectors_by_origin = {}
        
        # Precompute all point coordinates once
        point_coords = {
            pid: (self.points[pid]['lat'], self.points[pid]['lon'])
            for pid in self.points
        }
        
        logger.info(f"Loading Parquet file: {vectors_file}")
        df = pd.read_parquet(vectors_file)
        
        # Assuming Parquet structure: each row is an origin-destination pair
        # with columns: origin_id, destination_id, trips
        
        # Group by origin_id for processing
        grouped = df.groupby('origin_id')
        
        for origin_id, group in grouped:
            origin_coords = point_coords.get(origin_id)
            if not origin_coords:
                continue
                
            # Process destinations for this origin
            dests_data = group[['destination_id', 'trips']].to_dict('records')
            
            # Filter out zero or negative trips
            valid_dests = [d for d in dests_data if d['trips'] > 0]
            if not valid_dests:
                continue
                
            # Extract data for vectorized processing
            dest_ids = []
            trips_list = []
            dest_coords_list = []
            
            for dest in valid_dests:
                dest_id = dest['destination_id']
                trips = dest['trips']
                
                dest_ids.append(dest_id)
                trips_list.append(trips)
                dest_coords_list.append(point_coords[dest_id])
            
            lat1, lon1 = origin_coords
            
            # Convert list of tuples to numpy arrays
            dest_coords_arr = np.array(dest_coords_list)
            lat2_array = dest_coords_arr[:, 0]
            lon2_array = dest_coords_arr[:, 1]
            
            distances_arr = haversine_vectorized(lat1, lon1, lat2_array, lon2_array)
            
            # Use numpy arrays for faster operations
            dest_ids_arr = np.array(dest_ids)
            trips_arr = np.array(trips_list, dtype=np.float32)
            
            # Create masks for each band
            near_mask = distances_arr <= NEAR_THRESH
            med_mask = (distances_arr > NEAR_THRESH) & (distances_arr <= MED_THRESH)
            far_mask = distances_arr > MED_THRESH
            
            # Sample from each band (same logic as before)
            sampled_dests = []
            
            for mask, k in [(near_mask, K_NEAR), (med_mask, K_MED), (far_mask, K_FAR)]:
                band_dest_ids = dest_ids_arr[mask]
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
                        # Fallback to uniform sampling
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
            
            dest_osm_node = self.point_to_node.get(dest_id)  # Get OSM node for destination cell
            if not dest_osm_node:
                continue
            dest_idx = self.node_to_idx.get(dest_osm_node)  # Now get the sparse graph index
            if dest_idx is None:
                continue
            
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
    
    def visualize_coverage(self):
        """Create a map showing road network vs flows"""
        import matplotlib.pyplot as plt
        
        # Plot all roads
        fig, ax = ox.plot_graph(self.graph, show=False, close=False, 
                            edge_color='gray', edge_alpha=0.3, node_size=0)
        
        # Plot roads with flows
        edges_with_flow = []
        for (u, v, key), flow_dict in self.edge_flows.items():
            if flow_dict.get('total_flow', 0) > 0:
                edges_with_flow.append((u, v, key))
        
        # Highlight edges with flows
        ec = ['r' if (u, v, 0) in edges_with_flow else 'gray' 
            for u, v, k in self.graph.edges(keys=True)]
        
        fig, ax = ox.plot_graph(self.graph, edge_color=ec, node_size=0, 
                            show=False, close=False)
        
        # Add title
        ax.set_title(f'Road Coverage: {len(edges_with_flow)}/{len(self.graph.edges)} edges have flows')
        plt.savefig('coverage_debug.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_edge_flows(self, pcu_factors: dict = None, output_file: str = "flow_analysis.json"):
        """
        Analyze edge flows and compute:
        1. % of total PCU-km on different road types
        2. Gini coefficient for PCU-km distribution
        3. Save results to JSON and plot Lorenz curve
        
        Args:
            pcu_factors: PCU factors for different vehicles (default: car=1.0, motorbike=0.33)
            output_file: Path to save JSON results
        """
        if pcu_factors is None:
            pcu_factors = {"car": 1.0, "motorbike": 0.33}
        
        # Initialize road type categories
        road_type_categories = {
            'trunk': ['trunk', 'trunk_link'],
            'primary': ['primary', 'primary_link'],
            'secondary': ['secondary', 'secondary_link'],
            'tertiary': ['tertiary', 'tertiary_link'],
            'other': []  # All other types
        }
        
        # Initialize totals
        total_pcu_km = 0.0
        road_type_pcu_km = {category: 0.0 for category in road_type_categories}
        
        # Collect PCU-km for all edges for Gini calculation
        all_pcu_km = []
        
        for (u, v, key), flow_dict in self.edge_flows.items():
            car_flow = flow_dict.get("car_flow", 0)
            motorbike_flow = flow_dict.get("motorbike_flow", 0)
            
            # Convert flows to PCUs
            pcu_flow = (car_flow * pcu_factors.get("car", 1.0) + 
                       motorbike_flow * pcu_factors.get("motorbike", 0.33))
            
            if pcu_flow <= 0:
                continue
            
            edge = self.graph[u][v][key]
            length_m = edge.get("length", 0)
            length_km = length_m / 1000.0
            
            # Calculate PCU-km for this edge
            pcu_km = pcu_flow * length_km
            all_pcu_km.append(pcu_km)
            total_pcu_km += pcu_km
            
            # Categorize by road type
            highway = edge.get("highway", None)
            if isinstance(highway, list):
                highway = highway[0] if highway else None
            
            categorized = False
            for category, road_types in road_type_categories.items():
                if category == 'other':
                    continue
                if highway in road_types or highway == category:
                    road_type_pcu_km[category] += pcu_km
                    categorized = True
                    break
            
            if not categorized:
                road_type_pcu_km['other'] += pcu_km
        
        # Calculate percentages
        percentages = {}
        for category, pcu_km in road_type_pcu_km.items():
            if total_pcu_km > 0:
                percentages[category] = (pcu_km / total_pcu_km) * 100
            else:
                percentages[category] = 0.0
        
        # Calculate Gini coefficient
        gini_coefficient = self._calculate_gini(all_pcu_km) if all_pcu_km else 0.0
        
        # Prepare results
        results = {
            "total_pcu_km": total_pcu_km,
            "road_type_pcu_km": {k: float(v) for k, v in road_type_pcu_km.items()},
            "percentages": {k: float(v) for k, v in percentages.items()},
            "gini_coefficient": float(gini_coefficient),
            "num_edges": len(all_pcu_km),
            "pcu_factors": pcu_factors
        }
        
        # Save to JSON
        import json
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved PCU analysis results to {output_file}")
        
        # Generate Lorenz curve plot
        self._plot_lorenz_curve(all_pcu_km, gini_coefficient, output_file.replace('.json', '_lorenz.png'))
        
        return results
    
    def _calculate_gini(self, values):
        """
        Calculate Gini coefficient using the standard formula.
        Gini = (∑∑|x_i - x_j|) / (2n² * mean)
        """
        if not values:
            return 0.0
            
        values = np.array(values)
        n = len(values)
        
        if n == 0 or np.sum(values) == 0:
            return 0.0
        
        # Sort for better numerical stability
        values = np.sort(values)
        
        # Calculate using vectorized approach
        indices = np.arange(1, n + 1)
        
        # Gini = (2 * ∑(i * x_i) / (n * ∑x_i)) - (n + 1)/n
        gini = (2 * np.sum(indices * values)) / (n * np.sum(values)) - (n + 1) / n
        
        return gini
    
    def _plot_lorenz_curve(self, values, gini_coefficient, output_file="lorenz_curve.png"):
        """
        Plot and save Lorenz curve for PCU-km distribution.
        """
        if not values:
            logger.warning("No values to plot Lorenz curve")
            return
        
        sorted_values = np.sort(np.array(values))
        n = len(sorted_values)
        cum_values = np.cumsum(sorted_values)
        total = cum_values[-1]
        
        if total == 0:
            return
        
        # Lorenz curve
        lorenz_curve = cum_values / total
        
        # Perfect equality line
        perfect_equality = np.linspace(0, 1, n)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        plt.plot(perfect_equality, perfect_equality, 'k--', label='Perfect Equality', linewidth=2)
        plt.plot(perfect_equality, lorenz_curve, 'b-', label=f'Lorenz Curve (Gini = {gini_coefficient:.3f})', linewidth=2)
        plt.fill_between(perfect_equality, perfect_equality, lorenz_curve, alpha=0.3, color='blue')
        
        plt.xlabel('Cumulative Proportion of Edges', fontsize=12)
        plt.ylabel('Cumulative Proportion of PCU-km', fontsize=12)
        plt.title('Lorenz Curve: Distribution of PCU-km across Road Edges', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper left', fontsize=11)
        
        # Add text box with statistics
        stats_text = f'Total edges: {n:,}\n'
        stats_text += f'Total PCU-km: {total:,.2f}\n'
        stats_text += f'Gini coefficient: {gini_coefficient:.4f}\n'
        stats_text += f'Mean PCU-km: {np.mean(values):.2f}\n'
        stats_text += f'Median PCU-km: {np.median(values):.2f}'
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved Lorenz curve plot to {output_file}")

def haversine_vectorized(lon1, lat1, lon2_arr, lat2_arr):
    # Convert decimal degrees to radians 
    lon1, lat1, lon2_arr, lat2_arr = map(np.radians, [lon1, lat1, lon2_arr, lat2_arr])

    dlon = lon2_arr - lon1
    dlat = lat2_arr - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2_arr) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c  

def main():
    router = VectorRouter("Yogyakarta, Indonesia", cache_dir="./osm_cache")
    
    router.load_network(force_download=False)
    
    router.load_data(
        points_file="data/raw/rea_1000m_v2.geojson",
        car_vectors_file="data/raw/parquet/rea_1000m_car_vectors_v2.parquet",
        motorbike_vectors_file="data/raw/parquet/rea_1000m_motorbike_vectors_v2.parquet"
    )

    router.precompute_nearest_nodes()
    
    router.process_all(output_file='data/raw/rea_1000m_edge_flows_v3.geojson')

    analysis_results = router.analyze_edge_flows(
    pcu_factors={"car": 1.0, "motorbike": 0.33},
    output_file="data/analysis/flow_analysis.json"
    )
    
    logger.info("Complete!")


if __name__ == "__main__":
    main()