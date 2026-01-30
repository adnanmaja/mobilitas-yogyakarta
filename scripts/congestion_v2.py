import json
import random
import numpy as np
from collections import defaultdict
import osmnx as ox
import os
import pickle
import math
from vector_routing_v2 import VectorRouter

# Road-type capacity coefficients (relative capacity per unit length)
CAPACITY_COEFF_BY_TYPE = {
    "trunk": 5.0,
    "primary": 4.0,
    "secondary": 3.0,
    "tertiary": 2.0,
    "residential": 1.0,
    "living_street": 0.8,
    "unclassified": 1.5,
    "service": 0.5,
}
CAPACITY_COEFF_DEFAULT = 1.0  # fallback for unknown types

# Road-type utilization adjustment factors
UTILIZATION_FACTOR_BY_TYPE = {
    "trunk": 0.70,
    "primary": 0.75,
    "secondary": 0.70,
    "tertiary": 0.65,
    "residential": 0.60,
    "living_street": 0.50,
    "unclassified": 0.70,
    "service": 0.45,
}
UTILIZATION_FACTOR_DEFAULT = 0.7  # fallback for unknown types

NOISE = 0.05       # ±5% flow noise
R_MAX = 1.3        # clamp ratio
ALPHA = 0.15        # BPR    
BETA = 2           # BPR 
ITERATIONS = 5  # Number of feedback loop iterations
REASSIGN_METHOD = "proportional"  # "proportional" or "all-or-nothing"
CONVERGENCE_THRESHOLD = 0.01  # 1% change threshold

input_path = "data/raw/rea_1000m_edge_flows_v2.geojson"
output_path = "data/raw/rea_1000m_congestions_v2.geojson"


# Helper function to get bearing between two coordinates
def calculate_bearing(lon1, lat1, lon2, lat2):
    """
    Calculate the bearing between two points in degrees (0-360).
    """
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    
    y = math.sin(lon2 - lon1) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)
    
    bearing = math.atan2(y, x)
    bearing = math.degrees(bearing)
    bearing = (bearing + 360) % 360
    
    return bearing


def bin_bearing(bearing, bin_size=45):
    """
    Bin bearing into discrete directions.
    """
    if bearing is None:
        return 0
    
    return int(bearing // bin_size) * bin_size


def build_edge_graph(features):
    """
    Build a graph structure to find neighboring edges.
    Returns: dict mapping edge index to list of neighbor edge indices.
    """
    # Create a node-to-edges mapping
    node_to_edges = defaultdict(list)
    
    for i, feature in enumerate(features):
        geom = feature.get("geometry", {})
        if geom.get("type") != "LineString":
            continue
            
        coords = geom.get("coordinates", [])
        if len(coords) >= 2:
            # Get start and end nodes (simplified: use coordinates as node IDs)
            start_node = tuple(coords[0])
            end_node = tuple(coords[-1])
            
            node_to_edges[start_node].append(i)
            node_to_edges[end_node].append(i)
    
    # Build neighbor relationships
    edge_neighbors = defaultdict(list)
    
    for node, edges in node_to_edges.items():
        for i in range(len(edges)):
            for j in range(i + 1, len(edges)):
                edge_i = edges[i]
                edge_j = edges[j]
                
                if edge_j not in edge_neighbors[edge_i]:
                    edge_neighbors[edge_i].append(edge_j)
                if edge_i not in edge_neighbors[edge_j]:
                    edge_neighbors[edge_j].append(edge_i)
    
    return edge_neighbors


def calculate_edge_bearing(feature):
    """
    Calculate bearing of an edge from its geometry.
    """
    geom = feature.get("geometry", {})
    if geom.get("type") != "LineString":
        return None
        
    coords = geom.get("coordinates", [])
    if len(coords) < 2:
        return None
    
    # Use first and last coordinate for bearing
    lon1, lat1 = coords[0]
    lon2, lat2 = coords[-1]
    
    return calculate_bearing(lon1, lat1, lon2, lat2)

def update_travel_times(features, congestion_values):
    """
    Update edge travel times based on congestion using BPR formula.
    Returns new travel times and a mapping of edge_id -> new_time.
    """
    edge_times = {}
    
    for i, feature in enumerate(features):
        props = feature["properties"]
        
        # Get original travel time (if exists) or compute from length and speed
        if "time_original" not in props:
            # Estimate original time: length / speed
            length_m = props.get("length_m", 100.0)
            # Assume speed based on road type
            highway_type = props.get("highway")
            if isinstance(highway_type, list):
                highway_type = highway_type[0] if highway_type else "residential"
            
            # Simple speed estimates (km/h)
            speed_map = {
                "trunk": 70,
                "primary": 50,
                "secondary": 40,
                "tertiary": 30,
                "residential": 20,
                "living_street": 10,
                "unclassified": 30,
                "service": 15,
            }
            speed_kph = speed_map.get(highway_type, 30)
            speed_mps = speed_kph / 3.6  # Convert to m/s
            
            # Original time in seconds
            time_original = length_m / speed_mps if speed_mps > 0 else length_m / (30/3.6)
            props["time_original"] = time_original
        else:
            time_original = props["time_original"]
        
        # Apply congestion factor
        if i in congestion_values:
            congestion = congestion_values[i]
            # BPR formula: new_time = free_flow_time * (1 + α*(v/c)^β)
            # We already computed congestion = 1 + α*(utilization)^β
            # So: new_time = time_original * congestion
            new_time = time_original * congestion
        else:
            new_time = time_original
        
        edge_times[i] = new_time
        props["time_updated"] = new_time
    
    return edge_times

def reassign_flows(features, edge_times, router):
    # Update edge impedances in the router's graph
    for i, feature in enumerate(features):
        props = feature["properties"]
        u = props.get("u")
        v = props.get("v")
        
        if u is None or v is None:
            continue
            
        # Find the edge in the graph and update its impedance
        if router.graph.has_edge(u, v):
            for key in router.graph[u][v]:
                # Update with new travel time
                router.graph[u][v][key]['impedance'] = edge_times.get(i, 
                    router.graph[u][v][key].get('length', 0))
    
    # Rebuild sparse graph with updated impedances
    router.build_sparse_graph()
    
    # Clear old flows and re-route
    router.edge_flows.clear()
    router.process_all(output_file='temp_reassigned_flows.geojson', force=True)
    
    # Update flows in features based on new routing
    flow_lookup = {(u, v): flow for (u, v, key), flow in router.edge_flows.items()}

    for i, feature in enumerate(features):
        props = feature["properties"]
        u = props.get("u")
        v = props.get("v")
        
        if u is not None and v is not None:
            # Direct lookup instead of nested loop
            props["flow"] = flow_lookup.get((u, v), 0.0)

print("Initializing router...")
router = VectorRouter("Yogyakarta, Indonesia", cache_dir="./osm_cache")
router.load_network(force_download=False)
router.load_data(
    points_file="data/raw/rea_1000m.geojson",
    vectors_file="data/raw/rea_1000m_vectors_v2.json"
)
router.precompute_nearest_nodes()

# Load data
with open(input_path, "r") as f:
    data = json.load(f)

print(f"Starting congestion feedback loop ({ITERATIONS} iterations)...")

features = data["features"]

for feature in features:
    props = feature["properties"]
    if "flow" in props:
        props["flow_original"] = float(props["flow"])

for iteration in range(ITERATIONS):
    print(f"\n--- Iteration {iteration + 1}/{ITERATIONS} ---")

    for feature in features:
        props = feature["properties"]
        # Keep only essential properties
        keys_to_keep = ["u", "v", "length_m", "highway", "name", "flow", "flow_original"]
        new_props = {}
        for key in keys_to_keep:
            if key in props:
                new_props[key] = props[key]
        feature["properties"] = new_props

    # First pass: add noise to flows and calculate edge bearings
    for i, feature in enumerate(features):
        props = feature["properties"]
        
        # Add noise to flow
        if "flow" not in props or props["flow"] is None:
            continue
        f = float(props["flow"])
        f_noisy = f * (1 + random.uniform(-NOISE, NOISE))
        props["flow_noisy"] = f_noisy
        
        # Calculate bearing for corridor grouping
        bearing = calculate_edge_bearing(feature)
        props["_bearing"] = bearing
        props["_bearing_bin"] = bin_bearing(bearing) if bearing is not None else 0


    # Second pass: compute capacity, utilization, and congestion
    congestion_values = {}
    for i, feature in enumerate(features):
        props = feature["properties"]

        # Skip if no flow
        if "flow_noisy" not in props:
            continue

        f_noisy = props["flow_noisy"]
        
        # Get road length (in meters)
        length_m = float(props.get("length_m", 100.0))  # default to 100m if unknown
        
        # Get highway type
        highway_type = props.get("highway")
        if isinstance(highway_type, list):
            highway_type = highway_type[0] if highway_type else None
        
        # Get capacity coefficient and utilization factor
        capacity_coeff = CAPACITY_COEFF_BY_TYPE.get(highway_type, CAPACITY_COEFF_DEFAULT)
        utilization_factor = UTILIZATION_FACTOR_BY_TYPE.get(highway_type, UTILIZATION_FACTOR_DEFAULT)
        
        # Step 2: Define capacity proxy = road class coefficient * length
        capacity = capacity_coeff * length_m
        
        # Step 3: Compute utilization and apply road-type adjustment
        if capacity > 0:
            utilization = f_noisy / capacity
            # Apply utilization adjustment factor
            utilization *= utilization_factor
        else:
            utilization = 0.0
        
        # Clip utilization ratio
        utilization = min(utilization, R_MAX)
        
        # Step 4: Convert utilization to congestion using BPR (light) - power of 2
        congestion = 1 + ALPHA * (utilization ** BETA)
        
        # Store congestion for smoothing
        congestion_values[i] = congestion
        
        # Store initial results
        props["capacity_est"] = capacity
        props["capacity_coeff"] = capacity_coeff
        props["length_m"] = length_m
        props["utilization"] = utilization
        props["congestion_initial"] = congestion
        props["utilization_factor"] = utilization_factor
        props["highway_type"] = highway_type


    print("Updating travel times...")
    edge_times = update_travel_times(features, congestion_values)

    if iteration < ITERATIONS - 1:
        print("Reassigning flows...")
        reassign_flows(features, edge_times, router)
        print("Routes reassigned")

    # Track convergence
    if iteration > 0:
        # Simple convergence check: compare total flow difference
        total_flow = sum(float(f["properties"].get("flow", 0)) for f in features)
        prev_total_flow = sum(float(f["properties"].get("flow_prev_iter", 0)) for f in features)
        
        if prev_total_flow > 0:
            flow_change = abs(total_flow - prev_total_flow) / prev_total_flow
            print(f"Flow change: {flow_change:.3%}")

            # Optional: early stopping
            if flow_change < CONVERGENCE_THRESHOLD:
                print(f"Converged after {iteration + 1} iterations (change < {CONVERGENCE_THRESHOLD:.1%})")
                break

    for feature in features:
        props = feature["properties"]
        if "flow" in props:
            props["flow_prev_iter"] = float(props["flow"])


print("\n--- Feedback loop complete ---")

# Build edge graph for spatial smoothing
print("Building edge graph for spatial smoothing...")
edge_neighbors = build_edge_graph(features)

# Spatial smoothing
print("Applying spatial smoothing...")
for i, feature in enumerate(features):
    props = feature["properties"]
    
    if "congestion_initial" not in props:
        continue
        
    congestion_initial = props["congestion_initial"]
    
    # Get neighbor congestions
    neighbor_indices = edge_neighbors.get(i, [])
    neighbor_congestions = []
    
    for neighbor_idx in neighbor_indices:
        if neighbor_idx in congestion_values:
            neighbor_congestions.append(congestion_values[neighbor_idx])
    
    # Apply smoothing if we have neighbors
    if neighbor_congestions:
        avg_neighbor_congestion = sum(neighbor_congestions) / len(neighbor_congestions)
        congestion_smooth = (congestion_initial + avg_neighbor_congestion) / 2
    else:
        congestion_smooth = congestion_initial
    
    props["congestion_smooth"] = congestion_smooth
    # Update congestion_values for potential second pass
    congestion_values[i] = congestion_smooth


# Corridor-level congestion
print("Computing corridor-level congestion...")

# Group edges by: road class, name, bearing bin
corridor_groups = defaultdict(list)

for i, feature in enumerate(features):
    props = feature["properties"]
    
    if "congestion_smooth" not in props:
        continue
        
    # Get grouping keys
    highway_type = props.get("highway_type")
    if isinstance(highway_type, list):
        highway_type = highway_type[0] if highway_type else "unknown"
    
    name = props.get("name")
    if isinstance(name, list):
        name = name[0] if name else "unknown"
    name = name or "unknown"
    
    bearing_bin = props.get("_bearing_bin", 0)
    
    # Create corridor key
    corridor_key = (highway_type, name, bearing_bin)
    
    corridor_groups[corridor_key].append({
        "index": i,
        "congestion": props["congestion_smooth"],
        "length": props["length_m"]
    })

# Calculate length-weighted mean congestion for each corridor
corridor_congestion = {}
for corridor_key, edges in corridor_groups.items():
    total_length = 0
    weighted_congestion_sum = 0
    
    for edge in edges:
        weighted_congestion_sum += edge["congestion"] * edge["length"]
        total_length += edge["length"]
    
    if total_length > 0:
        corridor_congestion[corridor_key] = weighted_congestion_sum / total_length
    else:
        corridor_congestion[corridor_key] = 1.0  # default

# Assign corridor congestion back to edges
for corridor_key, edges in corridor_groups.items():
    corridor_cong = corridor_congestion[corridor_key]
    
    for edge in edges:
        feature = features[edge["index"]]
        props = feature["properties"]
        
        props["corridor_congestion"] = corridor_cong
        props["corridor_highway_type"] = corridor_key[0]
        props["corridor_name"] = corridor_key[1]
        props["corridor_bearing_bin"] = corridor_key[2]

        # Final congestion is a blend of smoothed and corridor congestion
        # You can adjust this blend ratio as needed
        SMOOTH_WEIGHT = 0.5
        CORRIDOR_WEIGHT = 0.5
        
        props["congestion_final"] = (
            props["congestion_smooth"] * SMOOTH_WEIGHT + 
            corridor_cong * CORRIDOR_WEIGHT
        )

# Only export select columns to save space
for feature in features:
    props = feature["properties"]
    
    export_props = {}

    if "u" in props:
        export_props["u"] = props["u"]
    if "v" in props:
        export_props["v"] = props["v"]
    if "length_m" in props:
        export_props["length_m"] = props["length_m"]
    if "highway" in props:
        export_props["highway"] = props["highway"]
    if "name" in props:
        export_props["name"] = props["name"]
    if "congestion_final" in props:
        export_props["congestion_final"] = props["congestion_final"]
    
    feature["properties"] = export_props

# Save output
with open(output_path, "w") as f:
    json.dump(data, f, indent=2)

print(f"Saved to {output_path}")
print(f"Processed {len(features)} features")
print(f"Created {len(corridor_groups)} corridor groups")

#w