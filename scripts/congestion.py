import json
import random
import numpy as np
from collections import defaultdict

# Road-type assumed utilization parameters
RHO_BY_TYPE = {
    "trunk": 0.70,
    "primary": 0.75,
    "secondary": 0.70,
    "tertiary": 0.65,
    "residential": 0.60,
    "living_street": 0.50,
    "unclassified": 0.70,
    "service": 0.45,
}
RHO_DEFAULT = 0.7  # fallback for unknown types

BETA = 4           # congestion sharpness
NOISE = 0.05       # ±5% flow noise
R_MAX = 1.3        # clamp ratio

input_path = "data/raw/weekend_routed_vectors_1000m_edge_flows.geojson"
output_path = "data/raw/weekend_routed_vectors_1000m_congestions.geojson"


# Load data
with open(input_path, "r") as f:
    data = json.load(f)

# First pass: add noise to flows and build node-to-edges map
node_to_edges = defaultdict(list)

for i, feature in enumerate(data["features"]):
    props = feature["properties"]
    
    # Add noise
    if "flow" not in props or props["flow"] is None:
        continue
    f = float(props["flow"])
    f_noisy = f * (1 + random.uniform(-NOISE, NOISE))
    props["flow_noisy"] = f_noisy
    
    # Map nodes to edge indices
    u = props.get("u")
    v = props.get("v")
    if u is not None:
        node_to_edges[u].append(i)
    if v is not None:
        node_to_edges[v].append(i)

# Build neighbor flows for each edge
neighbor_flows = defaultdict(list)

for i, feature in enumerate(data["features"]):
    props = feature["properties"]
    if "flow_noisy" not in props:
        continue
    
    u = props.get("u")
    v = props.get("v")
    
    # Get all edges connected to either endpoint
    neighbor_indices = set()
    if u is not None:
        neighbor_indices.update(node_to_edges[u])
    if v is not None:
        neighbor_indices.update(node_to_edges[v])
    
    # Remove self
    neighbor_indices.discard(i)
    
    # Collect neighbor flows
    for j in neighbor_indices:
        neighbor_props = data["features"][j]["properties"]
        if "flow_noisy" in neighbor_props:
            neighbor_flows[i].append(neighbor_props["flow_noisy"])

# Second pass: compute congestion with local scaling
for i, feature in enumerate(data["features"]):
    props = feature["properties"]

    # Skip if no flow
    if "flow_noisy" not in props:
        continue

    f_noisy = props["flow_noisy"]

    # Get highway type and corresponding rho
    highway_type = props.get("highway")
    if isinstance(highway_type, list):
        highway_type = highway_type[0] if highway_type else None
    rho = RHO_BY_TYPE.get(highway_type, RHO_DEFAULT)

    # Calculate local scaling factor
    if neighbor_flows[i]:
        median_neighbor_flow = sorted(neighbor_flows[i])[len(neighbor_flows[i]) // 2]
        se = f_noisy / median_neighbor_flow if median_neighbor_flow > 0 else 1.0
        se = np.clip(se, 0.5, 2.0)
    else:
        se = 1.0  # no neighbors, no scaling

    # Infer capacity with local scaling
    C = f_noisy / (rho * se) if f_noisy > 0 else 1.0

    # Flow ratio
    r = f_noisy / C if C > 0 else 0.0
    r = min(r, R_MAX)

    # Congestion score
    congestion = r ** BETA

    # Store results
    props["capacity_est"] = C
    props["flow_ratio"] = r
    props["congestion"] = congestion
    props["rho_used"] = rho
    props["scale_factor"] = se


# Save output
with open(output_path, "w") as f:
    json.dump(data, f, indent=2)

print(f"Saved to {output_path}")