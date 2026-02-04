import json
import os
from typing import Dict, List, Tuple
import copy

class CongestionFeedbackLoop:
    """
    Implements the congestion feedback loop with multi-class traffic assignment
    as described in sections F and G of the README.
    """
    
    def __init__(self, geojson_path: str):
        """
        Initialize with the road network GeoJSON.
        
        Args:
            geojson_path: Path to the GeoJSON file containing road network
        """
        with open(geojson_path, 'r') as f:
            self.geojson = json.load(f)
        
        self.edges = self.geojson['features']
        
        # BPR function parameters (assumed - needs calibration)
        self.bpr_params = {
            'car': {'alpha': 0.15, 'beta': 4.0},
            'motorbike': {'alpha': 0.10, 'beta': 3.5}  # Motorbikes less affected by congestion
        }
        
        # PCU factor for motorbikes (assumed)
        self.pcu_factor = 0.33  # φ in the equation
        
        # Road type to capacity mapping (relative capacities)
        self.road_capacities = {
            "trunk": 5.0,
            "primary": 4.0,
            "secondary": 3.0,
            "tertiary": 2.0,
            "residential": 1.0,
            "living_street": 0.8,
            "unclassified": 1.5,
            "service": 0.5,
        }
        
        # Default capacity if road type not found
        self.default_capacity = 1.5
        
        # Convergence threshold
        self.convergence_threshold = 0.01

    # Debugging
    def debug_edges(self):
        """
        Debug edge properties in the loaded GeoJSON.
        """
        print(f"\n{'='*60}")
        print("CONGESTION MODEL DEBUG")
        print('='*60)
        
        print(f"Total edges loaded: {len(self.edges)}")
        
        # Count edges with flow data
        edges_with_car_flow = sum(1 for e in self.edges if e['properties'].get('car_flow', 0) > 0)
        edges_with_motorbike_flow = sum(1 for e in self.edges if e['properties'].get('motorbike_flow', 0) > 0)
        
        print(f"Edges with car flow: {edges_with_car_flow}/{len(self.edges)}")
        print(f"Edges with motorbike flow: {edges_with_motorbike_flow}/{len(self.edges)}")
        
        # Check first few edges
        print("\nFirst 5 edge properties:")
        for i, edge in enumerate(self.edges[:5]):
            props = edge['properties']
            print(f"\nEdge {i}:")
            print(f"  u: {props.get('u')}, v: {props.get('v')}")
            print(f"  car_flow: {props.get('car_flow', 0):.2f}")
            print(f"  motorbike_flow: {props.get('motorbike_flow', 0):.2f}")
            print(f"  highway: {props.get('highway')}")
            print(f"  length_m: {props.get('length_m')}")
        
        # Check geographic extent
        lats = []
        lons = []
        for edge in self.edges:
            geom = edge.get('geometry')
            if geom and geom.get('type') == 'LineString':
                coords = geom.get('coordinates', [])
                for lon, lat in coords:
                    lats.append(lat)
                    lons.append(lon)
        
        if lats and lons:
            print(f"\nGeographic extent:")
            print(f"  Lat range: {min(lats):.4f} to {max(lats):.4f}")
            print(f"  Lon range: {min(lons):.4f} to {max(lons):.4f}")
        
        # Check for any None values in u,v
        edges_without_uv = sum(1 for e in self.edges if e['properties'].get('u') is None or e['properties'].get('v') is None)
        print(f"\nEdges missing u or v: {edges_without_uv}/{len(self.edges)}")    
    
    def calculate_free_flow_time(self, edge: Dict) -> float:
        """
        Calculate free-flow travel time for an edge.
        
        Args:
            edge: Edge feature dictionary
            
        Returns:
            Free-flow travel time in seconds
        """
        length_m = edge['properties']['length_m']
        highway_type = edge['properties']['highway']

        # Handle potential list
        if isinstance(highway_type, list):
            if highway_type and isinstance(highway_type[0], str):
                highway_type = highway_type[0]
            else:
                highway_type = None
        
        # Speed limits by road type (km/h)
        speed_limits = {
            "trunk": 70,
            "primary": 50,
            "secondary": 40,
            "tertiary": 30,
            "residential": 20,
            "living_street": 10,
            "unclassified": 30,
            "service": 15,
        }
        
        # Get speed limit, default to 30 km/h if not found
        speed_kmh = speed_limits.get(highway_type, 30)
        
        # Convert to m/s
        speed_ms = speed_kmh * 1000 / 3600
        
        # Calculate free-flow time (seconds)
        free_flow_time = length_m / speed_ms
        
        return free_flow_time
    
    def get_road_capacity(self, edge: Dict) -> float:
        """
        Get the capacity for a given road type.
        
        Args:
            edge: Edge feature dictionary
            
        Returns:
            Road capacity in PCU/hour
        """
        highway_type = edge['properties']['highway']
        
        # Handle cases where highway type might be a list
        if isinstance(highway_type, list):
            # Take the first element and ensure it's a string
            if highway_type and isinstance(highway_type[0], str):
                highway_type = highway_type[0]
            else:
                # If it's an empty list or not a string, use default
                return self.default_capacity
        elif not isinstance(highway_type, str):
            # If it's not a string or list, use default
            return self.default_capacity
        
        return self.road_capacities.get(highway_type, self.default_capacity)
    
    def calculate_effective_volume(self, car_flow: float, motorbike_flow: float) -> float:
        """
        Calculate effective traffic volume using Passenger Car Units (PCU).
        
        Args:
            car_flow: Car traffic volume
            motorbike_flow: Motorbike traffic volume
            
        Returns:
            Effective volume in PCU
        """
        return car_flow + (self.pcu_factor * motorbike_flow)
    
    def bpr_function(self, t0: float, v: float, c: float, alpha: float, beta: float) -> float:
        """
        Bureau of Public Roads (BPR) function for congestion modeling.
        
        Args:
            t0: Free-flow travel time
            v: Traffic volume (PCU)
            c: Road capacity (PCU/hour)
            alpha: BPR alpha parameter
            beta: BPR beta parameter
            
        Returns:
            Congested travel time
        """
        if c <= 0:
            return t0
        
        vc_ratio = v / c
        return t0 * (1 + alpha * (vc_ratio ** beta))
    
    def update_congestion(self, edges: List[Dict]) -> List[Dict]:
        """
        Update congestion levels for all edges based on current flows.
        
        Args:
            edges: List of edge features with current flow data
            
        Returns:
            Updated edges with new travel times
        """
        updated_edges = []

        debug_count = 0
        max_debug = 4
        
        for i, edge in enumerate(edges):
            props = edge['properties']
            
            # Get current flows
            car_flow = props.get('car_flow', 0)
            motorbike_flow = props.get('motorbike_flow', 0) 
            
            # Calculate effective volume
            effective_volume = self.calculate_effective_volume(car_flow, motorbike_flow)
            
            # Get free-flow time and capacity
            free_flow_time = self.calculate_free_flow_time(edge)
            capacity = self.get_road_capacity(edge)

            if debug_count < max_debug:
                print(f"\n= DEBUG1 Edge {i}")
                print(f"car_flow: {car_flow}")
                print(f"motorbike_flow: {motorbike_flow}")
                print(f"effective_volume: {effective_volume}")
                print(f"free_flow_time: {free_flow_time}")
                print(f"capacity: {capacity}")
                debug_count += 1
                
            
            # Calculate congested travel times for each mode
            car_alpha = self.bpr_params['car']['alpha']
            car_beta = self.bpr_params['car']['beta']
            motorbike_alpha = self.bpr_params['motorbike']['alpha']
            motorbike_beta = self.bpr_params['motorbike']['beta']
            
            # Calculate travel times
            car_travel_time = self.bpr_function(
                free_flow_time, effective_volume, capacity, car_alpha, car_beta
            )
            
            motorbike_travel_time = self.bpr_function(
                free_flow_time, effective_volume, capacity, motorbike_alpha, motorbike_beta
            )
            
            # Store updated travel times
            updated_props = props.copy()
            updated_props['car_travel_time'] = car_travel_time
            updated_props['motorbike_travel_time'] = motorbike_travel_time
            updated_props['effective_volume'] = effective_volume
            updated_props['vc_ratio'] = effective_volume / capacity if capacity > 0 else 0
            
            # Create updated edge
            updated_edge = edge.copy()
            updated_edge['properties'] = updated_props
            updated_edges.append(updated_edge)
        
        return updated_edges
    
    def check_convergence(self, old_edges: List[Dict], new_edges: List[Dict]) -> bool:
        """
        Check if the assignment has converged.
        
        Args:
            old_edges: Previous iteration's edges
            new_edges: Current iteration's edges
            
        Returns:
            True if converged, False otherwise
        """
        max_change = 0
        
        for old_edge, new_edge in zip(old_edges, new_edges):
            old_car_time = old_edge['properties'].get('car_travel_time', 0)
            new_car_time = new_edge['properties'].get('car_travel_time', 0)
            
            if old_car_time > 0:
                change = abs(new_car_time - old_car_time) / old_car_time
                max_change = max(max_change, change)
        
        return max_change < self.convergence_threshold
    
    def run_feedback_loop(self, max_iterations: int = 10) -> Dict:
        """
        Run the complete congestion feedback loop.
        
        Args:
            max_iterations: Maximum number of iterations
            
        Returns:
            Updated GeoJSON with final congestion levels
        """
        print("Starting congestion feedback loop...")
        
        # Initial edges (with initial flows from OD assignment)
        current_edges = self.edges
        
        for iteration in range(max_iterations):
            print(f"\nIteration {iteration + 1}/{max_iterations}")
            
            # Store previous edges for convergence check
            previous_edges = copy.deepcopy(current_edges)
            
            # Update congestion based on current flows
            updated_edges = self.update_congestion(current_edges)
            
            # Here you would re-route OD flows using the updated travel times
            # This requires a routing engine - for now, we'll simulate this
            # by adjusting flows based on the new travel times
            adjusted_edges = self.adjust_flows_based_on_congestion(updated_edges)
            
            # Update current edges
            current_edges = adjusted_edges
            
            # Check for convergence
            if iteration > 0 and self.check_convergence(previous_edges, current_edges):
                print(f"Converged after {iteration + 1} iterations")
                break
        
        # Create final GeoJSON
        final_geojson = self.geojson.copy()
        final_geojson['features'] = current_edges
        
        # Calculate summary statistics
        self.calculate_statistics(current_edges)
        
        return final_geojson
    
    def adjust_flows_based_on_congestion(self, edges: List[Dict], routing_module) -> List[Dict]:
        """
        Re-route OD flows using updated travel times.
        
        Args:
            edges: Current edges with congestion
            routing_module: VectorRouter instance for re-routing
                
        Returns:
            Edges with re-routed flows
        """
        print("  Re-routing based on updated travel times...")

        print(f"  DEBUG: Initial edge_flows count: {len(routing_module.edge_flows)}")
        if routing_module.edge_flows:
            sample_key = list(routing_module.edge_flows.keys())[0]
            print(f"  DEBUG: Sample flow before clear: {routing_module.edge_flows[sample_key]}")
        
        # Save current congestion state to temporary file
        temp_geojson = "temp_congestion.geojson"
        temp_data = {
            'type': 'FeatureCollection',
            'features': edges
        }
        
        with open(temp_geojson, 'w') as f:
            json.dump(temp_data, f)
        
        # Update routing module with new travel times
        routing_module.update_impedances_from_congestion(temp_geojson)
        routing_module.rebuild_sparse_graphs()
        
        # Clear previous flows
        old_flows = routing_module.edge_flows.copy()
        routing_module.edge_flows.clear()
        print(f"  DEBUG: Cleared {len(old_flows)} flows")
        
        # Re-route
        print("  Re-routing car trips...")
        routing_module.process_car()
        print(f"  DEBUG: After process_car, edge_flows count: {len(routing_module.edge_flows)}")
        
        print("  Re-routing motorbike trips...")
        routing_module.process_motorbike()
        print(f"  DEBUG: After process_motorbike, edge_flows count: {len(routing_module.edge_flows)}")

        flow_lookup = {}
        for (edge_u, edge_v, edge_id), flows in routing_module.edge_flows.items():
            if edge_id is not None:
                flow_lookup[edge_id] = {
                    'car_flow': flows.get('car_flow', 0),
                    'motorbike_flow': flows.get('motorbike_flow', 0)
                }
            else:
                # Fallback to (u, v) if no edge_id
                flow_lookup[(edge_u, edge_v)] = {
                    'car_flow': flows.get('car_flow', 0),
                    'motorbike_flow': flows.get('motorbike_flow', 0)
                }
            
        # Get updated edge flows
        debug_count = 0
        max_debug = 5

        sample_keys = list(routing_module.edge_flows.keys())[:5]
        print(f"\n  DEBUG: Sample edge_flows keys: {sample_keys}")
        print(f"  DEBUG: Total edge_flows: {len(routing_module.edge_flows)}")

        updated_edges = []
        for i, edge in enumerate(edges):
            props = edge['properties']
            u = props.get('u')
            v = props.get('v')
            edge_id = props.get('id') or props.get('edge_id') or props.get('osmid')
            
            flows = {'car_flow': 0, 'motorbike_flow': 0}

            # Try to find matching flow by checking all possible keys
            found = False
            if u is not None and v is not None:
                # Try different key values (0, 1, 2, etc. for multi-edges)
                for key_val in [0, 1, 2, edge_id]:  # Include edge_id as potential key
                    edge_key = (u, v, key_val)
                    if edge_key in routing_module.edge_flows:
                        flows = routing_module.edge_flows[edge_key]
                        found = True
                        if debug_count < max_debug:
                            print(f"  Found with key_val={key_val}: {flows}")
                        break
                
                # If not found, maybe it's using a different key format
                if not found:
                    # Check all keys that match (u, v) regardless of third element
                    matching_keys = [k for k in routing_module.edge_flows.keys() 
                                if k[0] == u and k[1] == v]
                    if matching_keys:
                        # Take the first matching key
                        flows = routing_module.edge_flows[matching_keys[0]]
                        found = True
                        if debug_count < max_debug:
                            print(f"  Found among {len(matching_keys)} matching keys: {flows}")
                            
            if flows is None and u is not None and v is not None:
                flows = flow_lookup.get((u, v))
            if flows is None and u is not None and v is not None and edge_id is not None:
                flows = flow_lookup.get((u, v, edge_id))
            car_flow = flows['car_flow']
            motorbike_flow = flows['motorbike_flow']
            
            # Update properties
            updated_props = props.copy()
            updated_props['car_flow'] = car_flow
            updated_props['motorbike_flow'] = motorbike_flow
            updated_props['total_flow'] = car_flow + motorbike_flow
            
            updated_edge = edge.copy()
            updated_edge['properties'] = updated_props
            updated_edges.append(updated_edge)

            if debug_count <= max_debug:
                print(f"\n= [DEBUG][ADJUST] Edge {i}")
                print(f"car_flow: {car_flow}")
                print(f"motorbike_flow: {motorbike_flow}")
                debug_count += 1
                
        
        # Clean up temp file
        if os.path.exists(temp_geojson):
            os.remove(temp_geojson)
        
        return updated_edges
    
    def calculate_statistics(self, edges: List[Dict]):
        """
        Calculate and print summary statistics about the network.
        
        Args:
            edges: Final edges after feedback loop
        """
        total_car_flow = 0
        total_motorbike_flow = 0
        total_length = 0
        congested_segments = 0
        total_segments = len(edges)
        
        for edge in edges:
            props = edge['properties']
            
            total_car_flow += props.get('car_flow', 0)
            total_motorbike_flow += props.get('motorbike_flow', 0)
            total_length += props.get('length_m', 0)
            
            vc_ratio = props.get('vc_ratio', 0)
            if vc_ratio > 0.8:
                congested_segments += 1
        
        print("\n" + "="*50)
        print("NETWORK STATISTICS")
        print("="*50)
        print(f"Total car flow: {total_car_flow:.0f} vehicles")
        print(f"Total motorbike flow: {total_motorbike_flow:.0f} vehicles")
        print(f"Total effective flow: {total_car_flow + total_motorbike_flow * self.pcu_factor:.0f} PCU")
        print(f"Total network length: {total_length/1000:.1f} km")
        print(f"Congested segments (v/c > 0.8): {congested_segments}/{total_segments} ({congested_segments/total_segments*100:.1f}%)")
        print("="*50)
    
    def save_results(self, geojson: Dict, output_path: str):
        """
        Save the results to a GeoJSON file.
        
        Args:
            geojson: GeoJSON data to save
            output_path: Output file path
        """
        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")


def main():
    """
    Main function to run the congestion feedback loop.
    """
    # Configuration
    INPUT_GEOJSON = "data/raw/rea_1000m_edge_flows_v3.geojson"  
    OUTPUT_GEOJSON = "data/raw/rea_1000m_congestions_v4.geojson"
    
    # Initialize the congestion feedback loop
    feedback_loop = CongestionFeedbackLoop(INPUT_GEOJSON)
    
    # Run the feedback loop
    result_geojson = feedback_loop.run_feedback_loop(max_iterations=10)
    
    # Save results
    feedback_loop.save_results(result_geojson, OUTPUT_GEOJSON)


if __name__ == "__main__":
    main()