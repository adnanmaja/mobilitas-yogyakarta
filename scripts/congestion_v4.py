import json
import os
from typing import Dict, List, Tuple, Optional
import time
import gc
from vector_routing_v2 import ImpedanceCalculator, SparseGraphBuilder, GraphManager, Router, VectorRouter, FlowAnalyzer
import yaml
from dataclasses import dataclass, fields

# Configuration
@dataclass
class Config:
    bpr: Dict[str, Dict[str, float]]
    motorbike_pcu: float
    road_capacities: Dict[str, float]
    default_capacity: float
    convergence_threshold: float
    speed_limits: Dict[str, int]
    default_speed_limit: int 
    congestion_iterations: int
    vc_cap: float
    export_paths: Dict[str, str]
    cache_paths: Dict[str, str]

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

class DataHandler:
    def __init__(self):
        self.config = Config.from_yaml()
        self.geojson = {}

    def load_data(self, edge_flows_path):
        """ Load edge flows data in GeoJSON """

        with open(edge_flows_path, 'r') as f:
            self.geojson = json.load(f)

    @staticmethod
    def save_results(geojson: Dict, output_path: str):
        """ Save the results to a GeoJSON file. """

        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")

    @staticmethod
    def caching(temp_path, data):
        with open(temp_path, 'w') as f:
            json.dump(data, f)

    @staticmethod
    def clear_cache(temp_path):
        if os.path.exists(temp_path):
            os.remove(temp_path)


class NetworkAnalyst:
    def __init__(self):
        self.config = Config.from_yaml()
        self.free_flow_time: Optional[float] = None
        self.road_capacity: Optional[float] = None
        self.effective_volume: Optional[float] = None

    def calculate_free_flow_time(self, edge: Dict) -> float:
        """ Calculate free-flow travel time for an edge. """
        
        length_m = edge['properties']['length_m']
        highway_type = edge['properties']['highway']

        # Handle potential list
        if isinstance(highway_type, list):
            if highway_type and isinstance(highway_type[0], str):
                highway_type = highway_type[0]
            else:
                highway_type = None
        
        # Get speed limit
        speed_kmh = self.config.speed_limits.get(highway_type, self.config.default_speed_limit)
        
        # Convert to m/s
        speed_ms = speed_kmh * 1000 / 3600
        
        # Calculate free-flow time (seconds)
        free_flow_time = length_m / speed_ms
        self.free_flow_time = free_flow_time
        
        return self.free_flow_time
    
    def get_road_capacity(self, edge: Dict) -> float:
        """ Get the capacity for a given road type. """
        
        highway_type = edge['properties']['highway']
        
        # Handle cases where highway type might be a list
        if isinstance(highway_type, list):
            if highway_type and isinstance(highway_type[0], str):
                highway_type = highway_type[0]
            else:
                return self.config.default_capacity
        elif not isinstance(highway_type, str):
            return self.config.default_capacity
        
        self.road_capacity = self.config.road_capacities.get(highway_type, self.config.default_capacity)
        return self.road_capacity

    def calculate_effective_volume(self, car_flow: float, motorbike_flow: float) -> float:
        """ Calculate effective traffic volume using Passenger Car Units (PCU). """

        return car_flow + (self.config.motorbike_pcu * motorbike_flow)


class CongestionEngine:
    def __init__(self, config: Config):
        self.config = config
        self.congested_time: Optional[float] = None
        self.updated_edges: Optional[List[Dict]] = None

        self.network_analyst = NetworkAnalyst()
        self.data_handler = DataHandler()
        self.impedance_calculator = ImpedanceCalculator(self.config)
        self.graph_manager = GraphManager()
        self.sparse_grapher = SparseGraphBuilder()
        self.router = Router(self.config)
        self.routing_module = VectorRouter()
        self.flow_Analyzer = FlowAnalyzer(self.config)

    def bpr_function(self, t0: float, v: float, c: float, alpha: float, beta: float) -> float:
        """ Bureau of Public Roads (BPR) function for congestion modeling. """

        if c <= 0:
            return t0
        
        vc_ratio = v / c
        self.congested_time = t0 * (1 + alpha * min((vc_ratio ** beta), self.config.vc_cap))
        return self.congested_time
    
    def update_congestion(self, edges: List[Dict]) -> List[Dict]:
        """ Update congestion levels for all edges based on current flows. """
        updated_edges = []
        
        for i, edge in enumerate(edges):
            props = edge['properties']
            
            # Get current flows
            car_flow = props.get('car_flow', 0)
            motorbike_flow = props.get('motorbike_flow', 0) 
            
            effective_volume = self.network_analyst.calculate_effective_volume(car_flow, motorbike_flow)
            
            # Get free-flow time and capacity
            free_flow_time = self.network_analyst.calculate_free_flow_time(edge)
            capacity = self.network_analyst.get_road_capacity(edge)
                        
            # Calculate travel times
            car_travel_time = self.bpr_function(
                free_flow_time, effective_volume, capacity, self.config.bpr['car']['alpha'], self.config.bpr['car']['beta']
            )
            
            motorbike_travel_time = self.bpr_function(
                free_flow_time, effective_volume, capacity, self.config.bpr['motorbike']['alpha'], self.config.bpr['motorbike']['beta']
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
        
        self.updated_edges = updated_edges
        return updated_edges
    
    def check_convergence(self, old_edges: List[Dict], new_edges: List[Dict]) -> bool:
        """ Check if the assignment has converged. """
        max_change = 0
        total_change = 0
        checked_edges = 0
        
        for old_edge, new_edge in zip(old_edges, new_edges):
            old_props = old_edge['properties']
            new_props = new_edge['properties']
            
            # Check car travel time
            old_car_time = old_props.get('car_travel_time', 0)
            new_car_time = new_props.get('car_travel_time', 0)
            
            # Check motorbike travel time
            old_bike_time = old_props.get('motorbike_travel_time', 0)
            new_bike_time = new_props.get('motorbike_travel_time', 0)
            
            # Check flows for detecting rerouting
            old_car_flow = old_props.get('car_flow', 0)
            new_car_flow = new_props.get('car_flow', 0)
            
            if old_car_time > 0:
                # Calculate relative change in travel time
                time_change = abs(new_car_time - old_car_time) / old_car_time
                max_change = max(max_change, time_change)
                total_change += time_change
                checked_edges += 1
                
            if old_car_flow > 0 or new_car_flow > 0:
                # Calculate flow change (handle division by zero)
                if old_car_flow + new_car_flow > 0:
                    flow_change = abs(new_car_flow - old_car_flow) / max(1, (old_car_flow + new_car_flow)/2)
                    max_change = max(max_change, flow_change)
        
        avg_change = total_change / max(1, checked_edges)
        
        print(f"    Convergence check: max_change={max_change:.4f}, avg_change={avg_change:.4f}, threshold={self.config.convergence_threshold}")
        
        return max_change < self.config.convergence_threshold
    
    def adjust_flows_based_on_congestion(self, edges: List[Dict], iteration) -> List[Dict]:
        """ Re-route OD flows using updated travel times. """

        print("  Re-routing based on updated travel times...")
        
        # Save current congestion state to temporary file
        temp_data = {
            'type': 'FeatureCollection',
            'features': edges
        }
        
        self.data_handler.caching(self.config.cache_paths['congetsion'], data=temp_data)
        
        # Update routing module with new travel times
        self.impedance_calculator.update_impedances_from_congestion(self.config.cache_paths['congetsion'])
        self.sparse_grapher.build_sparse_graph(self.graph_manager.graph, "car")
        self.sparse_grapher.build_sparse_graph(self.graph_manager.graph, "motorbike")
        
        # Clear previous flows
        self.router.edge_flows.clear()
        
        # Re-route both vehicle types
        print("  Re-routing car trips...")
        self.routing_module.process_vehicle_type("car")
        
        print("  Re-routing motorbike trips...")
        self.routing_module.process_vehicle_type("motorbike")

        pcu_km_by_edge, _ = self.flow_Analyzer.calculate_pcu_km(self.router.edge_flows, self.graph_manager.graph) 
        gini = self.flow_Analyzer.calculate_gini_coefficient(list(pcu_km_by_edge.values())) 
        print(f"Gini coffecient for iteration {iteration+1}: {float(gini)}")  
        
        # Create a more robust edge ID lookup
        edge_id_map = {}
        for edge in edges:
            props = edge['properties']
            u = props.get('u')
            v = props.get('v')
            edge_id = props.get('id') or props.get('edge_id') or props.get('osmid')
            
            # Create multiple possible lookup keys
            if u is not None and v is not None:
                # Key format used by OSMnx: (u, v, key) where key is usually 0
                key = props.get('key', 0)
                edge_id_map[(u, v, key)] = edge
                # Also map by edge_id if available
                if edge_id:
                    edge_id_map[edge_id] = edge
        
        # Update edges with new flows
        updated_edges = []
        edge_flows = self.router.edge_flows
        
        if not edge_flows:
            print("  WARNING: No edge flows returned from routing!")
            return edges
        
        # Debug: print sample of flows
        sample_keys = list(edge_flows.keys())[:3]
        print(f"  DEBUG: Sample flow keys: {sample_keys}")
        print(f"  DEBUG: Total flow entries: {len(edge_flows)}")
        
        flow_lookup = {}
        for (u_key, v_key, *rest), flow_data in edge_flows.items():
            flow_lookup[(u_key, v_key)] = flow_data

        matched_count = 0
        unmatched_count = 0

        n = iteration + 1
        new_weight = 1.0 / n
        old_weight = 1.0 - new_weight
        
        for edge in edges:
            props = edge['properties'].copy()
            u = props.get('u')
            v = props.get('v')

            old_car_flow = props.get('car_flow', 0)
            old_motorbike_flow = props.get('motorbike_flow', 0)
            
            found_flow = flow_lookup.get((u, v))
            
            if found_flow:
                found_car_flow = found_flow.get('car_flow', 0)
                found_motorbike_flow = found_flow.get('motorbike_flow', 0)
                
                props['car_flow'] = (old_weight * props.get('car_flow', 0)) + (new_weight * found_car_flow)
                props['motorbike_flow'] = (old_weight * props.get('motorbike_flow', 0)) + (new_weight * found_motorbike_flow)
                matched_count += 1
            else:
                unmatched_count += 1
            
            props['total_flow'] = props.get('car_flow', 0) + props.get('motorbike_flow', 0)
            edge['properties'] = props
            updated_edges.append(edge)
        
        print(f"  DEBUG: Matched {matched_count} edges, {unmatched_count} unmatched")
        
        # Clean up temp file
        self.data_handler.clear_cache(self.config.cache_paths['congestion'])

        self.updated_edges = updated_edges
        gc.collect()    
        return updated_edges
    

class Analytics:
    def __init__(self):
        self.config = Config.from_yaml()

    def calculate_statistics(self, edges: List[Dict]):
        """ Calculate and print summary statistics including mean and median travel times. """
        import statistics

        car_times = []
        bike_times = []
        
        # For flow-weighted averages
        total_car_travel_time = 0
        total_car_volume = 0
        total_bike_travel_time = 0
        total_bike_volume = 0

        congested_segments = 0
        total_segments = len(edges)
        
        for edge in edges:
            props = edge['properties']
            c_time = props.get('car_travel_time', 0)
            b_time = props.get('motorbike_travel_time', 0)
            c_flow = props.get('car_flow', 0)
            b_flow = props.get('motorbike_flow', 0)

            # Collect for Median and Simple Mean
            if c_time > 0: car_times.append(c_time)
            if b_time > 0: bike_times.append(b_time)

            # Collect for Flow-Weighted Mean
            total_car_travel_time += (c_time * c_flow)
            total_car_volume += c_flow
            total_bike_travel_time += (b_time * b_flow)
            total_bike_volume += b_flow
            
            if props.get('vc_ratio', 0) > 0.8:
                congested_segments += 1
        
        # Calculate Stats
        avg_car = statistics.mean(car_times) if car_times else 0
        med_car = statistics.median(car_times) if car_times else 0
        avg_bike = statistics.mean(bike_times) if bike_times else 0
        med_bike = statistics.median(bike_times) if bike_times else 0
        
        weighted_car = total_car_travel_time / total_car_volume if total_car_volume > 0 else 0
        weighted_bike = total_bike_travel_time / total_bike_volume if total_bike_volume > 0 else 0

        print("\n" + "="*50)
        print("NETWORK STATISTICS")
        print("="*50)
        print(f"CONGESTION: {congested_segments}/{total_segments} segments at v/c > 0.8")
        
        print("\nCAR TRAVEL TIMES (seconds per segment):")
        print(f"  Average: {avg_car:.2f}s")
        print(f"  Median:  {med_car:.2f}s")
        print(f"  Weighted Average: {weighted_car:.2f}s (based on flow)")

        print("\nMOTORBIKE TRAVEL TIMES (seconds per segment):")
        print(f"  Average: {avg_bike:.2f}s")
        print(f"  Median:  {med_bike:.2f}s")
        print(f"  Weighted Average: {weighted_bike:.2f}s (based on flow)")
        print("="*50)

class CongestionFeedbackLoop:
    """ 
    Main orchestrator class that coordinates all components.
    Implements the congestion feedback loop with multi-class traffic assignment
    as described in sections F and G of the README. """
    
    def __init__(self, config = Config):
        self.config = config
        self.data_handler = DataHandler()
        self.congestion_engine = CongestionEngine(self.config)
        self.graph_manager = GraphManager()
        self.analytics = Analytics()

        self.edges = {}
        self.updated_edges = {}
        self.geojson = {}

    def load_edge_flows(self, path):
        self.geojson = self.data_handler.load_data(path)
        self.edges = self.geojson['features']

    def load_network(self, force_download: bool = False) -> None:
        self.graph_manager.load_network(force_download)

    def calculate_congestion(self) :
        current_edges = self.edges.copy()
        self.updated_edges = self.congestion_engine.update_congestion(current_edges)
    
    def re_route(self, iteration: int) -> List[Dict]:
        re_routed_edges = self.congestion_engine.adjust_flows_based_on_congestion(
                self.updated_edges, iteration
            )
        return re_routed_edges
    
    def update_congestion(self):
        self.edges = self.re_route()

    def check_convergence(self) -> bool:
        return self.congestion_engine.check_convergence(self.edges, self.updated_edges)
    
    def calculate_statistics(self):
        self.analytics.calculate_statistics(edges=self.updated_edges)
    
    def save_results(self):
        self.data_handler.save_results(self.geojson, self.config.export_paths['congestion'])


def main():
    config = Config.from_yaml()
    congestion = CongestionFeedbackLoop(config)
    router = VectorRouter()

    router.load_data()
    router.load_network(force_download=False)
    router.precompute_nearest_nodes(force=False)
    router.process_all(output_file=config.cache_paths['congestion'])

    print(f"\n{'='*60}")
    print(f"STARTING CONGESTION FEEDBACK LOOP ({config.congestion_iterations} iterations)")
    print('='*60)
    start_time = time.time()

    for iteration in range(config.congestion_iterations):
        congestion.update_congestion()
        congestion.re_route(iteration)

        if iteration > 0:
            if congestion.check_convergence():
                print(f"\nConverged after {iteration + 1} iterations!")
                break

        elapsed = time.time() - start_time
        print(f"  Iteration completed in {elapsed:.1f} seconds")

    print(f"\n{'='*60}")
    print("FEEDBACK LOOP COMPLETE")
    print('='*60)

    congestion.calculate_statistics()
    congestion.save_results()
    
    
    
if __name__ == "__main__":
    main()
    