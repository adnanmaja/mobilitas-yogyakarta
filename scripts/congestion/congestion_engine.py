from typing import Dict, List, Tuple, Optional
import gc
from scripts.vector_routing.vector_router_model import ImpedanceCalculator, SparseGraphBuilder, GraphManager, Router, VectorRouter, FlowAnalyzer
from congestion_config import Config
from network_engine import NetworkAnalyst
from data_handler import DataHandler

class CongestionEngine:
    def __init__(self, config = Config):
        self.config = config.from_yaml()
        self.congested_time: Optional[float] = None
        self.updated_edges: Optional[List[Dict]] = None

        self.network_analyst = NetworkAnalyst(config)
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