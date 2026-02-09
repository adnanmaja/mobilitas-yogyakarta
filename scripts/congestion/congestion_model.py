from typing import Dict, List, Tuple, Optional
import time
import sys
import os
from scripts.vector_routing.vector_router_model import GraphManager, VectorRouter
from congestion_config import Config
from data_handler import DataHandler
from congestion_engine import CongestionEngine
from analytics import Analytics

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
    