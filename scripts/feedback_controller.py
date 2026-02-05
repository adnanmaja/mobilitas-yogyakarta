import json
from vector_routing_v2 import VectorRouter
from congestion_v4 import CongestionFeedbackLoop
import time

class FeedbackController:
    """Orchestrates the congestion feedback loop with actual re-routing"""
    
    def __init__(self, place_name: str, points_file: str, 
                 car_vectors_file: str, motorbike_vectors_file: str):
        # Initialize router
        self.router = VectorRouter(place_name, cache_dir="./osm_cache")
        self.router.load_network(force_download=False)
        self.router.load_data(points_file, car_vectors_file, motorbike_vectors_file)
        self.router.precompute_nearest_nodes()
        
        # Initial routing (produces initial edge flows)
        print("Initial routing...")
        self.router.process_all(output_file='temp_initial_flows.geojson', force=True)
        
        # Load initial flows into congestion model
        self.congestion = CongestionFeedbackLoop('temp_initial_flows.geojson')
    
    def run_iterations(self, max_iterations: int = 5):
        """Run the full feedback loop"""
        
        print(f"\n{'='*60}")
        print(f"STARTING CONGESTION FEEDBACK LOOP ({max_iterations} iterations)")
        print('='*60)
        
        for iteration in range(max_iterations):
            print(f"\n[Iteration {iteration + 1}/{max_iterations}]")
            start_time = time.time()
            
            # 1. Update congestion based on current flows
            print("  1. Calculating congestion...")
            current_edges = self.congestion.edges
            updated_edges = self.congestion.update_congestion(current_edges)
            
            # 2. Re-route based on new travel times
            print("  2. Re-routing...")
            re_routed_edges = self.congestion.adjust_flows_based_on_congestion(
                updated_edges, self.router
            )
            
            # 3. Update congestion model with new flows
            self.congestion.edges = re_routed_edges
            
            # Check convergence
            if iteration > 0:
                if self.congestion.check_convergence(previous_edges, re_routed_edges):
                    print(f"\nConverged after {iteration + 1} iterations!")
                    break
            
            previous_edges = current_edges.copy()
            
            elapsed = time.time() - start_time
            print(f"  Iteration completed in {elapsed:.1f} seconds")
        
        print(f"\n{'='*60}")
        print("FEEDBACK LOOP COMPLETE")
        print('='*60)
        
        # Calculate and display statistics
        self.congestion.calculate_statistics(self.congestion.edges)
        
        return self.congestion.geojson
    
    def save_results(self, output_path: str):
        """Save final results"""
        final_geojson = self.congestion.geojson.copy()
        final_geojson['features'] = self.congestion.edges
        
        with open(output_path, 'w') as f:
            json.dump(final_geojson, f, indent=2)
        print(f"\nResults saved to: {output_path}")


def main():
    PLACE_NAME = "Yogyakarta, Indonesia"
    POINTS_FILE = "data/raw/rea_1000m_v2.geojson"
    CAR_VECTORS_FILE = "data/raw/rea_1000m_car_vectors_v2.parquet"
    MOTORBIKE_VECTORS_FILE = "data/raw/rea_1000m_motorbike_vectors_v2.parquet"
    OUTPUT_FILE = "data/raw/rea_1000m_congestions_v4.geojson"
    
    # Run feedback loop
    controller = FeedbackController(
        PLACE_NAME, POINTS_FILE, CAR_VECTORS_FILE, MOTORBIKE_VECTORS_FILE
    )
    
    result = controller.run_iterations(max_iterations=5)
    controller.save_results(OUTPUT_FILE)


if __name__ == "__main__":
    main()