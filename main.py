
from scripts.od_matrix_v2 import ImprovedGravityModel
from scripts.vector_routing_v2 import VectorRouter
from scripts.congestion_v4 import CongestionFeedbackLoop
from typing import Tuple, List, Dict, Optional, Any    
import yaml
from dataclasses import dataclass, fields
import yaml
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# config
@dataclass
class Config:
    data_paths: Dict[str, str]
    export_paths: Dict[str, str]
    congestion_iterations: int

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

class RunTheWholeThing:
    def __init__(self, config = Config):
        self.config = config
        self.gravity_model = ImprovedGravityModel(config=config)
        self.vector_router = VectorRouter("Yogyakarta, Indonesia", cache_dir="./osm_cache", config=config)
        self.congestion = CongestionFeedbackLoop(config)

    def generate_vectors(self):
        self.gravity_model.run()

    def route_vectors(self):
        self.vector_router.load_network(force_download=False)
        self.vector_router.load_data()
        self.vector_router.precompute_nearest_nodes()
        self.vector_router.process_all(output_file=self.config.export_paths['edge_flow'])
        logger.info("Complete!")

    def compute_congestion(self):
        print(f"\n{'='*60}")
        print(f"STARTING CONGESTION FEEDBACK LOOP ({self.config.congestion_iterations} iterations)")
        print('='*60)
        start_time = time.time()

        for iteration in range(self.config.congestion_iterations):
            self.congestion.update_congestion()
            self.congestion.re_route(iteration)

            if iteration > 0:
                if self.congestion.check_convergence():
                    print(f"\nConverged after {iteration + 1} iterations!")
                    break

            elapsed = time.time() - start_time
            print(f"  Iteration completed in {elapsed:.1f} seconds")

        print(f"\n{'='*60}")
        print("FEEDBACK LOOP COMPLETE")
        print('='*60)

        self.congestion.calculate_statistics()
        self.congestion.save_results()

def main():
    config = Config.from_yaml()
    model = RunTheWholeThing(config)

    model.generate_vectors()
    model.route_vectors()
    model.compute_congestion()

if __name__=="__main__":
    main()
     
