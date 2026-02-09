import logging
import numpy as np
from scripts.vector_routing.config import Config
from scripts.vector_routing.graph_engine import GraphManager, SparseGraphBuilder, PointSnapper
from scripts.vector_routing.data_handler import DataLoader, FlowExporter
from scripts.vector_routing.router_engine import Router, ImpedanceCalculator
from scripts.vector_routing.stats_engine import FlowAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorRouter:
    """Main orchestrator class that coordinates all components"""
    
    def __init__(self, place_name: str = "Yogyakarta, Indonesia", cache_dir: str = "./cache", config = Config):
        self.place_name = place_name
        self.cache_dir = cache_dir
        self.config = config.from_yaml()
        
        # Initialize components
        self.graph_manager = GraphManager(self.config)
        self.impedance_calc = ImpedanceCalculator(self.config)
        self.sparse_graph_builder = SparseGraphBuilder()
        self.data_loader = DataLoader(self.config)
        self.point_snapper = PointSnapper(cache_dir)
        self.router = Router(self.config)
        self.flow_analyzer = FlowAnalyzer(self.config)
        
        np.random.seed(67)
    
    def load_network(self, force_download: bool = False) -> None:
        self.graph_manager.load_network(force_download)
    
    def load_data(self) -> None:
        self.data_loader.load_points(self.config.data_paths['grid'])
        self.data_loader.load_vectors(
            self.config.data_paths['car_vector'],
            self.config.data_paths['motorbike_vector']
        )
    
    def precompute_nearest_nodes(self, force: bool = False) -> None:
        self.point_snapper.snap_points(self.data_loader.points, self.graph_manager, force)
    
    def process_vehicle_type(self, vehicle_type: str) -> None:
        logger.info(f"Processing {vehicle_type} routes...")
        
        # Add impedance
        self.impedance_calc.add_impedance(self.graph_manager.graph, vehicle_type)
        
        # Build sparse graph
        self.sparse_graph_builder.build_sparse_graph(self.graph_manager.graph, vehicle_type)
        
        # Get vectors
        if vehicle_type == "car":
            vectors = self.data_loader.car_vectors_by_origin
        else:
            vectors = self.data_loader.motorbike_vectors_by_origin
        
        # Route
        self.router.route_vehicle_type(
            vehicle_type,
            vectors,
            self.point_snapper,
            self.sparse_graph_builder,
            self.graph_manager.graph,
            chunk_size=100
        )
    
    def process_all(self, output_file: str = 'routes.geojson') -> None:
        # Process car routes
        self.process_vehicle_type("car")
        
        # Process motorbike routes
        self.process_vehicle_type("motorbike")
        
        logger.info(f"Accumulated flows on {len(self.router.edge_flows)} edges")
        
        # Save edge flows
        FlowExporter.save_edge_flows(
            self.router.edge_flows,
            self.graph_manager.graph,
            output_file,
            self.config
        )
        
        # Analyze flow distribution
        analysis_path = "data/analysis/distribution_analysis.json".replace('.geojson', '')
        self.flow_analyzer.analyze_flow_distribution(
            self.router.edge_flows,
            self.graph_manager.graph,
            output_path=analysis_path,
            plot_lorenz=True
        )




def main():
    router = VectorRouter("Yogyakarta, Indonesia", cache_dir="./osm_cache")
    
    router.load_network(force_download=False)
    
    cfg = Config.from_yaml()

    router.load_data()

    router.precompute_nearest_nodes()
    
    router.process_all(output_file=cfg.export_paths['edge_flow'])
    
    logger.info("Complete!")


if __name__ == "__main__":
    main()