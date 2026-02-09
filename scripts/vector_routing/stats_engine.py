import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
from scripts.vector_routing.config import Config


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FlowAnalyzer:
    """Analyzes traffic flow distributions and generates statistics"""
    
    def __init__(self, config: Config):
        self.config = config.from_yaml()
    
    def calculate_pcu_km(self, edge_flows: Dict, graph: Any) -> Tuple[Dict, Dict]:
        """Calculate PCU-km for each edge"""
        pcu_km_by_edge = {}
        pcu_km_by_road_class = defaultdict(float)
        
        for (u, v, key), flow_dict in edge_flows.items():
            car_flow = flow_dict.get("car_flow", 0)
            motorbike_flow = flow_dict.get("motorbike_flow", 0)
            
            # Convert to PCU
            pcu_flow = (car_flow * self.config.pcu['car'] + 
                        motorbike_flow * self.config.pcu['motorbike'])
            
            # Get edge length in km
            edge = graph[u][v][key]
            length_km = edge.get("length", 0) / 1000.0
            
            # Calculate PCU-km
            pcu_km = pcu_flow * length_km
            
            if pcu_km > 0:
                pcu_km_by_edge[(u, v, key)] = pcu_km
                
                # Group by road class
                highway = edge.get("highway", "unclassified")
                if isinstance(highway, list):
                    highway = highway[0]
                
                # Categorize
                if highway in ['motorway', 'trunk']:
                    road_class = 'trunk'
                elif highway == 'primary':
                    road_class = 'primary'
                elif highway == 'secondary':
                    road_class = 'secondary'
                elif highway == 'tertiary':
                    road_class = 'tertiary'
                else:
                    road_class = 'other'
                
                pcu_km_by_road_class[road_class] += pcu_km
        
        return pcu_km_by_edge, pcu_km_by_road_class
    
    def calculate_gini_coefficient(self, values: List[float]) -> float:
        """Calculate Gini coefficient for a list of values"""
        if not values:
            return 0.0
        
        # Sort values
        sorted_values = np.sort(np.array(values))
        n = len(sorted_values)
        
        # Gini formula
        index = np.arange(1, n + 1)
        gini = (np.sum((2 * index - n - 1) * sorted_values)) / (n * np.sum(sorted_values))
        
        return gini
    
    def generate_lorenz_curve_data(self, values: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Generate data points for Lorenz curve"""
        if not values:
            return np.array([]), np.array([])
        
        sorted_values = np.sort(np.array(values))
        cumulative_values = np.cumsum(sorted_values)
        total = cumulative_values[-1]
        
        # Normalize
        cumulative_percentage = cumulative_values / total * 100
        population_percentage = np.arange(1, len(values) + 1) / len(values) * 100
        
        return population_percentage, cumulative_percentage
    
    def analyze_flow_distribution(self, 
                                 edge_flows: Dict, 
                                 graph: Any, 
                                 output_path: str = "analysis", 
                                 plot_lorenz: bool = True) -> Dict:
        """Analyze flow distribution and generate statistics"""
        
        # 1. Calculate PCU-km
        pcu_km_by_edge, pcu_km_by_road_class = self.calculate_pcu_km(edge_flows, graph)
        
        # 2. Calculate Gini coefficient for link flows (PCU-km)
        pcu_km_values = list(pcu_km_by_edge.values())
        gini = self.calculate_gini_coefficient(pcu_km_values)
        
        # 3. Generate Lorenz curve data
        pop_percent, cum_percent = self.generate_lorenz_curve_data(pcu_km_values)
        
        # 4. Calculate percentage by road class
        total_pcu_km = sum(pcu_km_by_road_class.values())
        percentages_by_class = {}
        for road_class, value in pcu_km_by_road_class.items():
            percentages_by_class[road_class] = (value / total_pcu_km * 100) if total_pcu_km > 0 else 0
        
        # 5. Plot Lorenz curve if requested
        if plot_lorenz and len(pop_percent) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(pop_percent, cum_percent, 'b-', linewidth=2, label='Lorenz Curve')
            plt.plot([0, 100], [0, 100], 'r--', linewidth=1, label='Perfect Equality')
            plt.fill_between(pop_percent, cum_percent, pop_percent, alpha=0.3)
            
            # Add Gini coefficient annotation
            plt.annotate(f'Gini = {gini:.3f}', 
                        xy=(0.6, 0.2), 
                        xycoords='axes fraction',
                        fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            plt.xlabel('Cumulative Percentage of Road Links (%)', fontsize=12)
            plt.ylabel('Cumulative Percentage of PCU-km (%)', fontsize=12)
            plt.title('Lorenz Curve of Traffic Flow Distribution', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.legend(loc='upper left')
            plt.axis('equal')
            
            # Save the plot
            plot_file = output_path.replace('.json', '_lorenz.png') if output_path.endswith('.json') else f"{output_path}_lorenz.png"
            plt.tight_layout()
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved Lorenz curve plot to {plot_file}")
        
        # 6. Save results to JSON
        results = {
            'gini_coefficient': float(gini),
            'total_pcu_km': float(total_pcu_km),
            'pcu_km_by_road_class': dict(percentages_by_class),
            'lorenz_curve': {
                'population_percentage': pop_percent.tolist(),
                'cumulative_percentage': cum_percent.tolist()
            },
            'edge_level_pcu_km': [
                {
                    'u': key[0],
                    'v': key[1],
                    'pcu_km': value,
                    'road_class': self._get_road_class(graph, key)
                }
                for key, value in pcu_km_by_edge.items()
            ]
        }
        
        # Save to JSON
        json_file = output_path if output_path.endswith('.json') else f"{output_path}.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved distribution analysis to {json_file}")
        
        # Print summary
        print(f"\n=== Flow Distribution Analysis ===")
        print(f"Gini Coefficient: {gini:.4f}")
        print(f"Total PCU-km: {total_pcu_km:.2f}")
        print("\nPCU-km by Road Class (%):")
        for road_class, percentage in percentages_by_class.items():
            print(f"  {road_class}: {percentage:.2f}%")
        
        return results
    
    def _get_road_class(self, graph: Any, edge_key: Tuple) -> str:
        """Helper to get road class for an edge"""
        u, v, key = edge_key
        edge = graph[u][v][key]
        highway = edge.get("highway", "unclassified")
        if isinstance(highway, list):
            highway = highway[0]
        
        if highway in ['motorway', 'trunk']:
            return 'trunk'
        elif highway == 'primary':
            return 'primary'
        elif highway == 'secondary':
            return 'secondary'
        elif highway == 'tertiary':
            return 'tertiary'
        elif highway == 'residential':
            return 'residential'
        else:
            return 'other'