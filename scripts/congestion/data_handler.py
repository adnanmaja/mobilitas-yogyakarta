import json
import os
from typing import Dict

class DataHandler:
    def __init__(self):
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