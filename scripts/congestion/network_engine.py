import json
import os
from typing import Dict, List, Tuple, Optional
import time
import gc
from scripts.vector_routing.vector_router_model import ImpedanceCalculator, SparseGraphBuilder, GraphManager, Router, VectorRouter, FlowAnalyzer
from congestion_config import Config

class NetworkAnalyst:
    def __init__(self, config = Config):
        self.config = config.from_yaml()
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