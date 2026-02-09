import yaml
from dataclasses import dataclass, fields
from typing import Dict, List, Tuple, Optional

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