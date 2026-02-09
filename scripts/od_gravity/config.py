from typing import Tuple, List, Dict, Optional, Any
import yaml
from dataclasses import dataclass, fields

# Configuration
@dataclass
class Config:
    distance_thresh: Dict[str, int]
    distance_shares: Dict[str, Dict[str, float]]
    purpose_shares: Dict[str, Dict[str, float]]
    total_trips: Dict[str, float]
    gravity: Dict[str, Dict[str, float]]
    time_weight: Dict[str, float]
    od_scale: float
    ipf: Dict[str, float]
    boundary_buffer: float
    leakage: Dict[str, float]
    data_paths: Dict[str, str]
    export_paths: Dict[str, str]

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