import yaml
from dataclasses import dataclass, fields
from typing import Dict, List, Tuple, Optional, Any

# Configuration
@dataclass
class Config:
    pcu: Dict[str, float]
    road_penalties: Dict[str, Dict[str, float]]
    noise_delta: Dict[str, float]
    turn_penalty: Dict[str, float]
    base_speeds: Dict[str, float]
    default_speed: float
    default_penalty: float
    near_thresh: Dict[str, float]
    medium_thresh: Dict[str, float]
    k_near: int
    k_med: int
    k_far: int
    data_paths: Dict[str, str]
    export_paths: Dict[str, str]
    cache_paths: Dict[str, str]
    city: str

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