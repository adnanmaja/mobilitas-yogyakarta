from typing import Tuple, List, Dict, Optional, Any
import yaml
from dataclasses import dataclass, fields

# Configuration
@dataclass
class Config:
    cell_size: str
    data_paths: Dict[str, str]
    export_paths: Dict[str, str]
    figure_paths: Dict[str, str]
    grid_weights: Dict[str, Dict[str, float]]
    scales: Dict[str, str]

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