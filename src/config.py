"""
Configuration management for 3D reconstruction pipeline.
"""

from pathlib import Path
from typing import List, Dict, Any
import yaml
import os


class ConfigManager:
    """Manages configuration files and parameters for reconstruction tools"""
    
    def __init__(self, base_dir: Path = None, config_file: str = "config.yaml"):
        """Initialize config manager with base directory and config file"""
        if base_dir is None:
            # current file is in src/, so parent is project root
            base_dir = Path(__file__).parent.parent
        
        self.base_dir = Path(base_dir)
        self.config_file_path = self.base_dir / config_file
        
        self.config = self._load_yaml_config()
        
        self.config_dir = self.base_dir / self.config['paths']['config_dir']
        self.bin_dir = self.base_dir / self.config['paths']['bin_dir']
        self.output_dir = self.base_dir / self.config['paths']['output_dir'] 
        self.sequence_dir = self.base_dir / self.config['paths']['sequence_dir']
        
        if not self.config_dir.exists():
            raise FileNotFoundError(f"Configuration directory not found: {self.config_dir}")
            
        if not self.bin_dir.exists():
            raise FileNotFoundError(f"Binary directory not found: {self.bin_dir}")
    
    def _load_yaml_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_file_path}")
        
        with open(self.config_file_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get_config(self, key: str = None) -> Any:
        """Get configuration value by key (dot notation supported)"""
        if key is None:
            return self.config
        
        keys = key.split('.')
        value = self.config
        for k in keys:
            value = value.get(k)
            if value is None:
                break
        return value
    
    def read_config(self, config_file: str) -> List[str]:
        """
        Read parameters from a configuration file.
        
        Args:
            config_file: Name of the .ini file (e.g., 'mstereo.ini')
        
        Returns:
            List of command line parameters
        """
        config_path = self.config_dir / config_file
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        params = []
        with open(config_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Skip image file references, we'll handle those dynamically
                    if not any(x in line.lower() for x in ['.bmp', '.tka', '.png', '.jpg']):
                        params.extend(line.split())
        
        return params
    
    def get_stereo_params(self) -> List[str]:
        """Get stereo processor parameters"""
        return self.read_config("stereo_processor.ini")
    
    def get_holes_params(self) -> List[str]:
        """Get hole filler parameters"""
        return self.read_config("hole_filler.ini")
    
    def get_filter_params(self) -> List[str]:
        """Get surface filter parameters"""
        return self.read_config("surface_filter.ini")
    
    def get_texture_params(self) -> List[str]:
        """Get texture mapper parameters (empty list, images added dynamically)"""
        return []  # texture_mapper.ini only contains image references
    
    def list_config_files(self) -> List[Path]:
        """List all available configuration files"""
        return list(self.config_dir.glob("*.ini"))
    
    def get_config_info(self) -> Dict[str, Dict]:
        """Get information about all configuration files"""
        info = {}
        for config_file in self.list_config_files():
            try:
                params = self.read_config(config_file.name)
                info[config_file.name] = {
                    'path': str(config_file),
                    'params': params,
                    'param_count': len(params)
                }
            except Exception as e:
                info[config_file.name] = {
                    'path': str(config_file),
                    'error': str(e)
                }
        return info
