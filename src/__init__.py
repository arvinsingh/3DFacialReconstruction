"""
3D Reconstruction Pipeline for 3dMD System

A modular Python package that processes stereo imaging data to create textured 3D models.
"""

from .config import ConfigManager
from .file_utils import FileManager
from .executor import WineExecutor
from .pipeline import ReconstructionPipeline

__version__ = "1.0.0"
__all__ = ["ConfigManager", "FileManager", "WineExecutor", "ReconstructionPipeline"]
