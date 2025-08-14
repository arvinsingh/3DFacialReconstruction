"""
3D Reconstruction Pipeline for 3dMD System

A modular Python package that processes stereo imaging data to create textured 3D models.
"""

from .config import ConfigManager
from .file_utils import FileManager
from .executor import CrossPlatformExecutor
from .pipeline import ReconstructionPipeline
from .parallel_pipeline import ParallelReconstructionPipeline

__version__ = "1.1.0"
__all__ = ["ConfigManager", "FileManager", "CrossPlatformExecutor", "ReconstructionPipeline", "ParallelReconstructionPipeline"]
