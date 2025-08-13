"""
Cross-platform executor for running Windows executables in the reconstruction pipeline.
Uses Wine on Linux/Unix systems, runs directly on Windows.
"""

from pathlib import Path
from typing import List, Tuple
import subprocess
import time
import platform


class CrossPlatformExecutor:
    """Executes Windows executables using wine on Linux/Unix, directly on Windows"""
    
    def __init__(self, bin_dir: Path = None):
        """
        Initialize executor.
        
        Args:
            bin_dir: Directory containing Windows executables
        """
        if bin_dir is None:
            raise ValueError("bin_dir must be provided")
        
        self.bin_dir = Path(bin_dir)
        
        if not self.bin_dir.exists():
            raise FileNotFoundError(f"Binary directory not found: {self.bin_dir}")
        
        self.is_windows = platform.system().lower() == 'windows'
        
        # check for binaries
        self.executables = {
            'stereo_processor': self.bin_dir / "stereo_processor.exe",
            'hole_filler': self.bin_dir / "hole_filler.exe",
            'surface_filter': self.bin_dir / "surface_filter.exe",
            'texture_mapper': self.bin_dir / "texture_mapper.exe",
            'mesh_converter': self.bin_dir / "mesh_converter.exe"
        }
        
        missing = [name for name, path in self.executables.items() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing executables: {missing}")
    
    def run_command(self, executable_name: str, args: List[str], cwd: Path = None, 
                   timeout: int = 300) -> Tuple[bool, str, float]:
        """
        Run a Windows executable.
        
        On Windows: Run directly
        On Linux/Unix: Run using wine
        
        Args:
            executable_name: Name of the executable ('mstereo', 'fillholes', etc.)
            args: Command line arguments
            cwd: Working directory for the command
            timeout: Timeout in seconds (default: 300)
            
        Returns:
            Tuple of (success, output/error, duration)
        """
        if executable_name not in self.executables:
            available = list(self.executables.keys())
            raise ValueError(f"Unknown executable '{executable_name}'. Available: {available}")
        
        executable_path = self.executables[executable_name]
        
        if self.is_windows:
            cmd = [str(executable_path)] + args
        else:
            # Use wine on Linux/Unix systems
            try:
                subprocess.run(['which', 'xvfb-run'], check=True, capture_output=True)
                cmd = ['xvfb-run', '-a', 'wine', str(executable_path)] + args
            except (subprocess.CalledProcessError, FileNotFoundError):
                # fall back to regular wine
                cmd = ['wine', str(executable_path)] + args
        
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd, 
                cwd=cwd, 
                capture_output=True, 
                text=True, 
                timeout=timeout
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                return True, result.stdout, duration
            else:
                error_msg = f"Command failed with code {result.returncode}\nError: {result.stderr}"
                return False, error_msg, duration
                
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return False, f"Command timed out after {timeout} seconds", duration
        except Exception as e:
            duration = time.time() - start_time
            return False, str(e), duration
    
    def run_stereo_processor(self, params: List[str], stereo_files: List[Tuple[str, str]], 
                   cwd: Path = None, frame_num: int = None) -> Tuple[bool, str, float]:
        """
        Run stereo_processor.exe with stereo image pairs.
        
        Args:
            params: MStereo parameters from config
            stereo_files: List of (image_path, calib_path) tuples
            cwd: Working directory
            frame_num: Frame number for output naming
            
        Returns:
            Tuple of (success, output/error, duration)
        """
        args = params.copy()
        
        # add output filename if frame number is provided
        if frame_num is not None:
            args.extend(["-o", f"out_{frame_num:03d}.tsb"])
        
        for img_path, calib_path in stereo_files:
            args.extend(["-g", str(img_path), str(calib_path)])
        
        return self.run_command('stereo_processor', args, cwd)
    
    def run_hole_filler(self, params: List[str], frame_num: int, cwd: Path = None) -> Tuple[bool, str, float]:
        """
        Run hole_filler.exe.
        
        Args:
            params: fillholes parameters from config
            frame_num: Frame number for input/output naming
            cwd: Working directory
            
        Returns:
            Tuple of (success, output/error, duration)
        """
        input_file = f"out_{frame_num:03d}.tsb"
        output_file = f"filled_{frame_num:03d}.tsb"
        args = ["-i", input_file, "-o", output_file] + params
        return self.run_command('hole_filler', args, cwd)
    
    def run_surface_filter(self, params: List[str], frame_num: int, cwd: Path = None) -> Tuple[bool, str, float]:
        """
        Run surface_filter.exe.
        
        Args:
            params: filtersurface parameters from config
            frame_num: Frame number for input/output naming
            cwd: Working directory
            
        Returns:
            Tuple of (success, output/error, duration)
        """
        input_file = f"filled_{frame_num:03d}.tsb"
        output_file = f"filtered_{frame_num:03d}.tsb"
        args = ["-i", input_file, "-o", output_file] + params
        return self.run_command('surface_filter', args, cwd)
    
    def run_texture_mapper(self, frame_num: int, texture_files: List[Tuple[str, str]], 
                      cwd: Path = None) -> Tuple[bool, str, float]:
        """
        Run texture_mapper.exe.
        
        Args:
            frame_num: Frame number for input/output naming
            texture_files: List of (texture_image_path, calib_path) tuples
            cwd: Working directory
            
        Returns:
            Tuple of (success, output/error, duration)
        """
        input_file = f"filtered_{frame_num:03d}.tsb"
        output_file = f"textured_{frame_num:03d}.tsb"
        args = ["-i", input_file, "-o", output_file]
        for tex_path, calib_path in texture_files:
            args.extend(["-t", str(tex_path), str(calib_path)])
        
        return self.run_command('texture_mapper', args, cwd)
    
    def run_mesh_converter(self, frame_num: int, cwd: Path = None) -> Tuple[bool, str, float]:
        """
        Run tsb2obj.exe.
        
        Args:
            frame_num: Frame number for input/output naming
            cwd: Working directory
            
        Returns:
            Tuple of (success, output/error, duration)
        """
        input_file = f"textured_{frame_num:03d}.tsb"
        output_file = f"frame_{frame_num:03d}.obj"
        args = ["-i", input_file, "-o", output_file]
        return self.run_command('mesh_converter', args, cwd)
    
    def get_executable_info(self) -> dict:
        """
        Get information about available executables.
        
        Returns:
            Dictionary with executable information
        """
        info = {}
        for name, path in self.executables.items():
            info[name] = {
                'path': str(path),
                'exists': path.exists(),
                'size_mb': path.stat().st_size / (1024 * 1024) if path.exists() else 0
            }
        return info
