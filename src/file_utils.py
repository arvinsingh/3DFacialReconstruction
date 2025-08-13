"""
File management utilities for 3D reconstruction pipeline.
"""

from pathlib import Path
from typing import Dict, List, Set
import re


class FileManager:
    """Manages file discovery and validation for reconstruction pipeline"""
    
    def __init__(self):
        """Initialize file manager"""
        # supported image formats
        self.image_formats = ['.bmp', '.png', '.jpg', '.jpeg']
        
        # camera configurations
        self.stereo_cameras = ['1A', '1B', '2A', '2B']  # Stereo pairs
        self.texture_cameras = ['1C', '2C']             # Texture cameras
        self.all_cameras = self.stereo_cameras + self.texture_cameras
    
    def find_image_files(self, image_dir: Path, frame_num: int) -> Dict[str, Path]:
        """
        Find image files for a specific frame.
        
        Args:
            image_dir: Directory containing images
            frame_num: Frame number to search for
            
        Returns:
            Dictionary mapping camera IDs to file paths
            
        Raises:
            Exception: If directory doesn't exist or cannot be accessed
        """
        if not image_dir.exists():
            raise Exception(f"Image directory not found: {image_dir}")
        
        if not image_dir.is_dir():
            raise Exception(f"Path is not a directory: {image_dir}")
        
        found_images = {}
        
        try:
            # Supported naming patterns (case-insensitive)
            patterns = [
                r'(STEREO|stereo)_(\d+[A-Za-z])_(\d{3,4})\.(bmp|png|jpg|jpeg)',
                r'(TEXTURE|texture)_(\d+[A-Za-z])_(\d{3,4})\.(bmp|png|jpg|jpeg)',
                r'(\d+[A-Za-z])_(\d{3,4})\.(bmp|png|jpg|jpeg)',
                r'(cam|CAM)(\d+[A-Za-z])_(\d{3,4})\.(bmp|png|jpg|jpeg)',
            ]
            
            for file_path in image_dir.glob("*"):
                if not file_path.is_file():
                    continue
                    
                filename = file_path.name
                
                for pattern in patterns:
                    match = re.search(pattern, filename, re.IGNORECASE)
                    if match:
                        groups = match.groups()
                        
                        if len(groups) >= 3:
                            # extract camera ID and frame number
                            if groups[0].upper() in ['STEREO', 'TEXTURE']:
                                camera_id = groups[1].upper()
                                file_frame = int(groups[2])
                            else:
                                camera_id = groups[0].upper() if groups[0].isalnum() else groups[1].upper()
                                file_frame = int(groups[1] if groups[0].isalnum() else groups[2])
                            
                            if file_frame == frame_num:
                                found_images[camera_id] = file_path
                                
        except Exception as e:
            raise Exception(f"Error searching for image files: {e}")
            
        return found_images
    
    def find_calibration_files(self, calib_dir: Path) -> Dict[str, Path]:
        """
        Find calibration files.
        
        Args:
            calib_dir: Directory containing calibration files
            
        Returns:
            Dictionary mapping camera IDs to calibration file paths
            
        Raises:
            Exception: If directory doesn't exist or cannot be accessed
        """
        if not calib_dir.exists():
            raise Exception(f"Calibration directory not found: {calib_dir}")
        
        if not calib_dir.is_dir():
            raise Exception(f"Path is not a directory: {calib_dir}")
        
        calib_files = {}
        
        try:
            for file_path in calib_dir.glob("*.tka"):
                # extract camera ID from filename (e.g., calib_1A.tka -> 1A)
                match = re.search(r'calib_(\d+[A-Za-z])', file_path.name, re.IGNORECASE)
                if match:
                    camera_id = match.group(1).upper()
                    # verify file is readable
                    if not file_path.exists() or file_path.stat().st_size == 0:
                        continue
                    calib_files[camera_id] = file_path
                    
        except Exception as e:
            raise Exception(f"Error searching for calibration files: {e}")
                
        return calib_files
    
    def validate_frame_files(self, images: Dict[str, Path], calibs: Dict[str, Path]) -> Dict[str, List[str]]:
        """
        Validate that all required files are present for a frame.
        
        Args:
            images: Dictionary of found image files
            calibs: Dictionary of found calibration files
            
        Returns:
            Dictionary with validation results and any missing files
        """
        required_stereo = set(self.stereo_cameras)
        required_texture = set(self.texture_cameras)
        required_calibs = required_stereo | required_texture
        
        found_stereo = set(images.keys()) & required_stereo
        found_texture = set(images.keys()) & required_texture
        found_calibs = set(calibs.keys()) & required_calibs
        
        results = {
            'valid': True,
            'missing_stereo': [],
            'missing_texture': [],
            'missing_calibs': [],
            'found_stereo': list(found_stereo),
            'found_texture': list(found_texture),
            'found_calibs': list(found_calibs)
        }
        
        if found_stereo != required_stereo:
            missing = required_stereo - found_stereo
            results['missing_stereo'] = list(missing)
            results['valid'] = False
            
        if found_texture != required_texture:
            missing = required_texture - found_texture
            results['missing_texture'] = list(missing)
            results['valid'] = False
            
        if found_calibs != required_calibs:
            missing = required_calibs - found_calibs
            results['missing_calibs'] = list(missing)
            results['valid'] = False
        
        return results
    
    def get_available_frames(self, image_dir: Path) -> List[int]:
        """
        Get list of available frame numbers in the image directory.
        
        Args:
            image_dir: Directory containing image files
            
        Returns:
            Sorted list of available frame numbers
        """
        frames = set()
        
        patterns = [
            r'(?:STEREO|stereo|TEXTURE|texture)_\d+[A-Za-z]_(\d{3,4})\.(bmp|png|jpg|jpeg)',
            r'\d+[A-Za-z]_(\d{3,4})\.(bmp|png|jpg|jpeg)',
            r'(?:cam|CAM)\d+[A-Za-z]_(\d{3,4})\.(bmp|png|jpg|jpeg)',
        ]
        
        for file_path in image_dir.glob("*"):
            if not file_path.is_file():
                continue
                
            filename = file_path.name
            
            for pattern in patterns:
                match = re.search(pattern, filename, re.IGNORECASE)
                if match:
                    frame_num = int(match.group(1))
                    frames.add(frame_num)
        
        return sorted(list(frames))
    
    def get_file_info(self, file_path: Path) -> Dict:
        """
        Get information about a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with file information
        """
        if not file_path.exists():
            return {'exists': False}
        
        stat = file_path.stat()
        return {
            'exists': True,
            'size_bytes': stat.st_size,
            'size_mb': stat.st_size / (1024 * 1024),
            'name': file_path.name,
            'path': str(file_path),
            'extension': file_path.suffix.lower()
        }
