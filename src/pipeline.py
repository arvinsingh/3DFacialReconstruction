"""
Main reconstruction pipeline orchestrator.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import time
import tempfile
import shutil

from config import ConfigManager
from file_utils import FileManager
from executor import CrossPlatformExecutor
from logging_config import ReconstructionLogger


class ReconstructionPipeline:
    """Main 3D reconstruction pipeline orchestrator"""
    
    def __init__(self, base_dir: Path = None, config_file: str = "config.yaml"):
        """
        Initialize reconstruction pipeline.
        
        Args:
            base_dir: Base directory for the project
            config_file: Configuration file name
        """
        if base_dir is None:
            # Current file is in src/, so parent is project root
            base_dir = Path(__file__).parent.parent
        
        self.base_dir = Path(base_dir)
        
        # Initialize components
        self.config = ConfigManager(base_dir, config_file)
        self.logger = ReconstructionLogger(self.config.config, base_dir)
        self.file_manager = FileManager()
        self.executor = CrossPlatformExecutor(self.config.bin_dir)
        
        self.logger.info(f"Pipeline initialized with base directory: {self.base_dir}")
        self.logger.info(f"Binary directory: {self.config.bin_dir}")
        self.logger.info(f"Config directory: {self.config.config_dir}")
    
    def process_single_frame(self, image_dir: Path, calib_dir: Path, output_dir: Path, 
                           frame_num: int) -> bool:
        """
        Process a single frame through the complete pipeline.
        
        Args:
            image_dir: Directory containing input images
            calib_dir: Directory containing calibration files
            output_dir: Directory for output files
            frame_num: Frame number to process
            
        Returns:
            True if successful, False otherwise
        """
        self.logger.info(f"Processing Frame {frame_num:03d}")
        
        # find input files
        try:
            images = self.file_manager.find_image_files(image_dir, frame_num)
            calibs = self.file_manager.find_calibration_files(calib_dir)
        except Exception as e:
            self.logger.error(f"Failed to find input files: {e}")
            return False
        
        validation = self.file_manager.validate_frame_files(images, calibs)
        if not validation['valid']:
            missing_items = []
            if validation['missing_stereo']:
                missing_items.extend([f"stereo: {','.join(validation['missing_stereo'])}"])
            if validation['missing_texture']:
                missing_items.extend([f"texture: {','.join(validation['missing_texture'])}"])
            if validation['missing_calibs']:
                missing_items.extend([f"calibration: {','.join(validation['missing_calibs'])}"])
            
            skip_reason = "Missing files - " + "; ".join(missing_items)
            self.logger.add_skipped_frame(frame_num, skip_reason)
            return False
        
        # log found files
        for camera in validation['found_stereo'] + validation['found_texture']:
            self.logger.info(f"Found {camera}: {images[camera].name}")
        for camera in validation['found_calibs']:
            self.logger.info(f"Found calibration {camera}: {calibs[camera].name}")
        
        self.logger.info(f"Output directory: {output_dir}")
        
        # Create temporary directory for format conversion if needed
        temp_dir = None
        converted_images = images
        
        try:
            needs_conversion = any(img_path.suffix.lower() in ['.png', '.jpg', '.jpeg'] 
                                 for img_path in images.values())
            
            if needs_conversion:
                temp_dir = Path(tempfile.mkdtemp(prefix="reconstruction_", suffix="_temp"))
                self.logger.info("Converting images to BMP format for compatibility...")
                converted_images = self.file_manager.convert_images_to_bmp(images, temp_dir, self.logger)
                self.logger.debug(f"Temporary conversion directory: {temp_dir}")
        
        except Exception as e:
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir)
            self.logger.error(f"Image format conversion failed: {e}")
            self.logger.add_failed_frame(frame_num, f"Image conversion failed: {e}", "image conversion")
            return False
        
        total_start = time.time()
        
        try:
            self.logger.info("Step 1: Stereo Processing")
            success = self._run_stereo_step(converted_images, calibs, output_dir, frame_num)
            if not success:
                self.logger.add_failed_frame(frame_num, "Stereo processing failed", "stereo processing")
                return False
            
            self.logger.info("Step 2: Fill Holes")
            success = self._run_holes_step(output_dir, frame_num)
            if not success:
                self.logger.add_failed_frame(frame_num, "Hole filling failed", "hole filling")
                return False
            
            self.logger.info("Step 3: Filter Surface")
            success = self._run_filter_step(output_dir, frame_num)
            if not success:
                self.logger.add_failed_frame(frame_num, "Surface filtering failed", "surface filtering")
                return False
            
            self.logger.info("Step 4: Add Texture")
            success = self._run_texture_step(converted_images, calibs, output_dir, frame_num)
            if not success:
                self.logger.add_failed_frame(frame_num, "Texture mapping failed", "texture mapping")
                return False

        except Exception as e:
            self.logger.add_failed_frame(frame_num, f"Pipeline error: {str(e)}", "pipeline")
            self._cleanup_failed_frame(output_dir, frame_num)
            # Clean up temporary directory if it was created
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir)
                self.logger.debug(f"Cleaned up temporary directory: {temp_dir}")
            return False
        
        try:
            self.logger.info("Step 5: Convert to OBJ")
            success = self._run_convert_step(output_dir, frame_num)
            if not success:
                self.logger.add_failed_frame(frame_num, "Mesh conversion failed", "mesh conversion")
                self._cleanup_failed_frame(output_dir, frame_num)
                return False
            
            # keep only .bmp and .obj
            self.logger.info("Step 6: Cleanup")
            if self.config.get_config('processing.cleanup_intermediate'):
                self._cleanup_intermediate_files(output_dir, frame_num)
            
            if not self._verify_final_output(output_dir, frame_num):
                self.logger.add_failed_frame(frame_num, "Final output verification failed", "output verification")
                self._cleanup_failed_frame(output_dir, frame_num)
                return False
            
        except Exception as e:
            self.logger.add_failed_frame(frame_num, f"Final processing error: {str(e)}", "final processing")
            self._cleanup_failed_frame(output_dir, frame_num)
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir)
                self.logger.debug(f"Cleaned up temporary directory: {temp_dir}")
            return False
        
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir)
            self.logger.debug(f"Cleaned up temporary directory: {temp_dir}")
        
        # Summary
        total_duration = time.time() - total_start
        self.logger.info(f"Frame {frame_num:03d} completed in {total_duration:.2f}s")
        
        # show timing if enabled
        if self.config.get_config('processing.show_timing'):
            self._show_output_summary(output_dir, frame_num)
        
        return True
    
    def process_frame_range(self, image_dir: Path, calib_dir: Path, output_dir: Path,
                          start_frame: int, end_frame: int) -> Dict[int, bool]:
        """
        Process a range of frames.
        
        Args:
            image_dir: Directory containing input images
            calib_dir: Directory containing calibration files
            output_dir: Directory for output files
            start_frame: Starting frame number
            end_frame: Ending frame number
            
        Returns:
            Dictionary mapping frame numbers to success status
        """
        results = {}
        total_frames = end_frame - start_frame + 1
        
        sequence_name = image_dir.name  # Use the sequence folder name
        sequence_output_dir = output_dir / sequence_name
        sequence_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Batch processing: Frame range {start_frame} to {end_frame} ({total_frames} frames)")
        self.logger.info(f"Sequence: {sequence_name}")
        self.logger.info(f"Image dir: {image_dir}")
        self.logger.info(f"Calib dir: {calib_dir}")
        self.logger.info(f"Output dir: {sequence_output_dir}")
        
        batch_start = time.time()
        
        progress_bar = None
        if self.config.get_config('processing.show_progress'):
            progress_bar = self.logger.create_progress_bar(
                total_frames, 
                f"Processing {sequence_name}"
            )
        
        for frame_num in range(start_frame, end_frame + 1):
            frame_start = time.time()
            success = self.process_single_frame(image_dir, calib_dir, sequence_output_dir, frame_num)
            frame_duration = time.time() - frame_start
            
            results[frame_num] = success
            
            if progress_bar:
                status = "OK" if success else "FAIL"
                self.logger.update_progress(1, f"Processing {sequence_name} [{status}]")
            
            self.logger.info(f"Frame {frame_num:03d}: {'SUCCESS' if success else 'FAILED'} ({frame_duration:.2f}s)")
        
        if progress_bar:
            self.logger.close_progress()
        
        batch_duration = time.time() - batch_start
        successful = sum(results.values())
        
        self.logger.print_summary(successful, total_frames, batch_duration)
        
        return results
    
    def _run_stereo_step(self, images: Dict[str, Path], calibs: Dict[str, Path], 
                        output_dir: Path, frame_num: int) -> bool:
        """Run stereo processing step"""
        try:
            params = self.config.get_stereo_params()
            self.logger.debug(f"Loaded stereo parameters: {' '.join(params)}")
            
            # build stereo file pairs
            stereo_files = []
            for camera in self.file_manager.stereo_cameras:
                if camera not in images or camera not in calibs:
                    self.logger.error(f"Missing stereo files for camera {camera}")
                    return False
                stereo_files.append((images[camera], calibs[camera]))
            
            self.logger.debug("Running stereo processor...")
            success, output, duration = self.executor.run_stereo_processor(params, stereo_files, output_dir, frame_num)
            
            if not success:
                self.logger.error(f"Stereo processing failed: {output}")
                return False
            
            self.logger.debug(f"Stereo processing completed in {duration:.2f}s")
            
            out_tsb = output_dir / f"out_{frame_num:03d}.tsb"
            if not out_tsb.exists():
                self.logger.error(f"Expected stereo output not found: {out_tsb.name}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Stereo processing step failed with exception: {e}")
            return False
    
    def _run_holes_step(self, output_dir: Path, frame_num: int) -> bool:
        """Run hole filling processing step"""
        try:
            params = self.config.get_holes_params()
            self.logger.debug(f"Loaded hole filling parameters: {' '.join(params)}")
            
            self.logger.debug("Running hole filler...")
            success, output, duration = self.executor.run_hole_filler(params, frame_num, cwd=output_dir)
            
            if not success:
                self.logger.error(f"Hole filling failed: {output}")
                return False
            
            self.logger.debug(f"Hole filling completed in {duration:.2f}s")
            
            filled_tsb = output_dir / f"filled_{frame_num:03d}.tsb"
            if not filled_tsb.exists():
                self.logger.error(f"Expected hole filling output not found: {filled_tsb.name}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Hole filling step failed with exception: {e}")
            return False
    
    def _run_filter_step(self, output_dir: Path, frame_num: int) -> bool:
        """Run filtersurface processing step"""
        try:
            params = self.config.get_filter_params()
            self.logger.debug(f"Loaded surface filter parameters: {' '.join(params)}")
            
            self.logger.debug("Running surface filter...")
            success, output, duration = self.executor.run_surface_filter(params, frame_num, cwd=output_dir)
            
            if not success:
                self.logger.error(f"Surface filtering failed: {output}")
                return False
            
            self.logger.debug(f"Surface filtering completed in {duration:.2f}s")
            
            filtered_tsb = output_dir / f"filtered_{frame_num:03d}.tsb"
            if not filtered_tsb.exists():
                self.logger.error(f"Expected surface filtering output not found: {filtered_tsb.name}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Surface filtering step failed with exception: {e}")
            return False
    
    def _run_texture_step(self, images: Dict[str, Path], calibs: Dict[str, Path], 
                         output_dir: Path, frame_num: int) -> bool:
        """Run addtexture processing step"""
        try:
            # build texture file pairs
            texture_files = []
            for camera in self.file_manager.texture_cameras:
                if camera not in images or camera not in calibs:
                    self.logger.error(f"Missing texture files for camera {camera}")
                    return False
                texture_files.append((images[camera], calibs[camera]))
            
            self.logger.debug("Running texture mapper...")
            success, output, duration = self.executor.run_texture_mapper(
                frame_num, texture_files, output_dir
            )
            
            if not success:
                self.logger.error(f"Texture mapping failed: {output}")
                return False
            
            self.logger.debug(f"Texture mapping completed in {duration:.2f}s")
            
            textured_tsb = output_dir / f"textured_{frame_num:03d}.tsb"
            if not textured_tsb.exists():
                self.logger.error(f"Expected texture mapping output not found: {textured_tsb.name}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Texture mapping step failed with exception: {e}")
            return False
    
    def _run_convert_step(self, output_dir: Path, frame_num: int) -> bool:
        """Run mesh conversion processing step"""
        try:
            self.logger.debug("Running mesh converter...")
            success, output, duration = self.executor.run_mesh_converter(frame_num, cwd=output_dir)
            
            if not success:
                self.logger.error(f"Mesh conversion failed: {output}")
                return False
            
            self.logger.debug(f"Mesh conversion completed in {duration:.2f}s")
            return True
            
        except Exception as e:
            self.logger.error(f"Mesh conversion step failed with exception: {e}")
            return False
    
    def _cleanup_intermediate_files(self, output_dir: Path, frame_num: int):
        """Clean up intermediate files, keeping only .bmp and .obj files"""
        files_to_keep = {'.bmp', '.obj'}
        removed_count = 0
        
        # find frame-specific files to clean
        for pattern in [f"*_{frame_num:03d}.*", f"*{frame_num:03d}.*"]:
            for file_path in output_dir.glob(pattern):
                if file_path.is_file() and file_path.suffix.lower() not in files_to_keep:
                    try:
                        file_path.unlink()
                        removed_count += 1
                    except Exception as e:
                        self.logger.warning(f"Could not remove {file_path.name}: {e}")
        
        if removed_count > 0:
            self.logger.debug(f"Removed {removed_count} intermediate files")
    
    def _cleanup_failed_frame(self, output_dir: Path, frame_num: int):
        """Clean up all files for a failed frame"""
        try:
            patterns = [f"*_{frame_num:03d}.*", f"*{frame_num:03d}.*"]
            removed_count = 0
            for pattern in patterns:
                for file_path in output_dir.glob(pattern):
                    if file_path.is_file():
                        try:
                            file_path.unlink()
                            removed_count += 1
                        except Exception as e:
                            print(f"Warning: Could not remove failed frame file {file_path.name}: {e}")
            
            if removed_count > 0:
                print(f"Cleaned up {removed_count} files from failed frame {frame_num:03d}")
        except Exception as e:
            print(f"Warning: Could not cleanup failed frame {frame_num:03d}: {e}")
    
    def _verify_final_output(self, output_dir: Path, frame_num: int) -> bool:
        """Verify that final output files exist and are valid"""
        try:
            obj_file = output_dir / f"frame_{frame_num:03d}.obj"
            bmp_file = output_dir / f"frame_{frame_num:03d}.bmp"
            
            if not obj_file.exists():
                print(f"Missing OBJ file: {obj_file.name}")
                return False
            
            if not bmp_file.exists():
                print(f"Missing BMP file: {bmp_file.name}")
                return False
            
            # check if files are not empty
            if obj_file.stat().st_size == 0:
                print(f"Empty OBJ file: {obj_file.name}")
                return False
            
            if bmp_file.stat().st_size == 0:
                print(f"Empty BMP file: {bmp_file.name}")
                return False
            
            return True
            
        except Exception as e:
            print(f"Error verifying output files: {e}")
            return False
    
    def _show_output_summary(self, output_dir: Path, frame_num: int):
        """Show summary of generated output files"""
        pattern = f"*_{frame_num:03d}.*"
        output_files = [f for f in output_dir.glob(pattern) if f.suffix.lower() in {'.bmp', '.obj'}]
        self.logger.debug(f"Generated {len(output_files)} final files:")
        for file_path in sorted(output_files):
            if file_path.is_file():
                file_info = self.file_manager.get_file_info(file_path)
                self.logger.debug(f"  {file_info['name']:<25} ({file_info['size_mb']:.2f} MB)")
    
    def get_system_info(self) -> Dict:
        """Get system information for debugging"""
        return {
            'base_dir': str(self.base_dir),
            'config_dir': str(self.config.config_dir),
            'bin_dir': str(self.executor.bin_dir),
            'available_configs': [f.name for f in self.config.list_config_files()],
            'executables': self.executor.get_executable_info(),
            'supported_cameras': {
                'stereo': self.file_manager.stereo_cameras,
                'texture': self.file_manager.texture_cameras
            }
        }
