"""
Multithreaded reconstruction pipeline for improved performance.
Supports both frame-level parallelism and pipeline-stage parallelism.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
import os
import tempfile
import shutil
import concurrent.futures
import threading
from queue import Queue, Empty
from dataclasses import dataclass
from enum import Enum

from config import ConfigManager
from file_utils import FileManager
from executor import CrossPlatformExecutor
from logging_config import ReconstructionLogger


class ProcessingStage(Enum):
    """Processing stages in the pipeline"""
    STEREO = "stereo"
    HOLES = "holes"
    FILTER = "filter"
    TEXTURE = "texture"
    CONVERT = "convert"


@dataclass
class FrameTask:
    """Task for processing a single frame"""
    frame_num: int
    image_dir: Path
    calib_dir: Path
    output_dir: Path
    images: Dict[str, Path] = None
    calibs: Dict[str, Path] = None
    stage: ProcessingStage = ProcessingStage.STEREO
    temp_dir: Path = None  # temp dir for converted img
    

@dataclass
class TaskResult:
    """Result of processing a task"""
    task: FrameTask
    success: bool
    error_msg: str = ""
    duration: float = 0.0


class ParallelReconstructionPipeline:
    """Multithreaded 3D reconstruction pipeline orchestrator"""
    
    def __init__(self, base_dir: Path = None, config_file: str = "config.yaml", 
                 max_workers: int = None):
        """
        Initialize parallel reconstruction pipeline.
        
        Args:
            base_dir: Base directory for the project
            config_file: Configuration file name
            max_workers: Maximum number of worker threads (default: CPU count)
        """
        if base_dir is None:
            base_dir = Path(__file__).parent.parent
        
        self.base_dir = Path(base_dir)
        
        # init components
        self.config = ConfigManager(base_dir, config_file)
        self.logger = ReconstructionLogger(self.config.config, base_dir)
        self.file_manager = FileManager()
        
        # create separate executor instances for thread safety
        self.max_workers = max_workers or min(8, (os.cpu_count() or 1) + 4)
        self.executors = [CrossPlatformExecutor(self.config.bin_dir) 
                         for _ in range(self.max_workers)]
        
        # thread-safe progress tracking
        self._progress_lock = threading.Lock()
        self._completed_frames = 0
        self._failed_frames = []
        self._skipped_frames = []
        
        self.logger.info(f"Parallel pipeline initialized with {self.max_workers} workers")
        self.logger.info(f"Base directory: {self.base_dir}")
    
    def process_frame_range_parallel(self, image_dir: Path, calib_dir: Path, output_dir: Path,
                                   start_frame: int, end_frame: int, 
                                   mode: str = "frame_parallel") -> Dict[int, bool]:
        """
        Process a range of frames using multithreading.
        
        Args:
            image_dir: Directory containing input images
            calib_dir: Directory containing calibration files  
            output_dir: Directory for output files
            start_frame: Starting frame number
            end_frame: Ending frame number
            mode: Threading mode - "frame_parallel" or "pipeline_parallel"
            
        Returns:
            Dictionary mapping frame numbers to success status
        """
        results = {}
        total_frames = end_frame - start_frame + 1
        
        sequence_name = image_dir.name
        sequence_output_dir = output_dir / sequence_name
        sequence_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Parallel batch processing: Frame range {start_frame} to {end_frame} ({total_frames} frames)")
        self.logger.info(f"Threading mode: {mode}")
        self.logger.info(f"Workers: {self.max_workers}")
        self.logger.info(f"Sequence: {sequence_name}")
        
        batch_start = time.time()
        
        # reset progress tracking
        self._completed_frames = 0
        self._failed_frames.clear()
        self._skipped_frames.clear()
        
        progress_bar = None
        if self.config.get_config('processing.show_progress'):
            progress_bar = self.logger.create_progress_bar(
                total_frames, 
                f"Processing {sequence_name} ({mode})"
            )
        
        if mode == "frame_parallel":
            results = self._process_frame_parallel(
                image_dir, calib_dir, sequence_output_dir, 
                start_frame, end_frame, progress_bar
            )
        elif mode == "pipeline_parallel":
            results = self._process_pipeline_parallel(
                image_dir, calib_dir, sequence_output_dir,
                start_frame, end_frame, progress_bar
            )
        else:
            raise ValueError(f"Unknown threading mode: {mode}")
        
        if progress_bar:
            self.logger.close_progress()
        
        batch_duration = time.time() - batch_start
        successful = sum(results.values())
        
        self.logger.print_summary(successful, total_frames, batch_duration)
        
        return results
    
    def _process_frame_parallel(self, image_dir: Path, calib_dir: Path, output_dir: Path,
                              start_frame: int, end_frame: int, progress_bar) -> Dict[int, bool]:
        """Process multiple frames in parallel (each frame through complete pipeline)"""
        results = {}
        failed_frames = []
        
        # pre-load calibration files once (performance optimization)
        self.logger.info("Loading calibration files...")
        try:
            calibs = self.file_manager.find_calibration_files(calib_dir)
        except Exception as e:
            self.logger.error(f"Failed to load calibration files: {e}")
            return {}
        
        # prep frame tasks in parallel (faster initialization)
        self.logger.info(f"Preparing {end_frame - start_frame + 1} frame tasks (parallel)...")
        frame_tasks = []
        
        def prepare_frame_task(frame_num):
            try:
                images = self.file_manager.find_image_files(image_dir, frame_num)
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
                    return {'frame_num': frame_num, 'status': 'failed', 'reason': skip_reason, 'stage': 'validation'}
                
                # check if imgs need conversion to BMP format
                converted_images = images
                temp_dir = None
                
                needs_conversion = any(img_path.suffix.lower() in ['.png', '.jpg', '.jpeg'] 
                                     for img_path in images.values())
                
                if needs_conversion:
                    temp_dir = Path(tempfile.mkdtemp(prefix=f"reconstruction_frame_{frame_num}_", suffix="_temp"))
                    try:
                        converted_images = self.file_manager.convert_images_to_bmp(images, temp_dir, self.logger)
                        self.logger.debug(f"Frame {frame_num}: Converted images to BMP format")
                    except Exception as e:
                        if temp_dir and temp_dir.exists():
                            shutil.rmtree(temp_dir)
                        return {'frame_num': frame_num, 'status': 'failed', 'reason': f"Image conversion failed: {e}", 'stage': 'conversion'}
                
                # create task with converted images and temp directory info
                task = FrameTask(frame_num, image_dir, calib_dir, output_dir, converted_images, calibs)
                task.temp_dir = temp_dir  # save ref for cleanup
                
                return {'frame_num': frame_num, 'status': 'success', 'task': task}
            except Exception as e:
                return {'frame_num': frame_num, 'status': 'failed', 'reason': str(e), 'stage': 'preparation'}
        
        # ThreadPoolExecutor for parallel task prep
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, self.max_workers)) as prep_executor:
            prep_futures = {prep_executor.submit(prepare_frame_task, frame_num): frame_num 
                           for frame_num in range(start_frame, end_frame + 1)}
            
            for future in concurrent.futures.as_completed(prep_futures):
                result = future.result()
                if result['status'] == 'failed':
                    self.logger.add_skipped_frame(result['frame_num'], result['reason'])
                    failed_frames.append({'frame': result['frame_num'], 'reason': result['reason'], 'stage': result['stage']})
                    results[result['frame_num']] = False
                else:
                    frame_tasks.append(result['task'])
        
        self.logger.info(f"Ready to process {len(frame_tasks)} frames with {self.max_workers} workers")
        
        # process frames in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_frame = {
                executor.submit(self._process_single_frame_complete, task, worker_id % len(self.executors)): task.frame_num
                for worker_id, task in enumerate(frame_tasks)
            }
            
            for future in concurrent.futures.as_completed(future_to_frame):
                frame_num = future_to_frame[future]
                try:
                    result = future.result()
                    success = result.get('success', False) if isinstance(result, dict) else result
                    results[frame_num] = success
                    
                    if not success and isinstance(result, dict):
                        failed_frames.append({
                            'frame': frame_num, 
                            'reason': result.get('error', 'Unknown error'),
                            'stage': result.get('failed_stage', 'unknown')
                        })
                    
                    with self._progress_lock:
                        self._completed_frames += 1
                        if progress_bar:
                            status = "OK" if success else "FAIL"
                            self.logger.update_progress(1, f"Frame {frame_num:03d} [{status}]")
                    
                except Exception as e:
                    error_msg = f"Frame {frame_num} failed with exception: {e}"
                    self.logger.error(error_msg)
                    failed_frames.append({'frame': frame_num, 'reason': str(e), 'stage': 'execution'})
                    results[frame_num] = False
                    
                    with self._progress_lock:
                        self._completed_frames += 1
                        if progress_bar:
                            self.logger.update_progress(1, f"Frame {frame_num:03d} [ERROR]")
        
        if failed_frames:
            self._generate_error_report(failed_frames, output_dir)
        
        return results
    
    def _process_pipeline_parallel(self, image_dir: Path, calib_dir: Path, output_dir: Path,
                                 start_frame: int, end_frame: int, progress_bar) -> Dict[int, bool]:
        """Process frames with pipeline parallelism (overlapping stages)"""
        results = {}
        total_frames = end_frame - start_frame + 1
        
        # stage queues for pipeline
        stage_queues = {
            ProcessingStage.STEREO: Queue(),
            ProcessingStage.HOLES: Queue(),
            ProcessingStage.FILTER: Queue(),
            ProcessingStage.TEXTURE: Queue(),
            ProcessingStage.CONVERT: Queue()
        }
        
        # result tracking
        completed_tasks = Queue()
        
        # init stereo queue with all frames
        for frame_num in range(start_frame, end_frame + 1):
            try:
                images = self.file_manager.find_image_files(image_dir, frame_num)
                calibs = self.file_manager.find_calibration_files(calib_dir)
                
                validation = self.file_manager.validate_frame_files(images, calibs)
                if validation['valid']:
                    task = FrameTask(frame_num, image_dir, calib_dir, output_dir, 
                                   images, calibs, ProcessingStage.STEREO)
                    stage_queues[ProcessingStage.STEREO].put(task)
                else:
                    results[frame_num] = False
                    
            except Exception as e:
                self.logger.error(f"Failed to prepare frame {frame_num}: {e}")
                results[frame_num] = False
        
        workers = []
        stages = list(ProcessingStage)
        
        for stage in stages:
            # use fewer workers for later stages that are typically faster
            stage_workers = max(1, self.max_workers // len(stages))
            if stage == ProcessingStage.STEREO:
                stage_workers = max(2, self.max_workers // 2)  # more workers for stereo (slowest step)
            
            for worker_id in range(stage_workers):
                worker = threading.Thread(
                    target=self._pipeline_worker,
                    args=(stage, stage_queues, completed_tasks, worker_id)
                )
                worker.daemon = True
                workers.append(worker)
                worker.start()
        
        # monitor progress
        completed_count = 0
        while completed_count < total_frames:
            try:
                result = completed_tasks.get(timeout=1.0)
                frame_num = result.task.frame_num
                results[frame_num] = result.success
                completed_count += 1
                
                if progress_bar:
                    status = "OK" if result.success else "FAIL"
                    self.logger.update_progress(1, f"Frame {frame_num:03d} [{status}]")
                    
            except Empty:
                continue
        
        # signal workers to stop
        for stage in stages:
            for _ in range(self.max_workers):
                stage_queues[stage].put(None)  # sentinel value
        
        # wait for workers to finish
        for worker in workers:
            worker.join(timeout=5.0)
        
        return results
    
    def _process_single_frame_complete(self, task: FrameTask, worker_id: int) -> Dict:
        """Process a single frame through the complete pipeline (for frame parallelism)"""
        executor = self.executors[worker_id % len(self.executors)]
        frame_num = task.frame_num
        
        self.logger.info(f"Worker {worker_id}: Processing Frame {frame_num:03d}")
        
        try:
            if not self._run_stereo_step_threaded(task, executor):
                return {'success': False, 'error': 'Stereo processing failed', 'failed_stage': 'stereo'}
            
            if not self._run_holes_step_threaded(task, executor):
                return {'success': False, 'error': 'Hole filling failed', 'failed_stage': 'holes'}
                
            if not self._run_filter_step_threaded(task, executor):
                return {'success': False, 'error': 'Surface filtering failed', 'failed_stage': 'filter'}
                
            if not self._run_texture_step_threaded(task, executor):
                return {'success': False, 'error': 'Texture mapping failed', 'failed_stage': 'texture'}
                
            if not self._run_convert_step_threaded(task, executor):
                return {'success': False, 'error': 'Mesh conversion failed', 'failed_stage': 'convert'}
            
            if self.config.get_config('processing.cleanup_intermediate'):
                self._cleanup_intermediate_files(task.output_dir, frame_num)
            
            if not self._verify_final_output(task.output_dir, frame_num):
                self._cleanup_failed_frame(task.output_dir, frame_num)
                return {'success': False, 'error': 'Final output verification failed', 'failed_stage': 'verification'}
            
            self.logger.info(f"Worker {worker_id}: Frame {frame_num:03d} completed successfully")
            
            if task.temp_dir and task.temp_dir.exists():
                shutil.rmtree(task.temp_dir)
                self.logger.debug(f"Worker {worker_id}: Cleaned up temporary directory for frame {frame_num:03d}")
            
            return {'success': True}
            
        except Exception as e:
            error_msg = f"Frame {frame_num:03d} failed: {e}"
            self.logger.error(f"Worker {worker_id}: {error_msg}")
            self._cleanup_failed_frame(task.output_dir, frame_num)
            
            if task.temp_dir and task.temp_dir.exists():
                shutil.rmtree(task.temp_dir)
                self.logger.debug(f"Worker {worker_id}: Cleaned up temporary directory for frame {frame_num:03d}")
            
            return {'success': False, 'error': error_msg, 'failed_stage': 'exception'}
    
    def _pipeline_worker(self, stage: ProcessingStage, stage_queues: Dict, 
                        completed_tasks: Queue, worker_id: int):
        """Worker thread for pipeline parallelism"""
        executor = self.executors[worker_id % len(self.executors)]
        
        while True:
            try:
                task = stage_queues[stage].get(timeout=1.0)
                if task is None:  # Sentinel value to stop
                    break
                
                start_time = time.time()
                success = False
                
                try:
                    if stage == ProcessingStage.STEREO:
                        success = self._run_stereo_step_threaded(task, executor)
                    elif stage == ProcessingStage.HOLES:
                        success = self._run_holes_step_threaded(task, executor)
                    elif stage == ProcessingStage.FILTER:
                        success = self._run_filter_step_threaded(task, executor)
                    elif stage == ProcessingStage.TEXTURE:
                        success = self._run_texture_step_threaded(task, executor)
                    elif stage == ProcessingStage.CONVERT:
                        success = self._run_convert_step_threaded(task, executor)
                        
                        # final processing for convert stage
                        if success:
                            if self.config.get_config('processing.cleanup_intermediate'):
                                self._cleanup_intermediate_files(task.output_dir, task.frame_num)
                            
                            if not self._verify_final_output(task.output_dir, task.frame_num):
                                success = False
                                self._cleanup_failed_frame(task.output_dir, task.frame_num)
                    
                    duration = time.time() - start_time
                    
                    if success:
                        next_stage = self._get_next_stage(stage)
                        if next_stage:
                            task.stage = next_stage
                            stage_queues[next_stage].put(task)
                        else:
                            result = TaskResult(task, True, "", duration)
                            completed_tasks.put(result)
                    else:
                        result = TaskResult(task, False, f"Stage {stage.value} failed", duration)
                        completed_tasks.put(result)
                        
                except Exception as e:
                    duration = time.time() - start_time
                    result = TaskResult(task, False, str(e), duration)
                    completed_tasks.put(result)
                
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"Pipeline worker {worker_id} error: {e}")
                break
    
    def _get_next_stage(self, current_stage: ProcessingStage) -> Optional[ProcessingStage]:
        """Get the next stage in the pipeline"""
        stage_order = [
            ProcessingStage.STEREO,
            ProcessingStage.HOLES,
            ProcessingStage.FILTER,
            ProcessingStage.TEXTURE,
            ProcessingStage.CONVERT
        ]
        
        try:
            current_index = stage_order.index(current_stage)
            if current_index < len(stage_order) - 1:
                return stage_order[current_index + 1]
        except ValueError:
            pass
        
        return None
    
    def _run_stereo_step_threaded(self, task: FrameTask, executor: CrossPlatformExecutor) -> bool:
        """Thread-safe version of stereo processing step"""
        try:
            params = self.config.get_stereo_params()
            
            stereo_files = []
            for camera in self.file_manager.stereo_cameras:
                if camera not in task.images or camera not in task.calibs:
                    self.logger.error(f"Frame {task.frame_num:03d}: Missing {camera} files")
                    return False
                stereo_files.append((task.images[camera], task.calibs[camera]))
            
            # add retries for stereo processing (critical step)
            max_retries = self.config.get_config('processing.retry_attempts')
            if max_retries is None:
                max_retries = 1
            for attempt in range(max_retries):
                try:
                    success, output, duration = executor.run_stereo_processor(
                        params, stereo_files, task.output_dir, task.frame_num
                    )
                    
                    if success:
                        out_tsb = task.output_dir / f"out_{task.frame_num:03d}.tsb"
                        if out_tsb.exists():
                            return True
                        else:
                            self.logger.info(f"Frame {task.frame_num:03d}: Stereo output file missing (attempt {attempt+1}/{max_retries})")
                    else:
                        self.logger.info(f"Frame {task.frame_num:03d}: Stereo processing failed (attempt {attempt+1}/{max_retries}): {output}")
                
                except Exception as e:
                    self.logger.info(f"Frame {task.frame_num:03d}: Stereo exception (attempt {attempt+1}/{max_retries}): {e}")
                    
                # wait briefly before retry to avoid race conditions
                if attempt < max_retries - 1:
                    import time
                    time.sleep(0.1)
            
            # only log as ERROR if all retries failed
            self.logger.error(f"Frame {task.frame_num:03d}: All stereo processing attempts failed after {max_retries} tries")
            return False
            
        except Exception as e:
            self.logger.error(f"Frame {task.frame_num:03d}: Critical stereo error: {e}")
            return False
    
    def _run_holes_step_threaded(self, task: FrameTask, executor: CrossPlatformExecutor) -> bool:
        """Thread-safe version of hole filling step"""
        try:
            params = self.config.get_holes_params()
            success, output, duration = executor.run_hole_filler(
                params, task.frame_num, cwd=task.output_dir
            )
            
            if not success:
                return False
            
            filled_tsb = task.output_dir / f"filled_{task.frame_num:03d}.tsb"
            return filled_tsb.exists()
            
        except Exception:
            return False
    
    def _run_filter_step_threaded(self, task: FrameTask, executor: CrossPlatformExecutor) -> bool:
        """Thread-safe version of surface filtering step"""
        try:
            params = self.config.get_filter_params()
            success, output, duration = executor.run_surface_filter(
                params, task.frame_num, cwd=task.output_dir
            )
            
            if not success:
                return False
            
            filtered_tsb = task.output_dir / f"filtered_{task.frame_num:03d}.tsb"
            return filtered_tsb.exists()
            
        except Exception:
            return False
    
    def _run_texture_step_threaded(self, task: FrameTask, executor: CrossPlatformExecutor) -> bool:
        """Thread-safe version of texture mapping step"""
        try:
            texture_files = []
            for camera in self.file_manager.texture_cameras:
                if camera not in task.images or camera not in task.calibs:
                    return False
                texture_files.append((task.images[camera], task.calibs[camera]))
            
            success, output, duration = executor.run_texture_mapper(
                task.frame_num, texture_files, task.output_dir
            )
            
            if not success:
                return False
            
            textured_tsb = task.output_dir / f"textured_{task.frame_num:03d}.tsb"
            return textured_tsb.exists()
            
        except Exception:
            return False
    
    def _run_convert_step_threaded(self, task: FrameTask, executor: CrossPlatformExecutor) -> bool:
        """Thread-safe version of mesh conversion step"""
        try:
            success, output, duration = executor.run_mesh_converter(
                task.frame_num, cwd=task.output_dir
            )
            return success
            
        except Exception:
            return False
    
    def _cleanup_intermediate_files(self, output_dir: Path, frame_num: int):
        """Clean up intermediate files (thread-safe)"""
        files_to_keep = {'.bmp', '.obj'}
        removed_count = 0
        
        for pattern in [f"*_{frame_num:03d}.*", f"*{frame_num:03d}.*"]:
            for file_path in output_dir.glob(pattern):
                if file_path.is_file() and file_path.suffix.lower() not in files_to_keep:
                    try:
                        file_path.unlink()
                        removed_count += 1
                    except Exception:
                        pass  # Ignore cleanup errors in threaded mode
    
    def _cleanup_failed_frame(self, output_dir: Path, frame_num: int):
        """Clean up all files for a failed frame (thread-safe)"""
        try:
            patterns = [f"*_{frame_num:03d}.*", f"*{frame_num:03d}.*"]
            for pattern in patterns:
                for file_path in output_dir.glob(pattern):
                    if file_path.is_file():
                        try:
                            file_path.unlink()
                        except Exception:
                            pass  # Ignore cleanup errors
        except Exception:
            pass
    
    def _verify_final_output(self, output_dir: Path, frame_num: int) -> bool:
        """Verify final output files (thread-safe)"""
        try:
            obj_file = output_dir / f"frame_{frame_num:03d}.obj"
            bmp_file = output_dir / f"frame_{frame_num:03d}.bmp"
            
            return (obj_file.exists() and obj_file.stat().st_size > 0 and
                    bmp_file.exists() and bmp_file.stat().st_size > 0)
        except Exception:
            return False
    
    def _generate_error_report(self, failed_frames: List[Dict], output_dir: Path):
        """Generate detailed error report for failed frames"""
        try:
            error_report_path = output_dir / "error_report.txt"
            
            with open(error_report_path, 'w') as f:
                f.write("3D Reconstruction Pipeline - Error Report\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total failed frames: {len(failed_frames)}\n\n")
                
                # Group by failure stage
                stage_groups = {}
                for failure in failed_frames:
                    stage = failure['stage']
                    if stage not in stage_groups:
                        stage_groups[stage] = []
                    stage_groups[stage].append(failure)
                
                for stage, failures in stage_groups.items():
                    f.write(f"=== {stage.upper()} STAGE FAILURES ({len(failures)}) ===\n")
                    for failure in failures:
                        f.write(f"Frame {failure['frame']:03d}: {failure['reason']}\n")
                    f.write("\n")
                
                f.write("=== CLEANUP RECOMMENDATIONS ===\n")
                for failure in failed_frames:
                    frame_num = failure['frame']
                    f.write(f"Frame {frame_num:03d}: Check/remove intermediate files in output directory\n")
            
            self.logger.info(f"Error report generated: {error_report_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate error report: {e}")
    
    def cleanup_failed_frames(self, output_dir: Path, failed_frames: List[int] = None):
        """Clean up failed frames and intermediate files"""
        if failed_frames is None:
            # Find failed frames by looking for incomplete outputs
            failed_frames = []
            for tsb_file in output_dir.glob("out_*.tsb"):
                frame_num = int(tsb_file.stem.split('_')[1])
                obj_file = output_dir / f"frame_{frame_num:03d}.obj"
                if not obj_file.exists() or obj_file.stat().st_size == 0:
                    failed_frames.append(frame_num)
        
        cleaned_count = 0
        for frame_num in failed_frames:
            try:
                patterns = [f"*_{frame_num:03d}.*", f"*{frame_num:03d}.*"]
                for pattern in patterns:
                    for file_path in output_dir.glob(pattern):
                        if file_path.is_file():
                            file_path.unlink()
                            cleaned_count += 1
            except Exception as e:
                self.logger.warning(f"Could not cleanup frame {frame_num}: {e}")
        
        if cleaned_count > 0:
            self.logger.info(f"Cleaned up {cleaned_count} files from {len(failed_frames)} failed frames")
        
        return cleaned_count
