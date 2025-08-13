"""
Logging configuration for 3D reconstruction pipeline.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from tqdm import tqdm


class ReconstructionLogger:
    """Custom logger for the reconstruction pipeline"""
    
    def __init__(self, config: dict, base_dir: Path):
        """Initialize logger with configuration"""
        self.config = config['logging']
        self.base_dir = base_dir
        self.logger = None
        self.progress_bar = None
        self.failed_frames = []
        self.skipped_frames = []
        self._setup_logging()
    
    def _setup_logging(self):
        """Set up logging configuration"""
        self.logger = logging.getLogger('reconstruction')
        self.logger.setLevel(getattr(logging, self.config['level']))
        
        self.logger.handlers.clear()
        
        if self.config['enable_file']:
            log_file = self.base_dir / self.config['file']
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(getattr(logging, self.config['level']))
            
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        
        # console handler (only for warnings/errors by default)
        if self.config['enable_console']:
            console_handler = logging.StreamHandler(sys.stdout)
            console_level = self.config.get('console_level', 'WARNING')
            console_handler.setLevel(getattr(logging, console_level))
            
            console_formatter = logging.Formatter('%(levelname)s: %(message)s')
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)
    
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
    
    def print_user(self, message: str):
        """Print message directly to user (always visible)"""
        print(message)
    
    def create_progress_bar(self, total: int, desc: str) -> tqdm:
        """Create a progress bar"""
        self.progress_bar = tqdm(
            total=total,
            desc=desc,
            unit="frame",
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} frames [{elapsed}<{remaining}]'
        )
        return self.progress_bar
    
    def update_progress(self, n: int = 1, desc: Optional[str] = None):
        """Update progress bar"""
        if self.progress_bar:
            self.progress_bar.update(n)
            if desc:
                self.progress_bar.set_description(desc)
    
    def close_progress(self):
        """Close progress bar"""
        if self.progress_bar:
            self.progress_bar.close()
            self.progress_bar = None
    
    def add_failed_frame(self, frame_num: int, reason: str, step: str = None):
        """Record a failed frame"""
        failure_info = {
            'frame': frame_num,
            'reason': reason,
            'step': step
        }
        self.failed_frames.append(failure_info)
        self.logger.info(f"Frame {frame_num:03d} failed: {reason}" + (f" (at {step})" if step else ""))
    
    def add_skipped_frame(self, frame_num: int, reason: str):
        """Record a skipped frame"""
        skip_info = {
            'frame': frame_num,
            'reason': reason
        }
        self.skipped_frames.append(skip_info)
        self.logger.info(f"Frame {frame_num:03d} skipped: {reason}")
    
    def save_error_report(self) -> Optional[str]:
        """Save detailed error report to file and return file path"""
        if not self.failed_frames and not self.skipped_frames:
            return None
        
        error_report_file = self.base_dir / "logs" / "error_report.txt"
        error_report_file.parent.mkdir(parents=True, exist_ok=True)
        
        report_lines = []
        report_lines.append("3D Reconstruction Pipeline - Error Report")
        report_lines.append("=" * 50)
        report_lines.append("")
        
        if self.skipped_frames:
            report_lines.append(f"SKIPPED FRAMES ({len(self.skipped_frames)}):")
            report_lines.append("-" * 30)
            for skip in self.skipped_frames:
                report_lines.append(f"Frame {skip['frame']:03d}: {skip['reason']}")
            report_lines.append("")
        
        if self.failed_frames:
            report_lines.append(f"FAILED FRAMES ({len(self.failed_frames)}):")
            report_lines.append("-" * 30)
            for failure in self.failed_frames:
                step_info = f" (at {failure['step']})" if failure['step'] else ""
                report_lines.append(f"Frame {failure['frame']:03d}: {failure['reason']}{step_info}")
            report_lines.append("")
        
        with open(error_report_file, 'w') as f:
            f.write("\n".join(report_lines))
        
        return str(error_report_file)
    
    def print_summary(self, successful: int, total: int, total_time: float):
        """Print processing summary"""
        avg_time = total_time / total if total > 0 else 0
        
        self.print_user(f"\nProcessing completed")
        self.print_user(f"Successful: {successful}/{total} frames")
        self.print_user(f"Total time: {total_time:.1f}s")
        self.print_user(f"Average: {avg_time:.1f}s per frame")
        
        if successful == total and total > 0:
            self.print_user("All frames processed successfully")
        else:
            # save error report and show brief message
            error_report_path = self.save_error_report()
            if error_report_path:
                failed_count = len(self.failed_frames)
                skipped_count = len(self.skipped_frames)
                
                if failed_count > 0 and skipped_count > 0:
                    self.print_user(f"{failed_count} frames failed, {skipped_count} frames skipped")
                elif failed_count > 0:
                    self.print_user(f"{failed_count} frames failed")
                elif skipped_count > 0:
                    self.print_user(f"{skipped_count} frames skipped")
                
                self.print_user(f"Check error report for details: {error_report_path}")
            elif successful == 0:
                self.print_user("No frames processed successfully")
