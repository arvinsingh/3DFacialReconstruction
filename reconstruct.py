#!/usr/bin/env python3
"""
3D Reconstruction Pipeline for 3dMD System
Main entry point for processing stereo imaging data into textured 3D models.
"""

import sys
from pathlib import Path
import argparse
import logging

sys.path.insert(0, str(Path(__file__).parent / "src"))

from pipeline import ReconstructionPipeline
from parallel_pipeline import ParallelReconstructionPipeline
from config import ConfigManager
from file_utils import FileManager
from executor import CrossPlatformExecutor


def main():
    """Main entry point for the reconstruction pipeline"""
    parser = argparse.ArgumentParser(
        description="3D Reconstruction Pipeline for 3dMD System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--frame', '-f', type=int, 
                       help='Process single frame number (e.g., --frame 0)')
    
    parser.add_argument('--frame-range', '-r', nargs=2, type=int, metavar=('START', 'END'),
                       help='Process frame range (e.g., --frame-range 0 5)')
    
    parser.add_argument('--sequence', '-s', action='store_true',
                       help='Process all frames in the sequence directory')
    
    parser.add_argument('--sequence-dir', type=Path,
                       help='Directory containing sequence files (default: from config.yaml)')
    
    parser.add_argument('--output-dir', '-o', type=Path,
                       help='Output directory (default: from config.yaml)')
    
    parser.add_argument('--base-dir', type=Path,
                       help='Base directory (default: script directory)')
    
    parser.add_argument('--config', '-c', type=str, default='config.yaml',
                       help='Configuration file (default: config.yaml)')
    
    parser.add_argument('--info', action='store_true',
                       help='Show system information and exit')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output (override config)')
    
    parser.add_argument('--parallel', '-p', action='store_true',
                       help='Enable parallel processing (multithreading)')
    
    parser.add_argument('--workers', '-w', type=int, default=None,
                       help='Number of worker threads (default: CPU count + 4)')
    
    parser.add_argument('--mode', '-m', choices=['frame_parallel', 'pipeline_parallel'],
                       default='frame_parallel',
                       help='Parallel processing mode: frame_parallel (process multiple frames simultaneously) or pipeline_parallel (overlap processing stages)')
    
    args = parser.parse_args()
    
    # validate
    exclusive_count = sum([
        args.frame is not None,
        args.frame_range is not None,
        args.sequence
    ])
    
    if exclusive_count != 1 and not args.info:
        print("ERROR: Must specify exactly one of: --frame, --frame-range, or --sequence")
        sys.exit(1)
    
    try:
        if args.parallel:
            pipeline = ParallelReconstructionPipeline(args.base_dir, args.config, args.workers)
        else:
            pipeline = ReconstructionPipeline(args.base_dir, args.config)
        
        if args.verbose:
            pipeline.logger.logger.handlers[1].setLevel(logging.INFO)
            
    except Exception as e:
        print(f"Failed to initialize pipeline: {e}")
        sys.exit(1)
    
    if args.info:
        info = pipeline.get_system_info()
        print("System Information:")
        print("=" * 50)
        print(f"Base directory: {info['base_dir']}")
        print(f"Config directory: {info['config_dir']}")
        print(f"Binary directory: {info['bin_dir']}")
        print(f"\nAvailable configurations:")
        for config in info['available_configs']:
            print(f"  - {config}")
        print(f"\nExecutables:")
        for name, exec_info in info['executables'].items():
            status = "OK" if exec_info['exists'] else "MISSING"
            print(f"  - {name:<15} {status:<8} ({exec_info['size_mb']:.1f} MB)")
        print(f"\nSupported cameras:")
        print(f"  - Stereo: {', '.join(info['supported_cameras']['stereo'])}")
        print(f"  - Texture: {', '.join(info['supported_cameras']['texture'])}")
        sys.exit(0)
    
    # use config defaults or command line overrides
    sequence_dir = args.sequence_dir or pipeline.config.sequence_dir
    output_dir = args.output_dir or pipeline.config.output_dir
    
    # abs paths
    sequence_dir = sequence_dir.resolve()
    output_dir = output_dir.resolve()
    
    # frame range
    if args.frame is not None:
        start_frame = end_frame = args.frame
    elif args.frame_range:
        start_frame, end_frame = args.frame_range
    elif args.sequence:
        # process entire seq - get available frames
        file_mgr = FileManager()
        available_frames = file_mgr.get_available_frames(sequence_dir)
        if not available_frames:
            pipeline.logger.print_user(f"ERROR: No frames found in {sequence_dir}")
            sys.exit(1)
        start_frame = min(available_frames)
        end_frame = max(available_frames)
        pipeline.logger.print_user(f"Found {len(available_frames)} frames")
    else:
        start_frame = end_frame = 0
    
    try:
        if args.parallel:
            results = pipeline.process_frame_range_parallel(
                sequence_dir,
                sequence_dir,
                output_dir,
                start_frame,
                end_frame,
                args.mode
            )
        else:
            results = pipeline.process_frame_range(
                sequence_dir,
                sequence_dir,
                output_dir,
                start_frame,
                end_frame
            )
        
        successful = sum(results.values())
        total = len(results)
        
        if successful == total:
            sys.exit(0)
        else:
            sys.exit(1)
        
    except Exception as e:
        pipeline.logger.error(f"FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
