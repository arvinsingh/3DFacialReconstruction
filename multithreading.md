# Multithreading Support


Following two parallel processing modes are available

#### 1. Frame-Parallel Processing (`frame_parallel`)
- **What it does**: Processes multiple frames simultaneously, each through the complete 5-step pipeline
- **Best for**: Sequences with many frames and sufficient CPU cores
- **Memory usage**: Higher (multiple frames in memory)
- **Expected speedup**: 2-8x depending on CPU cores

#### 2. Pipeline-Parallel Processing (`pipeline_parallel`) 
- **What it does**: Overlaps different processing stages across frames
- **Best for**: Sequences where pipeline stages have different processing times
- **Memory usage**: Moderate (streaming pipeline)
- **Expected speedup**: 1.5-4x depending on stage complexity

### Command Line

```bash
# Enable frame-parallel processing with auto-detected workers
python reconstruct.py --sequence --parallel --mode frame_parallel

# enable pipeline-parallel processing with 6 workers
python reconstruct.py --sequence --parallel --mode pipeline_parallel --workers 6

# Process specific frame range in parallel
python reconstruct.py --frame-range 0 10 --parallel --mode frame_parallel

# sequential processing (default)
python reconstruct.py --sequence
```

### Configuration File


```yaml
processing:
  enable_parallel: true              # Enable by default
  max_workers: 6                     # Number of worker threads (null = auto)
  parallel_mode: "frame_parallel"    # Default parallel mode
```

### Recommendations

#### Frame-Parallel
- Processing many frames (>10)
- High CPU core count (>4 cores)
- Sufficient memory (>8GB)
- Frames are independent

#### Pipeline-Parallel
- Limited memory (<4GB)
- Mixed processing stage complexity
- I/O bound operations
- Streaming-like processing needed

#### Sequential
- Single frame processing
- Limited CPU cores (<2)
- Memory constrained (<2GB)  
- Debugging issues
