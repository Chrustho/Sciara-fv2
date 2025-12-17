# Sciara-fv2: Multi-GPU Parallelization of Lava Flow Simulation

A CUDA-based parallelization and performance assessment of the Sciara-fv2 2D Cellular Automata lava flow simulation model.

**Course:** Massively Parallel Programming on GPUs  
**Institution:** Department of Mathematics and Computer Science, University of Calabria, Italy  
**Instructor:** Prof. Donato D'Ambrosio

---

## Table of Contents

- [Overview](#overview)
- [The Sciara-fv2 Model](#the-sciara-fv2-model)
- [CUDA Implementations](#cuda-implementations)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Building](#building)
- [Running the Simulation](#running-the-simulation)
- [Profiling and Performance Analysis](#profiling-and-performance-analysis)
- [Roofline Analysis](#roofline-analysis)
- [Dataset](#dataset)
- [Output Files](#output-files)
- [Correctness Verification](#correctness-verification)
- [References](#references)

---

## Overview

This project implements multiple CUDA parallelization strategies for the Sciara-fv2 lava flow simulation model, originally developed for simulating volcanic eruptions. The reference case study models the 2006 Mt. Etna (Italy) lava flow event.

The main objectives are:

1. Develop multiple CUDA parallel implementations with different memory optimization strategies
2. Perform comparative performance assessment across implementations
3. Apply the Roofline model to characterize kernel performance
4. Validate correctness against the reference serial/OpenMP implementation

---

## The Sciara-fv2 Model

Sciara-fv2 is a Cellular Automaton (CA) model that simulates lava flows through a discrete representation of the terrain. The model uses several state variables (substates):

| Substate | Description | Implementation |
|----------|-------------|----------------|
| Q_z | Topographic elevation above sea level | `Sz`, `Sz_next` |
| Q_h | Lava thickness | `Sh`, `Sh_next` |
| Q_T| Lava temperature | `ST`, `ST_next` |
| Q_f_8 | Outflows to 8 adjacent cells | `Mf` (buffered) |

Additional buffers (`Mv`, `Mb`, `Mhs`) support auxiliary computations including velocity, solidification, and cumulative thickness tracking.

The simulation loop executes the following elementary processes (kernels):

1. **Lava Emission** - Emits lava from active vents
2. **Outflow Computation** - Calculates lava distribution to neighboring cells
3. **Mass Balance** - Updates lava thickness based on inflows/outflows
4. **Temperature and Solidification** - Computes thermal evolution and solidification
5. **Boundary Conditions** - Enforces domain boundaries
6. **Reduction** - Computes total active lava for termination condition

---

## CUDA Implementations

Five CUDA implementations have been developed, all using CUDA Unified Memory (`cudaMallocManaged()`):

### 1. Global (`sciara_cuda`)
Straightforward parallelization using only global memory. Serves as the baseline implementation for performance comparison.

### 2. Tiled (`sciara_cuda_tiled`)
Shared memory-based tiled parallelization without halo cells. Loads tile data into shared memory for faster access within thread blocks.

### 3. Tiled with Halo (`sciara_cuda_tiled_halo`)
Enhanced tiled implementation where boundary threads perform additional work to copy halo cells from adjacent tiles into shared memory, enabling correct stencil computation at tile boundaries.

### 4. CfAMe (`sciara_cuda_cfame`)
Memory-Equivalent Conflict-Free tiled algorithm. This approach localizes outflows (`f⁴`) to shared memory and merges the outflow computation with mass balance kernels, avoiding global synchronization overhead. Requires model reformulation.

### 5. CfAMo (`sciara_cuda_cfamo`)
Memory-Optimized Conflict-Free tiled algorithm. An optimized variant of CfAMe with further memory access improvements. Also requires model revision.

---

## Project Structure

```
sciara-fv2-cuda/
├── Makefile                    # Main build configuration
├── README.md                   # This file
├── gpumembench.cu              # GPU microbenchmark for Roofline analysis
├── run_profiling.sh            # Automated profiling script
├── parse_metrics.py            # Profiling data parser
│
├── sciara_fv2.cu               # Global implementation entry point
├── sciara_fv2_tiled.cu         # Tiled implementation entry point
├── sciara_fv2_tiled_halo.cu    # Tiled with halo entry point
├── sciara_fv2_cfame.cu         # CfAMe implementation entry point
├── sciara_fv2_cfamo.cu         # CfAMo implementation entry point
│
├── Sciara.cu                   # Core data structures and initialization
├── constants.cu                # Simulation constants and parameters
│
├── implementations/
│   ├── global/                 # Global memory kernel implementations
│   │   └── kernel_global.cu
│   ├── tiled/                  # Tiled kernel implementations
│   ├── tiled_with_halos/       # Tiled with halo kernels
│   ├── cfame/                  # CfAMe kernels
│   └── cfamo/                  # CfAMo kernels
│
├── src/                        # C++ helper sources
│   ├── Makefile                # Serial/OpenMP build
│   ├── io.cpp                  # Input/Output operations
│   ├── util.cpp                # Utility functions
│   ├── configurationPathLib.cpp
│   └── *.h                     # Header files
│
├── data/
│   └── 2006/                   # Mt. Etna 2006 dataset
│       ├── 2006_000000000000.cfg
│       ├── 2006_000000000000_Morphology.asc
│       ├── 2006_000000000000_Vents.asc
│       ├── 2006_000000000000_EmissionRate.txt
│       └── ...
│
└── profiling_results/          # Generated profiling output
```

---

## Requirements

- **CUDA Toolkit** (tested with nvcc 11.8+)
- **GCC** (g++ 10.2.1 or compatible)
- **NVIDIA GPU** with Compute Capability 5.2+ (e.g., GTX 980 or newer)
- **GNU Make**
- **Python 3** (for profiling data parsing)
- **nvprof** or **Nsight Compute** (for profiling)

Optional:
- **QGIS** for visualizing output raster files
- **gnuplot** for generating performance plots

---

## Building

### Build All CUDA Implementations

```bash
make sciara_cuda
```

This compiles all five CUDA versions:
- `sciara_cuda` (Global)
- `sciara_cuda_tiled` (Tiled)
- `sciara_cuda_tiled_halo` (Tiled with Halo)
- `sciara_cuda_cfame` (CfAMe)
- `sciara_cuda_cfamo` (CfAMo)

### Build Individual Versions

```bash
make cuda              # Global only
make cuda_tiled        # Tiled only
make cuda_tiled_halo   # Tiled with Halo only
make cuda_cfame        # CfAMe only
make cuda_cfamo        # CfAMo only
```

### Build Serial/OpenMP Reference

```bash
cd src/
make
```

This builds `sciara_serial` and `sciara_omp` for correctness verification.

### Build GPU Microbenchmark

```bash
nvcc -arch=sm_52 -O3 gpumembench.cu -o gpumembench
```

### Clean Build Artifacts

```bash
make clean           # Remove executables
make wipe            # Remove object files
make wipe_data       # Remove output data files
make clean-profile   # Remove profiling results
make clean-all       # Full cleanup
```

---

## Running the Simulation

### Default Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Input Config | `./data/2006/2006_000000000000.cfg` | Initial configuration |
| Output Prefix | `./data/2006/output_2006` | Output file prefix |
| Steps | 16000 | Total simulation steps |
| Reduce Interval | 1000 | Steps between reduction operations |
| Thickness Threshold | 1.0 | Minimum active lava threshold |

### Run CUDA Versions

```bash
make run_cuda              # Global implementation
make run_cuda_tiled        # Tiled implementation
make run_cuda_tiled_halo   # Tiled with Halo
make run_cuda_cfame        # CfAMe implementation
make run_cuda_cfamo        # CfAMo implementation
```

### Run Reference Implementations

```bash
cd src/
make run       # Serial version
make run_omp   # OpenMP version (default: 8 threads)
```

To change OpenMP thread count, edit `THREADS` in `src/Makefile`.

### Manual Execution

```bash
./sciara_cuda <input_cfg> <output_prefix> <steps> <reduce_interval> <thickness_threshold>
```

Example:
```bash
./sciara_cuda ./data/2006/2006_000000000000.cfg ./data/2006/output_2006 16000 1000 1.0
```

---

## Profiling and Performance Analysis

### Automated Profiling

Run comprehensive profiling on all compiled CUDA executables:

```bash
make profile
```

This executes `run_profiling.sh`, which:
1. Runs `gpumembench` to collect GPU bandwidth specifications
2. Profiles each executable with nvprof collecting:
   - GPU execution summary
   - Compute metrics (FLOP counts: FP64, FP32, FP16)
   - Memory metrics (DRAM read/write throughput)
   - Occupancy statistics

Results are saved to `profiling_results/`.

### Parse Profiling Data

```bash
python3 parse_metrics.py
```

Generates:
- `roofline.dat` - Arithmetic intensity and GFLOPS per implementation
- `time.dat` - Execution times
- `occupancy.dat` - Achieved occupancy

### Manual Profiling with nvprof

```bash
# Execution time summary
nvprof --print-gpu-summary ./sciara_cuda <args>

# Detailed metrics
nvprof --metrics flop_count_dp,flop_count_sp,dram_read_throughput,dram_write_throughput ./sciara_cuda <args>

# Export to CSV
nvprof --csv --log-file profile.csv ./sciara_cuda <args>
```

---

## Dataset

The provided dataset represents the 2006 Mt. Etna lava flow:

| File | Description |
|------|-------------|
| `*.cfg` | Simulation parameters |
| `*_Morphology.asc` | Digital Elevation Model (DEM) |
| `*_Vents.asc` | Vent locations |
| `*_EmissionRate.txt` | Lava emission rates over time |
| `*_Thickness.asc` | Initial lava thickness |
| `*_Temperature.asc` | Initial temperature field |
| `*_SolidifiedLavaThickness.asc` | Solidified lava |

Files in `.asc` format are ASCII grid rasters compatible with GIS software (QGIS, ArcGIS).

---

## Output Files

After simulation completion, the following files are generated:

```
output_2006_000000016000.cfg                      # Final configuration
output_2006_000000016000_EmissionRate.txt         # Emission log
output_2006_000000016000_Morphology.asc           # Updated DEM
output_2006_000000016000_SolidifiedLavaThickness.asc
output_2006_000000016000_Temperature.asc          # Final temperature field
output_2006_000000016000_Thickness.asc            # Final lava thickness
output_2006_000000016000_Vents.asc
```

Visualize with QGIS by loading `.asc` files as raster layers.

---


## Implementation Notes

### Memory Access Optimization

- Use coalesced memory access patterns in reduction kernels (see Chapter 10 of CUDA textbook)
- The `GET/SET` and `BUF_GET/BUF_SET` macros handle 2D array indexing
- Flow divergence should be minimized in conditional kernels

### Compute Architecture

For GTX 980 (SM 5.2), use:
```makefile
NVCODE = -gencode=arch=compute_52,code="compute_52" -ftz=true
```

### Boundary Conditions

Cells at domain boundaries have their substates set to zero, preventing lava from leaving the simulation domain.

---

## References

1. D. D'Ambrosio et al., "Efficient Execution of Flow-Based Low-Order Stencil Algorithms on GPUs," SSRN, 2024. [Link](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5056757)

2. W. Spataro et al., "The latest release of the lava flows simulation model SCIARA: first application to Mt Etna (Italy)," Procedia Computer Science, vol. 1, pp. 17-26, 2010.

3. W.W. Hwu, D.B. Kirk, I. El Hajj, "Programming Massively Parallel Processors: A Hands-on Approach," 4th ed., Morgan Kaufmann, 2022.

4. E. Konstantinidis, Y. Cotronis, "A Quantitative Performance Evaluation of Fast on-Chip Memories of GPUs," PDP 2016.

5. S. Williams, A. Waterman, D. Patterson, "Roofline: An Insightful Visual Performance Model for Multicore Architectures," Communications of the ACM, 2009.

6. C. Yang et al., "Hierarchical Roofline Analysis for GPUs," Concurrency and Computation: Practice and Experience, 2020.

---

## License

This project is developed for academic purposes as part of the Massively Parallel Programming on GPUs course at the University of Calabria.

---

## Authors

Bruni Christian & Francesco Tieri 
Department of Mathematics and Computer Science  
University of Calabria, Italy

---

## Acknowledgments

- Prof. Donato D'Ambrosio for course instruction and project guidance
