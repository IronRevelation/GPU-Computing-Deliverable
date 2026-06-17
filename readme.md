# GPU SpMV Deliverable by Mattia Ferretti

## Implementations

- CPU CSR reference implementation
- GPU simple CSR kernel: one CUDA thread per row
- GPU adaptive CSR kernel:
  - normal rows: one thread per row
  - long rows: one CUDA block per row with shared-memory reduction
- cuSPARSE CSR SpMV comparison

## Build

make

## Data

Download the SuiteSparse Matrix Collection inputs used for the benchmarks:

```sh
./scripts/download_williams_matrices.sh
```

This creates `data/*.mtx` for the following matrices:

- pdb1HYS
- consph
- cant
- pwtk
- mac_econ_fwd500
- mc2depi
- cop20k_A
- scircuit
- webbase-1M
- rail4284

## Validation

make run

## Benchmark

./build/benchmark data/*.mtx

## Output

Benchmark CSV files are written to results/.

## Reproducibility Environment

The included benchmark results were produced with the following environment:

- Node: `edu01`
- Loaded module: `CUDA/12.3.2`
- GPU: NVIDIA A30, 24576 MiB
- NVIDIA driver: `550.144.03`
- NVIDIA-SMI reported CUDA version: `12.4`
- CUDA compiler: `nvcc` release `12.3`, `V12.3.107`
- GCC: `11.4.1 20230605 (Red Hat 11.4.1-2)`
- G++: `11.4.1 20230605 (Red Hat 11.4.1-2)`
- CPU: Intel Xeon Silver 4309Y CPU @ 2.80GHz

