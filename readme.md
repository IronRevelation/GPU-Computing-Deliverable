# GPU SpMV Deliverable

## Implementations

- CPU CSR reference implementation
- GPU simple CSR kernel: one CUDA thread per row
- GPU adaptive CSR kernel:
  - normal rows: one thread per row
  - long rows: one CUDA block per row with shared-memory reduction
- cuSPARSE CSR SpMV comparison

## Build

make

## Validation

make run

## Benchmark

./build/benchmark data/*.mtx

## Output

Benchmark CSV files are written to results/.