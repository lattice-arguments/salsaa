#!/bin/bash -l
#SBATCH --mem=500G
#SBATCH --cpus-per-task 96
#SBATCH --hint=nomultithread
#SBATCH --mem-bind=local
#SBATCH --time=0:30:00
#SBATCH --partition=gpu-h100-80g

module load gcc cmake
make hexl-triton
make wrapper
export LD_LIBRARY_PATH=./hexl-bindings/hexl/build/hexl/lib:$(pwd)
export RUSTFLAGS="-C linker=gcc -C target-feature=+avx512f -C target-cpu=native"
# cargo run
# cargo test
# cargo bench

for f in A0 A1 A2 B0 B1 B2 C0 C1 C2 F0 F1 F2; do
    echo "Building with feature: $f"
    cargo +nightly run --release --features "$f" > "./experiments/$f.log" 2>&1
done
