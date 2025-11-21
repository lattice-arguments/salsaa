# Salsaa

A Rust implementation of Salsaa, a framework for constructing efficient and versatile lattice-based succinct arguments. The repository further implements two applications of SALSAAA: a SNARK/PCS and a folding scheme.

The codebase is an auxiliary material for *SALSAA: Sumcheck-aided Lattice-based Succinct Arguments and Applications* by
Shuto Kuriyama, Russell W. F. Lai, Michał Osadnik, and Lorenzo Tucci from Aalto University, Espoo, Finland.

## Experiments
The codebase has been benchmarked on a Dell PowerEdge XE8640,  with a 2x48 core Xeon Platinum 8468 2.1GHz processor. Results of the benchmarks for each class of parameters can be found on the `/experiments` folder.
Note that in the final report, the commit phase runtime has been subtracted from the total prover runtime, whereas in the experiment logs it is included in the total.

Parameters have been chosen via the script located in `/est`.
Proof sizes of [KLNO25](https://eprint.iacr.org/2025/1220) reflected for a similar parameters regime as one considered by our work are estimated via a script in `/knlo25-baseline`. 


## Codebase structure

The Intel HEXL C++ library, located as a submodule in `hexl-bindings`, is used as a backend library for performing NTT operations through SIMD registers,  using AVX-512 when available and AVX2 as a fallback.
The Rust source code is contained in the `src` directory. The `src/protocol.rs` file implements both a SNARK and a folding scheme protocol, using subroutines defined in `src/subroutines` such as fold, split, decomposition, and sumcheck-aided norm checks. Various matrix arithmetic operations are defined in `src/arithmetic.rs`.

## Instructions
### Requirements
Building the program requires GCC, Cmake, and Cargo nightly to be pre-installed.

### Building
It is necessary to first clone and build the HEXL submodule. Run 
```
git submodule update --init --recursive
```
Then run
```
make hexl
make wrapper
export LD_LIBRARY_PATH=./hexl-bindings/hexl/build/hexl/lib:$(pwd)
```
Once compiled, it is necessary to manually enable AVX-512 feature or AVX2 for the Rust compiler.
For AVX-512 supported systems run:
```
export RUSTFLAGS="-C target-feature=+avx512f -C target-cpu=native"
```
Otherwise, fallback to AVX2
```
export RUSTFLAGS="-C target-feature=+avx2 -C target-cpu=native"

```
Now the codebase can be run with 
```
cargo +nightly run --release --features A2
```
Where the feature flag (in the example `A2`) specifies the parameters set to be run for the SNARK protocol.
The folding scheme can be run with the `F0`, `F1`, `F2` feature flags.

## License

Salsaa is licensed under the Apache 2.0 License.
