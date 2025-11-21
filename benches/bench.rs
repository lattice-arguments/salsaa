use criterion::{criterion_group, criterion_main, Criterion};
use ring_arith::cyclotomic_ring::*;
use std::hint::black_box;

const N: usize = 128;
// const MOD_Q: u64 = 4546383823830515713; // Example modulus
const MOD_Q: u64 = 1125899904679937; // Example modulus IFMA
                                     // 1125899904679937

fn bench_cyclotomic_arithmetic(c: &mut Criterion) {
    c.bench_function("naive multiplication", |b| {
        b.iter_with_setup(
            || {
                let mut operand1 = CyclotomicRing::<MOD_Q, N>::random();
                let mut operand2 = CyclotomicRing::<MOD_Q, N>::random();
                (operand1, operand2)
            },
            |(operand1, operand2)| {
                naive_multiply(black_box(&mut operand1), black_box(&mut operand2))
            },
        )
    });
    c.bench_function("full ntt multiplication", |b| {
        b.iter_with_setup(
            || {
                let mut operand1 = CyclotomicRing::<MOD_Q, N>::random();
                let mut operand2 = CyclotomicRing::<MOD_Q, N>::random();
                (operand1, operand2)
            },
            |(mut operand1, mut operand2)| {
                fully_splitting_ntt_multiplication(
                    black_box(&mut operand1),
                    black_box(&mut operand2),
                )
            },
        )
    });

    c.bench_function("partial ntt multiplication", |b| {
        b.iter_with_setup(
            || {
                let mut operand1 = CyclotomicRing::<MOD_Q, N>::random();
                let mut operand2 = CyclotomicRing::<MOD_Q, N>::random();
                (operand1, operand2)
            },
            |(mut operand1, mut operand2)| {
                incomplete_ntt_multiplication(black_box(&mut operand1), black_box(&mut operand2))
            },
        )
    });
}

criterion_group!(benches, bench_cyclotomic_arithmetic);
criterion_main!(benches);
