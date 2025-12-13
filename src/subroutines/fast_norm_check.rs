//use tokio::sync::Mutex;
use std::time::{Duration, Instant};

use crate::{
    arithmetic::{
        complete_power_series, parallel_dot_single_series_matrix, IncompletePowerSeries,
        PowerSeries,
    },
    cyclotomic_ring::{CyclotomicRing, Representation},
    helpers::println_with_timestamp,
    hexl::bindings::{eltwise_fma_mod, eltwise_mult_mod, eltwise_sub_mod, sum, sum_sq},
    subroutines::split::VerifierState,
};
use rand::Rng;

pub fn tensor<const MOD_Q: u64>(cs: &Vec<u64>) -> Vec<u64> {
    let mut result = vec![1u64];
    for &c in cs {
        let mut next = Vec::with_capacity(result.len() * 2);
        for &val in &result {
            let val = val as u128;
            let c = c as u128;
            let mod_q = MOD_Q as u128;
            next.push(((val * ((mod_q + 1 - c) % mod_q)) % mod_q) as u64);
            next.push(((val * (c % mod_q)) % mod_q) as u64);
        }
        result = next;
    }
    result
}

#[test]
fn test_tensor_basic() {
    const MOD_Q: u64 = 17;
    let cs = vec![3, 5];
    let result = tensor::<MOD_Q>(&cs);
    // For cs = [3, 5], MOD_Q = 17:
    // Step 1: result = [1]
    //   c = 3: next = [1 * (17 + 1 - 3) % 17, 1 * 3 % 17] = [15 % 17, 3 % 17] = [15, 3]
    // Step 2: result = [15, 3]
    //   c = 5:
    //     15 * (17 + 1 - 5) % 17 = 15 * 13 % 17 = 195 % 17 = 8
    //     15 * 5 % 17 = 75 % 17 = 7
    //     3 * 13 % 17 = 39 % 17 = 5
    //     3 * 5 % 17 = 15 % 17 = 15
    //   next = [8, 7, 5, 15]
    assert_eq!(result, vec![8, 7, 5, 15]);
}

fn eltwise_mult_mod_st(result: &mut [u64], a: &[u64], b: &[u64], modulus: u64) {
    unsafe {
        eltwise_mult_mod(
            result.as_mut_ptr(),
            a.as_ptr(),
            b.as_ptr(),
            a.len() as u64,
            modulus,
        );
    }
}

fn eltwise_sub_mod_st(result: &mut [u64], a: &[u64], b: &[u64], modulus: u64) {
    unsafe {
        eltwise_sub_mod(
            result.as_mut_ptr(),
            a.as_ptr(),
            b.as_ptr(),
            a.len() as u64,
            modulus,
        );
    }
}

fn eltwise_fma_mod_st(result: &mut [u64], a: &[u64], b: &[u64], c: u64, modulus: u64) {
    unsafe {
        eltwise_fma_mod(
            result.as_mut_ptr(),
            a.as_ptr(),
            c,
            b.as_ptr(),
            a.len() as u64,
            modulus,
        );
    }
}
pub fn sum_fast_st(a: &[u64], modulus: u64) -> u64 {
    unsafe { sum(a.as_ptr(), a.len() as u64, modulus) }
}

fn sum_sq_st(a: &[u64], modulus: u64) -> u64 {
    unsafe { sum_sq(a.as_ptr(), a.len() as u64, modulus) }
}

pub fn norm_check_2<const MOD_Q: u64, const N: usize>(
    prover_state: &mut SumCheckProverState,
    verifier_state: &mut SumCheckVerifierState<MOD_Q, N>,
) -> Duration {
    let m = prover_state.flattened_witness.len();

    let mut cur: Vec<Vec<u64>> = std::mem::take(&mut prover_state.flattened_witness);

    let mut wit_len = cur[0].len();
    debug_assert!(wit_len.is_power_of_two());
    let mut verifier_runtime = Instant::now().elapsed();

    while wit_len > 1 {
        let half = wit_len / 2;

        let mut sums: Vec<[u64; 3]> = vec![[0u64; 3]; m];

        for i in 0..m {
            let wit = &cur[i];
            let (w0, w1) = wit.split_at(half);

            let mut f0 = vec![0u64; half];
            let mut diff = vec![0u64; half];
            let mut f1 = vec![0u64; half];
            let mut f2 = vec![0u64; half];

            // f0 = w0^2
            eltwise_mult_mod_st(&mut f0, w0, w0, MOD_Q);
            // diff = w1 - w0
            eltwise_sub_mod_st(&mut diff, w1, w0, MOD_Q);
            // f1 = w0 * (w1 - w0)
            eltwise_mult_mod_st(&mut f1, w0, &diff, MOD_Q);
            // f2 = (w1 - w0)^2
            eltwise_mult_mod_st(&mut f2, &diff, &diff, MOD_Q);

            let sum_f_0 = sum_fast_st(&f0, MOD_Q);
            let sum_f_1 = sum_fast_st(&f1, MOD_Q);
            let sum_f_2 = sum_fast_st(&f2, MOD_Q);

            let now = Instant::now();
            verify_norm_check_2_p0::<MOD_Q, N>([sum_f_0, sum_f_1, sum_f_2], i, verifier_state);
            verifier_runtime = verifier_runtime + now.elapsed();

            sums[i] = [sum_f_0, sum_f_1, sum_f_2];
        }

        // sample one shared c for this round (replaces the barrier + i==0 thread)
        let mut rng = rand::rng();
        let c = rng.random_range(0..MOD_Q);
        verifier_state.cs.push(c);

        // second pass: update claims for all i and fold witnesses for all i
        let mut next: Vec<Vec<u64>> = (0..m).map(|_| vec![0u64; half]).collect();

        for i in 0..m {
            let [sum_f_0, sum_f_1, sum_f_2] = sums[i];

            let now = Instant::now();
            // update claim[i] with this round's c
            verifier_state.claims[i] = ((sum_f_0 as u128
                + (2u128 * sum_f_1 as u128 % MOD_Q as u128) * c as u128 % MOD_Q as u128
                + (sum_f_2 as u128 * c as u128 % MOD_Q as u128) * c as u128 % MOD_Q as u128)
                % MOD_Q as u128) as u64;

            verifier_runtime = verifier_runtime + now.elapsed();

            // fold witness: w_next = w0 + c*(w1 - w0)
            let wit = &cur[i];
            let (w0, w1) = wit.split_at(half);

            let mut diff = vec![0u64; half];
            eltwise_sub_mod_st(&mut diff, w1, w0, MOD_Q);
            eltwise_fma_mod_st(&mut next[i], &diff, w0, c, MOD_Q);
        }

        cur = next;
        wit_len = half;
    }

    prover_state.flattened_witness = cur;
    verifier_runtime
}

#[test]
fn test_sum_check() {
    const MOD_Q: u64 = 65537; // Example modulus
    let mut rng = rand::rng();
    let mut w = vec![0; 128];
    for i in 0..w.len() {
        w[i] = rng.random_range(0..MOD_Q);
    }
    let (cs, claim) = _sum_check_wrapper::<65537>(&w);

    let cs_extended = tensor::<MOD_Q>(&cs);

    let inner_product_w_cs = w
        .iter()
        .zip(cs_extended.iter())
        .map(|(&w, &c)| (w * c) % MOD_Q)
        .fold(0, |acc, x| (acc + x) % MOD_Q);

    assert_eq!(
        ((inner_product_w_cs as u128 * inner_product_w_cs as u128) % MOD_Q as u128) as u64,
        claim,
        "Inner product check failed: {} != {}",
        inner_product_w_cs,
        claim
    );
}

#[test]
fn test_sum_sq() {
    let w = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let sum_sq_result = unsafe { sum_sq(w.as_ptr(), w.len() as u64, 71) };
    assert_eq!(sum_sq_result, 62);
}

#[derive(Clone)]
pub struct SumCheckProverState {
    pub flattened_witness: Vec<Vec<u64>>,
}

pub struct VerifierSumcheckState<const MOD_Q: u64, const N: usize> {
    _g_claim: Vec<CyclotomicRing<MOD_Q, N>>,
    pub g_claim_curr: Vec<CyclotomicRing<MOD_Q, N>>,
    pub g_ts: Vec<CyclotomicRing<MOD_Q, N>>,
}

pub fn norm_check_1<'a, const MOD_Q: u64, const N: usize>(
    witness: &'a Vec<Vec<CyclotomicRing<MOD_Q, N>>>,
) -> (Vec<u64>, Vec<Vec<u64>>) {
    let row_sizes: Vec<usize> = witness
        .iter()
        .map(|row| row.iter().map(|elem| elem.data.len()).sum())
        .collect();

    let mut flattened_witness: Vec<Vec<u64>> = row_sizes
        .iter()
        .map(|&len| {
            let mut v = Vec::with_capacity(len);
            unsafe {
                v.set_len(len);
            }
            v
        })
        .collect();

    let mut claims = Vec::with_capacity(flattened_witness.len());

    for (flat_row, row) in flattened_witness.iter_mut().zip(witness.iter()) {
        for (dst, elem) in flat_row.chunks_mut(N).zip(row.iter()) {
            dst.copy_from_slice(&elem.data);
        }

        let res = sum_sq_st(flat_row, MOD_Q);
        claims.push(res);
    }

    (claims, flattened_witness)
}

#[derive(Clone, Debug)]
pub struct SumCheckVerifierState<const MOD_Q: u64, const N: usize> {
    pub claims: Vec<u64>,
    pub cs: Vec<u64>,
}

pub fn verify_norm_check_1<const MOD_Q: u64, const N: usize>(
    claims: Vec<u64>,
) -> SumCheckVerifierState<MOD_Q, N> {
    SumCheckVerifierState {
        claims: claims,
        cs: Vec::new(),
    }
}

pub fn verify_norm_check_2_p0<const MOD_Q: u64, const N: usize>(
    [sum_f_0, sum_f_1, sum_f_2]: [u64; 3],
    i: usize,
    verifier_state: &SumCheckVerifierState<MOD_Q, N>,
) {
    let claim = {
        let state = verifier_state;
        state.claims[i] % MOD_Q
    };

    assert_eq!(
        (((2 * sum_f_0 as u128) % MOD_Q as u128
            + (2 * sum_f_1 as u128) % MOD_Q as u128
            + sum_f_2 as u128)
            % MOD_Q as u128) as u64,
        claim,
        "Sum check failed" // {} + 2*{} + {} != {}, {}", sum_f_0, sum_f_1, sum_f_2, verifier_state.lock().claims[i], i
    );
}

pub fn verify_norm_check_2_p1<const MOD_Q: u64, const N: usize>(
    [sum_f_0, sum_f_1, sum_f_2]: [u64; 3],
    i: usize,
    c: &mut Option<u64>,
    verifier_state: &mut SumCheckVerifierState<MOD_Q, N>,
) -> u64 {
    if i == 0 {
        // This is the first thread, we generate the random number
        let mut rng = rand::rng();
        let new_c = rng.random_range(0..MOD_Q);
        {
            *c = Some(new_c);
        }
        verifier_state.cs.push(new_c);
    }

    let c = c.unwrap();

    verifier_state.claims[i] = ((sum_f_0 as u128
        + (2u128 * sum_f_1 as u128 % MOD_Q as u128) * c as u128 % MOD_Q as u128
        + (sum_f_2 as u128 * c as u128 % MOD_Q as u128) * c as u128 % MOD_Q as u128)
        % MOD_Q as u128) as u64;
    c
}

pub fn norm_check_3<const MOD_Q: u64, const N: usize>(
    fields_tensor_structure: &Vec<u64>,
    power_series: &mut Vec<PowerSeries<MOD_Q, N>>,
    witness: &Vec<Vec<CyclotomicRing<MOD_Q, N>>>,
) -> Vec<CyclotomicRing<MOD_Q, N>> {
    let split = N.ilog2() as usize;

    let (ring_tensor, base_element) =
        fields_tensor_structure.split_at(fields_tensor_structure.len() - split);
    let base_element_extended = tensor::<MOD_Q>(&base_element.to_vec());
    let mut base = CyclotomicRing::<MOD_Q, N> {
        data: base_element_extended.try_into().unwrap(),
        representation: Representation::Coefficient,
    }
    .conjugate();
    base.to_incomplete_ntt_representation();

    let now = Instant::now();
    let ps = complete_power_series(IncompletePowerSeries::<MOD_Q, N> {
        base_layer: vec![base],
        tensors: ring_tensor
            .iter()
            .map(|t| {
                let mut t0 = CyclotomicRing::<MOD_Q, N>::constant(MOD_Q + 1 - *t);
                t0.to_incomplete_ntt_representation();
                let mut t1 = CyclotomicRing::<MOD_Q, N>::constant(*t);
                t1.to_incomplete_ntt_representation();
                vec![t0, t1]
            })
            .collect(),
    });
    let elapsed = now.elapsed();
    println_with_timestamp!("Time to complete power series: {:.2?}", elapsed);

    let now = Instant::now();
    let new_rhs = parallel_dot_single_series_matrix(&ps, &witness);
    let elapsed = now.elapsed();
    println_with_timestamp!("Time to compute new RHS: {:.2?}", elapsed);
    power_series.push(ps);

    new_rhs
}

pub fn verify_norm_check_3<const MOD_Q: u64, const N: usize>(
    mut new_rhs: Vec<CyclotomicRing<MOD_Q, N>>,
    verifier_sumcheck_state: &mut SumCheckVerifierState<MOD_Q, N>,
    verifier_state: &mut VerifierState<MOD_Q, N>,
) {
    for j in 0..verifier_state.wit_rep {
        new_rhs[j].to_coeff_representation();
        assert_eq!(
            ((new_rhs[j].data[0] as u128 * new_rhs[j].data[0] as u128) % MOD_Q as u128) as u64,
            verifier_sumcheck_state.claims[j],
            "Inner product check failed for index {}: {} != {}",
            j,
            new_rhs[j].data[0],
            verifier_sumcheck_state.claims[j]
        );
        new_rhs[j].to_incomplete_ntt_representation();
    }

    verifier_state.rhs.push(new_rhs);
}
