//use tokio::sync::Mutex;
use parking_lot::Mutex;
use std::{
    sync::{Arc, Barrier},
    time::{Duration, Instant},
};

use crate::{
    arithmetic::{
        complete_power_series, parallel_dot_single_series_matrix, IncompletePowerSeries,
        PowerSeries, MAX_THREADS,
    },
    cyclotomic_ring::{CyclotomicRing, Representation},
    helpers::println_with_timestamp,
    hexl::bindings::{
        eltwise_fma_mod, eltwise_mult_mod, eltwise_sub_mod, sum, sum_sq, sum_sq_fast,
    },
    subroutines::split::VerifierState,
};
use rand::Rng;
use rayon::prelude::*;

const MIN_CHUNK_SIZE: usize = 1028;

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

fn eltwise_mult_mod_par(result: &mut [u64], a: &[u64], b: &[u64], modulus: u64, chunks: usize) {
    let chunk_size = (a.len() / chunks).max(MIN_CHUNK_SIZE);

    result
        .par_chunks_mut(chunk_size)
        .zip_eq(a.par_chunks(chunk_size))
        .zip_eq(b.par_chunks(chunk_size))
        .for_each(|((res, a_chunk), b_chunk)| unsafe {
            eltwise_mult_mod(
                res.as_mut_ptr(),
                a_chunk.as_ptr(),
                b_chunk.as_ptr(),
                a_chunk.len() as u64,
                modulus,
            );
        });
}

fn eltwise_sub_mod_par(result: &mut [u64], a: &[u64], b: &[u64], modulus: u64, chunks: usize) {
    let chunk_size = (a.len() / chunks).max(MIN_CHUNK_SIZE);

    result
        .par_chunks_mut(chunk_size)
        .zip_eq(a.par_chunks(chunk_size))
        .zip_eq(b.par_chunks(chunk_size))
        .for_each(|((res, a_chunk), b_chunk)| unsafe {
            eltwise_sub_mod(
                res.as_mut_ptr(),
                a_chunk.as_ptr(),
                b_chunk.as_ptr(),
                a_chunk.len() as u64,
                modulus,
            );
        });
}

fn eltwise_fma_mod_par(
    result: &mut [u64],
    a: &[u64],
    b: &[u64],
    c: u64,
    modulus: u64,
    chunks: usize,
) {
    let chunk_size = (a.len() / chunks).max(MIN_CHUNK_SIZE);

    result
        .par_chunks_mut(chunk_size)
        .zip_eq(a.par_chunks(chunk_size))
        .zip_eq(b.par_chunks(chunk_size))
        .for_each(|((res, a_chunk), b_chunk)| unsafe {
            eltwise_fma_mod(
                res.as_mut_ptr(),
                a_chunk.as_ptr(),
                c,
                b_chunk.as_ptr(),
                a_chunk.len() as u64,
                modulus,
            );
        });
}
pub fn sum_fast_par(a: &[u64], modulus: u64, chunks: usize) -> u64 {
    let chunk_size = (a.len() / chunks).max(MIN_CHUNK_SIZE);

    a.par_chunks(chunk_size)
        .map(|a_chunk| unsafe { sum(a_chunk.as_ptr(), a_chunk.len() as u64, modulus) })
        .sum()
}
fn sum_sq_par(a: &[u64], modulus: u64, chunks: usize) -> u64 {
    let chunk_size = (a.len() / chunks).max(MIN_CHUNK_SIZE);

    a.par_chunks(chunk_size)
        .map(|a_chunk| unsafe { sum_sq(a_chunk.as_ptr(), a_chunk.len() as u64, modulus) })
        .sum()
}

fn sum_check<const MOD_Q: u64>(
    w: &Vec<u64>,
    verifier: &mut dyn FnMut([u64; 3]) -> u64,
    _num_of_sumcheks: usize,
) {
    let mut wit_len = w.len();

    let mut wit_folded: Option<Vec<_>> = None;
    let mut w_f_0: Vec<u64> = Vec::with_capacity(wit_len / 2);
    let mut w_diff: Vec<u64> = Vec::with_capacity(wit_len / 2);
    let mut w_f_2: Vec<u64> = Vec::with_capacity(wit_len / 2);

    unsafe {
        w_f_0.set_len(wit_len / 2);
        w_diff.set_len(wit_len / 2);
        w_f_2.set_len(wit_len / 2);
    }

    while wit_len > 1 {
        let ip_threads = MAX_THREADS;

        let wit = match wit_folded {
            Some(ref w_in) => w_in,
            None => w,
        };

        let (w_0, w_1) = wit.split_at(wit.len() / 2);
        let len = w_0.len();

        let mut w_f_1 = vec![0u64; len];

        eltwise_mult_mod_par(&mut w_f_0[..len], w_0, &w_0, MOD_Q, ip_threads);

        // w_1 - w_0
        eltwise_sub_mod_par(&mut w_diff[..len], &w_1, &w_0, MOD_Q, ip_threads);
        // f_1 = w_0 * (w_1 - w_0)
        eltwise_mult_mod_par(&mut w_f_1, &w_0, &w_diff[..len], MOD_Q, ip_threads);
        // f_2 = (w_1 - w_0)^2
        eltwise_mult_mod_par(
            &mut w_f_2[..len],
            &w_diff[..len],
            &w_diff[..len],
            MOD_Q,
            ip_threads,
        );

        // Compute the three sums in parallel
        let (sum_f_0, (sum_f_1, sum_f_2)) = rayon::join(
            || sum_fast_par(&w_f_0[..len], MOD_Q, ip_threads),
            || {
                rayon::join(
                    || sum_fast_par(&w_f_1, MOD_Q, ip_threads),
                    || sum_fast_par(&w_f_2[..len], MOD_Q, ip_threads),
                )
            },
        );

        let c = verifier([sum_f_0, sum_f_1, sum_f_2]);

        eltwise_fma_mod_par(&mut w_f_1, &w_diff[..len], &w_0, c, MOD_Q, ip_threads);

        wit_len = w_f_1.len();
        wit_folded = Some(w_f_1);
    }
}

fn _sum_check_wrapper<const MOD_Q: u64>(w: &Vec<u64>) -> (Vec<u64>, u64) {
    let mut claim = unsafe { sum_sq_fast(w.as_ptr(), w.len() as u64, MOD_Q) };
    let mut cs = Vec::new();
    sum_check::<MOD_Q>(
        w,
        &mut |[sum_f_0, sum_f_1, sum_f_2]| {
            // This is a placeholder for the verifier function.
            // In a real application, this would be replaced with actual verification logic.
            assert_eq!(
                (((2 * sum_f_0 as u128) % MOD_Q as u128
                    + (2 * sum_f_1 as u128) % MOD_Q as u128
                    + sum_f_2 as u128)
                    % MOD_Q as u128) as u64,
                claim % MOD_Q,
                "Sum check failed: {} + 2*{} + {} != {}",
                sum_f_0,
                sum_f_1,
                sum_f_2,
                claim
            );

            let mut rng = rand::rng();
            let c = rng.random_range(0..MOD_Q);
            cs.push(c);

            claim = ((sum_f_0 as u128
                + (2u128 * sum_f_1 as u128 % MOD_Q as u128) * c as u128 % MOD_Q as u128
                + (sum_f_2 as u128 * c as u128 % MOD_Q as u128) * c as u128 % MOD_Q as u128)
                % MOD_Q as u128) as u64;
            c
        },
        1,
    );
    (cs, claim)
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
pub struct SumCheckProverState<'a> {
    pub flattened_witness: rayon::iter::Enumerate<rayon::slice::Iter<'a, Vec<u64>>>,
}

pub struct VerifierSumcheckState<const MOD_Q: u64, const N: usize> {
    _g_claim: Vec<CyclotomicRing<MOD_Q, N>>,
    pub g_claim_curr: Vec<CyclotomicRing<MOD_Q, N>>,
    pub g_ts: Vec<CyclotomicRing<MOD_Q, N>>,
}

pub fn norm_check_1<'a, const MOD_Q: u64, const N: usize>(
    witness: &'a Vec<Vec<CyclotomicRing<MOD_Q, N>>>,
) -> (Vec<u64>, Vec<Vec<u64>>) {
    let ip_threads = MAX_THREADS;

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

    let claims: Vec<u64> = flattened_witness
        .par_iter_mut()
        .zip(witness.par_iter())
        .map(|(flat_row, row)| {
            flat_row
                .par_chunks_mut(N)
                .zip(row.par_iter())
                .for_each(|(dst, elem)| {
                    dst.copy_from_slice(&elem.data);
                });

            let res = sum_sq_par(flat_row, MOD_Q, ip_threads);
            res
        })
        .collect();

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

pub fn norm_check_2<const MOD_Q: u64, const N: usize>(
    prover_state: SumCheckProverState,
    verifier: Arc<dyn Fn([u64; 3], usize) -> u64 + Send + Sync>,
) {
    let len = prover_state.flattened_witness.len();
    prover_state.flattened_witness.for_each(|(i, w)| {
        let verifier = Arc::clone(&verifier);
        sum_check::<MOD_Q>(
            &w,
            &mut |[sum_f_0, sum_f_1, sum_f_2]| verifier([sum_f_0, sum_f_1, sum_f_2], i),
            len,
        );
    });
}

pub fn verify_norm_check_2<const MOD_Q: u64, const N: usize>(
    [sum_f_0, sum_f_1, sum_f_2]: [u64; 3],
    i: usize,
    verifier_state: &Arc<Mutex<SumCheckVerifierState<MOD_Q, N>>>,
    sync_barrier: &Barrier,
    shared_c: &Arc<Mutex<Option<u64>>>,
) -> u64 {
    let claim = {
        let state = verifier_state.lock();
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

    sync_barrier.wait();

    if i == 0 {
        let mut state = verifier_state.lock();
        let mut shared_c = shared_c.lock();
        // This is the first thread, we generate the random number
        let mut rng = rand::rng();
        let c = rng.random_range(0..MOD_Q);
        {
            *shared_c = Some(c);
        }
        state.cs.push(c);
    }

    sync_barrier.wait();
    let c = {
        let shared_c = shared_c.lock();
        shared_c.unwrap()
    };

    let mut state = verifier_state.lock();
    state.claims[i] = ((sum_f_0 as u128
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
            .par_iter()
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

pub fn make_verify_norm_check_2_wrapped<const MOD_Q: u64, const N: usize>(
    verifier_state: &VerifierState<MOD_Q, N>,
    sumcheck_verifier_state: SumCheckVerifierState<MOD_Q, N>,
) -> (
    Arc<Mutex<SumCheckVerifierState<MOD_Q, N>>>,
    Arc<Mutex<Duration>>,
    Arc<dyn Fn([u64; 3], usize) -> u64 + Send + Sync>,
) {
    let inner_verifier_runtime = Instant::now().elapsed();
    println!("number of witness rep: {}", verifier_state.wit_rep);
    let sync_barrier = Arc::new(Barrier::new(verifier_state.wit_rep));
    let sumcheck_verifier_state = Arc::new(Mutex::new(sumcheck_verifier_state));
    let inner_verifier_runtime = Arc::new(Mutex::new(inner_verifier_runtime));
    let shared_c = Arc::new(Mutex::new(None));

    let verify_norm_check_2_wrapped = {
        let sumcheck_verifier_state = Arc::clone(&sumcheck_verifier_state);
        let sync_barrier = Arc::clone(&sync_barrier);
        let shared_c = Arc::clone(&shared_c);
        Arc::new(move |[sum_f_0, sum_f_1, sum_f_2]: [u64; 3], i| {
            let c = verify_norm_check_2::<MOD_Q, N>(
                [sum_f_0, sum_f_1, sum_f_2],
                i,
                &sumcheck_verifier_state,
                &sync_barrier,
                &shared_c,
            );
            c
        }) as Arc<dyn Fn([u64; 3], usize) -> u64 + Send + Sync>
    };

    (
        sumcheck_verifier_state.clone(),
        inner_verifier_runtime,
        verify_norm_check_2_wrapped,
    )
}
