use std::time::{Duration, Instant};

use num_traits::Zero;
use rayon::prelude::*;

use crate::{
    arithmetic::{
        add_no_reduction, complete_power_series, parallel_dot_series_matrix,
        parallel_dot_single_series_matrix, IncompletePowerSeries, PowerSeries,
    },
    cyclotomic_ring::CyclotomicRing,
    helpers::println_with_timestamp,
    hexl::bindings::eltwise_reduce_mod,
    subroutines::split::VerifierState,
};

const HEIGHT: usize = 256;

pub struct Challenge {
    pub is_zero: Vec<Vec<bool>>,
    pub sign: Vec<Vec<bool>>,
}

// we project with challennges from 0, \pm 1 so we represent this as two booleans.
// one for the sign and one for the zero challenge
pub fn challenge_for_project<const MOD_Q: u64, const N: usize>(
    verifier_state: &VerifierState<MOD_Q, N>,
) -> Challenge {
    let width = HEIGHT * verifier_state.wit_rep;
    println_with_timestamp!("Generating challenge for project with width: {}", width);
    Challenge {
        is_zero: (0..HEIGHT)
            .map(|_| (0..width).map(|_| rand::random::<bool>()).collect())
            .collect(),
        sign: (0..HEIGHT)
            .map(|_| (0..width).map(|_| rand::random::<bool>()).collect())
            .collect(),
    }
}

#[inline]
fn reduce<const MOD_Q: u64, const N: usize>(
    /*zn_base: &ZnBase,*/ x: &mut CyclotomicRing<MOD_Q, N>,
) {
    unsafe {
        eltwise_reduce_mod(x.data.as_mut_ptr(), x.data.as_ptr(), N as u64, MOD_Q);
    }
}

pub fn project<const MOD_Q: u64, const N: usize>(
    power_series: &Vec<PowerSeries<MOD_Q, N>>,
    witness: &Vec<Vec<CyclotomicRing<MOD_Q, N>>>,
    challenge: &Challenge,
) -> (
    Vec<Vec<CyclotomicRing<MOD_Q, N>>>,
    Vec<Vec<CyclotomicRing<MOD_Q, N>>>,
) {
    let chunk_width = witness.len() * HEIGHT;
    if chunk_width > witness[0].len() {
        panic!("Chunk width is greater than witness width");
    }

    let indices: Vec<_> = (0..witness.len())
        .flat_map(|i| (0..witness[i].len() / chunk_width).map(move |j| (i, j)))
        .collect();

    let mut projected_witness: Vec<CyclotomicRing<MOD_Q, N>> =
        Vec::with_capacity(indices.len() * HEIGHT);
    unsafe { projected_witness.set_len(indices.len() * HEIGHT) };

    let time = Instant::now();
    projected_witness
        .par_chunks_mut(HEIGHT)
        .enumerate()
        .for_each(|(idx, chunk)| {
            let (i, j) = indices[idx];
            let row_chunk = &witness[i][j * chunk_width..(j + 1) * chunk_width];

            chunk.par_iter_mut().enumerate().for_each(|(k, slot)| {
                let mut positive_sum = CyclotomicRing::<MOD_Q, N>::zero();
                let mut negative_sum = CyclotomicRing::<MOD_Q, N>::zero();

                for l in 0..chunk_width {
                    if !challenge.is_zero[k][l] {
                        if challenge.sign[k][l] {
                            add_no_reduction(&mut positive_sum, &row_chunk[l]);
                        } else {
                            add_no_reduction(&mut negative_sum, &row_chunk[l]);
                        }
                    }
                }
                reduce(&mut positive_sum);
                reduce(&mut negative_sum);
                *slot = positive_sum - negative_sum;
            });
        });
    println!("time to project: {:?}", time.elapsed());

    let projected_witness_matrix = vec![projected_witness];
    let rhs = parallel_dot_series_matrix::<MOD_Q, N>(&power_series, &projected_witness_matrix);

    (projected_witness_matrix, rhs)
}

pub fn batching_challenge_for_project<const MOD_Q: u64, const N: usize>() -> (
    CyclotomicRing<MOD_Q, N>,
    CyclotomicRing<MOD_Q, N>,
    CyclotomicRing<MOD_Q, N>,
) {
    // we have the first challenge to batch challenge matrix, second to batch across projection within a single column, third to batch across the columns
    (
        CyclotomicRing::constant(1),
        CyclotomicRing::random(),
        CyclotomicRing::random(),
    )
}

fn consecutive_powers<const MOD_Q: u64, const N: usize>(
    challenge: CyclotomicRing<MOD_Q, N>,
    width: usize,
) -> Vec<CyclotomicRing<MOD_Q, N>> {
    let mut consecutive_powers = Vec::with_capacity(width);
    let mut current_power = CyclotomicRing::<MOD_Q, N>::one();
    for _ in 0..width {
        consecutive_powers.push(current_power.clone());
        current_power = current_power * challenge;
    }
    consecutive_powers
}

pub fn batch_projections<const MOD_Q: u64, const N: usize>(
    projected_witness: &Vec<Vec<CyclotomicRing<MOD_Q, N>>>,
    witness: &Vec<Vec<CyclotomicRing<MOD_Q, N>>>,
    projection_challenge: &Challenge,
    challenge: (
        CyclotomicRing<MOD_Q, N>,
        CyclotomicRing<MOD_Q, N>,
        CyclotomicRing<MOD_Q, N>,
    ),
    power_series: &mut Vec<PowerSeries<MOD_Q, N>>,
) -> (
    PowerSeries<MOD_Q, N>,
    Vec<CyclotomicRing<MOD_Q, N>>,
    Vec<CyclotomicRing<MOD_Q, N>>,
    Duration
) {
    let chunk_width = witness.len() * HEIGHT;
    let c_0_consecutive_powers = consecutive_powers::<MOD_Q, N>(challenge.0, HEIGHT);

    // Pre-allocate cj
    let mut cj = Vec::with_capacity(chunk_width);
    unsafe {
        cj.set_len(chunk_width);
    }

    let now = Instant::now();

    cj.par_iter_mut().enumerate().for_each(|(j, cj_j)| {
        let acc = c_0_consecutive_powers
            .par_iter()
            .enumerate()
            .filter(|(i, _)| !projection_challenge.is_zero[*i][j])
            .map(|(i, power)| {
                if projection_challenge.sign[i][j] {
                    *power
                } else {
                    -power
                }
            })
            .reduce(|| CyclotomicRing::<MOD_Q, N>::zero(), |a, b| a + b);

        *cj_j = acc;
    });
    let verifier_runtime = now.elapsed();

    let nof_tensor_layers = (witness[0].len() / chunk_width).ilog2() as usize;
    let nof_tensor_layers_projection = nof_tensor_layers + witness.len().ilog2() as usize;

    let witness_lhs = complete_power_series(IncompletePowerSeries {
        tensors: vec![vec![CyclotomicRing::one(); 2]; nof_tensor_layers],
        base_layer: cj,
    });

    let projected_lhs = complete_power_series(IncompletePowerSeries {
        tensors: vec![vec![CyclotomicRing::one(); 2]; nof_tensor_layers_projection],
        base_layer: c_0_consecutive_powers,
    });

    let witness_rhs = parallel_dot_single_series_matrix::<MOD_Q, N>(&witness_lhs, &witness);

    let projection_rhs =
        parallel_dot_single_series_matrix::<MOD_Q, N>(&projected_lhs, &projected_witness);

    let sum_rhs = witness_rhs
        .iter()
        .fold(CyclotomicRing::<MOD_Q, N>::zero(), |acc, x| acc + *x);

    assert_eq!(projection_rhs[0], sum_rhs, "Inner product mismatch");

    power_series.push(witness_lhs);

    (projected_lhs, projection_rhs, witness_rhs, verifier_runtime)
}

pub fn verify_batching<const MOD_Q: u64, const N: usize>(
    witness_new_rhs: Vec<CyclotomicRing<MOD_Q, N>>,
    projection_rhs: &Vec<CyclotomicRing<MOD_Q, N>>,
    mut verifier_state: VerifierState<MOD_Q, N>,
) -> VerifierState<MOD_Q, N> {
    let sum_rhs = witness_new_rhs
        .iter()
        .fold(CyclotomicRing::<MOD_Q, N>::zero(), |acc, x| acc + *x);

    assert_eq!(projection_rhs[0], sum_rhs, "Inner product mismatch");

    println_with_timestamp!(
        "rhs dimensions: {} x {}",
        verifier_state.rhs.len(),
        verifier_state.rhs[0].len()
    );
    verifier_state.rhs.push(witness_new_rhs);
    VerifierState {
        wit_len: verifier_state.wit_len,
        wit_rep: verifier_state.wit_rep,
        rhs: verifier_state.rhs,
    }
}

pub fn send_cross_terms_before_join<const MOD_Q: u64, const N: usize>(
    projected_lhs: PowerSeries<MOD_Q, N>,
    power_series: &mut Vec<PowerSeries<MOD_Q, N>>,
    witness_wrong_orientation: &Vec<Vec<CyclotomicRing<MOD_Q, N>>>,
    projected_witness_wrong_orientation: &Vec<Vec<CyclotomicRing<MOD_Q, N>>>,
) -> (CyclotomicRing<MOD_Q, N>, CyclotomicRing<MOD_Q, N>) {
    let l0 =
        parallel_dot_single_series_matrix::<MOD_Q, N>(&projected_lhs, &witness_wrong_orientation);

    let l1 = parallel_dot_single_series_matrix::<MOD_Q, N>(
        &power_series.last().unwrap(),
        &projected_witness_wrong_orientation,
    );
    power_series.push(projected_lhs);

    (l0[0], l1[0])
}

pub fn verifier_join<const MOD_Q: u64, const N: usize>(
    verifier_state: VerifierState<MOD_Q, N>,
    l0: CyclotomicRing<MOD_Q, N>,
    l1: CyclotomicRing<MOD_Q, N>,
    projection_rhs: Vec<CyclotomicRing<MOD_Q, N>>,
    commitment_projection: Vec<Vec<CyclotomicRing<MOD_Q, N>>>,
) -> VerifierState<MOD_Q, N> {
    let mut new_rhs = Vec::new();
    for i in 0..verifier_state.rhs.len() {
        if i == verifier_state.rhs.len() - 1 {
            // For the last row ignore for now
        } else {
            // For other rows, concatenate projection_rhs[i] and verifier_state.rhs[i]
            let mut row = verifier_state.rhs[i].clone();
            row.extend(commitment_projection[i].clone());
            new_rhs.push(row);
        }
    }

    new_rhs.push(vec![verifier_state.rhs.last().unwrap()[0], l1]);

    new_rhs.push(vec![l0, projection_rhs[0]]);

    VerifierState {
        wit_len: verifier_state.wit_len,
        wit_rep: verifier_state.wit_rep + 1,
        rhs: new_rhs,
    }
}
