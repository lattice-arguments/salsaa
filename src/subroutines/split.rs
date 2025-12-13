use std::time::Instant;

use crate::arithmetic::{
    add_matrices, first_n_columns, last_n_columns, parallel_dot_series_matrix, row_wise_tensor,
    PowerSeries,
};
use crate::cyclotomic_ring::CyclotomicRing;
use crate::helpers::println_with_timestamp;

/// Splits the provided vector into three parts: L (left), C (center), and R (right).
///
/// # Arguments
///
/// * `vec` - The vector to be split.
/// * `chunk_size` - The size of each chunk.
///
/// # Returns
///
/// A tuple containing three vectors: `vec_L`, `vec_C`, and `vec_R`.
fn split_vec<const MOD_Q: u64, const N: usize>(
    mut vec: Vec<CyclotomicRing<MOD_Q, N>>,
    _chunk_size: usize,
) -> (Vec<CyclotomicRing<MOD_Q, N>>, Vec<CyclotomicRing<MOD_Q, N>>) {
    let mid = vec.len() / 2;
    let right = vec.split_off(mid);
    (vec, right)
}

/// The output of the split operation, containing the new RHS, the witness center.
pub struct SplitOutput<const MOD_Q: u64, const N: usize> {
    pub(crate) rhs: Vec<Vec<CyclotomicRing<MOD_Q, N>>>,
}

/// Splits the given power series and witness into components and computes the necessary matrices.
///
/// # Arguments
///
/// * `power_series` - The reference to the power series matrix.
/// * `witness` - The witness matrix to be split.
///
/// # Returns
///
/// A `SplitOutput` containing the new RHS, witness center, and new witness matrices.
pub fn split<const MOD_Q: u64, const N: usize>(
    power_series: &Vec<PowerSeries<MOD_Q, N>>,
    witness: Vec<Vec<CyclotomicRing<MOD_Q, N>>>,
) -> (Vec<Vec<CyclotomicRing<MOD_Q, N>>>, SplitOutput<MOD_Q, N>) {
    println_with_timestamp!(" Splitting {:?}", witness.len());

    let (witness_split_l, mut witness_split_r): (Vec<_>, Vec<_>) =
        witness.into_iter().map(|w| split_vec(w, 1)).unzip();

    println_with_timestamp!(
        " into {:?} {:?}",
        witness_split_l[0].len(),
        witness_split_r[0].len()
    );

    let mut witness_split = witness_split_l;
    witness_split.append(&mut witness_split_r);

    // Compute the new RHS

    let time = Instant::now();
    println_with_timestamp!(
        "performing paralleld dot series {:?}, {:?} with {:?}",
        witness_split.len(),
        witness_split[0].len(),
        power_series.len()
    );
    let new_rhs = parallel_dot_series_matrix(&power_series, &witness_split);
    println_with_timestamp!("performed parallel dot series in {:?}", time.elapsed());

    (witness_split, SplitOutput { rhs: new_rhs })
}

// fn test_split_vec() {
//     let series = PowerSeries {
//         coeffs: map_vector_to_prime_ring(vec![1, 2, 4, 8, 16, 32, 64, 128]),
//         factor: PrimeRing::constant(2),
//         chunks: 8,
//         expanded_layers: vec![],
//         tensors: vec![],
//     };
//
//     let expected_L = map_vector_to_prime_ring(vec![1, 2, 4, 8]);
//     let expected_C: Vec<DPrimeRingElement> = Vec::new();
//     let expected_R = map_vector_to_prime_ring(vec![16, 32, 64, 128]);
//
//     let (vec_L, vec_C, vec_R) = split_power_series(&series);
//
//     assert_eq!(vec_L, expected_L);
//     assert_eq!(vec_C, expected_C);
//     assert_eq!(vec_R, expected_R);
// }
#[cfg(test)]
mod tests {
    use crate::{
        arithmetic::{
            map_matrix_to_prime_ring, map_vector_to_prime_ring, matrix_to_incomplete_ntt,
        },
        subroutines::crs::CRS,
    };

    use super::*;
    const MOD_Q: u64 = 4546383823830515713;
    const N: usize = 8;
    #[test]
    fn test_split() {
        let mut series = vec![PowerSeries::<MOD_Q, N> {
            expanded_layers: map_matrix_to_prime_ring(vec![
                vec![1, 2, 4, 8, 16, 32, 64, 128],
                vec![1, 2, 4, 8],
                vec![1, 2],
                vec![1],
            ]),
            tensors: map_matrix_to_prime_ring(vec![vec![1, 16], vec![1, 4], vec![1, 2]]),
        }];

        let witness = vec![map_vector_to_prime_ring(vec![1, 2, 3, 4, 5, 6, 7, 8])];

        let mut rhs = vec![map_vector_to_prime_ring(vec![1793])];
        rhs[0][0].to_incomplete_ntt_representation();

        assert_eq!(parallel_dot_series_matrix(&series, &witness), rhs);

        let (witness_split, split_output) = split(&mut series, witness.into_par_iter());

        let mut new_rhs = vec![map_vector_to_prime_ring(vec![49, 109])];
        matrix_to_incomplete_ntt(&mut new_rhs);

        assert_eq!(split_output.rhs, new_rhs);

        let new_witness = vec![
            map_vector_to_prime_ring(vec![1, 2, 3, 4]),
            map_vector_to_prime_ring(vec![5, 6, 7, 8]),
        ];

        assert_eq!(witness_split, new_witness);
    }
    #[test]
    fn test_split_2() {
        let mut series = CRS::<MOD_Q, N>::gen_crs(2, 1).ck;

        let multiplier = series[0].expanded_layers[0][0];
        let witness = vec![map_vector_to_prime_ring(vec![1, 1])];

        let rhs = parallel_dot_series_matrix(&series, &witness);

        let (_witness_split, split_output) = split(&mut series, witness.into_par_iter());

        let rhs_l = first_n_columns(&split_output.rhs, 1);
        let rhs_r = last_n_columns(&split_output.rhs, 1);

        let rhs_r_multiplied = row_wise_tensor(&rhs_r, &vec![vec![multiplier]]);

        println!("{:?} {:?}", rhs_l, rhs_r_multiplied);
        assert_eq!(add_matrices(&rhs_r_multiplied, &rhs_l), rhs);
    }
}

pub struct VerifierState<const MOD_Q: u64, const N: usize> {
    pub(crate) wit_len: usize,
    pub(crate) wit_rep: usize,
    pub(crate) rhs: Vec<Vec<CyclotomicRing<MOD_Q, N>>>,
}

pub fn verifier_split<const MOD_Q: u64, const N: usize>(
    power_series: &Vec<PowerSeries<MOD_Q, N>>,
    output_split: SplitOutput<MOD_Q, N>,
    verifier_state: &VerifierState<MOD_Q, N>,
) -> VerifierState<MOD_Q, N> {
    let now = Instant::now();
    let ck_l = first_n_columns(&output_split.rhs, verifier_state.wit_rep);
    let ck_r = last_n_columns(&output_split.rhs, verifier_state.wit_rep);
    let elapsed = now.elapsed();
    println_with_timestamp!(
        "  Time to extract left and right columns from RHS: {:.2?}",
        elapsed
    );
    // Compute the multiplier and row-wise tensor product
    let now = Instant::now();

    let mut multiplier_l = Vec::with_capacity(power_series.len());
    let mut multiplier_r = Vec::with_capacity(power_series.len());

    power_series.iter().for_each(|ps| {
        let index = ps
            .expanded_layers
            .iter()
            .position(|el| el.len() == verifier_state.wit_len)
            .unwrap();
        multiplier_l.push(ps.tensors[index][0]);
        multiplier_r.push(ps.tensors[index][1]);
    });

    let elapsed = now.elapsed();
    println_with_timestamp!("  Time to compute the multiplier: {:.2?}", elapsed);

    let now = Instant::now();
    let ck_l_multiplied = row_wise_tensor(&ck_l, &vec![multiplier_l]);
    let ck_r_multiplied = row_wise_tensor(&ck_r, &vec![multiplier_r]);
    let elapsed = now.elapsed();
    println_with_timestamp!("  Time for row-wise tensor product: {:.2?}", elapsed);

    let now = Instant::now();
    let ck_lr = add_matrices(&ck_r_multiplied, &ck_l_multiplied);
    assert_eq!(ck_lr, verifier_state.rhs);

    let elapsed = now.elapsed();
    println_with_timestamp!("  Time verification: {:.2?}", elapsed);

    VerifierState {
        wit_len: verifier_state.wit_len / 2,
        wit_rep: verifier_state.wit_rep * 2,
        rhs: output_split.rhs,
    }
}
