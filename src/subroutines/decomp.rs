use crate::arithmetic::{
    compose_with_radix_mod, decompose_matrix_by_chunks, parallel_dot_series_matrix, PowerSeries,
};
use crate::cyclotomic_ring::CyclotomicRing;
use crate::helpers::println_with_timestamp;
use crate::subroutines::split::VerifierState;

/// Struct representing the decomposition output.
pub struct DecompOutput<const MOD_Q: u64, const N: usize> {
    /// The number of parts the original witness matrix is decomposed into.
    pub(crate) parts: usize,

    pub(crate) radix: u64,

    /// The resulting right-hand side (RHS) matrix.
    pub(crate) rhs: Vec<Vec<CyclotomicRing<MOD_Q, N>>>,
}

/// Decomposes a witness matrix based on a given power series matrix, using the maximal infinity norm
/// to determine the number of parts for decomposition.
///
/// # Arguments
///
/// * `power_series` - A reference to the power series matrix, represented as `Vec<Vec<RingElement>>`.
/// * `witness` - A reference to the witness matrix, represented as `Vec<Vec<RingElement>>`.
///
/// # Returns
///
/// A `DecompOutput` struct containing the new witness matrix, the number of parts, and the resulting RHS matrix.
/// ```

pub fn decomp_ell<const MOD_Q: u64, const N: usize>(
    power_series: &Vec<PowerSeries<MOD_Q, N>>,
    witness: Vec<Vec<CyclotomicRing<MOD_Q, N>>>,
    parts: usize,
) -> (Vec<Vec<CyclotomicRing<MOD_Q, N>>>, DecompOutput<MOD_Q, N>) {
    use std::time::Instant;
    // Decompose each column of the decomposition of the witness matrix
    let now = Instant::now();

    let (new_witness, radix) = decompose_matrix_by_chunks(witness, parts);
    let elapsed = now.elapsed();

    println_with_timestamp!("  Time to decompose witness: {:.2?}", elapsed);

    // Extract relevant columns from the power series matrix to form a submatrix
    let now = Instant::now();
    let elapsed = now.elapsed();
    println_with_timestamp!(
        "Power series dim {:?} new_witnes dim {:?} {:?}",
        power_series.len(),
        new_witness.len(),
        new_witness[0].len()
    );
    println_with_timestamp!(
        "  Time to extract relevant columns from power series: {:.2?}",
        elapsed
    );

    // Compute the resulting RHS matrix
    let now = Instant::now();
    let rhs = parallel_dot_series_matrix(&power_series, &new_witness);
    let elapsed = now.elapsed();
    println_with_timestamp!("  Time to compute RHS matrix: {:.2?}", elapsed);

    (new_witness, DecompOutput { radix, parts, rhs })
}

#[cfg(test)]
mod tests {
    use super::*;
    const MOD_Q: u64 = 4546383823830515713;
    const N: usize = 8;
    use crate::arithmetic::{compose_with_radix_mod, sample_random_short_mat};
    use crate::subroutines::crs::CRS;
    #[test]
    fn test_decomp() {
        let parts: usize = 1;
        let ck = CRS::<MOD_Q, N>::gen_crs(8, 2).ck;
        let wit = sample_random_short_mat(8, 2, 3);
        let (new_witness, output) = decomp_ell(&ck, wit.clone(), parts);
        let composed_witness = compose_with_radix_mod(&new_witness, output.radix, parts);
        assert_eq!(composed_witness, wit);

        let rhs = parallel_dot_series_matrix(&ck, &wit);

        let composed_rhs = compose_with_radix_mod(&output.rhs, output.radix, parts);

        assert_eq!(composed_rhs, rhs);
    }
}

pub fn verify_decomp<const MOD_Q: u64, const N: usize>(
    decomp_output: DecompOutput<MOD_Q, N>,
    verifier_state: &VerifierState<MOD_Q, N>,
) -> VerifierState<MOD_Q, N> {
    let composed_rhs = compose_with_radix_mod::<MOD_Q, N>(
        &decomp_output.rhs,
        decomp_output.radix,
        decomp_output.parts,
    );

    assert_eq!(composed_rhs, verifier_state.rhs);

    VerifierState::<MOD_Q, N> {
        wit_len: verifier_state.wit_len,
        wit_rep: verifier_state.wit_rep * decomp_output.parts,
        rhs: decomp_output.rhs,
    }
}
