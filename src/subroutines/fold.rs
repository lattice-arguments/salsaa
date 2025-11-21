use crate::arithmetic::{
    parallel_dot_matrix_matrix, sample_random_biased_mat, transpose,
};
use crate::cyclotomic_ring::CyclotomicRing;
use crate::subroutines::split::VerifierState;

pub struct FoldOutput<const MOD_Q: u64, const N: usize> {
    pub(crate) _new_witness: Vec<Vec<CyclotomicRing<MOD_Q, N>>>,
}

/// Performs a fold operation on the given witness and challenge matrices.
///
/// # Arguments
///
/// * `witness` - A reference to a matrix represented as a vector of vectors of `RingElement`.
/// * `challenge` - A reference to a matrix represented as a vector of vectors of `RingElement`.
///
/// # Returns
///
/// A `FoldOutput` instance containing the new witness matrix resulting from the dot product of the witness and challenge matrices.
///
pub fn fold<const MOD_Q: u64, const N: usize>(
    witness: Vec<Vec<CyclotomicRing<MOD_Q, N>>>,
    challenge: &Vec<Vec<CyclotomicRing<MOD_Q, N>>>,
) -> Vec<Vec<CyclotomicRing<MOD_Q, N>>> {
    parallel_dot_matrix_matrix(witness, challenge)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arithmetic::{parallel_dot_series_matrix, sample_random_mat};
    use crate::subroutines::crs::CRS;
    const MOD_Q: u64 = 4546383823830515713;
    const N: usize = 8;
    #[test]
    fn test_fold() {
        // Generate the CRS and sample a random witness
        let ck = CRS::<MOD_Q, N>::gen_crs(8, 1).ck;
        let witness = sample_random_mat(ck[0].expanded_layers[0].len(), 4);

        let commitment = parallel_dot_series_matrix(&ck, &witness);

        let challenge = sample_random_mat(8, 2);

        assert_eq!(parallel_dot_series_matrix(&ck, &witness), commitment);

        let folded_witness = fold(witness, &transpose(&challenge));
        let folded_commitment = parallel_dot_matrix_matrix(challenge, &commitment);

        assert_eq!(
            parallel_dot_series_matrix(&ck, &folded_witness),
            folded_commitment
        );
    }
}

pub fn verifier_fold<const MOD_Q: u64, const N: usize>(
    verifier_state: &VerifierState<MOD_Q, N>,
    challenge: &Vec<Vec<CyclotomicRing<MOD_Q, N>>>,
    rep: usize,
) -> VerifierState<MOD_Q, N> {
    VerifierState {
        wit_rep: rep,
        wit_len: verifier_state.wit_len,
        rhs: parallel_dot_matrix_matrix(transpose(&challenge), &verifier_state.rhs),
    }
}

pub fn challenge_for_fold<const MOD_Q: u64, const N: usize>(
    verifier_state: &VerifierState<MOD_Q, N>,
    rep: usize,
) -> Vec<Vec<CyclotomicRing<MOD_Q, N>>> {
    sample_random_biased_mat::<MOD_Q, N>(rep, verifier_state.wit_rep)
}
