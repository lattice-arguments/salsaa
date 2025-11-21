use crate::{
    arithmetic::{sample_random_vector, PowerSeries},
    cyclotomic_ring::CyclotomicRing,
};
use rayon::prelude::*;

/// Struct representing the Common Reference String (CRS) for cryptographic operations.
pub struct CRS<const MOD_Q: u64, const N: usize> {
    pub(crate) ck: Vec<PowerSeries<MOD_Q, N>>,
}

/// Generates a Common Reference String (CRS).
///
/// # Returns
///
/// A `CRS` containing commitment keys (`ck`) a randomly sampled vector (`a`), and a challenge set.
///
/// # Panics
///
/// This function will panic if the dimensions of `V_COEFFS` do not match the expected values.

impl<const MOD_Q: u64, const N: usize> CRS<MOD_Q, N> {
    pub fn gen_crs(wit_dim: usize, module_size: usize) -> CRS<MOD_Q, N> {
        let v_module = sample_random_vector(module_size);

        let ck = compute_commitment_keys(v_module, wit_dim);

        CRS { ck }
    }
}

/// Computes commitment keys by raising the given module to successive powers.
///
/// # Arguments
///
/// * `module` - A vector of `RingElement`
/// * `chunk_size` - The chunk size.
/// * `log_q` - The logarithmic size of Q.
///
/// # Returns
///
/// A vector of vectors representing the computed commitment keys.
pub fn compute_commitment_keys<const MOD_Q: u64, const N: usize>(
    module: Vec<CyclotomicRing<MOD_Q, N>>,
    wit_dim: usize,
) -> Vec<PowerSeries<MOD_Q, N>> {
    module
        .into_par_iter()
        .map(|mut m| {
            let mut row = Vec::with_capacity(wit_dim);
            let mut power = m.clone();
            row.push(m.clone());
            for _ in 1..wit_dim {
                power = &mut power * &mut m;
                row.push(power.clone());
            }
            let mut ps = PowerSeries {
                expanded_layers: vec![],
                tensors: vec![],
            };
            let mut current_dim = wit_dim;
            while current_dim % 2 == 0 {
                ps.expanded_layers.push(row[0..current_dim].to_vec());
                current_dim /= 2;
                ps.tensors
                    .push(vec![CyclotomicRing::one(), row[current_dim - 1]]);
            }
            ps.expanded_layers.push(row[0..current_dim].to_vec());
            ps
        })
        .collect()
}
