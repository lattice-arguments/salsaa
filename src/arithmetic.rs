use crate::{
    cyclotomic_ring::{CyclotomicRing, Representation},
    hexl::bindings,
};
use num_traits::Zero;
use rayon::prelude::*;
use std::cmp::max;
use std::simd::cmp::SimdPartialOrd;
use std::simd::*;
use std::{
    iter::Sum,
    ops::{Add, Mul},
};

pub const MAX_THREADS: usize = 120;

#[cfg(target_feature = "avx512f")]
#[inline]
pub fn add_no_reduction<const MOD_Q: u64, const N: usize>(
    a: &mut CyclotomicRing<MOD_Q, N>,
    b: &CyclotomicRing<MOD_Q, N>,
) {
    use std::arch::x86_64::*;
    let ptr_a = a.data.as_ptr();
    let ptr_b = b.data.as_ptr();
    let ptr_res = a.data.as_mut_ptr();
    let mut i = 0;
    while i + 8 <= N {
        unsafe {
            let va = _mm512_loadu_epi64(ptr_a.add(i) as *const i64);
            let vb = _mm512_loadu_epi64(ptr_b.add(i) as *const i64);
            let vsum = _mm512_add_epi64(va, vb);
            _mm512_storeu_epi64(ptr_res.add(i) as *mut i64, vsum);
        }
        i += 8;
    }
    while i < N {
        a.data[i] = a.data[i] + b.data[i];
        i += 1;
    }
}

#[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
#[inline]
pub fn add_no_reduction<const MOD_Q: u64, const N: usize>(
    a: &mut CyclotomicRing<MOD_Q, N>,
    b: &CyclotomicRing<MOD_Q, N>,
) {
    use std::arch::x86_64::*;

    //let mut result = CyclotomicRing::<MOD_Q, N>::zero();
    let ptr_a = a.data.as_ptr();
    let ptr_b = b.data.as_ptr();
    let ptr_res = a.data.as_mut_ptr();

    let mut i = 0;
    while i + 4 <= N {
        unsafe {
            let va = _mm256_loadu_si256(ptr_a.add(i) as *const __m256i);
            let vb = _mm256_loadu_si256(ptr_b.add(i) as *const __m256i);
            let vsum = _mm256_add_epi64(va, vb);
            _mm256_storeu_si256(ptr_res.add(i) as *mut __m256i, vsum);
        }
        i += 4;
    }

    while i < N {
        a.data[i] = a.data[i] + b.data[i];
        i += 1;
    }
}
#[cfg(target_feature = "avx512f")]
#[inline]
pub fn split_simd<const MOD_Q: u64, const N: usize>(input: &[u64; N]) -> ([u64; N], [u64; N]) {
    let mut pos_data = [0u64; N];
    let mut neg_data = [0u64; N];

    let mut i = 0;
    while i + 8 <= N {
        let vals = Simd::<u64, 8>::from_slice(&input[i..i + 8]);
        let doubled = vals * Simd::splat(2);

        // Create mask: true if 2*val > MOD_Q
        let mask: Mask<_, 8> = doubled.simd_gt(Simd::splat(MOD_Q));

        // Masked selection: mask selects from first argument, !mask selects from second
        let neg_selected = mask.select(Simd::splat(MOD_Q) - vals, Simd::splat(0));
        let pos_selected = mask.select(Simd::splat(0), vals);

        neg_data[i..i + 8].copy_from_slice(&neg_selected.to_array());
        pos_data[i..i + 8].copy_from_slice(&pos_selected.to_array());

        i += 8;
    }

    // handle tail
    while i < N {
        let val = input[i];
        if val * 2 > MOD_Q {
            neg_data[i] = MOD_Q - val;
        } else {
            pos_data[i] = val;
        }
        i += 1;
    }
    (pos_data, neg_data)
}

#[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
#[inline]
pub fn split_simd<const MOD_Q: u64, const N: usize>(input: &[u64; N]) -> ([u64; N], [u64; N]) {
    let mut pos_data = [0u64; N];
    let mut neg_data = [0u64; N];

    let mut i = 0;
    while i + 4 <= N {
        let vals = Simd::<u64, 4>::from_slice(&input[i..i + 4]);
        let doubled = vals * Simd::splat(2);

        // Create mask: true if 2*val > MOD_Q
        let mask: Mask<_, 4> = doubled.simd_gt(Simd::splat(MOD_Q));

        // Masked selection: mask selects from first argument, !mask selects from second
        let neg_selected = mask.select(Simd::splat(MOD_Q) - vals, Simd::splat(0));
        let pos_selected = mask.select(Simd::splat(0), vals);

        neg_data[i..i + 4].copy_from_slice(&neg_selected.to_array());
        pos_data[i..i + 4].copy_from_slice(&pos_selected.to_array());

        i += 4;
    }

    // handle tail
    while i < N {
        let val = input[i];
        if val * 2 > MOD_Q {
            neg_data[i] = MOD_Q - val;
        } else {
            pos_data[i] = val;
        }
        i += 1;
    }
    (pos_data, neg_data)
}

#[derive(Clone)]
pub struct PowerSeries<const MOD_Q: u64, const N: usize> {
    pub expanded_layers: Vec<Vec<CyclotomicRing<MOD_Q, N>>>,
    pub tensors: Vec<Vec<CyclotomicRing<MOD_Q, N>>>,
}

#[derive(Clone)]
pub struct IncompletePowerSeries<const MOD_Q: u64, const N: usize> {
    pub base_layer: Vec<CyclotomicRing<MOD_Q, N>>,
    pub tensors: Vec<Vec<CyclotomicRing<MOD_Q, N>>>, // note the inverse order
}

pub fn complete_power_series<const MOD_Q: u64, const N: usize>(
    ps: IncompletePowerSeries<MOD_Q, N>,
) -> PowerSeries<MOD_Q, N> {
    let mut expanded_layers = vec![ps.base_layer.clone()];
    let mut tensors = ps.tensors;
    tensors.reverse();

    for i in 0..tensors.len() {
        expanded_layers.push(kronecker_product(&tensors[i], &expanded_layers[i]));
    }

    tensors.reverse();
    expanded_layers.reverse();

    PowerSeries {
        expanded_layers,
        tensors,
    }
}

// // Helper function to map a vector into PrimeRingElement::constant
pub fn map_vector_to_prime_ring<const MOD_Q: u64, const N: usize>(
    vector: Vec<u64>,
) -> Vec<CyclotomicRing<MOD_Q, N>> {
    vector
        .into_iter()
        .map(|v| CyclotomicRing::<MOD_Q, N>::constant(v))
        .collect()
}

// // Helper function to map a matrix into PrimeRingElement::constant
pub fn map_matrix_to_prime_ring<const MOD_Q: u64, const N: usize>(
    matrix: Vec<Vec<u64>>,
) -> Vec<Vec<CyclotomicRing<MOD_Q, N>>> {
    matrix
        .into_iter()
        .map(|row| map_vector_to_prime_ring(row))
        .collect()
}

pub fn matrix_to_incomplete_ntt<const MOD_Q: u64, const N: usize>(
    matrix: &mut Vec<Vec<CyclotomicRing<MOD_Q, N>>>,
) {
    matrix.into_iter().for_each(|row| {
        row.into_iter().for_each(|cyclotomic_elem| {
            cyclotomic_elem.to_incomplete_ntt_representation();
        })
    })
}

pub fn matrix_to_coeff<const MOD_Q: u64, const N: usize>(
    matrix: &mut Vec<Vec<CyclotomicRing<MOD_Q, N>>>,
) {
    matrix.into_par_iter().for_each(|row| {
        row.into_par_iter().for_each(|cyclotomic_elem| {
            cyclotomic_elem.to_coeff_representation();
        })
    })
}

// /// Computes the dot product of a vector and a matrix in parallel.
// ///
// /// # Arguments
// ///
// /// * `matrix` - A reference to a matrix represented as a slice of vectors.
// /// * `vector` - A reference to a vector.
// ///
// /// # Returns
// ///
// /// A new vector containing the result of the vector-matrix multiplication.
// ///
// /// # Type Parameters
// ///
// /// * `T` - The type of the elements in the matrix and the vector. It must implement the `Mul`, `Zero`, `Copy`, `Send`, `Sync`,
// /// and `Add` traits.
// ///
// /// # Panics
// ///
// /// This function will panic if the number of rows in the matrix does not match the length of the vector.
// ///
// /// # Examples
// ///
// /// ```
// /// # fn main() {
// /// let matrix = vec![
// ///     vec![1, 2],
// ///     vec![3, 4],
// ///     vec![5, 6],
// /// ];
// /// let vector = vec![7, 8, 9];
// /// let result = parallel_dot_vector_matrix(&matrix, &vector);
// /// assert_eq!(result, vec![76, 100]);
// /// # }
// /// ```
pub fn parallel_dot_vector_matrix<T>(vector: &Vec<T>, matrix: &Vec<Vec<T>>) -> Vec<T>
where
    T: Mul<Output = T> + Zero + Copy + Send + Sync + Add<Output = T>,
{
    assert!(
        matrix.len() == vector.len(),
        "Number of rows in the matrix must match the length of the vector"
    );

    (0..matrix[0].len())
        .into_par_iter()
        .map(|col| {
            matrix
                .iter()
                .zip(vector.iter())
                .map(|(row, &v)| row[col] * v)
                .fold(T::zero(), |acc, x| acc + x)
        })
        .collect()
}

// /// Computes the dot product of a matrix and a vector in parallel.
// ///
// /// # Arguments
// ///
// /// * `matrix` - A reference to a matrix represented as a slice of vectors (each inner vector is a row).
// /// * `vector` - A reference to a vector.
// ///
// /// # Returns
// ///
// /// A new vector containing the result of the matrix-vector multiplication.
// ///
// /// # Type Parameters
// ///
// /// * `T` - The type of the elements in the matrix and the vector. It must implement the `Mul`, `Zero`, `Copy`, `Send`, `Sync`,
// /// and `Add` traits.
// ///
// /// # Panics
// ///
// /// This function will panic if the number of columns in the matrix does not match the length of the vector.
// ///
// /// # Examples
// ///
// /// ```
// /// # fn main() {
// /// let matrix = vec![
// ///     vec![1, 2, 3],
// ///     vec![4, 5, 6],
// /// ];
// /// let vector = vec![7, 8, 9];
// /// let result = parallel_dot_matrix_vector(&matrix, &vector);
// /// assert_eq!(result, vec![50, 122]);
// /// # }
// /// ```
// pub fn parallel_dot_matrix_vector<T>(matrix: &[Vec<T>], vector: &[T]) -> Vec<T>
// where
//     T: Mul<Output = T> + Zero + Copy + Send + Sync + Add<Output = T>,
// {
//     assert!(
//         matrix[0].len() == vector.len(),
//         "Number of columns in the matrix must match the length of the vector"
//     );

//     matrix
//         .par_iter()
//         .map(|row| {
//             row.iter()
//                 .zip(vector.iter())
//                 .map(|(&a, &b)| a * b)
//                 .fold(T::zero(), |acc, x| acc + x)
//         })
//         .collect()
// }

// #[test]
// fn test_parallel_dot_vector_matrix_integers() {
//     let matrix = vec![
//         vec![1, 2],
//         vec![3, 4],
//         vec![5, 6],
//     ];
//     let vector = vec![7, 8, 9];
//     let result = parallel_dot_vector_matrix(&vector, &matrix);
//     assert_eq!(result, vec![76, 100]);
// }

// #[test]
// fn test_parallel_dot_vector_matrix_rings() {
//     let matrix = vec![
//         vec![1, 2],
//         vec![3, 4],
//         vec![5, 6],
//     ];
//     let vector = vec![7, 8, 9];

//     let mapped_matrix: Vec<Vec<DPrimeRingElement>> = map_matrix_to_prime_ring(matrix);
//     let mapped_vector: Vec<DPrimeRingElement> = map_vector_to_prime_ring(vector);

//     let result = parallel_dot_vector_matrix(&mapped_vector, &mapped_matrix);

//     let expected_result = map_vector_to_prime_ring(vec![76, 100]); // Assuming these are the correct results

//     assert_eq!(result, expected_result);
// }

// /// Multiplies each element in the given vector by a given scalar.
// ///
// /// # Arguments
// ///
// /// * `vector` - A reference to a vector of elements to be multiplied.
// /// * `ell` - A reference to a scalar value by which each element of the vector will be multiplied.
// ///
// /// # Returns
// ///
// /// A new vector containing the result of element-wise multiplication.
// ///
// /// # Type Parameters
// ///
// /// * `T` - The type of elements in the vector and the scalar. It must implement the `Mul`, `Zero`, `Copy`, `Send`, `Sync`,
// /// and `Add` traits.
// ///
// /// # Examples
// ///
// /// ```
// /// # fn main() {
// /// let vector = vec![1, 2, 3, 4];
// /// let scalar = 2;
// /// let result = vector_element_product(&vector, &scalar);
// /// assert_eq!(result, vec![2, 4, 6, 8]);
// /// # }
// /// ```
// pub fn vector_element_product<T>(vector: &Vec<T>, ell: &T) -> Vec<T>
// where
//     T: Mul<Output = T> + Zero + Copy + Send + Sync + Add<Output = T>,
// {
//     vector.par_iter().map(|x| *x * *ell).collect()
// }

// #[test]
// fn test_vector_element_product_integers() {
//     let vector = vec![1, 2, 3, 4];
//     let scalar = 2;
//     let result = vector_element_product(&vector, &scalar);
//     assert_eq!(result, vec![2, 4, 6, 8]);
// }

// #[test]
// fn test_vector_element_product_ring() {
//     let vector = vec![1, 2, 3, 4];
//     let scalar = 2;

//     let mapped_vector = map_vector_to_prime_ring(vector);
//     let mapped_scalar = PrimeRing::constant(scalar);

//     let result = vector_element_product(&mapped_vector, &mapped_scalar);

//     let expected_result = map_vector_to_prime_ring(vec![2, 4, 6, 8]);

//     assert_eq!(result, expected_result);
// }
// #[test]
// fn test_vector_element_product_floats() {
//     let vector = vec![1.0, 2.0, 3.0, 4.0];
//     let scalar = 0.5;
//     let result = vector_element_product(&vector, &scalar);
//     assert_eq!(result, vec![0.5, 1.0, 1.5, 2.0]);
// }

// #[test]
// fn test_vector_element_product_zeros() {
//     let vector = vec![0, 0, 0];
//     let scalar = 999;
//     let result = vector_element_product(&vector, &scalar);
//     assert_eq!(result, vec![0, 0, 0]);
// }


pub fn parallel_dot_matrix_matrix<T>(matrix_a: Vec<Vec<T>>, matrix_b: &Vec<Vec<T>>) -> Vec<Vec<T>>
where
    T: Mul<Output = T> + Zero + Copy + Send + Sync + Add<Output = T> + Sum,
{
    let ncols = matrix_a[0].len();
    let nrows = matrix_b.len();
    let inner_dim = matrix_a.len();

    println!(
        "ncols {:?} nrows {:?} inner_dim {:?}",
        ncols, nrows, inner_dim
    );

    assert_eq!(
        inner_dim,
        matrix_b[0].len(),
        "Matrix dimensions incompatible"
    );

    (0..nrows)
        .map(|j| {
            let b_row = &matrix_b[j];
            (0..ncols)
                .into_par_iter()
                .map(|i| (0..inner_dim).map(|k| matrix_a[k][i] * b_row[k]).sum())
                .collect::<Vec<T>>()
        })
        .collect()
}

pub fn parallel_dot_series_matrix<const MOD_Q: u64, const N: usize>(
    matrix_a: &[PowerSeries<MOD_Q, N>],
    matrix_b: &Vec<Vec<CyclotomicRing<MOD_Q, N>>>,
) -> Vec<Vec<CyclotomicRing<MOD_Q, N>>> {
    let ncols = matrix_b[0].len();
    let inner_dim = matrix_b.len();

    let extracted_rows: Vec<&[CyclotomicRing<MOD_Q, N>]> = matrix_a
        .par_iter()
        .map(|series| {
            series
                .expanded_layers
                .par_iter()
                .find_first(|layer| layer.len() == ncols)
                .expect("No matching layer found")
                .as_slice()
        })
        .collect();

    // Pre allocate a result matrix skipping zero-initialization
    let mut result: Vec<Vec<CyclotomicRing<MOD_Q, N>>> = (0..extracted_rows.len())
        .into_iter()
        .map(|_| {
            let mut v = Vec::with_capacity(inner_dim);
            unsafe {
                v.set_len(inner_dim);
            }
            v
        })
        .collect();

    result.par_iter_mut().enumerate().for_each(|(j, row_res)| {
        let row = extracted_rows[j];

        row_res.par_iter_mut().enumerate().for_each(|(k, res)| {
            *res = (0..ncols)
                .into_par_iter()
                .fold(
                    || CyclotomicRing::zero(),
                    |acc, i| acc + row[i] * matrix_b[k][i],
                )
                .sum()
        });
    });

    result
}

pub fn parallel_dot_single_series_matrix<const MOD_Q: u64, const N: usize>(
    series: &PowerSeries<MOD_Q, N>,
    matrix: &Vec<Vec<CyclotomicRing<MOD_Q, N>>>,
) -> Vec<CyclotomicRing<MOD_Q, N>> {
    let ncols = matrix[0].len();
    let inner_dim = matrix.len();

    let row_a = series
        .expanded_layers
        .iter()
        .find(|l| l.len() == ncols)
        .expect("No matching expanded layer found");

    let len = row_a.len();

    // Pre allocate a result vec skipping zero-initialization
    let mut result: Vec<CyclotomicRing<MOD_Q, N>> = Vec::with_capacity(inner_dim);
    unsafe {
        result.set_len(inner_dim);
    }

    result.par_iter_mut().enumerate().for_each(|(j, res)| {
        *res = (0..len)
            .into_par_iter()
            .map(|i| row_a[i] * matrix[j][i])
            .reduce(CyclotomicRing::zero, |a, b| a + b);
    });

    result
}

// #[test]
// fn test_parallel_dot_matrix_matrix() {
//     let matrix_a = vec![
//         vec![1, 2, 3],
//         vec![4, 5, 6],
//         vec![7, 8, 9],
//     ];
//     let matrix_b = vec![
//         vec![1, 4, 7],
//         vec![2, 5, 8],
//         vec![3, 6, 9],
//     ];
//     let result = parallel_dot_matrix_matrix(&matrix_a, &matrix_b);
//     let expected = vec![
//         vec![14, 32, 50],
//         vec![32, 77, 122],
//         vec![50, 122, 194],
//     ];
//     assert_eq!(result, expected);
// }

// #[test]
// fn test_parallel_dot_matrix_matrix_ring() {
//     let matrix_a = vec![
//         vec![1, 2, 3],
//         vec![4, 5, 6],
//         vec![7, 8, 9],
//     ];
//     let matrix_b = vec![
//         vec![1, 4, 7],
//         vec![2, 5, 8],
//         vec![3, 6, 9],
//     ];

//     let mapped_matrix_a = map_matrix_to_prime_ring(matrix_a);
//     let mapped_matrix_b = map_matrix_to_prime_ring(matrix_b);

//     let result = parallel_dot_matrix_matrix(&mapped_matrix_a, &mapped_matrix_b);

//     let expected = vec![
//         vec![14, 32, 50],
//         vec![32, 77, 122],
//         vec![50, 122, 194],
//     ];

//     let mapped_expected = map_matrix_to_prime_ring(expected);

//     assert_eq!(result, mapped_expected);
// }

// /// Transposes a matrix represented as a vector of vectors.
// ///
// /// # Arguments
// ///
// /// * `matrix` - A vector of vectors representing the matrix to be transposed.
// ///
// /// # Returns
// ///
// /// A new vector of vectors representing the transposed matrix.
// ///
// /// # Example
// ///
// /// ```rust
// /// let matrix = vec![
// ///     vec![1, 2, 3],
// ///     vec![4, 5, 6],
// ///     vec![7, 8, 9],
// /// ];
// /// let result = transpose(matrix);
// /// assert_eq!(result, vec![
// ///     vec![1, 4, 7],
// ///     vec![2, 5, 8],
// ///     vec![3, 6, 9],
// /// ]);
// /// ```
pub fn transpose<T>(matrix: &Vec<Vec<T>>) -> Vec<Vec<T>>
where
    T: Copy + Zero + Send + Sync,
{
    if matrix.is_empty() {
        return vec![];
    }

    let rows = matrix.len();
    let cols = matrix[0].len();
    let mut transposed = vec![vec![T::zero(); rows]; cols];

    // Transpose using parallel iteration
    transposed.par_iter_mut().enumerate().for_each(|(i, row)| {
        for (j, value) in row.iter_mut().enumerate() {
            *value = matrix[j][i];
        }
    });

    transposed
}

// #[test]
// fn test_transpose_square_matrix() {
//     let matrix = vec![
//         vec![1, 2, 3],
//         vec![4, 5, 6],
//         vec![7, 8, 9],
//     ];
//     let result = transpose(&matrix);
//     let expected = vec![
//         vec![1, 4, 7],
//         vec![2, 5, 8],
//         vec![3, 6, 9],
//     ];
//     assert_eq!(result, expected);
// }

// /// Extracts the first `n` columns from the input matrix.
// ///
// /// # Arguments
// ///
// /// * `matrix` - A 2D vector representing the input matrix.
// /// * `n` - The number of columns to extract from the start.
// ///
// /// # Returns
// ///
// /// A new 2D vector (submatrix) containing the first `n` columns.
// ///
// /// # Panics
// ///
// /// Panics if the matrix is empty or if `n` is greater than the number of columns in the matrix.
// ///
// /// # Examples
// ///
// /// ```
// /// let mat = vec![
// ///     vec![1, 2, 3, 4],
// ///     vec![5, 6, 7, 8],
// ///     vec![9, 10, 11, 12],
// /// ];
// ///
// /// let sub_mat = first_n_columns(mat, 2);
// /// assert_eq!(sub_mat, vec![
// ///     vec![1, 2],
// ///     vec![5, 6],
// ///     vec![9, 10]
// /// ]);
// /// ```
pub fn first_n_columns<T: Copy>(matrix: &Vec<Vec<T>>, n: usize) -> Vec<Vec<T>> {
    if matrix.is_empty() {
        panic!("Matrix is empty");
    }
    let num_columns = matrix[0].len();
    if n > num_columns {
        panic!("Invalid range: `n` is greater than the number of columns in the matrix");
    }

    let mut submatrix = Vec::new();
    for row in matrix.iter() {
        let sub_row = row[0..n].to_vec();
        submatrix.push(sub_row);
    }

    submatrix
}

pub fn last_n_columns<T: Copy>(mat: &Vec<Vec<T>>, n: usize) -> Vec<Vec<T>> {
    if mat.is_empty() {
        panic!("Matrix is empty");
    }
    let num_columns = mat[0].len();
    if n > num_columns {
        panic!("Invalid range: `n` is greater than the number of columns in the matrix");
    }

    let from = num_columns - n;
    let to = num_columns - 1;

    let mut submatrix = Vec::new();
    for row in mat.iter() {
        let sub_row = row[from..=to].to_vec();
        submatrix.push(sub_row);
    }

    submatrix
}

// pub fn random(len: usize, mod_q: u64) -> Vec<u64> {
//     let mut rng = rand::thread_rng();
//     let mut result = Vec::with_capacity(len);
//     for i in 0..len {
//         let number = rng.gen_range(0..mod_q);
//         result.push(number);
//     }
//     result
// }

/// Samples a random matrix of size n x m where each element is a random RingElement.
///
/// # Arguments
///
/// * `n` - The number of rows in the matrix.
/// * `m` - The number of columns in the matrix.
///
/// # Returns
///
/// A vector of vectors (matrix) where each element is a random RingElement.
pub fn sample_random_mat<const MOD_Q: u64, const N: usize>(
    n: usize,
    m: usize,
) -> Vec<Vec<CyclotomicRing<MOD_Q, N>>> {
    (0..n)
        .map(|_| {
            (0..m)
                .map(|_| CyclotomicRing::<MOD_Q, N>::random())
                .collect()
        })
        .collect()
}

// pub fn sample_short_random_mat(n: usize, m: usize) -> Vec<Vec<DPrimeRingElement>> {
//     // Create a matrix of size n x m
//     (0..n).map(|_| {
//         (0..m).map(|_| PrimeRing::random_short()).collect()
//     }).collect()
// }

// pub fn sample_bin_random_mat(n: usize, m: usize) -> Vec<Vec<DPrimeRingElement>> {
//     // Create a matrix of size n x m
//     (0..n).map(|_| {
//         (0..m).map(|_| PrimeRing::random_bin()).collect()
//     }).collect()
// }

// /// Samples a random vector of the given size where each element is a random RingElement.
// ///
// /// # Arguments
// ///
// /// * `size` - The number of elements in the vector.
// ///
// /// # Returns
// ///
// /// A vector where each element is a random RingElement.
pub fn sample_random_vector<const MOD_Q: u64, const N: usize>(
    size: usize,
) -> Vec<CyclotomicRing<MOD_Q, N>> {
    // Create a vector of the given size with random RingElement values
    (0..size).map(|_| CyclotomicRing::random()).collect()
}

// pub fn sample_short_random_vector(size: usize) -> Vec<DPrimeRingElement> {
//     // Create a vector of the given size with random RingElement values
//     (0..size).map(|_| PrimeRing::random_short()).collect()
// }

// /// Computes the row-wise tensor product of two matrices `a` and `b`.
// ///
// /// # Arguments
// ///
// /// * `a` - A matrix represented as a vector of vectors of generic type `T`.
// /// * `b` - A matrix represented as a vector of vectors of generic type `T`.
// ///
// /// # Returns
// ///
// /// A matrix represented as a vector of vectors of generic type `T`, where each row of the result
// /// is the tensor product of the corresponding rows of `a` and `b`.
// ///
// /// # Panics
// ///
// /// Panics if the number of rows in `a` and `b` do not match.
// ///
// /// # Examples
// ///
// /// ```
// /// let a = vec![
// ///     vec![1, 2],
// ///     vec![3, 4],
// /// ];
// ///
// /// let b = vec![
// ///     vec![5, 6],
// ///     vec![7, 8],
// /// ];
// ///
// /// let result = row_wise_tensor(a, b);
// /// assert_eq!(result, vec![
// ///     vec![5, 6, 10, 12],
// ///     vec![21, 24, 28, 32],
// /// ]);
// /// ```

pub fn row_wise_tensor<T>(a: &Vec<Vec<T>>, b: &Vec<Vec<T>>) -> Vec<Vec<T>>
where
    T: Copy + Mul<Output = T> + Send + Sync,
{
    if a.len() != b[0].len() {
        panic!("Number of rows in `a` must equal number of columns in transposed `b`");
    }

    let num_rows = a.len();
    let mut result = Vec::with_capacity(num_rows);

    result.par_extend(a.par_iter().enumerate().map(|(i, row_a)| {
        row_a
            .par_iter()
            .flat_map(|&elem_a| b.par_iter().map(move |row_b_t| elem_a * row_b_t[i]))
            .collect()
    }));
    result
}

#[inline]
pub fn kronecker_product<T>(a: &Vec<T>, b: &Vec<T>) -> Vec<T>
where
    T: Copy + Mul<Output = T> + Send + Sync,
{
    let a_len = a.len();
    let b_len = b.len();
    let mut result = Vec::with_capacity(a_len * b_len);
    unsafe {
        result.set_len(a_len * b_len);
    }

    result.par_iter_mut().enumerate().for_each(|(idx, elem)| {
        let i = idx / b_len;
        let j = idx % b_len;
        *elem = a[i] * b[j];
    });

    result
}

// #[test]
// fn test_kronecker_product_basic() {
//     let a = vec![1, 2, 3];
//     let b = vec![4, 5];
//     let expected = vec![4, 5, 8, 10, 12, 15];
//     let result = kronecker_product(&a, &b);
//     assert_eq!(result, expected);
// }
// #[test]
// fn test_row_wise_tensor_normal_case() {
//     let a = vec![
//         vec![1, 2],
//         vec![3, 4],
//     ];

//     let b = vec![
//         vec![5, 6],
//         vec![7, 8],
//     ];

//     let result = row_wise_tensor(&a, &b);
//     assert_eq!(result, vec![
//         vec![5, 6, 10, 12],
//         vec![21, 24, 28, 32],
//     ]);
// }

// /// Adds two matrices element-wise.
// ///
// /// # Arguments
// ///
// /// * `matrix_a` - A reference to the first matrix.
// /// * `matrix_b` - A reference to the second matrix.
// ///
// /// # Returns
// ///
// /// A new matrix which is the element-wise sum of `matrix_a` and `matrix_b`.
// ///
// /// # Panics
// ///
// /// This function will panic if the dimensions of the two matrices are not the same.
// ///
// /// # Example
// ///
// /// ```rust
// /// let matrix_a = vec![
// ///     vec![1, 2, 3],
// ///     vec![4, 5, 6],
// /// ];
// /// let matrix_b = vec![
// ///     vec![7, 8, 9],
// ///     vec![10, 11, 12],
// /// ];
// /// let result = add_matrices(&matrix_a, &matrix_b);
// /// assert_eq!(result, vec![
// ///     vec![8, 10, 12],
// ///     vec![14, 16, 18],
// /// ]);
// /// ```
pub fn add_matrices<T>(matrix_a: &Vec<Vec<T>>, matrix_b: &Vec<Vec<T>>) -> Vec<Vec<T>>
where
    T: Add<Output = T> + Copy + Zero + Send + Sync,
{
    let nrows = matrix_a.len();
    let ncols = matrix_a[0].len();

    assert_eq!(
        nrows,
        matrix_b.len(),
        "The number of rows in the matrices must be the same"
    );
    assert_eq!(
        ncols,
        matrix_b[0].len(),
        "The number of columns in the matrices must be the same"
    );

    let mut result = vec![vec![T::zero(); ncols]; nrows];

    result.par_iter_mut().enumerate().for_each(|(i, row)| {
        for j in 0..ncols {
            row[j] = matrix_a[i][j] + matrix_b[i][j];
        }
    });

    result
}

// Add two matrices from their iterators, transposing the result
pub fn add_matrices_transposed<'a, const MOD_Q: u64, const N: usize>(
    matrix_a: impl ParallelIterator<Item = Vec<CyclotomicRing<MOD_Q, N>>>
        + IndexedParallelIterator
        + 'a
        + Clone,
    matrix_b: impl ParallelIterator<Item = Vec<CyclotomicRing<MOD_Q, N>>>
        + IndexedParallelIterator
        + 'a
        + Clone,
) -> Vec<Vec<CyclotomicRing<MOD_Q, N>>> {
    let first_col = matrix_a
        .clone()
        .find_first(|_| true)
        .expect("matrix_a is empty");
    let nrows = first_col.len();

    let zipped_cols = matrix_a.zip(matrix_b);

    (0..nrows)
        .into_iter()
        .map(move |row_idx| {
            zipped_cols
                .clone()
                .into_par_iter()
                .map(|(col_a, col_b)| col_a[row_idx] + col_b[row_idx])
                .collect::<Vec<_>>()
        })
        .collect()
}

// /// Adds corresponding elements of two vectors and returns the result as a new vector.
// ///
// /// # Arguments
// ///
// /// * `vector_a` - A reference to the first vector.
// /// * `vector_b` - A reference to the second vector.
// ///
// /// # Returns
// ///
// /// A new vector containing the result of element-wise addition.
// ///
// /// # Type Parameters
// ///
// /// * `T` - The type of the elements in the vectors. It must implement the `Add`, `Zero`, `Copy`, `Send`, and `Sync` traits.
// ///
// /// # Panics
// ///
// /// This function will panic if the input vectors are not of the same length.
// ///
// /// # Examples
// ///
// /// ```
// /// # fn main() {
// /// let vector_a = vec![1, 2, 3, 4];
// /// let vector_b = vec![5, 6, 7, 8];
// /// let result = add_vectors(&vector_a, &vector_b);
// /// assert_eq!(result, vec![6, 8, 10, 12]);
// /// # }
// /// ```
pub fn add_vectors<T>(vector_a: &Vec<T>, vector_b: &Vec<T>) -> Vec<T>
where
    T: Add<Output = T> + Copy + Send + Sync,
{
    assert_eq!(
        vector_a.len(),
        vector_b.len(),
        "Vectors must be of the same length"
    );

    vector_a
        .iter()
        .zip(vector_b)
        .map(|(&a, &b)| a + b)
        .collect()
}

pub fn multiply_vectors<T>(vector_a: &Vec<T>, vector_b: &Vec<T>) -> Vec<T>
where
    T: Mul<Output = T> + Copy + Send + Sync,
{
    assert_eq!(
        vector_a.len(),
        vector_b.len(),
        "Vectors must be of the same length"
    );

    vector_a
        .iter()
        .zip(vector_b)
        .map(|(&a, &b)| a * b)
        .collect()
}

// #[test]
// fn test_add_vectors_integers() {
//     let vector_a = vec![1, 2, 3, 4];
//     let vector_b = vec![5, 6, 7, 8];
//     let result = add_vectors(&vector_a, &vector_b);
//     assert_eq!(result, vec![6, 8, 10, 12]);
// }

// #[test]
// fn test_add_vectors_floats() {
//     let vector_a = vec![1.0, 2.0, 3.0, 4.0];
//     let vector_b = vec![0.5, 1.5, 2.5, 3.5];
//     let result = add_vectors(&vector_a, &vector_b);
//     assert_eq!(result, vec![1.5, 3.5, 5.5, 7.5]);
// }

// #[test]
// #[should_panic(expected = "Vectors must be of the same length")]
// fn test_add_vectors_different_lengths() {
//     let vector_a = vec![1, 2, 3];
//     let vector_b = vec![1, 2, 3, 4];
//     add_vectors(&vector_a, &vector_b); // This should panic
// }

// #[test]
// fn test_add_vectors_zeros() {
//     let vector_a = vec![0, 0, 0];
//     let vector_b = vec![1, 2, 3];
//     let result = add_vectors(&vector_a, &vector_b);
//     assert_eq!(result, vec![1, 2, 3]);
// }

// #[test]
// fn test_add_matrices_basic() {
//     let matrix_a = vec![
//         vec![1, 2, 3],
//         vec![4, 5, 6],
//     ];
//     let matrix_b = vec![
//         vec![7, 8, 9],
//         vec![10, 11, 12],
//     ];
//     let result = add_matrices(&matrix_a, &matrix_b);
//     let expected = vec![
//         vec![8, 10, 12],
//         vec![14, 16, 18],
//     ];
//     assert_eq!(result, expected);
// }

// pub fn extract_matrix_from_power_series(
//     series: &Vec<PowerSeries>,
//     size: usize
// ) -> Vec<Vec<DPrimeRingElement>> {
//     series.iter().filter_map(|s| {
//         s.expanded_layers.iter()
//             .filter(|l| l.len() == size)
//             .take(1)
//             .cloned()
//             .next()
//     }).collect()
// }

// // Computes `a` raised to the power of `pow` using exponentiation by squaring.
// ///
// /// This method works for `pow` being non-negative. If `pow` is 0, the result is 1.
// ///
// /// # Arguments
// ///
// /// * `a` - The base value of type `T`.
// /// * `pow` - The exponent value of type `u32`.
// ///
// /// # Returns
// ///
// /// A value of type `T` representing `a` raised to the power of `pow`.
// ///
// /// # Example
// ///
// /// ```
// /// let result = fast_power(2, 10); // result should be 1024
// /// ```
// pub fn fast_power<T>(a: T, pow: u32) -> T
// where
//     T: Mul<Output = T> + Copy + One,
// {
//     if pow == 0 {
//         return T::one();
//     }

//     let mut base = a;
//     let mut exponent = pow;
//     let mut result = T::one();

//     while exponent > 0 {
//         if exponent % 2 != 0 {
//             result = result * base;
//         }
//         base = base * base;
//         exponent /= 2;
//     }

//     result
// }

// #[test]
// fn test_fast_power() {
//     assert_eq!(fast_power(2, 10), 1024);
//     assert_eq!(fast_power(3, 0), 1); // a^0 should be 1
//     assert_eq!(fast_power(5, 3), 125); // 5^3 = 5 * 5 * 5
//     assert_eq!(fast_power(2, 1), 2); // 2^1 should be 2
// }

pub fn sample_random_short_mat<const MOD_Q: u64, const N: usize>(
    n: usize,
    m: usize,
    bound: u64,
) -> Vec<Vec<CyclotomicRing<MOD_Q, N>>> {
    (0..n)
        .map(|_| {
            (0..m)
                .map(|_| {
                    let t = CyclotomicRing::random_bounded(bound);
                    t
                })
                .collect()
        })
        .collect()
}

pub fn sample_random_biased_mat<const MOD_Q: u64, const N: usize>(
    n: usize,
    m: usize,
) -> Vec<Vec<CyclotomicRing<MOD_Q, N>>> {
    (0..n)
        .map(|_| {
            (0..m)
                .map(|_| {
                    let t = CyclotomicRing::random_biased();
                    t
                })
                .collect()
        })
        .collect()
}

// /// Joins two matrices horizontally.
// ///
// /// This function takes two matrices and joins them horizontally, i.e., concatenates them column-wise.
// ///
// /// # Arguments
// ///
// /// * `mat1` - A reference to the first matrix.
// /// * `mat2` - A reference to the second matrix.
// ///
// /// # Returns
// ///
// /// A single matrix that is the result of concatenating the input matrices column-wise.
// ///
// /// # Panics
// ///
// /// This function will panic if the matrices do not have the same number of rows.
// ///
// /// # Example
// ///
// /// ```rust
// /// let mat1 = vec![
// ///     vec![1, 2],
// ///     vec![3, 4],
// /// ];
// /// let mat2 = vec![
// ///     vec![5, 6],
// ///     vec![7, 8],
// /// ];
// /// let result = join_matrices_horizontally(&mat1, &mat2);
// /// assert_eq!(result, vec![
// ///     vec![1, 2, 5, 6],
// ///     vec![3, 4, 7, 8],
// /// ]);
// /// ```
// pub fn join_matrices_horizontally<T>(mat1: &Vec<Vec<T>>, mat2: &Vec<Vec<T>>) -> Vec<Vec<T>>
// where
//     T: Clone,
// {
//     // Ensure both matrices have the same number of rows
//     assert_eq!(mat1.len(), mat2.len(), "Both matrices must have the same number of rows");

//     // Initialize the result matrix
//     let mut result = Vec::with_capacity(mat1.len());

//     // Extend each row of the result with rows from both matrices
//     for (row1, row2) in mat1.iter().zip(mat2) {
//         let mut new_row = row1.clone();
//         new_row.extend(row2.clone());
//         result.push(new_row);
//     }

//     result
// }

// // Unit testing
// #[cfg(test)]
// mod join_matrices_horizontally_tests {
//     use super::*;

//     #[test]
//     fn test_join_matrices_horizontally_basic() {
//         let mat1 = vec![
//             vec![1, 2],
//             vec![3, 4],
//         ];
//         let mat2 = vec![
//             vec![5, 6],
//             vec![7, 8],
//         ];
//         let result = join_matrices_horizontally(&mat1, &mat2);
//         let expected = vec![
//             vec![1, 2, 5, 6],
//             vec![3, 4, 7, 8],
//         ];
//         assert_eq!(result, expected);
//     }

//     #[test]
//     fn test_join_matrices_horizontally_single_row() {
//         let mat1 = vec![vec![1, 2, 3]];
//         let mat2 = vec![vec![4, 5, 6]];
//         let result = join_matrices_horizontally(&mat1, &mat2);
//         let expected = vec![vec![1, 2, 3, 4, 5, 6]];
//         assert_eq!(result, expected);
//     }

//     #[test]
//     #[should_panic(expected = "Both matrices must have the same number of rows")]
//     fn test_join_matrices_horizontally_diff_row_count() {
//         let mat1 = vec![
//             vec![1, 2],
//             vec![3, 4],
//         ];
//         let mat2 = vec![
//             vec![5, 6],
//         ];
//         join_matrices_horizontally(&mat1, &mat2);
//     }

//     #[test]
//     fn test_join_matrices_horizontally_empty() {
//         let mat1: Vec<Vec<i32>> = vec![];
//         let mat2: Vec<Vec<i32>> = vec![];
//         let result = join_matrices_horizontally(&mat1, &mat2);
//         let expected: Vec<Vec<i32>> = vec![];
//         assert_eq!(result, expected);
//     }
// }

// /// Calls a Sage script to compute the inverse of a polynomial.
// ///
// /// # Arguments
// ///
// /// * `a` - A `RingElement` to find the inverse for.
// ///
// /// # Returns
// ///
// /// * `Option<RingElement>` - The inverse of the given polynomial if it exists, otherwise `None`.
// pub fn call_sage_inverse_polynomial<
//     const phi: usize,
//     const two_phi_minus_one: usize,
//     const mod_q: u64
// >(a: &PrimeRingElement<phi, two_phi_minus_one, mod_q>) -> PrimeRingElement<phi, two_phi_minus_one, mod_q> {
//         // Prepare the polynomial coefficients as a string.
//         let coeffs_a_str = format!("{:?}", a.coeffs).replace(" ", "");

//         // Call the Sage script with the required arguments.
//         let output = Command::new("sage")
//             .arg("inverse.sage")
//             .arg(&coeffs_a_str)
//             .arg(format!("{:?}", phi + 1).replace(" ", ""))
//             .arg(mod_q.to_string())
//             .output()
//             .expect("Failed to execute Sage script");

//         // Check if the Sage script execution was successful.
//         if !output.status.success() {
//             panic!("Error running Sage script: {:?}", output);
//         }

//         // Process the output from the Sage script.
//         let stdout = String::from_utf8_lossy(&output.stdout);
//         if stdout.trim() == "None" {
//             panic!("Inverse does not exist!");
//         }
//         let output = stdout.trim();
//         let result: Vec<u64> = output
//             .trim_matches(&['[', ']'] as &[_])
//             .split(',')
//             .map(|s| s.trim().parse().expect("Invalid number"))
//             .collect();

//     PrimeRingElement {
//         coeffs: <[u64; phi]>::try_from(result).unwrap()
//     }
// }

// #[test]
// pub fn test_inverse_sage() {
//     let a = PrimeRing::random();
//     let b = call_sage_inverse_polynomial(&a);
//     assert_eq!(a * b, PrimeRing::constant(1));
// }

// /// Computes a power series for the given element, prefixing the series with a `1`.
// ///
// /// # Arguments
// ///
// /// * `element` - A `RingElement` which serves as the base for the power series.
// /// * `len` - The length of the series.
// ///
// /// # Returns
// ///
// /// A vector containing a single vector of `RingElement`s, representing a power series prefixed with `1`.
// pub fn compute_one_prefixed_power_series(element: &DPrimeRingElement, len: usize) -> PowerSeries {
//     let mut series = Vec::with_capacity(len);
//     series.push(PrimeRing::constant(1));
//     series.push(element.clone());

//     let mut power = element.clone();
//     for _ in 2..len {
//         power = power.clone() * element.clone();
//         series.push(power.clone());
//     }

//     let mut ps = PowerSeries {
//         expanded_layers: vec![],
//         tensors: vec![],
//     };
//     let mut current_dim = len;
//     while current_dim % 2 == 0 {
//         ps.expanded_layers.push(series[0..current_dim].to_vec());
//         current_dim /= 2;
//         ps.tensors.push(vec![PrimeRingElement::one(), series[current_dim]]);
//     }
//     ps.expanded_layers.push(series[0..current_dim].to_vec());
//     ps
// }

pub fn compute_hp_power_series<const MOD_Q: u64, const N: usize>(
    elements: &Vec<CyclotomicRing<MOD_Q, N>>,
) -> PowerSeries<MOD_Q, N> {
    let mut ps = PowerSeries {
        expanded_layers: vec![],
        tensors: vec![],
    };

    ps.expanded_layers.push(vec![CyclotomicRing::constant(1)]);

    for t in elements.iter().rev() {
        let l_factor = CyclotomicRing::constant(1) - t.clone();
        let r_factor = t.clone();
        ps.tensors.push(vec![l_factor.clone(), r_factor.clone()]);
        ps.expanded_layers.push(kronecker_product(
            ps.tensors.last().unwrap(),
            ps.expanded_layers.last().unwrap(),
        ));
    }
    PowerSeries {
        expanded_layers: ps.expanded_layers.iter().rev().cloned().collect(),
        tensors: ps.tensors.iter().rev().cloned().collect(),
    }
}

// #[test]
// fn test_compute_one_prefixed_power_series() {
//     let element = PrimeRing::constant(2);
//     let result = compute_one_prefixed_power_series(&element, 4);
//     assert_eq!(
//         result,
//         PowerSeries {
//             expanded_layers: map_matrix_to_prime_ring(vec![
//                 vec![1, 2, 4, 8],
//                 vec![1, 2],
//                 vec![1],
//             ]),
//             tensors: map_matrix_to_prime_ring(vec![
//                 vec![1, 4],
//                 vec![1, 2]
//             ]),
//         }
//     );
// }

// pub fn ring_inner_product(a: &Vec<DPrimeRingElement<>>, b: &Vec<DPrimeRingElement>) -> DPrimeRingElement {
//     assert_eq!(a.len(), b.len(), "Input vectors must have the same length");

//     a.par_iter()
//         .zip(b.par_iter())
//         .map(|(a_i, b_i)| *a_i * *b_i)
//         .reduce(DPrimeRingElement::zero, |acc, prod| acc + prod)
// }

// /// Computes the conjugate of each element in a `RingElement` vector.
// ///
// /// # Arguments
// ///
// /// * `series` - A vector of `RingElement`s.
// ///
// /// # Returns
// ///
// /// A new vector where each `RingElement` in the input has been replaced by its conjugate.
// pub fn conjugate_vector(row: &Vec<DPrimeRingElement>) -> Vec<DPrimeRingElement> {
//     row.iter().map(DPrimeRingElement::conjugate).collect()
// }

// // RNS: assume small elements.

fn determine_bit_width<const MOD_Q: u64, const N: usize>(
    matrix: &Vec<Vec<CyclotomicRing<MOD_Q, N>>>,
) -> usize {
    let bit_accumulator = matrix
        .par_iter()
        .map(|row| {
            row.par_iter()
                .map(|cell| cell.data.iter().fold(0u64, |a, b| a | b))
                .reduce(|| 0u64, |a, b| a | b)
        })
        .reduce(|| 0u64, |a, b| a | b);

    64 - bit_accumulator.leading_zeros() as usize
}

pub fn neg_matrix<'a, const MOD_Q: u64, const N: usize>(
    matrix: impl ParallelIterator<Item = Vec<CyclotomicRing<MOD_Q, N>>>
        + rayon::iter::IndexedParallelIterator
        + 'a
        + Clone,
) -> impl ParallelIterator<Item = Vec<CyclotomicRing<MOD_Q, N>>>
       + rayon::iter::IndexedParallelIterator
       + 'a
       + Clone {
    matrix.map(|row| row.iter().map(|el| CyclotomicRing::zero() - *el).collect())
}
pub fn neg_matrix_orig<'a, const MOD_Q: u64, const N: usize>(
    matrix: &Vec<Vec<CyclotomicRing<MOD_Q, N>>>,
) -> Vec<Vec<CyclotomicRing<MOD_Q, N>>> {
    matrix
        .par_iter()
        .map(|row| row.iter().map(|el| CyclotomicRing::zero() - *el).collect())
        .collect()
}

pub fn split_into_positive_and_negative<const MOD_Q: u64, const N: usize>(
    matrix: Vec<Vec<CyclotomicRing<MOD_Q, N>>>,
) -> (
    Vec<Vec<CyclotomicRing<MOD_Q, N>>>,
    Vec<Vec<CyclotomicRing<MOD_Q, N>>>,
) {
    let (positive_matrix, negative_matrix): (Vec<_>, Vec<_>) = matrix
        .into_par_iter()
        .map(|row| {
            let row_len = row.len();

            let mut pos_vec: Vec<CyclotomicRing<MOD_Q, N>> = Vec::with_capacity(row_len);
            let mut neg_vec: Vec<CyclotomicRing<MOD_Q, N>> = Vec::with_capacity(row_len);

            unsafe {
                pos_vec.set_len(row_len);
                neg_vec.set_len(row_len);
            }
            pos_vec
                .par_iter_mut()
                .zip(neg_vec.par_iter_mut())
                .enumerate()
                .for_each(|(i, (pos, neg))| {
                    let (pos_data, neg_data) = split_simd::<MOD_Q, N>(&row[i].data);
                    *pos = CyclotomicRing {
                        data: pos_data,
                        representation: Representation::Coefficient,
                    };
                    *neg = CyclotomicRing {
                        data: neg_data,
                        representation: Representation::Coefficient,
                    };
                });

            (pos_vec, neg_vec)
        })
        .collect();

    (positive_matrix, negative_matrix)
}

pub fn decompose<const MOD_Q: u64, const N: usize>(
    matrix: &Vec<Vec<CyclotomicRing<MOD_Q, N>>>,
    radix: u64,
    num_chunks: usize,
) -> impl ParallelIterator<Item = Vec<CyclotomicRing<MOD_Q, N>>>
       + rayon::iter::IndexedParallelIterator
       + '_
       + Clone {
    let chunk_mask = radix - 1;
    let shift_amount = radix.trailing_zeros() as usize;

    let rows = matrix.len();
    let cols = matrix[0].len();

    (0..cols).into_par_iter().map(move |col_idx| {
        (0..rows)
            .flat_map(|row_idx| {
                let cell = &matrix[row_idx][col_idx];

                (0..num_chunks).map(move |chunk_idx| {
                    let mut chunk = CyclotomicRing::zero();
                    for k in 0..N {
                        let mut val = cell.data[k];
                        for _ in 0..chunk_idx {
                            val >>= shift_amount;
                        }
                        chunk.data[k] = val & chunk_mask;
                    }
                    chunk
                })
            })
            .collect::<Vec<_>>()
    })
}

pub fn decompose_matrix_by_chunks<const MOD_Q: u64, const N: usize>(
    mut matrix: Vec<Vec<CyclotomicRing<MOD_Q, N>>>,
    num_chunks: usize,
) -> (Vec<Vec<CyclotomicRing<MOD_Q, N>>>, u64) {
    matrix_to_coeff(&mut matrix);
    let (positive_matrix, negative_matrix) = split_into_positive_and_negative(matrix);

    let bit_width = max(
        determine_bit_width(&positive_matrix),
        determine_bit_width(&negative_matrix),
    );

    let radix_log = (bit_width as f64 / num_chunks as f64).ceil() as u32;
    let radix = (2u64).pow(radix_log);

    let decomposed_matrix = add_matrices_transposed(
        decompose(&positive_matrix, radix, num_chunks),
        neg_matrix(decompose(&negative_matrix, radix, num_chunks)),
    );

    (decomposed_matrix, radix)
}

// #[test]
// fn test_determine_bit_width() {
//     let matrix = vec![
//         vec![PrimeRing::constant(12345), PrimeRing::constant(67890)]
//     ];
//     assert_eq!(determine_bit_width(&matrix), 17);

//     let matrix = vec![
//         vec![PrimeRing::constant(1), PrimeRing::constant(2)],
//         vec![PrimeRing::constant(3), PrimeRing::constant(4)]
//     ];
//     assert_eq!(determine_bit_width(&matrix), 3);

//     let matrix = vec![
//         vec![PrimeRing::constant(u64::MAX), PrimeRing::constant(u64::MAX)]
//     ];
//     assert_eq!(determine_bit_width(&matrix), 64);
// }

// #[test]
// fn test_decompose_by_radix() {
//     let matrix = map_matrix_to_prime_ring(vec![
//         vec![17, 19],
//         vec![17, 19],
//     ]);
//     let radix = 16;

//     let (decomposed_matrix, num_chunks) = decompose_matrix_by_radix(&matrix, radix);
//     assert_eq!(num_chunks, 2);
//     assert_eq!(decomposed_matrix, map_matrix_to_prime_ring(vec![
//         vec![1, 1, 3, 1],
//         vec![1, 1, 3, 1]
//     ]));
// }

// #[test]
// fn test_decompose_by_radix_2() {
//     let matrix = map_matrix_to_prime_ring(vec![
//         vec![15, 15],
//         vec![15, 15],
//     ]);
//     let radix = 16;

//     let (decomposed_matrix, num_chunks) = decompose_matrix_by_radix(&matrix, radix);
//     assert_eq!(num_chunks, 1);
//     assert_eq!(decomposed_matrix, map_matrix_to_prime_ring(vec![
//         vec![15, 15],
//         vec![15, 15],
//     ]));
// }

// #[test]
// fn test_decompose_by_radix_3() {
//     let matrix = map_matrix_to_prime_ring(vec![
//         vec![15, 15],
//         vec![15, 16],
//     ]);
//     let radix = 16;

//     let (decomposed_matrix, num_chunks) = decompose_matrix_by_radix(&matrix, radix);
//     assert_eq!(num_chunks, 2);
//     assert_eq!(decomposed_matrix, map_matrix_to_prime_ring(vec![
//         vec![15, 0, 15, 0],
//         vec![15, 0, 0, 1],
//     ]));
// }

// #[test]
// fn test_decompose_by_chunks() {
//     let matrix = map_matrix_to_prime_ring(vec![
//         vec![15, 240],
//         vec![255, 16]
//     ]);
//     let num_chunks = 4;

//     let (decomposed_matrix, num_chunks) = decompose_matrix_by_chunks(&matrix, num_chunks);

//     assert_eq!(num_chunks, 4);
//     assert_eq!(decomposed_matrix, map_matrix_to_prime_ring(vec![
//         vec![3, 3, 0, 0, 0, 0, 3, 3],
//         vec![3, 3, 3, 3, 0, 0, 1, 0],
//     ]));
// }

pub fn compose_with_radix<const MOD_Q: u64, const N: usize>(
    mut decomposed_matrix: Vec<Vec<CyclotomicRing<MOD_Q, N>>>,
    radix: u64,
    num_chunks: usize, // into how many parts split
) -> Vec<Vec<CyclotomicRing<MOD_Q, N>>> {
    matrix_to_coeff(&mut decomposed_matrix);
    let radix_shift = radix.trailing_zeros() as usize;

    let (positive_matrix, negative_matrix) = split_into_positive_and_negative(decomposed_matrix);

    let compose = |decomposed_matrix: &Vec<Vec<CyclotomicRing<MOD_Q, N>>>| {
        decomposed_matrix
            .iter()
            .map(|chunked_row| {
                let num_elements_in_chunk = chunked_row.len() / num_chunks;
                (0..num_elements_in_chunk)
                    .map(|i| {
                        let mut coeffs = [0u64; N];
                        for k in 0..N {
                            coeffs[k] = (0..num_chunks).fold(0u64, |acc, el_chunk| {
                                acc | ((chunked_row[i * num_chunks + el_chunk].data[k] as u64)
                                    << (radix_shift * el_chunk))
                            });
                        }
                        CyclotomicRing {
                            representation: Representation::Coefficient,
                            data: coeffs,
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    };

    add_matrices(
        &compose(&positive_matrix),
        &neg_matrix_orig(&compose(&negative_matrix)),
    )
}

pub fn compose_with_radix_mod<const MOD_Q: u64, const N: usize>(
    decomposed_matrix: &Vec<Vec<CyclotomicRing<MOD_Q, N>>>,
    radix: u64,
    num_chunks: usize, // into how many parts split,
) -> Vec<Vec<CyclotomicRing<MOD_Q, N>>> {
    let radix_shift = radix.trailing_zeros() as usize;
    decomposed_matrix
        .iter()
        .map(|chunked_row| {
            let num_elements_in_chunk = chunked_row.len() / num_chunks;
            (0..num_elements_in_chunk)
                .map(|i| {
                    let mut coeffs = [0u64; N];
                    for k in 0..N {
                        coeffs[k] = (0..num_chunks).fold(0u64, |acc, el_chunk| {
                            unsafe {
                                return bindings::add_mod(
                                    acc,
                                    bindings::multiply_mod(
                                        chunked_row[i * num_chunks + el_chunk].data[k],
                                        2u64.pow((radix_shift * el_chunk) as u32),
                                        MOD_Q,
                                    ), // TODO eltwise FMA?
                                    MOD_Q,
                                );
                            }
                        });
                    }
                    CyclotomicRing {
                        representation: decomposed_matrix[0][0].representation,
                        data: coeffs,
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>()
}

// #[test]
// fn test_compose_with_radix() {
//     let matrix = sample_random_mat(2, 2);
//     let radix = 4;
//     let (decomposed_matrix, num_chunks) = decompose_matrix_by_radix(&matrix, radix);
//     let composed_matrix = compose_with_radix_mod(&decomposed_matrix, radix, num_chunks);

//     assert_eq!(composed_matrix, matrix);
// }

// pub fn reduce_mod_vec(a: &mut [u64], mod_q: u64) {
//     #[cfg(all(target_arch = "x86_64", feature = "use-hardware"))]
//     unsafe {
//         eltwise_reduce_mod(a.as_mut_ptr(), a.as_mut_ptr(), a.len() as u64, mod_q);
//     }
//     #[cfg(any(target_arch = "aarch64", not(feature = "use-hardware")))]
//     {
//         for i in 0..a.len() {
//             a[i] = a[i] % mod_q;
//         }
//     }
// }

// pub fn reduce_mod(a: &mut u64, mod_q: u64) {
//     reduce_mod_vec(&mut [*a], mod_q)
// }

// pub fn add_mod(a: u64, b: u64, mod_q: u64) -> u64 {
//     #[cfg(any(target_arch = "aarch64", not(feature = "use-hardware")))]
//     {
//         return ((a as u128 + b as u128) % (mod_q as u128)) as u64;
//     }
//     unsafe { bindings::add_mod(a, b, mod_q) }
// }

// pub fn sub_mod(a: u64, b: u64, mod_q: u64) -> u64 {
//     #[cfg(any(target_arch = "aarch64", not(feature = "use-hardware")))]
//     {
//         return ((mod_q as u128 + a as u128 - b as u128) % (mod_q as u128)) as u64;
//     }
//     unsafe { bindings::sub_mod(a, b, mod_q) }
// }

// pub fn multiply_mod(a: u64, b: u64, mod_q: u64) -> u64 {
//     #[cfg(any(target_arch = "aarch64", not(feature = "use-hardware")))]
//     {
//         return ((a as u128 * b as u128) % (mod_q as u128)) as u64;
//     };
//     unsafe { bindings::multiply_mod(a, b, mod_q) }
// }

// fn inverse_rns_slow(remainders: &[u64], primes: &[u64]) -> Integer {
//     assert_eq!(remainders.len(), primes.len(), "Mismatched remainders and primes length");

//     let product = primes.iter()
//         .fold(Integer::from(1), |acc, &p| acc * Integer::from(p));

//     let mut result = Integer::new();
//     for (&r, &p) in remainders.iter().zip(primes.iter()) {
//         let p_int = Integer::from(p);
//         let n = (&product / &p_int).complete(); // Convert division result to Integer
//         let m = n.clone().invert(&p_int).expect("Factors must be pairwise coprime");
//         result += Integer::from(r) * &n * &m;
//     }

//     result % product
// }
