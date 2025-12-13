use crate::hexl::bindings::{
    eltwise_add_mod, eltwise_mult_mod, eltwise_reduce_mod, eltwise_sub_mod, inv_mod,
    ntt_forward_in_place, ntt_inverse_in_place, power_mod,
};
use crate::quadratic_extension::QuadraticExtension;
use num_traits::Zero;
use rand::Rng;
use std::collections::HashMap;
use std::iter::Sum;
use std::ops::{Add, Mul, Neg, Sub};
use std::sync::{Mutex, OnceLock};

#[derive(Clone, Debug, PartialEq, Eq, Copy)]
pub enum Representation {
    Coefficient,
    NTT,
    IncompleteNTT,
}

const TEMP_N: usize = crate::protocol::N / 2;

impl<const MOD_Q: u64, const N: usize> Zero for CyclotomicRing<MOD_Q, N> {
    fn zero() -> Self {
        let mut data = [0u64; N];
        data[0] = 0;
        CyclotomicRing {
            data,
            representation: Representation::Coefficient,
        }
    }

    fn is_zero(&self) -> bool {
        self.data.iter().all(|&x| x == 0)
    }
}

#[derive(Clone, Debug, Copy, PartialEq)]
pub struct CyclotomicRing<const MOD_Q: u64, const N: usize> {
    pub data: [u64; N],
    pub representation: Representation,
}

impl<const MOD_Q: u64, const N: usize> Add<&CyclotomicRing<MOD_Q, N>>
    for &mut CyclotomicRing<MOD_Q, N>
{
    type Output = CyclotomicRing<MOD_Q, N>;

    fn add(self, other: &CyclotomicRing<MOD_Q, N>) -> Self::Output {
        self.adjust_representation(other.representation);
        let mut result = CyclotomicRing::<MOD_Q, N>::new();
        unsafe {
            eltwise_add_mod(
                result.data.as_mut_ptr(),
                self.data.as_ptr(),
                other.data.as_ptr(),
                N as u64,
                MOD_Q,
            )
        }
        result.representation = self.representation.clone();
        result
    }
}

impl<const MOD_Q: u64, const N: usize> Add for CyclotomicRing<MOD_Q, N> {
    type Output = CyclotomicRing<MOD_Q, N>;

    fn add(mut self, other: Self) -> Self::Output {
        &mut self + &other
    }
}

impl<const MOD_Q: u64, const N: usize> Sum for CyclotomicRing<MOD_Q, N> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut result = CyclotomicRing::<MOD_Q, N>::zero();
        for item in iter {
            result = result + item;
        }
        result
    }
}

impl<const MOD_Q: u64, const N: usize> Sub<&CyclotomicRing<MOD_Q, N>>
    for &mut CyclotomicRing<MOD_Q, N>
{
    type Output = CyclotomicRing<MOD_Q, N>;

    fn sub(self, other: &CyclotomicRing<MOD_Q, N>) -> Self::Output {
        self.adjust_representation(other.representation);
        let mut result = CyclotomicRing::<MOD_Q, N>::new();
        unsafe {
            eltwise_sub_mod(
                result.data.as_mut_ptr(),
                self.data.as_ptr(),
                other.data.as_ptr(),
                N as u64,
                MOD_Q,
            )
        }
        result.representation = self.representation.clone();
        result
    }
}
impl<const MOD_Q: u64, const N: usize> Neg for &CyclotomicRing<MOD_Q, N> {
    type Output = CyclotomicRing<MOD_Q, N>;

    fn neg(self) -> Self::Output {
        let mut other = CyclotomicRing::zero();
        unsafe {
            eltwise_sub_mod(
                other.data.as_mut_ptr(),
                other.data.as_ptr(),
                self.data.as_ptr(),
                N as u64,
                MOD_Q,
            )
        }

        other.representation = self.representation.clone();
        other
    }
}

impl<const MOD_Q: u64, const N: usize> Neg for CyclotomicRing<MOD_Q, N> {
    type Output = CyclotomicRing<MOD_Q, N>;

    fn neg(self) -> Self::Output {
        let mut other = CyclotomicRing::zero();
        unsafe {
            eltwise_sub_mod(
                other.data.as_mut_ptr(),
                other.data.as_ptr(),
                self.data.as_ptr(),
                N as u64,
                MOD_Q,
            )
        }

        other.representation = self.representation.clone();
        other
    }
}

impl<const MOD_Q: u64, const N: usize> Neg for &mut CyclotomicRing<MOD_Q, N> {
    type Output = CyclotomicRing<MOD_Q, N>;

    fn neg(self) -> Self::Output {
        let mut other = CyclotomicRing::zero();
        unsafe {
            eltwise_sub_mod(
                other.data.as_mut_ptr(),
                other.data.as_ptr(),
                self.data.as_ptr(),
                N as u64,
                MOD_Q,
            )
        }

        other.representation = self.representation.clone();
        other
    }
}

impl<const MOD_Q: u64, const N: usize> Sub for CyclotomicRing<MOD_Q, N> {
    type Output = CyclotomicRing<MOD_Q, N>;

    fn sub(mut self, other: Self) -> Self::Output {
        &mut self - &other
    }
}

// impl<const MOD_Q: u64, const N: usize> Sub for CyclotomicRing<MOD_Q, N> {
//     type Output = CyclotomicRing<MOD_Q, N>;
//     fn sub(mut self, other: Self) -> Self::Output {
//         self.adjust_representation(other.representation);
//         let mut result = CyclotomicRing::<MOD_Q, N>::new();
//         for i in 0..N {
//             result.data[i] = (self.data[i] + MOD_Q - other.data[i]) % MOD_Q;
//         }
//         result.representation = self.representation.clone();
//         result
//     }
// }

impl<const MOD_Q: u64, const N: usize> Mul for &mut CyclotomicRing<MOD_Q, N> {
    type Output = CyclotomicRing<MOD_Q, N>;

    fn mul(self, other: Self) -> Self::Output {
        incomplete_ntt_multiplication::<MOD_Q, N>(self, other, true)
    }
}

impl<const MOD_Q: u64, const N: usize> Mul for CyclotomicRing<MOD_Q, N> {
    type Output = CyclotomicRing<MOD_Q, N>;

    fn mul(mut self, mut other: Self) -> Self::Output {
        &mut self * &mut other
    }
}

#[test]
fn test_addition_same_representation() {
    const MOD_Q: u64 = 17;
    const N: usize = 4;
    let mut a = CyclotomicRing::<MOD_Q, N>::new();
    let mut b = CyclotomicRing::<MOD_Q, N>::new();
    a.data = [1, 2, 3, 4];
    b.data = [4, 3, 2, 1];

    let c = &mut a + &b;

    assert_eq!(c.data, [5, 5, 5, 5]);
}

#[test]
fn test_subtraction_same_representation() {
    const MOD_Q: u64 = 17;
    const N: usize = 4;
    let mut a = CyclotomicRing::<MOD_Q, N>::new();
    let mut b = CyclotomicRing::<MOD_Q, N>::new();
    a.data = [5, 6, 7, 8];
    b.data = [4, 3, 2, 1];

    let c = &mut a - &b;

    assert_eq!(c.data, [1, 3, 5, 7]);
}

#[test]
fn test_multiplication_same_representation() {
    const MOD_Q: u64 = 17;
    const N: usize = 4;
    let mut a = CyclotomicRing::<MOD_Q, N>::new();
    let mut b = CyclotomicRing::<MOD_Q, N>::new();
    a.data = [1, 2, 1, 0];
    b.data = [1, 1, 1, 0];

    let mut c = &mut a * &mut b;
    c.to_coeff_representation();

    assert_eq!(c.data, [0, 3, 4, 3]);
}

static NORMALIZE_INCOMPLETE_NTT_FACTORS_CACHE: OnceLock<Mutex<HashMap<usize, Vec<u64>>>> =
    OnceLock::new();
static NORMALIZE_INCOMPLETE_NTT_FACTORS_INVERSE_CACHE: OnceLock<Mutex<HashMap<usize, Vec<u64>>>> =
    OnceLock::new();

impl<const MOD_Q: u64, const N: usize> CyclotomicRing<MOD_Q, N> {
    pub fn new() -> Self {
        Self {
            data: [0u64; N],
            representation: Representation::Coefficient,
        }
    }

    pub fn random() -> Self {
        let mut rng = rand::rng();
        let mut data = [0u64; N];
        for i in 0..N {
            data[i] = rng.random_range(0..MOD_Q);
        }
        let mut t = Self {
            data,
            representation: Representation::Coefficient,
        };
        t.to_coeff_representation();
        t
    }

    pub fn random_real() -> Self {
        let t = CyclotomicRing::random();
        let res = t + t.conjugate();
        // assert_eq!(res, res.conjugate());
        res
    }

    pub fn random_with_equal_fields_slot() -> Self {
        // let shift = (|| {
        //     let cache = SHIFT_FACTORS_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        //     let mut cache_guard = cache.lock().unwrap();
        //     cache_guard
        //     .entry(N)
        //     .or_insert_with(|| get_shift_factors::<MOD_Q, N>())
        //     [0]
        // })();

        let shift = get_shift_factors_cached::<MOD_Q, N>()[0];
        let t = QuadraticExtension::random(shift);
        CyclotomicRing::<MOD_Q, N>::from_quadratic_fields(vec![t; N / 2])
    }

    pub fn random_bounded(bound: u64) -> Self {
        let mut rng = rand::rng();
        let mut data = [0u64; N];
        for i in 0..N {
            data[i] = rng.random_range(0..bound);
            if rng.random_bool(0.5) {
                data[i] = MOD_Q - data[i];
            }
        }
        unsafe {
            eltwise_reduce_mod(
                data.as_mut_ptr(),
                data.as_mut_ptr(),
                data.len() as u64,
                MOD_Q,
            );
        }

        let mut t = Self {
            data,
            representation: Representation::Coefficient,
        };
        t.to_coeff_representation();
        t
    }

    pub fn random_biased() -> Self {
        let mut rng = rand::rng();
        let mut data = [0u64; N];

        for i in 0..N {
            let r = rng.random_range(0..3);
            data[i] = match r {
                0 => MOD_Q - 1,
                1 => 0,
                2 => 1,
                _ => unreachable!(),
            };
        }

        let mut t = Self {
            data,
            representation: Representation::Coefficient,
        };
        t.to_coeff_representation();
        t
    }

    pub fn constant(value: u64) -> Self {
        let mut data = [0u64; N];
        data[0] = value;
        Self {
            data,
            representation: Representation::Coefficient,
        }
    }

    pub fn one() -> Self {
        let mut data = [0u64; N];
        data[0] = 1;
        Self {
            data,
            representation: Representation::Coefficient,
        }
    }

    fn normalize_incomplete_ntt(&mut self) {
        assert!(
            self.representation == Representation::IncompleteNTT,
            "Cannot normalize unless in Incomplete NTT representation"
        );

        // Use cached normalization factors and their inverses
        let normalize_factors = (|| {
            let cache =
                NORMALIZE_INCOMPLETE_NTT_FACTORS_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
            let mut cache_guard = cache.lock().unwrap();
            cache_guard
                .entry(N)
                .or_insert_with(|| get_roots_of_unity_trans::<MOD_Q, N>().0)
                .clone()
        })();

        // Multiply the odd part by the normalization factors
        unsafe {
            eltwise_mult_mod(
                self.data.as_mut_ptr().add(N / 2),
                self.data.as_ptr().add(N / 2),
                normalize_factors.as_ptr(),
                (N / 2) as u64,
                MOD_Q,
            );
        }
    }

    fn normalize_incomplete_ntt_inverse(&mut self) {
        assert!(
            self.representation == Representation::IncompleteNTT,
            "Cannot inverse normalize unless in Incomplete NTT representation"
        );

        // Use cached normalization factors and their inverses
        let normalize_factors = (|| {
            let cache = NORMALIZE_INCOMPLETE_NTT_FACTORS_INVERSE_CACHE
                .get_or_init(|| Mutex::new(HashMap::new()));
            let mut cache_guard = cache.lock().unwrap();
            cache_guard
                .entry(N)
                .or_insert_with(|| get_roots_of_unity_trans::<MOD_Q, N>().1)
                .clone()
        })();

        // Multiply the odd part by the normalization factors
        unsafe {
            eltwise_mult_mod(
                self.data.as_mut_ptr().add(N / 2),
                self.data.as_ptr().add(N / 2),
                normalize_factors.as_ptr(),
                (N / 2) as u64,
                MOD_Q,
            );
        }
    }

    pub fn spit_into_quadratic_fields(&mut self) -> Vec<QuadraticExtension<MOD_Q>> {
        // let shift = (|| {
        //     let cache = SHIFT_FACTORS_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        //     let mut cache_guard = cache.lock().unwrap();
        //     cache_guard
        //     .entry(N)
        //     .or_insert_with(|| get_shift_factors::<MOD_Q, N>())
        //     [0]
        // })();

        let shift = get_shift_factors_cached::<MOD_Q, N>()[0];

        self.to_incomplete_ntt_representation();
        self.normalize_incomplete_ntt();

        let mut quadratic_fields = Vec::with_capacity(N / 2);
        for i in 0..N / 2 {
            let mut coeffs = [0u64; 2];
            coeffs[0] = self.data[i];
            coeffs[1] = self.data[i + N / 2];
            quadratic_fields.push(QuadraticExtension::<MOD_Q>::new(coeffs, shift as u64));
        }

        self.normalize_incomplete_ntt_inverse();
        quadratic_fields
    }

    pub fn from_quadratic_fields(quadratic_fields: Vec<QuadraticExtension<MOD_Q>>) -> Self {
        let mut cyclotomic_ring = CyclotomicRing::<MOD_Q, N>::new();
        for i in 0..N / 2 {
            cyclotomic_ring.data[i] = quadratic_fields[i].coeffs[0];
            cyclotomic_ring.data[i + N / 2] = quadratic_fields[i].coeffs[1];
        }
        cyclotomic_ring.representation = Representation::IncompleteNTT;
        cyclotomic_ring.normalize_incomplete_ntt_inverse();
        cyclotomic_ring
    }

    pub fn conjugate(&self) -> Self {
        let mut conjugated = self.clone();
        conjugated.to_coeff_representation();
        let conjugated_clone = conjugated.clone();

        for i in 1..N {
            if conjugated_clone.data[N - i] == 0 {
                conjugated.data[i] = 0;
                continue;
            }
            conjugated.data[i] = MOD_Q - conjugated_clone.data[N - i];
        }
        conjugated.adjust_representation(self.representation);
        conjugated
    }

    fn adjust_representation(&mut self, new_representation: Representation) {
        if self.representation == new_representation {
            return; // already in the desired representation
        }

        match new_representation {
            Representation::Coefficient => self.to_coeff_representation(),
            Representation::NTT => self.to_ntt_representation(),
            Representation::IncompleteNTT => self.to_incomplete_ntt_representation(),
        }
    }

    #[inline]
    pub fn to_incomplete_ntt_representation(&mut self) {
        if self.representation == Representation::IncompleteNTT {
            return;
        }
        if self.representation == Representation::NTT {
            self.to_coeff_representation();
        }

        let mut even_data = [0u64; TEMP_N];
        let mut odd_data = [0u64; TEMP_N];

        for i in 0..(N / 2) {
            even_data[i] = self.data[i * 2];
            odd_data[i] = self.data[i * 2 + 1];
        }

        unsafe {
            ntt_forward_in_place(even_data.as_mut_ptr(), N / 2, MOD_Q);
            ntt_forward_in_place(odd_data.as_mut_ptr(), N / 2, MOD_Q);
        }

        for i in 0..(N / 2) {
            self.data[i] = even_data[i];
            self.data[i + N / 2] = odd_data[i];
        }

        self.representation = Representation::IncompleteNTT;
    }

    #[inline]
    pub fn to_ntt_representation(&mut self) {
        if self.representation == Representation::NTT {
            return;
        }

        if self.representation == Representation::IncompleteNTT {
            self.to_coeff_representation();
        }

        unsafe { ntt_forward_in_place(self.data.as_mut_ptr(), self.data.len(), MOD_Q) };
        self.representation = Representation::NTT;
    }

    #[inline]
    pub fn to_coeff_representation(&mut self) {
        if self.representation == Representation::Coefficient {
            return;
        }

        if self.representation == Representation::IncompleteNTT {
            let mut even_odd = [0u64; N];
            for i in 0..N / 2 {
                even_odd[i] = self.data[i]; // even part
                even_odd[i + N / 2] = self.data[i + N / 2]; // odd part
            }

            unsafe {
                ntt_inverse_in_place(even_odd.as_mut_ptr(), N / 2, MOD_Q);
                ntt_inverse_in_place(even_odd.as_mut_ptr().add(N / 2), N / 2, MOD_Q);
            }

            for i in 0..N / 2 {
                self.data[i * 2] = even_odd[i];
                self.data[i * 2 + 1] = even_odd[i + N / 2];
            }
        }

        if self.representation == Representation::NTT {
            unsafe { ntt_inverse_in_place(self.data.as_mut_ptr(), self.data.len(), MOD_Q) };
        }

        self.representation = Representation::Coefficient;
    }
}

fn get_shift_factors<const MOD_Q: u64, const N: usize>() -> Vec<u64> {
    let mut factors = vec![0u64; N / 2];
    factors[1] = 1;
    unsafe { ntt_forward_in_place(factors.as_mut_ptr(), factors.len(), MOD_Q) };
    factors
}

pub fn get_roots_of_unity_trans<const MOD_Q: u64, const N: usize>() -> (Vec<u64>, Vec<u64>) {
    // Get or compute shift factors
    // let shift_factors = (|| {
    //     let cache = SHIFT_FACTORS_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    //     let mut cache_guard = cache.lock().unwrap();
    //     cache_guard
    //         .entry(N)
    //         .or_insert_with(|| get_shift_factors::<MOD_Q, N>()).clone()
    // })();
    let shift_factors = get_shift_factors::<MOD_Q, N>();
    let mut roots_translations = vec![0u64; N / 2];
    for i in 0..N / 2 {
        let mut t = 0;
        while (|| {
            let mut ex = CyclotomicRing::<MOD_Q, N>::constant(0);
            ex.to_incomplete_ntt_representation();
            ex.data[N / 2 + i] = 1;
            let ex_0 = incomplete_ntt_multiplication(&mut (ex.clone()), &mut ex, true);
            ex.data[N / 2 + i] = unsafe { power_mod(shift_factors[i], t, MOD_Q) };

            let ex_1 = incomplete_ntt_multiplication(&mut (ex.clone()), &mut ex, false);
            ex_0 != ex_1
        })() {
            t = t + 1;
            // if t > 10 {
            //     panic!("Failed to find roots translation for i = {} after 10 iterations", i);
            // }
        }

        roots_translations[i] = unsafe { power_mod(shift_factors[i], t, MOD_Q) };
    }

    let mut roots_translations_inv = vec![0u64; N / 2];

    for i in 0..N / 2 {
        roots_translations_inv[i] = unsafe { inv_mod(roots_translations[i], MOD_Q) };
    }

    (roots_translations, roots_translations_inv)
}

#[cfg(test)]
mod tests {
    use crate::arithmetic::{add_vectors, multiply_vectors};

    use super::*;
    const MOD_Q: u64 = 4546383823830515713;
    const N: usize = 32;
    #[test]
    fn test_get_powers_root() {
        let (_translation_factors, _translation_factors_inv) =
            get_roots_of_unity_trans::<MOD_Q, N>();
        let mut ex = CyclotomicRing::<MOD_Q, N>::random();
        ex.to_incomplete_ntt_representation();
        let ex_0 = incomplete_ntt_multiplication(&mut (ex.clone()), &mut ex, true);

        ex.normalize_incomplete_ntt();

        let mut ex_1 = incomplete_ntt_multiplication(&mut (ex.clone()), &mut ex, false);
        ex_1.normalize_incomplete_ntt_inverse();
        assert_eq!(ex_0, ex_1);
    }

    #[test]
    fn to_and_from_quadratic_fields() {
        let mut cyclotomic_ring = CyclotomicRing::<MOD_Q, N>::random();
        let quadratic_fields = cyclotomic_ring.spit_into_quadratic_fields();
        let cyclotomic_ring_from_fields =
            CyclotomicRing::<MOD_Q, N>::from_quadratic_fields(quadratic_fields);
        assert_eq!(cyclotomic_ring, cyclotomic_ring_from_fields);
    }

    #[test]
    fn to_and_from_quadratic_fields_with_multiplication() {
        let mut a = CyclotomicRing::<MOD_Q, N>::random();
        let mut b = CyclotomicRing::<MOD_Q, N>::random();
        let res = &mut a * &mut b;
        let quadratic_fields_a = a.spit_into_quadratic_fields();
        let quadratic_fields_b = b.spit_into_quadratic_fields();
        let quadratic_fields_res = multiply_vectors(&quadratic_fields_a, &quadratic_fields_b);
        let res_2 = CyclotomicRing::<MOD_Q, N>::from_quadratic_fields(quadratic_fields_res);
        assert_eq!(res, res_2);
    }

    #[test]
    fn to_and_from_quadratic_fields_with_add() {
        let mut a = CyclotomicRing::<MOD_Q, N>::random();
        let mut b = CyclotomicRing::<MOD_Q, N>::random();
        let mut res = &mut a + &mut b;
        res.to_incomplete_ntt_representation();
        let quadratic_fields_a = a.spit_into_quadratic_fields();
        let quadratic_fields_b = b.spit_into_quadratic_fields();
        let quadratic_fields_res = add_vectors(&quadratic_fields_a, &quadratic_fields_b);
        let res_2 = CyclotomicRing::<MOD_Q, N>::from_quadratic_fields(quadratic_fields_res);
        assert_eq!(res, res_2);
    }

    #[test]
    fn multiply_conjugate() {
        let mut a = CyclotomicRing::<MOD_Q, N>::random();
        let mut a_conj = a.conjugate();
        let mut b = CyclotomicRing::<MOD_Q, N>::random();
        let mut b_conj = b.conjugate();
        let res = &mut a * &mut b;
        let res_conj = &mut a_conj * &mut b_conj;
        assert_eq!(res, res_conj.conjugate());
    }

    #[test]
    fn multiply_real() {
        let mut a = CyclotomicRing::<MOD_Q, N>::random_real();
        let mut b = CyclotomicRing::<MOD_Q, N>::random_real();
        let res = &mut a * &mut b;
        let res_2 = a * b;
        assert_eq!(res, res.conjugate());
        assert_eq!(res_2, res_2.conjugate());
        assert_eq!(a, a.conjugate());
        assert_eq!(b, b.conjugate());
    }
}

static SHIFT_FACTORS_CACHE: OnceLock<Vec<u64>> = OnceLock::new();

fn get_shift_factors_cached<const MOD_Q: u64, const N: usize>() -> Vec<u64> {
    // Ensure the cache is initialized
    // Safe to access without locking since OnceLock + HashMap is read-only after init
    if cfg!(test) {
        return get_shift_factors::<MOD_Q, N>();
    }
    SHIFT_FACTORS_CACHE
        .get_or_init(get_shift_factors::<MOD_Q, N>)
        .clone()
}

#[inline]
pub fn incomplete_ntt_multiplication<const MOD_Q: u64, const N: usize>(
    operand1: &mut CyclotomicRing<MOD_Q, N>,
    operand2: &mut CyclotomicRing<MOD_Q, N>,
    use_shift_factors: bool,
) -> CyclotomicRing<MOD_Q, N> {
    let shift_factors = get_shift_factors_cached::<MOD_Q, N>();

    operand1.to_incomplete_ntt_representation();
    operand2.to_incomplete_ntt_representation();

    let mut result = CyclotomicRing::<MOD_Q, N>::new();

    let mut temp = [0u64; TEMP_N];

    let op1_data = &operand1.data;
    let op2_data = &operand2.data;

    unsafe {
        // result_even = op1_even * op2_even
        eltwise_mult_mod(
            result.data.as_mut_ptr(),
            op1_data.as_ptr(),
            op2_data.as_ptr(),
            (N / 2) as u64,
            MOD_Q,
        );

        // result_odd = op1_odd * op2_even
        eltwise_mult_mod(
            result.data.as_mut_ptr().add(N / 2),
            op1_data.as_ptr().add(N / 2),
            op2_data.as_ptr(),
            (N / 2) as u64,
            MOD_Q,
        );

        // temp = op1_odd * op2_odd
        eltwise_mult_mod(
            temp.as_mut_ptr(),
            op1_data.as_ptr().add(N / 2),
            op2_data.as_ptr().add(N / 2),
            (N / 2) as u64,
            MOD_Q,
        );

        // Apply shift factors
        if use_shift_factors {
            eltwise_mult_mod(
                temp.as_mut_ptr(),
                temp.as_ptr(),
                shift_factors.as_ptr(),
                (N / 2) as u64,
                MOD_Q,
            );
        } else if shift_factors[0] != 1 {
            let factor = shift_factors[0];
            for i in 0..(N / 2) {
                temp[i] = ((temp[i] as u128 * factor as u128) % MOD_Q as u128) as u64;
            }
        }

        // result_even += temp
        eltwise_add_mod(
            result.data.as_mut_ptr(),
            result.data.as_ptr(),
            temp.as_ptr(),
            (N / 2) as u64,
            MOD_Q,
        );

        // Reuse temp for op1_even * op2_odd
        eltwise_mult_mod(
            temp.as_mut_ptr(),
            op1_data.as_ptr(),
            op2_data.as_ptr().add(N / 2),
            (N / 2) as u64,
            MOD_Q,
        );

        // result_odd += temp
        eltwise_add_mod(
            result.data.as_mut_ptr().add(N / 2),
            result.data.as_ptr().add(N / 2),
            temp.as_ptr(),
            (N / 2) as u64,
            MOD_Q,
        );
    }

    result.representation = Representation::IncompleteNTT;
    result
}

pub fn fully_splitting_ntt_multiplication<const MOD_Q: u64, const N: usize>(
    operand1: &mut CyclotomicRing<MOD_Q, N>,
    operand2: &mut CyclotomicRing<MOD_Q, N>,
) -> CyclotomicRing<MOD_Q, N> {
    operand1.to_ntt_representation();
    operand2.to_ntt_representation();

    let mut result = CyclotomicRing::<MOD_Q, N>::new();

    unsafe {
        eltwise_mult_mod(
            result.data.as_mut_ptr(),
            operand1.data.as_ptr(),
            operand2.data.as_ptr(),
            result.data.len() as u64,
            MOD_Q,
        )
    };

    result.representation = Representation::NTT;
    result
}

pub fn naive_multiply<const MOD_Q: u64, const N: usize>(
    operand1: &mut CyclotomicRing<MOD_Q, N>,
    operand2: &mut CyclotomicRing<MOD_Q, N>,
) -> CyclotomicRing<MOD_Q, N> {
    operand1.to_coeff_representation();
    operand2.to_coeff_representation();
    let mut result = CyclotomicRing::<MOD_Q, N>::new();
    for i in 0..N {
        for j in 0..N {
            if i + j < N {
                result.data[i + j] = (result.data[i + j]
                    + ((operand1.data[i] as u128 * operand2.data[j] as u128) % MOD_Q as u128)
                        as u64)
                    % MOD_Q;
            } else {
                result.data[i + j - N] = (result.data[i + j - N] + MOD_Q
                    - ((operand1.data[i] as u128 * operand2.data[j] as u128) % MOD_Q as u128)
                        as u64)
                    % MOD_Q;
            }
        }
    }
    result
}
