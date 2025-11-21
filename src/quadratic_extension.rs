use std::ops::{Add, Mul};

use crate::hexl::bindings::{add_mod, multiply_mod};

use rand::Rng;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct QuadraticExtension<const MOD_Q: u64> {
    pub coeffs: [u64; 2],
    shift: u64,
}

impl<const MOD_Q: u64> Add for QuadraticExtension<MOD_Q> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let coeffs = unsafe {
            [
                add_mod(self.coeffs[0], other.coeffs[0], MOD_Q as u64),
                add_mod(self.coeffs[1], other.coeffs[1], MOD_Q as u64),
            ]
        };
        Self {
            coeffs,
            shift: self.shift, // Assuming shift remains unchanged
        }
    }
}

impl<const MOD_Q: u64> Mul for QuadraticExtension<MOD_Q> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let a = self.coeffs[0];
        let b = self.coeffs[1];
        let c = other.coeffs[0];
        let d = other.coeffs[1];

        let coeffs = unsafe {
            [
                add_mod(
                    multiply_mod(a, c, MOD_Q as u64),
                    multiply_mod(self.shift, multiply_mod(b, d, MOD_Q as u64), MOD_Q as u64),
                    MOD_Q as u64,
                ),
                add_mod(
                    multiply_mod(a, d, MOD_Q as u64),
                    multiply_mod(b, c, MOD_Q as u64),
                    MOD_Q as u64,
                ),
            ]
        };
        Self {
            coeffs,
            shift: self.shift, // Assuming shift remains unchanged
        }
    }
}

impl<const MOD_Q: u64> QuadraticExtension<MOD_Q> {
    pub fn new(coeffs: [u64; 2], shift: u64) -> Self {
        Self { coeffs, shift }
    }

    pub fn random(shift: u64) -> Self {
        let mut rng = rand::rng();
        let coeffs = [rng.random_range(0..MOD_Q), rng.random_range(0..MOD_Q)];
        Self { coeffs, shift }
    }
}

// impl<const MOD_Q: u64> Zero for QuadraticExtension<MOD_Q> {
//     fn zero() -> Self {
//         Self {
//             coeffs: [0, 0],
//             shift: 0, // Assuming a zero shift for the zero element
//         }
//     }

//     fn is_zero(&self) -> bool {
//         self.coeffs[0] == 0 && self.coeffs[1] == 0
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quadratic_extension_addition() {
        const MOD_Q: u64 = 17;
        let a = QuadraticExtension::<MOD_Q>::new([3, 5], 1);
        let b = QuadraticExtension::<MOD_Q>::new([7, 9], 1);

        let result = a + b;

        assert_eq!(result.coeffs, [10, 14]);
        assert_eq!(result.shift, 1);
    }

    #[test]
    fn test_quadratic_extension_multiplication() {
        const MOD_Q: u64 = 17;
        let a = QuadraticExtension::<MOD_Q>::new([3, 5], 1);
        let b = QuadraticExtension::<MOD_Q>::new([7, 9], 1);

        let result = a * b;

        assert_eq!(result.coeffs, [15, 11]);
        assert_eq!(result.shift, 1);
    }
}
