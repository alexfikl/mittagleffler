// SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
// SPDX-License-Identifier: MIT

mod algorithm;
mod garrappa;

use crate::algorithm::mittag_leffler_special;
use num::complex::Complex64;

pub use algorithm::MittagLefflerAlgorithm;
pub use garrappa::GarrappaMittagLeffler;

/// Mittag-Leffler function.
///
/// Evaluates the Mittag-Leffler function using default parameters. It can be
/// used as
/// ```rust
///     let alpha: f64 = 1.0;
///     let beta: f64 = 1.0;
///     let z: f64 = 1.0;
///     let result = z.mittag_leffler(alpha, beta);
/// ```
/// on real or complex arguments.
pub trait MittagLeffler
where
    Self: Sized,
{
    fn mittag_leffler(&self, alpha: f64, beta: f64) -> Option<Complex64>;
}

impl MittagLeffler for f64 {
    fn mittag_leffler(&self, alpha: f64, beta: f64) -> Option<Complex64> {
        let ml = GarrappaMittagLeffler::default();
        let z = Complex64::new(*self, 0.0);

        match mittag_leffler_special(z, alpha, beta) {
            Some(value) => Some(value),
            None => ml.evaluate(z, alpha, beta),
        }
    }
}

impl MittagLeffler for Complex64 {
    fn mittag_leffler(&self, alpha: f64, beta: f64) -> Option<Complex64> {
        let ml = GarrappaMittagLeffler::default();

        match mittag_leffler_special(*self, alpha, beta) {
            Some(value) => Some(value),
            None => ml.evaluate(*self, alpha, beta),
        }
    }
}
