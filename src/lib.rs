// SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
// SPDX-License-Identifier: MIT

//! # Mittag-Leffler Function
//!
//! This library implements the two-parameter Mittag-Leffler function.
//!
//! Currently only the algorithm described in the paper by
//! [Roberto Garrapa (2015)](<https://doi.org/10.1137/140971191>) is implemented.
//! This seems to be the most accurate and computationally efficient method to
//! date for evaluating the Mittag-Leffler function.
//!
//! ```rust
//! use num::complex::Complex64;
//! use mittagleffler::MittagLeffler;
//!
//! let alpha = 0.75;
//! let beta = 1.25;
//! let z = Complex64::new(1.0, 2.0);
//! println!("E({}; {}, {}) = {:?}", z, alpha, beta, z.mittag_leffler(alpha, beta));
//!
//! let z: f64 = 3.1415;
//! println!("E({}; {}, {}) = {:?}", z, alpha, beta, z.mittag_leffler(alpha, beta));
//! ```
//!
//! # Acknowledgments
//!
//! Work on ``pycaputo`` was sponsored, in part, by
//!
//! * the West University of Timișoara (Romania) under START Grant No. 33580/25.05.2023,
//! * the CNCS-UEFISCDI (Romania), under Project No. ROSUA-2024-0002,
//! * the "Romanian Hub for Artificial Intelligence - HRIA", Smart Growth,
//!   Digitization and Financial Instruments Program, 2021-2027, MySMIS no. 351416.
//!
//! The views and opinions expressed herein do not necessarily reflect those of the
//! funding agencies.

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
/// use mittagleffler::MittagLeffler;
///
/// let alpha: f64 = 1.0;
/// let beta: f64 = 1.0;
/// let z: f64 = 1.0;
/// let result = z.mittag_leffler(alpha, beta);
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
