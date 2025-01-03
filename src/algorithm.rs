// SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
// SPDX-License-Identifier: MIT

use num::complex::Complex64;

// {{{ evaluate

/// An algorithm used to evaluate the Mittag-Leffler function.
pub trait MittagLefflerAlgorithm {
    /// Evaluate the Mittag-Leffler function $E_{\alpha, \beta}(z)$.
    ///
    /// Note that this function does not need to evaluate the function to
    /// machine precision. Consult each implementation for its accuracy guarantees.
    ///
    /// If the algorithm cannot compute the Mittag-Leffler function for a given
    /// set of $(\alpha, \beta)$ or in a region of the complex plane, then *None*
    /// is returned instead.
    fn evaluate(&self, z: Complex64, alpha: f64, beta: f64) -> Option<Complex64>;
}

// }}}

// {{{ known values

pub fn mittag_leffler_special(z: Complex64, alpha: f64, beta: f64) -> Option<Complex64> {
    let eps = f64::EPSILON;

    if alpha.abs() < eps && (z.norm() - 1.0).abs() < eps {
        return Some(1.0 / (1.0 - z) / special::Gamma::gamma(beta));
    }

    if (beta - 1.0).abs() < eps {
        if (alpha - 1.0).abs() < eps {
            return Some(z.exp());
        } else if (alpha - 2.0).abs() < eps {
            return Some(z.sqrt().cosh());
        } else if (alpha - 3.0).abs() < eps {
            let z3 = z.cbrt();
            return Some(z3.exp() + 2.0 * (-z3 / 2.0).exp() * (3.0_f64.sqrt() / 2.0 * z3).cos());
        } else if (alpha - 4.0).abs() < eps {
            let z4 = z.sqrt().sqrt();
            return Some((z4.cos() + z4.cosh()) / 2.0);
        }

        // FIXME: alpha = 0.5 is also known to be `exp(z^2) * erfc(-z)`, but we
        // cannot evaluate the complex `erfc` function at the moment.
    }

    if (beta - 2.0).abs() < eps {
        if (alpha - 1.0).abs() < eps {
            return Some((z.exp() - 1.0) / z);
        } else if (alpha - 2.0).abs() < eps {
            let z2 = z.sqrt();
            return Some(z2.sinh() / z2);
        }
    }

    None
}

// }}}
