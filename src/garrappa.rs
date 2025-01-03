// SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
// SPDX-License-Identifier: MIT

use std::f64::consts::PI;
use std::fmt;

use num::complex::Complex64;
use num::{Float, Num};

use crate::algorithm::MittagLefflerAlgorithm;

/// Parameters for evaluating the Mittag-Leffler function.
///
/// This implements the algorithm described in
/// [Garrappa 2015](https://doi.org/10.1137/140971191). It is largely a direct
/// port of the [MATLAB implementation](https://www.mathworks.com/matlabcentral/fileexchange/48154-the-mittag-leffler-function). For a more thorough description
/// of the parameters see the paper.
///
/// ## References
///
/// 1. R. Garrappa, *Numerical Evaluation of Two and Three Parameter Mittag-Leffler
///     Functions*, SIAM Journal on Numerical Analysis, Vol. 53, pp. 1350--1369, 2015,
///     DOI: [10.1137/140971191](https://doi.org/10.1137/140971191).
#[derive(Clone, Debug)]
pub struct GarrappaMittagLeffler {
    /// Tolerance used to control the accuracy of evaluating the inverse Laplace
    /// transform used to compute the function. This should match the tolerance
    /// for evaluating the Mittag-Leffler function itself.
    pub eps: f64,
    /// Factor used to perturb integration regions.
    pub fac: f64,
    /// Tolerance used to compute integration regions.
    pub p_eps: f64,
    /// Tolerance used to compute integration regions.
    pub q_eps: f64,
    /// If true, a more conservative method is used to estimate the integration
    /// region. This should not be necessary for most evaluation.
    pub conservative_error_analysis: bool,
}

impl GarrappaMittagLeffler {
    pub fn new(eps: Option<f64>) -> Self {
        let mach_eps = f64::epsilon();
        let ml_eps = match eps {
            Some(value) => value,
            None => 5.0 * mach_eps,
        };

        GarrappaMittagLeffler {
            eps: ml_eps,
            fac: 1.01,
            p_eps: 100.0 * f64::epsilon(),
            q_eps: 100.0 * f64::epsilon(),
            conservative_error_analysis: false,
        }
    }

    pub fn with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }
}

impl MittagLefflerAlgorithm for GarrappaMittagLeffler {
    fn evaluate(&self, z: Complex64, alpha: f64, beta: f64) -> Option<Complex64> {
        if alpha < 0.0 {
            return None;
        }

        if z.norm() < self.eps {
            return Some(Complex64::new(special::Gamma::gamma(beta).recip(), 0.0));
        }

        laplace_transform_inversion(self, 1.0, z, alpha, beta)
    }
}

impl Default for GarrappaMittagLeffler {
    fn default() -> Self {
        Self::new(None)
    }
}

impl fmt::Display for GarrappaMittagLeffler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GarrappaMittagLeffler(eps={})", self.eps)
    }
}

// }}}

// {{{ impl

fn argsort<T: Float>(data: &[T]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..data.len()).collect();
    indices.sort_by(|&i, &j| {
        data[i]
            .partial_cmp(&data[j])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    indices
}

fn argmin<T: Float>(data: &[T]) -> Option<usize> {
    data.iter()
        .enumerate()
        .min_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(j, _)| j)
}

fn pick<T: Num + Copy>(data: &[T], indices: &[usize]) -> Vec<T> {
    indices.iter().map(|&i| data[i]).collect()
}

fn laplace_transform_inversion(
    ml: &GarrappaMittagLeffler,
    t: f64,
    z: Complex64,
    alpha: f64,
    beta: f64,
) -> Option<Complex64> {
    let znorm = z.norm();

    // get precision constants
    let log_mach_eps = f64::EPSILON.ln();
    let mut log_eps = ml.eps.ln();
    let log_10 = 10.0_f64.ln();
    let d_log_eps = log_eps - log_mach_eps;

    // evaluate poles
    let theta = z.arg();
    let kmin = (-alpha / 2.0 - theta / (2.0 * PI)).ceil() as i64;
    let kmax = (alpha / 2.0 - theta / (2.0 * PI)).floor() as i64;
    let mut s_star: Vec<Complex64> = (kmin..=kmax)
        .map(|k| {
            znorm.powf(alpha.recip())
                * Complex64::new(0.0, (theta + 2.0 * PI * (k as f64)) / alpha).exp()
        })
        .collect();

    // sort poles
    let mut phi_star: Vec<f64> = s_star.iter().map(|s| (s.re + s.norm()) / 2.0).collect();

    let mut phi_star_index = argsort(&phi_star);
    phi_star_index.retain(|&i| phi_star[i] > ml.eps);
    s_star = pick(&s_star, &phi_star_index);
    phi_star = pick(&phi_star, &phi_star_index);

    // add back the origin
    s_star.insert(0, Complex64::new(0.0, 0.0));
    phi_star.insert(0, 0.0);
    phi_star.push(f64::INFINITY);

    // evaluate the strength of the singularities
    let n_star = s_star.len();
    let mut p = vec![1.0; n_star];
    p[0] = (-2.0 * (alpha - beta + 1.0)).max(0.0);
    let mut q = vec![1.0; n_star];
    q[n_star - 1] = f64::INFINITY;

    // find admissible regions
    let region_index: Vec<usize> = phi_star
        .windows(2)
        .map(|x| x[0] < d_log_eps / t && x[0] < x[1])
        .enumerate()
        .filter_map(|(i, value)| if value { Some(i) } else { None })
        .collect();

    // evaluate parameters of the Laplace Transform inversion in each region
    let nregions = region_index.last().unwrap() + 1;
    let mut mu = vec![f64::INFINITY; nregions];
    let mut npoints = vec![f64::INFINITY; nregions];
    let mut h = vec![f64::INFINITY; nregions];

    let mut found_region = false;
    while !found_region {
        for j in &region_index {
            let j = *j;
            if j < n_star - 1 {
                (mu[j], npoints[j], h[j]) = find_optimal_bounded_param(
                    ml,
                    t,
                    phi_star[j],
                    phi_star[j + 1],
                    p[j],
                    q[j],
                    log_eps,
                );
            } else {
                (mu[j], npoints[j], h[j]) =
                    find_optional_unbounded_param(ml, t, phi_star[j], p[j], log_eps);
            }
        }

        let n_min = npoints
            .iter()
            .fold(f64::INFINITY, |min, &x| if x < min { x } else { min });
        if n_min > 200.0 {
            log_eps += log_10;
        } else {
            found_region = true;
        }

        if log_eps >= 0.0 {
            println!("Failed to find an admissible region");
            return None;
        }
    }

    // select region that contains the minimum number of nodes
    let jmin = argmin(&npoints).unwrap();
    let mu_min = mu[jmin];
    let n_min = npoints[jmin] as i64;
    let h_min = h[jmin];

    // evaluate inverse Laplace Transform integral
    let hk: Vec<f64> = (-n_min..=n_min).map(|k| h_min * (k as f64)).collect();
    let zk: Vec<Complex64> = hk
        .iter()
        .map(|&hk_i| mu_min * Complex64::new(1.0, hk_i).powi(2))
        .collect();
    let zd: Vec<Complex64> = hk
        .iter()
        .map(|&hk_i| Complex64::new(-2.0 * mu_min * hk_i, 2.0 * mu_min))
        .collect();

    let sv: Complex64 = zk
        .iter()
        .zip(zd.iter())
        .map(|(&zk_i, &zd_i)| zk_i.powf(alpha - beta) / (zk_i.powf(alpha) - z) * zd_i)
        .zip(zk.iter())
        .map(|(f_i, &zk_i)| f_i * (zk_i * t).exp())
        .sum();
    let integral = h_min * sv / Complex64::new(0.0, 2.0 * PI);

    // evaluate residues
    let residues: Complex64 = (jmin + 1..n_star)
        .map(|j| 1.0 / alpha * s_star[j].powf(1.0 - beta) * (s_star[j] * t).exp())
        .sum();

    Some(residues + integral)
}

fn find_optimal_bounded_param(
    ml: &GarrappaMittagLeffler,
    t: f64,
    phi_star0: f64,
    phi_star1: f64,
    p: f64,
    q: f64,
    log_eps: f64,
) -> (f64, f64, f64) {
    // set maximum value for fbar (the ratio of the tolerance to the machine tolerance)
    let log_mach_eps = f64::EPSILON.ln();
    let f_max = (log_eps - log_mach_eps).exp();
    let thresh = 2.0 * ((log_eps - log_mach_eps) / t).sqrt();

    // starting values
    let phi_star0_sq = phi_star0.sqrt();
    let phi_star1_sq = phi_star1.sqrt().min(thresh - phi_star0_sq);

    // determine phibar and admissible region
    let mut f_bar = f_max;
    let mut phibar_star0_sq = phi_star0_sq;
    let mut phibar_star1_sq = phi_star1_sq;
    let mut adm_region = false;

    if p < ml.p_eps {
        if q < ml.q_eps {
            phibar_star0_sq = phi_star0_sq;
            phibar_star1_sq = phi_star1_sq;
            adm_region = true;
        } else {
            phibar_star0_sq = phi_star0_sq;
            let f_min = if phi_star0_sq > 0.0 {
                ml.fac * (phi_star0_sq / (phi_star1 - phi_star0_sq)).powf(q)
            } else {
                ml.fac
            };

            if f_min < f_max {
                f_bar = f_min + f_min / f_max * (f_max - f_min);
                let fq = f_bar.powf(-q.recip());
                phibar_star1_sq = (2.0 * phi_star1_sq - fq * phi_star0_sq) / (2.0 + fq);
                adm_region = true;
            }
        }
    } else if q < ml.q_eps {
        phibar_star1_sq = phi_star1_sq;
        let f_min = ml.fac * (phi_star1_sq / (phi_star1_sq - phi_star0_sq)).powf(p);

        if f_min < f_max {
            f_bar = f_min + f_min / f_max * (f_max - f_min);
            let fp = f_bar.powf(-p.recip());
            phibar_star0_sq = (2.0 * phi_star0_sq - fp * phi_star1_sq) / (2.0 - fp);
            adm_region = false;
        }
    } else {
        let mut f_min =
            ml.fac * (phi_star1_sq + phi_star0_sq) / (phi_star1_sq - phi_star0_sq).powf(p.max(q));

        if f_min < f_max {
            f_min = f_min.max(1.5);
            f_bar = f_min + f_min / f_max * (f_max - f_min);
            let fp = f_bar.powf(-p.recip());
            let fq = f_bar.powf(-q.recip());
            let w = if ml.conservative_error_analysis {
                -2.0 * phi_star1 * t / (log_eps - phi_star1 * t)
            } else {
                -phi_star1 * t / log_eps
            };

            let den = 2.0 + w - (1.0 + w) * fp + fq;
            phibar_star0_sq = ((2.0 + w + fq) * phi_star0_sq + fp * phi_star1_sq) / den;
            phibar_star1_sq =
                (-(1.0 + w) * fq * phi_star0_sq + (2.0 + w - (1.0 + w) * fp) * phi_star1_sq) / den;
            adm_region = true;
        }
    }

    if adm_region {
        let log_eps_bar = log_eps - f_bar.ln();
        let w = if ml.conservative_error_analysis {
            -2.0 * t * phibar_star1_sq.powi(2) / (log_eps_bar - phibar_star1_sq.powi(2) * t)
        } else {
            -t * phibar_star1_sq.powi(2) / log_eps_bar
        };

        let mu = (((1.0 + w) * phibar_star0_sq + phibar_star1_sq) / (2.0 + w)).powi(2);
        let h = -2.0 * PI / log_eps_bar * (phibar_star1_sq - phibar_star0_sq)
            / ((1.0 + w) * phibar_star0_sq + phibar_star1_sq);
        let npoints = ((1.0 - log_eps_bar / t / mu).sqrt() / h).ceil();

        (mu, npoints, h)
    } else {
        (0.0, f64::INFINITY, 0.0)
    }
}

fn find_optional_unbounded_param(
    ml: &GarrappaMittagLeffler,
    t: f64,
    phi_star: f64,
    p: f64,
    log_eps: f64,
) -> (f64, f64, f64) {
    const F_MIN: f64 = 1.0;
    const F_MAX: f64 = 10.0;
    const F_TAR: f64 = 5.0;

    let log_mach_eps = f64::EPSILON.ln();
    let phi_star_sq = phi_star.sqrt();
    let mut phibar_star = if phi_star > 0.0 {
        ml.fac * phi_star
    } else {
        0.01
    };
    let mut phibar_star_sq = phibar_star.sqrt();

    // search for fbar in [f_min, f_max]
    let mut a;
    let mut mu;
    let mut n;
    let mut h;
    let mut f_bar;

    loop {
        let phi = phibar_star * t;
        let log_eps_phi = log_eps / phi;

        n = (phi / PI * (1.0 - 3.0 * log_eps_phi / 2.0 + (1.0 - 2.0 * log_eps_phi).sqrt())).ceil();
        a = PI * n / phi;
        mu = phibar_star_sq * (4.0 - a).abs() / (7.0 - (1.0 + 12.0 * a).sqrt()).abs();
        f_bar = ((phibar_star_sq - phi_star_sq) / mu).powf(-p);

        let found = (p < ml.p_eps) || (F_MIN < f_bar && f_bar < F_MAX);
        if found {
            break;
        }

        phibar_star_sq = F_TAR.powf(-p.recip()) * mu + phi_star_sq;
        phibar_star = phibar_star_sq.powi(2);
    }

    mu = mu.powi(2);
    h = (-3.0 * a - 2.0 + 2.0 * (1.0 + 12.0 * a).sqrt()) / (4.0 - a) / n;

    // adjust the integration parameters
    let thresh = (log_eps - log_mach_eps) / t;
    if mu > thresh {
        let q = if p.abs() < ml.p_eps {
            0.0
        } else {
            F_TAR.powf(-p.recip()) * mu.sqrt()
        };
        phibar_star = (q + phi_star.sqrt()).powi(2);

        if phibar_star < thresh {
            let w = (log_mach_eps / (log_mach_eps - log_eps)).sqrt();
            let u = (-phibar_star * t / log_mach_eps).sqrt();

            mu = thresh;
            n = (w * log_eps / (2.0 * PI * (u * w - 1.0))).ceil();
            h = w / n;
        } else {
            n = f64::INFINITY;
            h = 0.0;
        }
    }

    (mu, n, h)
}

// }}}
