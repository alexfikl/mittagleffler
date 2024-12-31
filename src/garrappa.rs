// SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
// SPDX-License-Identifier: MIT

use std::f64::consts::PI;

use num::complex::{c64, Complex64};
use num::{Float, Num};

// {{{ trait

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
        let ml = MittagLefflerParam::new(None);
        ml.evaluate(c64(*self, 0.0), alpha, beta)
    }
}

impl MittagLeffler for Complex64 {
    fn mittag_leffler(&self, alpha: f64, beta: f64) -> Option<Complex64> {
        let ml = MittagLefflerParam::new(None);
        ml.evaluate(*self, alpha, beta)
    }
}

// }}}

// {{{ MittagLeffler

/// Parameters for the Mittag-Leffler function evaluation.
///
/// This implements the algorithm described in [1]. It is largely a direct port
/// of the MATLAB implementation at [2].
///
/// [1] R. Garrappa, *Numerical Evaluation of Two and Three Parameter Mittag-Leffler
///     Functions*, SIAM Journal on Numerical Analysis, Vol. 53, pp. 1350--1369, 2015,
///     DOI: [10.1137/140971191](https://doi.org/10.1137/140971191).
/// [2] https://www.mathworks.com/matlabcentral/fileexchange/48154-the-mittag-leffler-function
pub struct MittagLefflerParam {
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

    /// Log of epsilon.
    log_eps: f64,
    /// Log of machine epsilon for *T*.
    log_mach_eps: f64,
}

impl MittagLefflerParam {
    pub fn new(eps: Option<f64>) -> Self {
        let mach_eps = f64::epsilon();
        let ml_eps = match eps {
            Some(value) => value,
            None => 5.0 * mach_eps,
        };

        MittagLefflerParam {
            eps: ml_eps,
            fac: 1.01,
            p_eps: 100.0 * f64::epsilon(),
            q_eps: 100.0 * f64::epsilon(),
            conservative_error_analysis: false,

            log_eps: ml_eps.ln(),
            log_mach_eps: mach_eps.ln(),
        }
    }

    pub fn with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self.log_eps = eps.ln();
        self
    }

    pub fn evaluate(&self, z: Complex64, alpha: f64, beta: f64) -> Option<Complex64> {
        if z.norm() < self.eps {
            return Some(Complex64 {
                re: special::Gamma::gamma(beta).recip(),
                im: 0.0,
            });
        }

        laplace_transform_inversion(self, 1.0, z, alpha, beta)
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
    ml: &MittagLefflerParam,
    t: f64,
    z: Complex64,
    alpha: f64,
    beta: f64,
) -> Option<Complex64> {
    let znorm = z.norm();

    // get precision constants
    let log_mach_eps = ml.log_mach_eps;
    let mut log_eps = ml.log_eps;
    let log_10 = 10.0_f64.ln();
    let d_log_eps = log_eps - log_mach_eps;

    // evaluate poles
    let theta = z.arg();
    let kmin = (-alpha / 2.0 - theta / (2.0 * PI)).ceil() as i64;
    let kmax = (alpha / 2.0 - theta / (2.0 * PI)).floor() as i64;
    let mut s_star: Vec<Complex64> = (kmin..=kmax)
        .map(|k| {
            znorm.powf(alpha.recip()) * c64(0.0, (theta + 2.0 * PI * (k as f64)) / alpha).exp()
        })
        .collect();
    // println!("s_star: {:?}", s_star);

    // sort poles
    let mut phi_star: Vec<f64> = s_star.iter().map(|s| (s.re + s.norm()) / 2.0).collect();
    // println!("phi_star: {:?}", phi_star);

    let mut phi_star_index = argsort(&phi_star);
    phi_star_index.retain(|&i| phi_star[i] > ml.eps);
    // println!("phi_star_index: {:?}", phi_star_index);
    s_star = pick(&s_star, &phi_star_index);
    // println!("s_star: {:?}", s_star);
    phi_star = pick(&phi_star, &phi_star_index);
    // println!("phi_star: {:?}", phi_star);

    // add back the origin
    s_star.insert(0, c64(0.0, 0.0));
    phi_star.insert(0, 0.0);
    phi_star.push(f64::INFINITY);
    // println!("s_star: {:?}", s_star);
    // println!("phi_star: {:?}", phi_star);

    // evaluate the strength of the singularities
    let n_star = s_star.len();
    let mut p = vec![1.0; n_star];
    p[0] = (-2.0 * (alpha - beta + 1.0)).max(0.0);
    let mut q = vec![1.0; n_star];
    q[n_star - 1] = f64::INFINITY;
    // println!("p: {:?}", p);
    // println!("q: {:?}", q);

    // find admissible regions
    let region_index: Vec<usize> = phi_star
        .windows(2)
        .map(|x| x[0] < d_log_eps / t && x[0] < x[1])
        .enumerate()
        .filter_map(|(i, value)| if value { Some(i) } else { None })
        .collect();
    // println!("region_index: {:?}", region_index);

    // evaluate parameters of the Laplace Transform inversion in each region
    let nregions = region_index.last().unwrap() + 1;
    let mut mu = vec![f64::INFINITY; nregions];
    let mut npoints = vec![f64::INFINITY; nregions];
    let mut h = vec![f64::INFINITY; nregions];
    // println!("nregions {}", nregions);
    // println!("mu {:?}", mu);

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
        // println!("mu {:?} N {:?} h {:?}", mu, npoints, h);

        let n_min = npoints
            .iter()
            .fold(f64::INFINITY, |min, &x| if x < min { x } else { min });
        // println!("n_min {}", n_min);
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
    // println!(
    //     "jmin {} mu_min {} n_min {} h_min {}",
    //     jmin, mu_min, n_min, h_min
    // );

    // evaluate inverse Laplace Transform integral
    let hk: Vec<f64> = (-n_min..=n_min).map(|k| h_min * (k as f64)).collect();
    let zk: Vec<Complex64> = hk
        .iter()
        .map(|&hk_i| mu_min * c64(1.0, hk_i).powi(2))
        .collect();
    let zd: Vec<Complex64> = hk
        .iter()
        .map(|&hk_i| c64(-2.0 * mu_min * hk_i, 2.0 * mu_min))
        .collect();

    let sv: Complex64 = zk
        .iter()
        .zip(zd.iter())
        .map(|(&zk_i, &zd_i)| zk_i.powf(alpha - beta) / (zk_i.powf(alpha) - z) * zd_i)
        .zip(zk.iter())
        .map(|(f_i, &zk_i)| f_i * (zk_i * t).exp())
        .sum();
    let integral = h_min * sv / c64(0.0, 2.0 * PI);

    // evaluate residues
    let residues: Complex64 = (jmin + 1..n_star)
        .map(|j| 1.0 / alpha * s_star[j].powf(1.0 - beta) * (s_star[j] * t).exp())
        .sum();

    Some(residues + integral)
}

fn find_optimal_bounded_param(
    ml: &MittagLefflerParam,
    t: f64,
    phi_star0: f64,
    phi_star1: f64,
    p: f64,
    q: f64,
    log_eps: f64,
) -> (f64, f64, f64) {
    // set maximum value for fbar (the ratio of the tolerance to the machine tolerance)
    let f_max = (log_eps - ml.log_mach_eps).exp();
    let thresh = 2.0 * ((log_eps - ml.log_mach_eps) / t).sqrt();

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
    ml: &MittagLefflerParam,
    t: f64,
    phi_star: f64,
    p: f64,
    log_eps: f64,
) -> (f64, f64, f64) {
    const F_MIN: f64 = 1.0;
    const F_MAX: f64 = 10.0;
    const F_TAR: f64 = 5.0;

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
    let thresh = (log_eps - ml.log_mach_eps) / t;
    if mu > thresh {
        let q = if p.abs() < ml.p_eps {
            0.0
        } else {
            F_TAR.powf(-p.recip()) * mu.sqrt()
        };
        phibar_star = (q + phi_star.sqrt()).powi(2);

        if phibar_star < thresh {
            let w = (ml.log_mach_eps / (ml.log_mach_eps - log_eps)).sqrt();
            let u = (-phibar_star * t / ml.log_mach_eps).sqrt();

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
