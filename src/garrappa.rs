// SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
// SPDX-License-Identifier: MIT

use num::complex::{c64, Complex64};
use num::Num;

const MACHINE_EPSILON: f64 = f64::EPSILON;

pub struct MittagLeffler {
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

impl MittagLeffler {
    pub fn new() -> Self {
        MittagLeffler {
            eps: 5.0 * MACHINE_EPSILON,
            fac: 1.01,
            p_eps: 1.0e-14,
            q_eps: 1.0e-14,
            conservative_error_analysis: false,
        }
    }

    pub fn with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    pub fn evaluate(&self, z: Complex64, alpha: f64, beta: f64) -> Complex64 {
        if z.norm() < self.eps {
            return c64(1.0 / special::Gamma::gamma(beta), 0.0);
        }

        laplace_transform_inversion(
            1.0,
            z,
            alpha,
            beta,
            self.eps,
            self.fac,
            self.p_eps,
            self.q_eps,
            self.conservative_error_analysis,
        )
    }
}

// {{{ impl

fn argsort(data: &[f64]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..data.len()).collect();
    indices.sort_by(|&i, &j| data[j].partial_cmp(&data[i]).unwrap());
    indices
}

fn argmin(data: &[f64]) -> usize {
    data.iter()
        .enumerate()
        .max_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap())
        .map(|(j, _)| j)
        .unwrap()
}

fn reorder<T: Num + Clone>(data: Vec<T>, indices: &[usize]) -> Vec<T> {
    indices.iter().map(|&i| data[i].clone()).collect()
}

fn laplace_transform_inversion(
    t: f64,
    z: Complex64,
    alpha: f64,
    beta: f64,
    eps: f64,
    fac: f64,
    p_eps: f64,
    q_eps: f64,
    conservative_error_analysis: bool,
) -> Complex64 {
    let znorm = z.norm();

    // get precision constants
    let log_mach_eps: f64 = MACHINE_EPSILON.ln();
    let mut log_eps = eps.ln();
    let log_10 = 10.0_f64.ln();
    let d_log_eps = log_eps - log_mach_eps;

    // evaluate poles
    let pi = std::f64::consts::PI;
    let (_, theta) = z.to_polar();
    let kmin = (-alpha / 2.0 - theta / (2.0 * pi)).ceil() as u64;
    let kmax = (alpha / 2.0 - theta / (2.0 * pi)).floor() as u64;
    let mut s_star: Vec<Complex64> = (kmin..=kmax)
        .map(|k| znorm.powf(1.0 / alpha) * c64(0.0, (theta + 2.0 * pi * (k as f64)) / alpha).exp())
        .collect();

    // sort poles
    let mut phi_star: Vec<f64> = s_star
        .iter()
        .map(|&s| (s.re + s.norm()) / 2.0)
        .filter(|&phi| phi > eps)
        .collect();
    let s_star_index = argsort(&phi_star);
    s_star = reorder(s_star, &s_star_index);
    phi_star = reorder(phi_star, &s_star_index);

    // add back the origin
    s_star.insert(0, c64(0.0, 0.0));
    phi_star.insert(0, 0.0);
    phi_star.push(f64::INFINITY);
    let n_star = s_star.len();

    // evaluate the strength of the singularities
    let mut p = vec![0.0; n_star];
    p[0] = f64::max(0.0, -2.0 * (alpha - beta + 1.0));
    let mut q = vec![0.0; n_star];
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
                    t,
                    phi_star[j],
                    phi_star[j + 1],
                    p[j],
                    q[j],
                    log_eps,
                    log_mach_eps,
                    fac,
                    p_eps,
                    q_eps,
                    conservative_error_analysis,
                );
            } else {
                (mu[j], npoints[j], h[j]) = find_optional_unbounded_param(
                    t,
                    phi_star[j],
                    p[j],
                    log_eps,
                    log_mach_eps,
                    fac,
                    p_eps,
                );
            }
        }

        let n_min = npoints
            .iter()
            .fold(0.0, |max, &x| if x > max { x } else { max });
        if n_min > 200.0 {
            log_eps += log_10;
        } else {
            found_region = true;
        }

        if log_eps >= 0.0 {
            panic!("Failed to find admissible region");
        }
    }

    // select region that contains the minimum number of nodes
    let jmin = argmin(&npoints);
    let n_min = npoints[jmin] as i64;
    let mu_min = mu[jmin];
    let h_min = h[jmin];

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
    let fv: Vec<Complex64> = zk
        .iter()
        .zip(zd.iter())
        .map(|(&zk_i, &zd_i)| zk_i.powf(alpha - beta) / (zk_i.powf(alpha) - z) * zd_i)
        .collect();

    let sv: Complex64 = fv
        .iter()
        .zip(zk.iter())
        .map(|(&f_i, &zk_i)| f_i * (zk_i * t).exp())
        .sum();
    let integral = h_min * sv / c64(0.0, 2.0 * pi);

    // evaluate residues
    let residues: Complex64 = (jmin + 1..n_star)
        .map(|j| 1.0 / alpha * s_star[j].powf(1.0 - beta) * (s_star[j] * t).exp())
        .sum();

    residues + integral
}

fn find_optimal_bounded_param(
    t: f64,
    phi_star0: f64,
    phi_star1: f64,
    p: f64,
    q: f64,
    log_eps: f64,
    log_mach_eps: f64,
    fac: f64,
    p_eps: f64,
    q_eps: f64,
    conservative_error_analysis: bool,
) -> (f64, f64, f64) {
    (0.0, 0.0, 0.0)
}

fn find_optional_unbounded_param(
    t: f64,
    phi_star: f64,
    p: f64,
    log_eps: f64,
    log_mach_eps: f64,
    fac: f64,
    p_eps: f64,
) -> (f64, f64, f64) {
    (0.0, 0.0, 0.0)
}

// }}}

// {{{ tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mittag_leffler() {}
}

// }}}
