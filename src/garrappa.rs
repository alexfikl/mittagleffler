// SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
// SPDX-License-Identifier: MIT

use std::f64::consts::{LN_10, PI};
use std::fmt;

// ln(f64::EPSILON) — a fixed mathematical constant, never changes.
const LOG_MACH_EPS: f64 = -36.043_653_389_117_15;

use num::Float;
use num::complex::Complex64;
use smallvec::SmallVec;

use crate::algorithm::MittagLefflerAlgorithm;

// Inline capacities for SmallVec buffers.
//
// The number of poles is bounded by floor(alpha) * 2, so N_POLES = 16 covers
// alpha up to ~7.5 without a heap fallback. N_REGIONS is almost always 1-2.
const N_POLES: usize = 16;
const N_REGIONS: usize = 4;

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
///    Functions*, SIAM Journal on Numerical Analysis, Vol. 53, pp. 1350--1369, 2015,
///    DOI: [10.1137/140971191](https://doi.org/10.1137/140971191).
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

fn argmin<T: Float>(data: &[T]) -> Option<usize> {
    data.iter()
        .enumerate()
        .min_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(j, _)| j)
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
    let mut log_eps = ml.eps.ln();
    let d_log_eps = log_eps - LOG_MACH_EPS;

    // evaluate poles
    let theta = z.arg();
    let theta_over_2pi = theta / (2.0 * PI);
    let half_alpha = alpha / 2.0;
    let kmin = (-half_alpha - theta_over_2pi).ceil() as i64;
    let kmax = (half_alpha - theta_over_2pi).floor() as i64;

    // evaluate poles: build (phi, s) pairs, filter out phi <= eps, sort by phi,
    // then prepend the origin sentinel and append phi = +inf.
    let alpha_recip = alpha.recip();
    let mut poles: SmallVec<[(f64, Complex64); N_POLES]> = (kmin..=kmax)
        .filter_map(|k| {
            let s = znorm.powf(alpha_recip)
                * Complex64::new(0.0, (theta + 2.0 * PI * (k as f64)) / alpha).exp();
            let phi = (s.re + s.norm()) / 2.0;
            if phi > ml.eps { Some((phi, s)) } else { None }
        })
        .collect();
    poles.sort_unstable_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // s_star and phi_star: origin at index 0, poles follow, phi = +inf as sentinel at the end.
    let n_poles = poles.len();
    let mut s_star: SmallVec<[Complex64; N_POLES]> = SmallVec::with_capacity(n_poles + 1);
    let mut phi_star: SmallVec<[f64; N_POLES]> = SmallVec::with_capacity(n_poles + 2);
    s_star.push(Complex64::new(0.0, 0.0));
    phi_star.push(0.0);
    for (phi, s) in poles {
        s_star.push(s);
        phi_star.push(phi);
    }
    phi_star.push(f64::INFINITY);

    // evaluate the strength of the singularities
    // p[j] = 1.0 for all j > 0; p[0] = max(0, -2*(alpha - beta + 1))
    // q[j] = 1.0 for all j < n_star-1; q[n_star-1] = infinity
    let n_star = s_star.len();
    let p = |j: usize| -> f64 {
        if j == 0 {
            (-2.0 * (alpha - beta + 1.0)).max(0.0)
        } else {
            1.0
        }
    };
    let q = |j: usize| -> f64 { if j == n_star - 1 { f64::INFINITY } else { 1.0 } };

    // find admissible regions
    let region_index: SmallVec<[usize; N_REGIONS]> = phi_star
        .windows(2)
        .map(|x| x[0] < d_log_eps / t && x[0] < x[1])
        .enumerate()
        .filter_map(|(i, value)| if value { Some(i) } else { None })
        .collect();

    // evaluate parameters of the Laplace Transform inversion in each region
    let nregions = region_index.last().unwrap() + 1;
    let mut mu: SmallVec<[f64; N_REGIONS]> = smallvec::smallvec![f64::INFINITY; nregions];
    let mut npoints: SmallVec<[f64; N_REGIONS]> = smallvec::smallvec![f64::INFINITY; nregions];
    let mut h: SmallVec<[f64; N_REGIONS]> = smallvec::smallvec![f64::INFINITY; nregions];

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
                    p(j),
                    q(j),
                    log_eps,
                );
            } else {
                (mu[j], npoints[j], h[j]) =
                    find_optional_unbounded_param(ml, t, phi_star[j], p(j), log_eps);
            }
        }

        let n_min = npoints
            .iter()
            .fold(f64::INFINITY, |min, &x| if x < min { x } else { min });
        if n_min > 200.0 {
            log_eps += LN_10;
        } else {
            found_region = true;
        }

        if log_eps >= 0.0 {
            return None;
        }
    }

    // select region that contains the minimum number of nodes
    let jmin = argmin(&npoints).unwrap();
    let mu_min = mu[jmin];
    let n_min = npoints[jmin] as i64;
    let h_min = h[jmin];

    // evaluate inverse Laplace Transform integral
    //
    // NOTE: in the MATLAB code, the inner computation was originally equivalent to
    //
    //      k = -N:N;
    //      u = h*k;
    //      z = mu*(1i*u+1).^2;
    //      zd = -2*mu*u + 2*mu*1i;
    //      zexp = exp(z*t);
    //      F = z.^(alpha*gama-beta)./(z.^alpha - lambda).^gama.*zd;
    //      S = zexp.*F;
    //      Integral = h*sum(S)/2/pi/1i;
    //
    // The version below is faster: it avoids complex powf and other transcendental
    // functions + it precomputes a bunch of values that are constant. The compiler
    // can probably do some of that, but better safe than sorry.
    //
    // All quadrature nodes have the form zk = mu * (1 + i*h*k)^2, which means their
    // polar components reduce to exact closed forms - avoiding more costly
    // transcendental function evaluation. We have
    //
    //   |zk| = mu * (1 + (h*k)^2)
    //   ln|zk| = ln(mu) + ln(1 + (h*k)^2)
    //   arg(zk) = 2 * atan(h*k)
    //   exp(zk*t) = exp(mu*t) * exp(-mu*t*(h*k)^2) * cis(2*mu*t*h*k)
    //
    // plus some more polar decompositions below.
    //
    // We exploit k/−k symmetry: for a pair (k, -k) with k > 0:
    //   - hk_sq, ln1p_hk_sq, ln_r, r_ab, r_a, exp(-mu_t*hk_sq) are identical
    //   - theta(-k) = -theta(k), so sin_cos of angle just flips the sin sign
    // This halves the number of atan, ln_1p, and shared exp calls.
    let ln_mu = mu_min.ln();
    let alpha_minus_beta = alpha - beta;
    let exp_mu_t = (mu_min * t).exp();
    let mu_t = mu_min * t;
    let two_mu = 2.0 * mu_min;

    // Inline helper: compute the integrand contribution for a single node
    // given hk_i (= h*k), ln_r, theta, and the shared real exp factor.
    //
    // The real exp factor exp(-mu_t * hk_sq) * exp_mu_t is passed in as
    // `exp_real` to allow sharing between +k and -k.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn node_contrib(
        hk_i: f64,
        theta: f64,
        ln_r: f64,
        exp_real: f64,
        alpha: f64,
        alpha_minus_beta: f64,
        z: Complex64,
        mu_t: f64,
        two_mu: f64,
    ) -> Complex64 {
        // zk^(alpha − beta)
        let (s_ab, c_ab) = (alpha_minus_beta * theta).sin_cos();
        let r_ab = (alpha_minus_beta * ln_r).exp();
        let zk_alpha_beta = Complex64::new(r_ab * c_ab, r_ab * s_ab);

        // zk^alpha
        let (s_a, c_a) = (alpha * theta).sin_cos();
        let r_a = (alpha * ln_r).exp();
        let zk_alpha = Complex64::new(r_a * c_a, r_a * s_a);

        // zd = derivative factor: 2*mu*(i - hk)
        let zd = Complex64::new(-two_mu * hk_i, two_mu);
        let f = zk_alpha_beta / (zk_alpha - z) * zd;

        // exp(zk*t): only the cis part varies between +k and -k
        let (s_exp, c_exp) = (2.0 * mu_t * hk_i).sin_cos();
        let exp_zk_t = exp_real * Complex64::new(c_exp, s_exp);

        f * exp_zk_t
    }

    // k = 0 term
    let sv0 = node_contrib(
        0.0,
        0.0,
        ln_mu,
        exp_mu_t,
        alpha,
        alpha_minus_beta,
        z,
        mu_t,
        two_mu,
    );

    // k = 1..N pairs: each pair shares the expensive transcendentals
    let sv_pairs: Complex64 = (1..=n_min)
        .map(|k| {
            let hk_i = h_min * (k as f64);
            let hk_sq = hk_i * hk_i;

            // shared between +k and -k
            let ln_r = ln_mu + hk_sq.ln_1p();
            let theta = 2.0 * hk_i.atan(); // theta for +k; -k uses -theta
            let exp_real = exp_mu_t * (-mu_t * hk_sq).exp(); // shared real factor

            // +k and -k contributions
            node_contrib(
                hk_i,
                theta,
                ln_r,
                exp_real,
                alpha,
                alpha_minus_beta,
                z,
                mu_t,
                two_mu,
            ) + node_contrib(
                -hk_i,
                -theta,
                ln_r,
                exp_real,
                alpha,
                alpha_minus_beta,
                z,
                mu_t,
                two_mu,
            )
        })
        .sum();

    let sv = sv0 + sv_pairs;
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
    let f_max = (log_eps - LOG_MACH_EPS).exp();
    let thresh = 2.0 * ((log_eps - LOG_MACH_EPS) / t).sqrt();

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
    let mut sqrt_1_12a;

    loop {
        let phi = phibar_star * t;
        let log_eps_phi = log_eps / phi;

        n = (phi / PI * (1.0 - 3.0 * log_eps_phi / 2.0 + (1.0 - 2.0 * log_eps_phi).sqrt())).ceil();
        a = PI * n / phi;
        sqrt_1_12a = (1.0 + 12.0 * a).sqrt();
        mu = phibar_star_sq * (4.0 - a).abs() / (7.0 - sqrt_1_12a).abs();
        f_bar = ((phibar_star_sq - phi_star_sq) / mu).powf(-p);

        let found = (p < ml.p_eps) || (F_MIN < f_bar && f_bar < F_MAX);
        if found {
            break;
        }

        phibar_star_sq = F_TAR.powf(-p.recip()) * mu + phi_star_sq;
        phibar_star = phibar_star_sq.powi(2);
    }

    mu = mu.powi(2);
    h = (-3.0 * a - 2.0 + 2.0 * sqrt_1_12a) / (4.0 - a) / n;

    // adjust the integration parameters
    let thresh = (log_eps - LOG_MACH_EPS) / t;
    if mu > thresh {
        let q = if p.abs() < ml.p_eps {
            0.0
        } else {
            F_TAR.powf(-p.recip()) * mu.sqrt()
        };
        phibar_star = (q + phi_star.sqrt()).powi(2);

        if phibar_star < thresh {
            let w = (LOG_MACH_EPS / (LOG_MACH_EPS - log_eps)).sqrt();
            let u = (-phibar_star * t / LOG_MACH_EPS).sqrt();

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
