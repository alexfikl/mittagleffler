// SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
// SPDX-License-Identifier: MIT

use num::complex::c64;
use rand::prelude::*;

mod reference_data;
use mittagleffler::{GarrappaMittagLeffler, MittagLeffler, MittagLefflerAlgorithm};
use reference_data::MATHEMATICA_RESULTS;

#[test]
fn test_param() {
    let ml = GarrappaMittagLeffler::new(None);
    assert!(ml.eps > f64::EPSILON);
    assert!(ml.fac >= 1.0);
    assert!(ml.p_eps > f64::EPSILON);
    assert!(ml.q_eps > f64::EPSILON);
    assert!(!ml.conservative_error_analysis);
}

#[test]
fn test_vs_exponential() {
    let mut rng = StdRng::from_seed([42; 32]);

    let (a, b) = (0.0, 10.0);
    let alpha = 1.0;
    let beta = 1.0;
    let eps = 5.0 * f64::EPSILON;
    let ml = GarrappaMittagLeffler::new(Some(eps));

    // test real only
    for _ in 0..512 {
        let z = c64(a + (b - a) * rng.random::<f64>(), 0.0);
        let result = ml.evaluate(z, alpha, beta).unwrap();
        let e_ref = z.exp();

        let error = (result - e_ref).norm();
        let rtol = 100.0 * eps * e_ref.norm();
        assert!(
            error < rtol,
            "Result {result} Reference {e_ref} Error {error:.8e} (rtol {rtol:.8e})",
        );
    }

    // test complex
    for _ in 0..512 {
        let z = c64(
            a + (b - a) * rng.random::<f64>(),
            a + (b - a) * rng.random::<f64>(),
        );
        let result = ml.evaluate(z, alpha, beta).unwrap();
        let e_ref = z.exp();

        let error = (result - e_ref).norm();
        let rtol = 100.0 * eps * e_ref.norm();
        assert!(
            error < rtol,
            "Result {result} Reference {e_ref} Error {error:.8e} (rtol {rtol:.8e})",
        );
    }
}

#[test]
fn test_vs_cosine() {
    let mut rng = StdRng::from_seed([42; 32]);

    let (a, b) = (0.0, 10.0);
    let alpha = 2.0;
    let beta = 1.0;
    let eps = 5.0 * f64::EPSILON;
    let ml = GarrappaMittagLeffler::new(Some(eps));

    // test real only
    for _ in 0..512 {
        let z = c64(a + (b - a) * rng.random::<f64>(), 0.0);
        let result = ml.evaluate(-z.powi(2), alpha, beta).unwrap();
        let e_ref = z.cos();

        let error = (result - e_ref).norm();
        let rtol = 100.0 * eps * e_ref.norm();
        assert!(
            error < rtol,
            "Result {result} Reference {e_ref} Error {error:.8e} (rtol {rtol:.8e})",
        );
    }

    // test complex
    for _ in 0..512 {
        let z = c64(
            a + (b - a) * rng.random::<f64>(),
            a + (b - a) * rng.random::<f64>(),
        );
        let result = ml.evaluate(-z.powi(2), alpha, beta).unwrap();
        let e_ref = z.cos();

        let error = (result - e_ref).norm();
        let rtol = 100.0 * eps * e_ref.norm();
        assert!(
            error < rtol,
            "Result {result} Reference {e_ref} Error {error:.8e} (rtol {rtol:.8e})",
        );
    }
}

#[test]
fn test_vs_hyperbolic_cosine() {
    let mut rng = StdRng::from_seed([42; 32]);

    let (a, b) = (0.0, 100.0);
    let alpha = 2.0;
    let beta = 1.0;
    let eps = 5.0 * f64::EPSILON;
    let ml = GarrappaMittagLeffler::new(Some(eps));

    // test real only
    for _ in 0..512 {
        let z = c64(a + (b - a) * rng.random::<f64>(), 0.0);
        let result = ml.evaluate(z, alpha, beta).unwrap();
        let e_ref = z.sqrt().cosh();

        let error = (result - e_ref).norm();
        let rtol = 100.0 * eps * e_ref.norm();
        assert!(
            error < rtol,
            "Result {result} Reference {e_ref} Error {error:.8e} (rtol {rtol:.8e})",
        );
    }

    // test complex
    for _ in 0..512 {
        let z = c64(
            a + (b - a) * rng.random::<f64>(),
            a + (b - a) * rng.random::<f64>(),
        );
        let result = ml.evaluate(z, alpha, beta).unwrap();
        let e_ref = z.sqrt().cosh();

        let error = (result - e_ref).norm();
        let rtol = 100.0 * eps * e_ref.norm();
        assert!(
            error < rtol,
            "Result {result} Reference {e_ref} Error {error:.8e} (rtol {rtol:.8e})",
        );
    }
}

#[test]
fn test_vs_exponential_inv() {
    let mut rng = StdRng::from_seed([42; 32]);

    let (a, b) = (0.0, 10.0);
    let alpha = 1.0;
    let beta = 2.0;
    let eps = 5.0 * f64::EPSILON;
    let ml = GarrappaMittagLeffler::new(Some(eps));

    // test real only
    for _ in 0..512 {
        let z = c64(a + (b - a) * rng.random::<f64>(), 0.0);
        let result = ml.evaluate(z, alpha, beta).unwrap();
        let e_ref = (z.exp() - 1.0) / z;

        let error = (result - e_ref).norm();
        let rtol = 100.0 * eps * e_ref.norm();
        assert!(
            error < rtol,
            "Result {result} Reference {e_ref} Error {error:.8e} (rtol {rtol:.8e})",
        );
    }

    // test complex
    for _ in 0..512 {
        let z = c64(
            a + (b - a) * rng.random::<f64>(),
            a + (b - a) * rng.random::<f64>(),
        );
        let result = ml.evaluate(z, alpha, beta).unwrap();
        let e_ref = (z.exp() - 1.0) / z;

        let error = (result - e_ref).norm();
        let rtol = 100.0 * eps * e_ref.norm();
        assert!(
            error < rtol,
            "Result {result} Reference {e_ref} Error {error:.8e} (rtol {rtol:.8e})",
        );
    }
}

#[test]
fn test_vs_hyperbolic_sine() {
    let mut rng = StdRng::from_seed([42; 32]);

    let (a, b) = (0.0, 10.0);
    let alpha = 2.0;
    let beta = 2.0;
    let eps = 5.0 * f64::EPSILON;
    let ml = GarrappaMittagLeffler::new(Some(eps));

    // test real only
    for _ in 0..512 {
        let z = c64(a + (b - a) * rng.random::<f64>(), 0.0);
        let result = ml.evaluate(z, alpha, beta).unwrap();
        let e_ref = z.sqrt().sinh() / z.sqrt();

        let error = (result - e_ref).norm();
        let rtol = 100.0 * eps * e_ref.norm();
        assert!(
            error < rtol,
            "Result {result} Reference {e_ref} Error {error:.8e} (rtol {rtol:.8e})",
        );
    }

    // test complex
    for _ in 0..512 {
        let z = c64(
            a + (b - a) * rng.random::<f64>(),
            a + (b - a) * rng.random::<f64>(),
        );
        let result = ml.evaluate(z, alpha, beta).unwrap();
        let e_ref = z.sqrt().sinh() / z.sqrt();

        let error = (result - e_ref).norm();
        let rtol = 100.0 * eps * e_ref.norm();
        assert!(
            error < rtol,
            "Result {result} Reference {e_ref} Error {error:.8e} (rtol {rtol:.8e})",
        );
    }
}

#[test]
fn test_vs_mathematica() {
    let eps = 1.0e-5;

    for result in MATHEMATICA_RESULTS {
        let alpha = result.alpha;
        let beta = result.beta;

        for (z, e_ref) in result.z.iter().zip(result.result.iter()) {
            let result = z.mittag_leffler(alpha, beta).unwrap();
            println!("z: {z}");

            // FIXME: All mathematica results seem to be accurate to 8-ish
            // digits. Not quite sure what the problem is..
            let error = (result - e_ref).norm();
            let rtol = 10.0 * eps * e_ref.norm();
            assert!(
                error < rtol,
                "Result {result} Reference {e_ref} Error {error:.8e} (rtol {rtol:.8e})",
            );
        }
    }
}
