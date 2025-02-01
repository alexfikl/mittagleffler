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
    let mut rng = rand::rng();

    let (a, b) = (0.0, 10.0);
    let alpha = 1.0;
    let beta = 1.0;
    let eps = 5.0 * f64::EPSILON;
    let ml = GarrappaMittagLeffler::new(Some(eps));

    // test real only
    for _ in 0..512 {
        let z = c64(a + (b - a) * rng.random::<f64>(), 0.0);
        let result = ml.evaluate(z, alpha, beta).unwrap();
        let reference = z.exp();

        let error = (result - reference).norm();
        let rtol = 100.0 * eps * reference.norm();
        assert!(
            error < rtol,
            "Result {} Reference {} Error {:.8e} (rtol {:.8e})",
            result,
            reference,
            error,
            rtol
        );
    }

    // test complex
    for _ in 0..512 {
        let z = c64(
            a + (b - a) * rng.random::<f64>(),
            a + (b - a) * rng.random::<f64>(),
        );
        let result = ml.evaluate(z, alpha, beta).unwrap();
        let reference = z.exp();

        let error = (result - reference).norm();
        let rtol = 100.0 * eps * reference.norm();
        assert!(
            error < rtol,
            "Result {} Reference {} Error {:.8e} (rtol {:.8e})",
            result,
            reference,
            error,
            rtol
        );
    }
}

#[test]
fn test_vs_cosine() {
    let mut rng = rand::rng();

    let (a, b) = (0.0, 10.0);
    let alpha = 2.0;
    let beta = 1.0;
    let eps = 5.0 * f64::EPSILON;
    let ml = GarrappaMittagLeffler::new(Some(eps));

    // test real only
    for _ in 0..512 {
        let z = c64(a + (b - a) * rng.random::<f64>(), 0.0);
        let result = ml.evaluate(-z.powi(2), alpha, beta).unwrap();
        let reference = z.cos();

        let error = (result - reference).norm();
        let rtol = 100.0 * eps * reference.norm();
        assert!(
            error < rtol,
            "Result {} Reference {} Error {:.8e} (rtol {:.8e})",
            result,
            reference,
            error,
            rtol
        );
    }

    // test complex
    for _ in 0..512 {
        let z = c64(
            a + (b - a) * rng.random::<f64>(),
            a + (b - a) * rng.random::<f64>(),
        );
        let result = ml.evaluate(-z.powi(2), alpha, beta).unwrap();
        let reference = z.cos();

        let error = (result - reference).norm();
        let rtol = 100.0 * eps * reference.norm();
        assert!(
            error < rtol,
            "Result {} Reference {} Error {:.8e} (rtol {:.8e})",
            result,
            reference,
            error,
            rtol
        );
    }
}

#[test]
fn test_vs_hyperbolic_cosine() {
    let mut rng = rand::rng();

    let (a, b) = (0.0, 100.0);
    let alpha = 2.0;
    let beta = 1.0;
    let eps = 5.0 * f64::EPSILON;
    let ml = GarrappaMittagLeffler::new(Some(eps));

    // test real only
    for _ in 0..512 {
        let z = c64(a + (b - a) * rng.random::<f64>(), 0.0);
        let result = ml.evaluate(z, alpha, beta).unwrap();
        let reference = z.sqrt().cosh();

        let error = (result - reference).norm();
        let rtol = 100.0 * eps * reference.norm();
        assert!(
            error < rtol,
            "Result {} Reference {} Error {:.8e} (rtol {:.8e})",
            result,
            reference,
            error,
            rtol
        );
    }

    // test complex
    for _ in 0..512 {
        let z = c64(
            a + (b - a) * rng.random::<f64>(),
            a + (b - a) * rng.random::<f64>(),
        );
        let result = ml.evaluate(z, alpha, beta).unwrap();
        let reference = z.sqrt().cosh();

        let error = (result - reference).norm();
        let rtol = 100.0 * eps * reference.norm();
        assert!(
            error < rtol,
            "Result {} Reference {} Error {:.8e} (rtol {:.8e})",
            result,
            reference,
            error,
            rtol
        );
    }
}

#[test]
fn test_vs_exponential_inv() {
    let mut rng = rand::rng();

    let (a, b) = (0.0, 10.0);
    let alpha = 1.0;
    let beta = 2.0;
    let eps = 5.0 * f64::EPSILON;
    let ml = GarrappaMittagLeffler::new(Some(eps));

    // test real only
    for _ in 0..512 {
        let z = c64(a + (b - a) * rng.random::<f64>(), 0.0);
        let result = ml.evaluate(z, alpha, beta).unwrap();
        let reference = (z.exp() - 1.0) / z;

        let error = (result - reference).norm();
        let rtol = 100.0 * eps * reference.norm();
        assert!(
            error < rtol,
            "Result {} Reference {} Error {:.8e} (rtol {:.8e})",
            result,
            reference,
            error,
            rtol
        );
    }

    // test complex
    for _ in 0..512 {
        let z = c64(
            a + (b - a) * rng.random::<f64>(),
            a + (b - a) * rng.random::<f64>(),
        );
        let result = ml.evaluate(z, alpha, beta).unwrap();
        let reference = (z.exp() - 1.0) / z;

        let error = (result - reference).norm();
        let rtol = 100.0 * eps * reference.norm();
        assert!(
            error < rtol,
            "Result {} Reference {} Error {:.8e} (rtol {:.8e})",
            result,
            reference,
            error,
            rtol
        );
    }
}

#[test]
fn test_vs_hyperbolic_sine() {
    let mut rng = rand::rng();

    let (a, b) = (0.0, 10.0);
    let alpha = 2.0;
    let beta = 2.0;
    let eps = 5.0 * f64::EPSILON;
    let ml = GarrappaMittagLeffler::new(Some(eps));

    // test real only
    for _ in 0..512 {
        let z = c64(a + (b - a) * rng.random::<f64>(), 0.0);
        let result = ml.evaluate(z, alpha, beta).unwrap();
        let reference = z.sqrt().sinh() / z.sqrt();

        let error = (result - reference).norm();
        let rtol = 100.0 * eps * reference.norm();
        assert!(
            error < rtol,
            "Result {} Reference {} Error {:.8e} (rtol {:.8e})",
            result,
            reference,
            error,
            rtol
        );
    }

    // test complex
    for _ in 0..512 {
        let z = c64(
            a + (b - a) * rng.random::<f64>(),
            a + (b - a) * rng.random::<f64>(),
        );
        let result = ml.evaluate(z, alpha, beta).unwrap();
        let reference = z.sqrt().sinh() / z.sqrt();

        let error = (result - reference).norm();
        let rtol = 100.0 * eps * reference.norm();
        assert!(
            error < rtol,
            "Result {} Reference {} Error {:.8e} (rtol {:.8e})",
            result,
            reference,
            error,
            rtol
        );
    }
}

#[test]
fn test_vs_mathematica() {
    let eps = 1.0e-5;

    for result in MATHEMATICA_RESULTS {
        let alpha = result.alpha;
        let beta = result.beta;

        for (z, reference) in result.z.iter().zip(result.result.iter()) {
            let result = z.mittag_leffler(alpha, beta).unwrap();
            println!("z: {}", z);

            // FIXME: All mathematica results seem to be accurate to 8-ish
            // digits. Not quite sure what the problem is..
            let error = (result - reference).norm();
            let rtol = 10.0 * eps * reference.norm();
            assert!(
                error < rtol,
                "Result {} Reference {} Error {:.8e} (rtol {:.8e})",
                result,
                reference,
                error,
                rtol
            );
        }
    }
}
