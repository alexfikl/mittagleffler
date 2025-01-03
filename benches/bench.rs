// SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
// SPDX-License-Identifier: MIT

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use mittagleffler::{GarrappaMittagLeffler, MittagLefflerAlgorithm};
use num::complex::Complex64;

pub fn benchmark_vs_tolerance(c: &mut Criterion) {
    let mut group = c.benchmark_group("Tolerance");

    let alpha = 1.0;
    let beta = 1.0;
    let z = Complex64::new(1.0, 1.0);

    for n in 2..16 {
        let eps = 10.0_f64.powi(-n);
        let ml = GarrappaMittagLeffler::new(Some(eps));

        group.bench_with_input(BenchmarkId::new("ML", n), &n, |b, &_n| {
            b.iter(|| ml.evaluate(z, alpha, beta))
        });
    }

    group.finish()
}

pub fn benchmark_vs_alpha(c: &mut Criterion) {
    let mut group = c.benchmark_group("Alpha");
    let ml = GarrappaMittagLeffler::new(None);

    let n = 16;
    let (amin, amax) = (0.1, 7.0);
    let beta = 1.0;
    let z = Complex64::new(1.0, 1.0);

    for i in 0..n {
        let alpha = amin + (amax - amin) * (i as f64) / ((n - 1) as f64);
        group.bench_with_input(BenchmarkId::new("ML", i), &i, |b, &_i| {
            b.iter(|| ml.evaluate(z, alpha, beta))
        });
    }

    group.finish()
}

pub fn benchmark_vs_beta(c: &mut Criterion) {
    let mut group = c.benchmark_group("Beta");
    let ml = GarrappaMittagLeffler::new(None);

    let n = 16;
    let alpha = 1.0;
    let (bmin, bmax) = (0.1, 7.0);
    let z = Complex64::new(1.0, 1.0);

    for i in 0..n {
        let beta = bmin + (bmax - bmin) * (i as f64) / ((n - 1) as f64);
        group.bench_with_input(BenchmarkId::new("ML", i), &i, |b, &_i| {
            b.iter(|| ml.evaluate(z, alpha, beta))
        });
    }

    group.finish()
}

pub fn benchmark_vs_radial(c: &mut Criterion) {
    let mut group = c.benchmark_group("Radial");
    let ml = GarrappaMittagLeffler::new(None);

    let n = 16;
    let alpha = 0.5;
    let beta = 1.0;
    let (rmin, rmax) = (0.01, 12.0);

    for i in 0..n {
        let r = rmin + (rmax - rmin) * (i as f64) / ((n - 1) as f64);
        let z = Complex64::new(0.0, r);

        group.bench_with_input(BenchmarkId::new("ML", i), &i, |b, &_i| {
            b.iter(|| ml.evaluate(z, alpha, beta))
        });
    }

    group.finish()
}

criterion_group!(
    benches,
    benchmark_vs_tolerance,
    benchmark_vs_alpha,
    benchmark_vs_beta,
    benchmark_vs_radial
);
criterion_main!(benches);
