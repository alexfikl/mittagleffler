// SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
// SPDX-License-Identifier: CC0-1.0

use num::complex::Complex64;

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

#[pyclass]
pub struct GarrappaMittagLeffler {
    inner: mittagleffler::GarrappaMittagLeffler,
}

#[pymethods]
impl GarrappaMittagLeffler {
    #[new]
    #[pyo3(signature = (eps=None))]
    pub fn new(eps: Option<f64>) -> Self {
        GarrappaMittagLeffler {
            inner: mittagleffler::GarrappaMittagLeffler::new(eps),
        }
    }

    pub fn evaluate(&self, z: Complex64, alpha: f64, beta: f64) -> Option<Complex64> {
        self.inner.evaluate(z, alpha, beta)
    }

    #[setter]
    pub fn set_eps(&mut self, eps: f64) {
        self.inner.eps = eps;
    }

    #[getter]
    pub fn get_eps(&self) -> f64 {
        self.inner.eps
    }
}

#[pyfunction]
pub fn mittag_leffler<'py>(
    py: Python<'py>,
    z: PyReadonlyArray1<Complex64>,
    alpha: f64,
    beta: f64,
) -> Bound<'py, PyArray1<Complex64>> {
    let z = z.as_array();
    let z: Vec<Complex64> = z
        .iter()
        .map(
            |z_i| match mittagleffler::MittagLeffler::mittag_leffler(z_i, alpha, beta) {
                Some(value) => value,
                None => Complex64 {
                    re: f64::NAN,
                    im: f64::NAN,
                },
            },
        )
        .collect();

    z.into_pyarray(py)
}

#[pymodule]
#[pyo3(name = "_bindings")]
fn _bindings(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<GarrappaMittagLeffler>()?;
    m.add_function(wrap_pyfunction!(mittag_leffler, m)?)?;

    Ok(())
}
