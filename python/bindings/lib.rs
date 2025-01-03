// SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
// SPDX-License-Identifier: MIT

use num::complex::{c64, Complex32, Complex64};

use numpy::{IntoPyArray, PyReadonlyArrayDyn};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyComplex;

use mittagleffler::{GarrappaMittagLeffler, MittagLeffler, MittagLefflerAlgorithm};

#[pyclass]
#[pyo3(name = "GarrappaMittagLeffler")]
pub struct PyGarrappaMittagLeffler {
    inner: GarrappaMittagLeffler,
}

#[pymethods]
impl PyGarrappaMittagLeffler {
    #[new]
    #[pyo3(signature = (*, eps=None))]
    pub fn new(eps: Option<f64>) -> Self {
        PyGarrappaMittagLeffler {
            inner: GarrappaMittagLeffler::new(eps),
        }
    }

    pub fn evaluate(&self, z: Complex64, alpha: f64, beta: f64) -> Option<Complex64> {
        self.inner.evaluate(z, alpha, beta)
    }

    #[setter(eps)]
    pub fn set_eps(&mut self, eps: f64) {
        self.inner.eps = eps;
    }

    #[getter(eps)]
    pub fn get_eps(&self) -> f64 {
        self.inner.eps
    }
}

fn mittag_leffler_always_c64(z: &Complex64, alpha: f64, beta: f64) -> Complex64 {
    match z.mittag_leffler(alpha, beta) {
        Some(value) => value,
        None => Complex64 {
            re: f64::NAN,
            im: f64::NAN,
        },
    }
}

#[pyfunction]
pub fn mittag_leffler<'py>(
    py: Python<'py>,
    z: Bound<'py, PyAny>,
    alpha: f64,
    beta: f64,
) -> PyResult<PyObject> {
    if let Ok(ary) = z.extract::<Complex64>() {
        let result = mittag_leffler_always_c64(&ary, alpha, beta);
        return Ok(PyComplex::from_doubles(py, result.re, result.im).into());
    }

    if let Ok(ary) = z.extract::<PyReadonlyArrayDyn<f32>>() {
        let ary = ary
            .as_array()
            .map(|x| mittag_leffler_always_c64(&c64(*x as f64, 0.0), alpha, beta));
        #[allow(deprecated)]
        return Ok(ary.into_pyarray(py).into_py(py));
    }

    if let Ok(ary) = z.extract::<PyReadonlyArrayDyn<f64>>() {
        let ary = ary
            .as_array()
            .map(|x| mittag_leffler_always_c64(&c64(*x, 0.0), alpha, beta));
        #[allow(deprecated)]
        return Ok(ary.into_pyarray(py).into_py(py));
    }

    if let Ok(ary) = z.extract::<PyReadonlyArrayDyn<Complex32>>() {
        let ary = ary
            .as_array()
            .map(|x| mittag_leffler_always_c64(&c64(x.re, x.im), alpha, beta));
        #[allow(deprecated)]
        return Ok(ary.into_pyarray(py).into_py(py));
    }

    if let Ok(ary) = z.extract::<PyReadonlyArrayDyn<Complex64>>() {
        let ary = ary
            .as_array()
            .map(|x| mittag_leffler_always_c64(x, alpha, beta));
        #[allow(deprecated)]
        return Ok(ary.into_pyarray(py).into_py(py));
    }

    if let Ok(ary) = z.extract::<PyReadonlyArrayDyn<i32>>() {
        let ary = ary
            .as_array()
            .map(|x| mittag_leffler_always_c64(&c64(*x as f64, 0.0), alpha, beta));
        #[allow(deprecated)]
        return Ok(ary.into_pyarray(py).into_py(py));
    }

    if let Ok(ary) = z.extract::<PyReadonlyArrayDyn<i64>>() {
        let ary = ary
            .as_array()
            .map(|x| mittag_leffler_always_c64(&c64(*x as f64, 0.0), alpha, beta));
        #[allow(deprecated)]
        return Ok(ary.into_pyarray(py).into_py(py));
    }

    Err(PyTypeError::new_err(format!(
        "Input 'z' has unsupported type {}",
        z.get_type()
    )))
}

#[pymodule]
#[pyo3(name = "_bindings")]
fn _bindings(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGarrappaMittagLeffler>()?;
    m.add_function(wrap_pyfunction!(mittag_leffler, m)?)?;

    Ok(())
}
