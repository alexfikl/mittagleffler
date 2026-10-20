// SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
// SPDX-License-Identifier: MIT

use num::complex::{Complex32, Complex64, c64};

use numpy::ndarray::Zip;
use numpy::{IntoPyArray, PyReadonlyArrayDyn};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyComplex;

use mittagleffler::{GarrappaMittagLeffler, MittagLefflerAlgorithm, mittag_leffler_special};

/// NOTE: below this many elements, the overhead of spawning parallel work
/// outweighs the gain, so arrays are evaluated serially.
const PARALLEL_THRESHOLD: usize = 32;

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

#[inline]
fn eval(ml: &GarrappaMittagLeffler, z: Complex64, alpha: f64, beta: f64) -> Complex64 {
    match mittag_leffler_special(z, alpha, beta).or_else(|| ml.evaluate(z, alpha, beta)) {
        Some(value) => value,
        None => Complex64 {
            re: f64::NAN,
            im: f64::NAN,
        },
    }
}

#[inline]
fn evaluate_array<'py, T, F>(
    py: Python<'py>,
    ary: PyReadonlyArrayDyn<'py, T>,
    alpha: f64,
    beta: f64,
    to_complex: F,
) -> PyResult<Py<PyAny>>
where
    T: numpy::Element + Copy,
    F: Fn(T) -> Complex64 + Send + Sync,
{
    let ml = GarrappaMittagLeffler::default();
    let view = ary.as_array();

    let result = if view.len() < PARALLEL_THRESHOLD || rayon::current_num_threads() <= 1 {
        view.map(|x| eval(&ml, to_complex(*x), alpha, beta))
    } else {
        // NOTE: the GIL is released while evaluating
        py.detach(|| Zip::from(view).par_map_collect(|x| eval(&ml, to_complex(*x), alpha, beta)))
    };

    Ok(result
        .into_pyarray(py)
        .into_pyobject(py)?
        .into_any()
        .unbind())
}

#[pyfunction]
pub fn mittag_leffler<'py>(
    py: Python<'py>,
    z: Bound<'py, PyAny>,
    alpha: f64,
    beta: f64,
) -> PyResult<Py<PyAny>> {
    let ml = GarrappaMittagLeffler::default();

    if let Ok(ary) = z.extract::<Complex64>() {
        let result = eval(&ml, ary, alpha, beta);
        return Ok(PyComplex::from_doubles(py, result.re, result.im).into());
    }

    if let Ok(ary) = z.extract::<PyReadonlyArrayDyn<f32>>() {
        return evaluate_array(py, ary, alpha, beta, |x| c64(x as f64, 0.0));
    }

    if let Ok(ary) = z.extract::<PyReadonlyArrayDyn<f64>>() {
        return evaluate_array(py, ary, alpha, beta, |x| c64(x, 0.0));
    }

    if let Ok(ary) = z.extract::<PyReadonlyArrayDyn<Complex32>>() {
        return evaluate_array(py, ary, alpha, beta, |x| c64(x.re as f64, x.im as f64));
    }

    if let Ok(ary) = z.extract::<PyReadonlyArrayDyn<Complex64>>() {
        return evaluate_array(py, ary, alpha, beta, |x| x);
    }

    if let Ok(ary) = z.extract::<PyReadonlyArrayDyn<i32>>() {
        return evaluate_array(py, ary, alpha, beta, |x| c64(x as f64, 0.0));
    }

    if let Ok(ary) = z.extract::<PyReadonlyArrayDyn<i64>>() {
        return evaluate_array(py, ary, alpha, beta, |x| c64(x as f64, 0.0));
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
