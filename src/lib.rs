use pyo3::prelude::*;
use num::complex::Complex;

use ml_series::ml_series;

mod ml_series;

#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[pyfunction]
fn ml(z: f64, alpha: f64, beta: f64) -> PyResult<usize> {
    let _r = ml_series(Complex::new(z, z), alpha, beta);
    Ok(0)
}

#[pymodule]
fn _mittagleffler(_py: Python, m: &PyModule) -> PyResult<()> {
    ml(0.0, 0.0, 0.0)?;
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}
