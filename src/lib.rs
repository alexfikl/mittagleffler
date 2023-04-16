use num_complex::Complex64;
use pyo3::prelude::*;
use pyo3::types::PyComplex;

mod ml_series;
use ml_series::ml_series;

#[derive(Clone)]
#[pyclass]
enum MittagLefflerAlgorithm {
    Series = 0,
    Diethelm = 1,
    Garrappa = 2,
    Ortigueira = 3,
}

#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[pyfunction]
fn ml(
    _py: Python,
    z_py: PyObject,
    alpha_py: PyObject,
    beta_py: PyObject,
    alg: MittagLefflerAlgorithm,
) -> PyResult<&PyComplex> {
    let z: Complex64 = z_py.extract(_py)?;
    let alpha: f64 = alpha_py.extract(_py)?;
    let beta: f64 = beta_py.extract(_py)?;

    let r = match alg {
        MittagLefflerAlgorithm::Series => ml_series(z, alpha, beta),
        MittagLefflerAlgorithm::Diethelm => Complex64::new(0.0, 0.0),
        MittagLefflerAlgorithm::Garrappa => Complex64::new(0.0, 0.0),
        MittagLefflerAlgorithm::Ortigueira => Complex64::new(0.0, 0.0),
    };

    Ok(PyComplex::from_doubles(_py, r.re, r.im))
}

#[pymodule]
fn _mittagleffler(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<MittagLefflerAlgorithm>()?;

    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(ml, m)?)?;

    Ok(())
}
