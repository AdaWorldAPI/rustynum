// bindings/python/src/lib.rs

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

mod array_f32;
mod array_f64;
mod array_u8;
mod functions;

use array_f32::PyNumArrayF32;
use array_f64::PyNumArrayF64;
use array_u8::PyNumArrayU8;

use functions::*;

#[pymodule]
fn _rustynum(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyNumArrayF32>()?;
    m.add_class::<PyNumArrayF64>()?;
    m.add_class::<PyNumArrayU8>()?;

    m.add_function(wrap_pyfunction!(zeros_f32, m)?)?;
    m.add_function(wrap_pyfunction!(ones_f32, m)?)?;
    m.add_function(wrap_pyfunction!(matmul_f32, m)?)?;
    m.add_function(wrap_pyfunction!(dot_f32, m)?)?;
    m.add_function(wrap_pyfunction!(arange_f32, m)?)?;
    m.add_function(wrap_pyfunction!(linspace_f32, m)?)?;
    m.add_function(wrap_pyfunction!(mean_f32, m)?)?;
    m.add_function(wrap_pyfunction!(median_f32, m)?)?;
    m.add_function(wrap_pyfunction!(min_f32, m)?)?;
    m.add_function(wrap_pyfunction!(min_axis_f32, m)?)?;
    m.add_function(wrap_pyfunction!(max_f32, m)?)?;
    m.add_function(wrap_pyfunction!(max_axis_f32, m)?)?;
    m.add_function(wrap_pyfunction!(exp_f32, m)?)?;
    m.add_function(wrap_pyfunction!(log_f32, m)?)?;
    m.add_function(wrap_pyfunction!(sigmoid_f32, m)?)?;
    m.add_function(wrap_pyfunction!(concatenate_f32, m)?)?;

    m.add_function(wrap_pyfunction!(zeros_f64, m)?)?;
    m.add_function(wrap_pyfunction!(ones_f64, m)?)?;
    m.add_function(wrap_pyfunction!(matmul_f64, m)?)?;
    m.add_function(wrap_pyfunction!(dot_f64, m)?)?;
    m.add_function(wrap_pyfunction!(arange_f64, m)?)?;
    m.add_function(wrap_pyfunction!(linspace_f64, m)?)?;
    m.add_function(wrap_pyfunction!(mean_f64, m)?)?;
    m.add_function(wrap_pyfunction!(median_f64, m)?)?;
    m.add_function(wrap_pyfunction!(min_f64, m)?)?;
    m.add_function(wrap_pyfunction!(min_axis_f64, m)?)?;
    m.add_function(wrap_pyfunction!(max_f64, m)?)?;
    m.add_function(wrap_pyfunction!(max_axis_f64, m)?)?;
    m.add_function(wrap_pyfunction!(exp_f64, m)?)?;
    m.add_function(wrap_pyfunction!(log_f64, m)?)?;
    m.add_function(wrap_pyfunction!(sigmoid_f64, m)?)?;
    m.add_function(wrap_pyfunction!(concatenate_f64, m)?)?;
    m.add_function(wrap_pyfunction!(norm_f32, m)?)?;
    m.add_function(wrap_pyfunction!(norm_f64, m)?)?;

    Ok(())
}
