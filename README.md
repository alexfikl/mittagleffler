<div align="center">
<img width="600" src="https://raw.githubusercontent.com/alexfikl/mittagleffler/refs/heads/main/python/docs/_static/mittag-leffler-accuracy-contour.png"/><br>
</div>

# mittagleffler

[![Build Status](https://github.com/alexfikl/mittagleffler/workflows/CI/badge.svg)](https://github.com/alexfikl/mittagleffler/actions?query=branch%3Amain+workflow%3ACI)
[![REUSE](https://api.reuse.software/badge/github.com/alexfikl/mittagleffler)](https://api.reuse.software/info/github.com/alexfikl/mittagleffler)
[![PyPI](https://badge.fury.io/py/pymittagleffler.svg)](https://pypi.org/project/pymittagleffler/)
[![crates.io](https://img.shields.io/crates/v/mittagleffler)](https://crates.io/crates/mittagleffler)
[![readthedocs.io](https://img.shields.io/readthedocs/mittagleffler?label=rtd.io&color=%234280B2)](https://mittagleffler.readthedocs.io/en/latest)
[![docs.rs](https://img.shields.io/docsrs/mittagleffler?label=docs.rs&color=%23F58042)](https://docs.rs/mittagleffler/latest/mittagleffler/)

This library implements the two-parameter Mittag-Leffler function.

Currently only the algorithm described in the paper by [Roberto Garrapa (2015)](<https://doi.org/10.1137/140971191>)
is implemented. This seems to be the most accurate and computationally efficient
method to date for evaluating the Mittag-Leffler function.

**Links**

* *Documentation*: [Rust (docs.rs)](https://docs.rs/mittagleffler/latest/mittagleffler/)
  and [Python (readthedocs.io)](https://mittagleffler.readthedocs.io).
* *Code*: [Github](https://github.com/alexfikl/mittagleffler).
* *License*: [MIT](https://spdx.org/licenses/MIT.html) (see `LICENSES/MIT.txt`).

**Other implementations**

* [ml.m](https://www.mathworks.com/matlabcentral/fileexchange/48154-the-mittag-leffler-function) (MATLAB):
  implements three-parameter Mittag-Leffler function.
* [ml_matrix.m](https://www.mathworks.com/matlabcentral/fileexchange/66272-mittag-leffler-function-with-matrix-arguments) (MATLAB):
  implements the matrix-valued two-parameter Mittag-Leffler function.
* [MittagLeffler.jl](https://github.com/JuliaMath/MittagLeffler.jl) (Julia):
  implements the two-parameter Mittag-Leffler function and its derivative.
* [MittagLeffler](https://github.com/gurteksinghgill/MittagLeffler) (R):
  implements the three-parameter Mittag-Leffler function.
* [mittag-leffler](https://github.com/khinsen/mittag-leffler) (Python):
  implements the three-parameter Mittag-Leffler function.
* [mlf](https://github.com/tranqv/Mittag-Leffler-function-and-its-derivative) (Fortran 90):
  implements the three-parameter Mittag-Leffler function.
* [mlpade](https://github.com/matt-black/mlpade) (MATLAB):
  implements the two-parameter Mittag-Leffler function.
* [MittagLeffler](https://github.com/droodman/Mittag-Leffler-for-Stata) (Stata):
  implements the three-parameter Mittag-Leffler function.
* [MittagLefflerE](https://reference.wolfram.com/language/ref/MittagLefflerE.html.en) (Mathematica):
  implements the two-parameter Mittag-Leffler function.

# Rust Crate

The library is available as a Rust crate that implements the main algorithms.
Evaluating the Mittag Leffler function can be performed directly by

```rust
use mittagleffler::MittagLeffler;

let alpha = 0.75;
let beta = 1.25;
let z = Complex64::new(1.0, 2.0);
println!("E_{}_{}({}) = {}", alpha, beta, z, z.mittag_leffler(alpha, beta));

let z: f64 = 3.1415;
println!("E_{}_{}({}) = {}", alpha, beta, z, z.mittag_leffler(alpha, beta));
```

This method will call the best underlying algorithm and take care of any special
cases that are known in the literature, e.g. for `(alpha, beta) = (1, 1)` we
know that the Mittag-Leffler function is equivalent to the standard exponential.
To call a specific algorithm, we can do

```rust
use mittagleffler::GarrappaMittagLeffler

let eps = 1.0e-8;
let ml = GarrappaMittagLeffler::new(eps);

let z = Complex64::new(1.0, 2.0);
println!("E_{}_{}({}) = {}", alpha, beta, z, ml.evaluate(z, alpha, beta));
```

The algorithm from Garrappa (2015) has several parameters that can be tweaked
for better performance or accuracy. They can be found in the documentation of the
structure, but should not be changed unless there is good reason!

Python Bindings
===============

The library also has Python bindings (using [pyo3](https://github.com/PyO3/pyo3))
that can be found in the `python` directory. The bindings are written to work
with scalars and with `numpy` arrays equally. For example

```python
import numpy as np
from pymittagleffler import mittag_leffler

alpha, beta = 2.0, 2.0
z = np.linspace(0.0, 1.0, 128)
result = mittag_leffler(z, alpha, beta)
```

These are available on PyPI under the name `pymittagleffler`.
