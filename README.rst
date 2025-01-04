.. |badge-ci| image:: https://github.com/alexfikl/mittagleffler/workflows/CI/badge.svg
    :alt: Build Status
    :target: https://github.com/alexfikl/mittagleffler/actions?query=branch%3Amain+workflow%3ACI

.. |badge-pypi| image:: https://badge.fury.io/py/pymittagleffler.svg
    :alt: PyPI
    :target: https://pypi.org/project/pymittagleffler/

.. |badge-reuse| image:: https://api.reuse.software/badge/github.com/alexfikl/mittagleffler
    :alt: REUSE
    :target: https://api.reuse.software/info/github.com/alexfikl/mittagleffler

.. |badge-rtd| image:: https://readthedocs.org/projects/mittagleffler/badge/?version=latest
    :alt: Documentation
    :target: https://mittagleffler.readthedocs.io/en/latest/?badge=latest

|badge-ci| |badge-reuse| |badge-rtd| |badge-pypi|

mittagleffler
=============

This library implements the two-parameter Mittag-Leffler function.

Currently only the algorithm described in the paper by `Roberto Garrapa (2015)
<https://doi.org/10.1137/140971191>`__ is implemented. This seems to be the
most accurate and computationally efficient method to date for evaluating the
Mittag-Leffler function.

**Links**

* `Documentation <https://mittagleffler.readthedocs.io>`__.
* `Code <https://github.com/alexfikl/mittagleffler>`__.
* License: MIT (see ``LICENSES/MIT.txt``).

**Other implementations**

* `ml.m <https://www.mathworks.com/matlabcentral/fileexchange/48154-the-mittag-leffler-function>`__ (MATLAB):
  implements three-parameter Mittag-Leffler function.
* `ml_matrix.m <https://www.mathworks.com/matlabcentral/fileexchange/66272-mittag-leffler-function-with-matrix-arguments>`__ (MATLAB):
  implements the matrix-valued two-parameter Mittag-Leffler function.
* `MittagLeffler.jl <https://github.com/JuliaMath/MittagLeffler.jl>`__ (Julia):
  implements the two-parameter Mittag-Leffler function and its derivative.
* `MittagLeffler <https://github.com/gurteksinghgill/MittagLeffler>`__ (R):
  implements the three-parameter Mittag-Leffler function.
* `mittag-leffler <https://github.com/khinsen/mittag-leffler>`__ (Python):
  implements the three-parameter Mittag-Leffler function.
* `mlf <https://github.com/tranqv/Mittag-Leffler-function-and-its-derivative>`__ (Fortran 90):
  implements the three-parameter Mittag-Leffler function.
* `mlpade <https://github.com/matt-black/mlpade>`__ (MATLAB):
  implements the two-parameter Mittag-Leffler function.
* `MittagLeffler <https://github.com/droodman/Mittag-Leffler-for-Stata>`__ (Stata):
  implements the three-parameter Mittag-Leffler function.
* `MittagLefflerE <https://reference.wolfram.com/language/ref/MittagLefflerE.html.en>`__ (Mathematica):
  implements the two-parameter Mittag-Leffler function.

Rust Crate
==========

The library is available as a Rust crate that implements the main algorithms.
Evaluating the Mittag Leffler function can be performed directly by

.. code:: rust

    use mittagleffler::MittagLeffler;

    let alpha = 0.75;
    let beta = 1.25;
    let z = Complex64::new(1.0, 2.0);
    println!("E_{}_{}({}) = {}", alpha, beta, z, z.mittag_leffler(alpha, beta));

    let z: f64 = 3.1415;
    println!("E_{}_{}({}) = {}", alpha, beta, z, z.mittag_leffler(alpha, beta));

This method will call the best underlying algorithm and take care of any special
cases that are known in the literature, e.g. for $`(\alpha, \beta) = (1, 1)`$ we
know that the Mittag-Leffler function is equivalent to the standard exponential.
To call a specific algorithm, we can do

.. code:: rust

    use mittagleffler::GarrappaMittagLeffler

    let eps = 1.0e-8;
    let ml = GarrappaMittagLeffler::new(eps);

    let z = Complex64::new(1.0, 2.0);
    println!("E_{}_{}({}) = {}", alpha, beta, z, ml.evaluate(z, alpha, beta));

The algorithm from ``Garrappa2015`` has several parameters that can be tweaked
for better performance or accuracy. They can be found in the documentation of the
structure, but should not be changed unless there is good reason!

Python Bindings
===============

The library also has Python bindings (using `pyo3 <https://github.com/PyO3/pyo3>`__)
that can be found in the ``python`` directory. The bindings are written to work
with scalars and with ``numpy`` arrays equally. For example

.. code:: python

    import numpy as np
    from pymittagleffler import mittag_leffler

    alpha, beta = 2.0, 2.0
    z = np.linspace(0.0, 1.0, 128)
    result = mittag_leffler(z, alpha, beta)

These are available on PyPI under the name ``pymittagleffler``.
