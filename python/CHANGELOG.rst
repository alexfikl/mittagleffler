Version 0.1.0 (TDB)
-------------------

This is the initial release of the Python bindings for the ``mittagleffler``
Rust crate. It mainly provides a simple function to evaluate the Mittag-Leffler
function as:

.. code:: python

    from pymittagleffler import mittag_leffler

    z = np.linspace(0.0, 1.0) + 1j * np.linspace(0.0, 1.0)
    ml = mittag_leffler(z, alpha=1.0, beta=2.0)

The function accepts real and complex inputs, both scalars and :mod:`numpy` arrays
of any shape. The function is applied component wise, i.e. this is different than
the existing "Matrix" Mittag-Leffler function.
