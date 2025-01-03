# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
from collections.abc import Callable

import numpy as np
import rich.logging

from pymittagleffler import mittag_leffler
from pymittagleffler.fallback import mittag_leffler_garrappa

log = logging.getLogger(pathlib.Path(__file__).stem)
log.setLevel(logging.INFO)
log.addHandler(rich.logging.RichHandler())


Array = np.ndarray[tuple[int, ...], np.dtype[np.inexact]]


# {{{ utils


def timeit(
    stmt: str | Callable[[], object],
    *,
    repeat: int = 16,
    number: int = 1,
) -> tuple[float, float, float]:
    import timeit as _timeit

    r = np.array(_timeit.repeat(stmt=stmt, repeat=repeat + 1, number=number))
    return np.min(r), np.mean(r), np.std(r, ddof=1)


# }}}


# {{{ benchmark

MATLAB_ML_URL = "https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/48154/versions/6/download/zip/ml.zip"


def benchmark_rust(z: Array, alpha: Array, beta: float) -> Array:
    log.info("Running Rust benchmark...")

    result = np.empty((alpha.size, 3))
    for i in range(alpha.size):
        result[i] = timeit(lambda i=i: mittag_leffler(z, float(alpha[i]), beta))

    return result


def benchmark_python(z: Array, alpha: Array, beta: float) -> Array:
    log.info("Running Python benchmark...")

    result = np.empty((alpha.size, 3))
    for i in range(alpha.size):
        result[i] = timeit(
            lambda i=i: mittag_leffler_garrappa(z, float(alpha[i]), beta)
        )

    return result


def benchmark_matlab(z: Array, alpha: Array, beta: Array) -> Array:
    # {{{ download

    import requests

    response = requests.get(MATLAB_ML_URL, timeout=10)
    if not response.ok:
        log.error("Could not download '%s'.", MATLAB_ML_URL)
        raise SystemExit(1)

    import tempfile

    with tempfile.NamedTemporaryFile(
        prefix="mittag-leffler-", suffix=".zip", delete=False
    ) as f:
        filename = pathlib.Path(f.name)
        f.write(response.content)
        log.info("Downloaded 'ml.zip' to '%s'.", filename)

    import zipfile

    dirname = filename.with_suffix("")
    with zipfile.ZipFile(filename, "r") as f:
        f.extractall(dirname)
        log.info("File extracted to '%s'.", dirname)

    # }}}

    # {{{ copy files

    import shutil

    benchfile = pathlib.Path(__file__).parent.parent / "benches" / "bench_matlab.m"
    if not benchfile.exists():
        log.error("Could not find benchmark file '%s'", benchfile.name)
        raise SystemExit(1)

    shutil.copyfile(benchfile, dirname / benchfile.name)
    log.info("Copied file to bench directory: '%s'", benchfile)

    from scipy.io import savemat

    benchmat = dirname / "bench_data.mat"
    savemat(benchmat, {"z": z, "alpha": alpha, "beta": beta})
    log.info("Saved data to file: '%s'", benchmat)

    # }}}

    # {{{ run benchmark

    import subprocess  # noqa: S404

    log.info("Running MATLAB benchmark...")
    try:
        subprocess.check_call(  # noqa: S603
            [  # noqa: S607
                "matlab",
                "-nodisplay",
                "-nosplash",
                "-nodesktop",
                "-r",
                "run('bench_matlab.m'); exit;",
            ],
            cwd=dirname,
        )
    except subprocess.CalledProcessError as exc:
        log.error("Failed to run MATLAB benchmark. Check folder '%s'", dirname)
        raise SystemExit(exc.returncode) from exc

    from scipy.io import loadmat

    data = loadmat(dirname / "bench_result.mat")
    result = data["result"]

    # }}}

    # {{{ cleanup

    # if filename.exists():
    #     filename.unlink()

    # if dirname.exists():
    #     shutil.rmtree(dirname)

    # }}}

    return result


# {{{ main


def main(
    filename: pathlib.Path | None,
    outfile: pathlib.Path | None,
    *,
    overwrite: bool = False,
) -> int:
    if outfile is None:
        outfile = pathlib.Path("bench-mittag-leffler")

    if not overwrite and outfile.exists():
        log.error("File already exists (use --overwrite): '%s'", outfile)
        return 1

    if filename is not None:
        data = np.load(filename, allow_pickle=True)

        alpha = data["alpha"]
        result_rust = data["result_rust"]
        result_python = data["result_python"]
        result_matlab = data["result_matlab"]
    else:
        alpha = np.linspace(0.1, 5.0, 32)
        beta = 1.25

        arg = np.pi / 2
        r = np.linspace(0.0, 12.0, 256)
        z = r * np.exp(1j * arg)

        result_rust = benchmark_rust(z, alpha, beta)
        result_python = benchmark_python(z, alpha, beta)
        result_matlab = benchmark_matlab(z, alpha, beta)

        np.savez(
            outfile.with_suffix(".npz"),
            alpha=alpha,
            result_rust=result_rust,
            result_python=result_python,
            result_matlab=result_matlab,
        )

    # {{{ plot

    try:
        import matplotlib.pyplot as mp
    except ImportError as exc:
        print("'matplotlib' package is not available for plotting")
        raise SystemExit(0) from exc

    from pymittagleffler import _set_recommended_matplotlib  # noqa: E402,RUF100

    _set_recommended_matplotlib()
    fig = mp.figure()
    ax = fig.gca()

    # mean = result_rust[:, 1] * 1000
    # std = result_rust[:, 2] * 1000
    # (line,) = ax.plot(alpha, mean, label="Rust")
    # ax.fill_between(alpha, mean + std, mean - std, alpha=0.2, color=line.get_color())

    # mean = result_python[:, 1] * 1000
    # std = result_python[:, 2] * 1000
    # (line,) = ax.plot(alpha, mean, label="Python")
    # ax.fill_between(alpha, mean + std, mean - std, alpha=0.2, color=line.get_color())

    # mean = result_matlab[:, 1] * 1000
    # std = result_matlab[:, 2] * 1000
    # (line,) = ax.plot(alpha, mean, label="MATLAB")
    # ax.fill_between(alpha, mean + std, mean - std, alpha=0.2, color=line.get_color())

    width = 0.3
    size = result_rust[:, 1]
    n = np.arange(size.size)

    ax.bar(n, result_rust[:, 1] / size, width, label="Rust")
    ax.bar(n + width, result_python[:, 1] / size, width, label="Python")
    ax.bar(n + 2 * width, result_matlab[:, 1] / size, width, label="MATLAB")

    ax.set_xlabel(r"$n$")
    ax.set_ylabel("Improvement (mean time)")
    ax.legend()

    fig.savefig(outfile)
    log.info("Saved plot in '%s'", outfile)

    # }}}

    return 0


# }}}

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("filename", nargs="?", type=pathlib.Path, default=None)
    parser.add_argument("-o", "--outfile", type=pathlib.Path, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only show error messages",
    )
    args = parser.parse_args()

    if not args.quiet:
        log.setLevel(logging.INFO)

    raise SystemExit(main(args.filename, args.outfile, overwrite=args.overwrite))
