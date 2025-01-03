# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import math

import numpy as np

Array = np.ndarray[tuple[int, ...], np.dtype[np.complexfloating]]


def _find_optimal_bounded_param(
    t: float,
    phi_star0: float,
    phi_star1: float,
    p: float,
    q: float,
    *,
    log_eps: float,
    log_machine_eps: float,
    fac: float = 1.01,
    p_eps: float = 1.0e-14,
    q_eps: float = 1.0e-14,
    conservative_error_analysis: bool = False,
) -> tuple[float, float, float]:
    # set maximum value of fbar (the ratio of the tolerance to the machine epsilon)
    f_max = math.exp(log_eps - log_machine_eps)
    threshold = 2 * np.sqrt((log_eps - log_machine_eps) / t)

    # starting values
    phi_star0_sq = np.sqrt(phi_star0)
    phi_star1_sq = min(np.sqrt(phi_star1), threshold - phi_star0_sq)

    # determine phibar and admissible region
    if p < p_eps:
        if q < q_eps:
            phibar_star0_sq = phi_star0_sq
            phibar_star1_sq = phi_star1_sq
            adm_region = True
        else:
            phibar_star0_sq = phi_star0_sq
            if phi_star0_sq > 0:
                f_min = fac * (phi_star0_sq / (phi_star1_sq - phi_star0_sq)) ** q
            else:
                f_min = fac

            if f_min < f_max:
                f_bar = f_min + f_min / f_max * (f_max - f_min)
                fq = f_bar ** (-1 / q)
                phibar_star1_sq = (2 * phi_star1_sq - fq * phi_star0_sq) / (2 + fq)
                adm_region = True
            else:
                adm_region = False
    else:
        if q < q_eps:
            phibar_star1_sq = phi_star1_sq
            f_min = fac * (phi_star1_sq / (phi_star1_sq - phi_star0_sq)) ** p
            if f_min < f_max:
                f_bar = f_min + f_min / f_max * (f_max - f_min)
                fp = f_bar ** (-1 / p)
                phibar_star0_sq = (2 * phi_star0_sq - fp * phi_star1_sq) / (2 - fp)
                adm_region = True
            else:
                adm_region = False
        else:
            f_min = (
                fac
                * (phi_star1_sq + phi_star0_sq)
                / (phi_star1_sq - phi_star0_sq) ** max(p, q)
            )
            if f_min < f_max:
                f_min = max(f_min, 1.5)
                f_bar = f_min + f_min / f_max * (f_max - f_min)
                fp = f_bar ** (-1 / p)
                fq = f_bar ** (-1 / q)

                if not conservative_error_analysis:
                    w = -phi_star1 * t / log_eps
                else:
                    w = -2 * phi_star1 * t / (log_eps - phi_star1 * t)

                den = 2 + w - (1 + w) * fp + fq
                phibar_star0_sq = (
                    (2 + w + fq) * phi_star0_sq + fp * phi_star1_sq
                ) / den
                phibar_star1_sq = (
                    -(1 + w) * fq * phi_star0_sq + (2 + w - (1 + w) * fp) * phi_star1_sq
                ) / den
                adm_region = True
            else:
                adm_region = False

    if adm_region:
        log_eps = log_eps - math.log(f_bar)
        if not conservative_error_analysis:
            w = -(phibar_star1_sq**2) * t / log_eps
        else:
            w = -2 * phibar_star1_sq**2 * t / (log_eps - phibar_star1_sq**2 * t)

        mu = (((1 + w) * phibar_star0_sq + phibar_star1_sq) / (2 + w)) ** 2
        h = (
            -2
            * np.pi
            / log_eps
            * (phibar_star1_sq - phibar_star0_sq)
            / ((1 + w) * phibar_star0_sq + phibar_star1_sq)
        )
        N = math.ceil(np.sqrt(1 - log_eps / t / mu) / h)
    else:
        mu = 0
        h = 0
        N = np.inf

    return mu, N, h


def _find_optimal_unbounded_param(
    t: float,
    phi_star: float,
    p: float,
    *,
    log_eps: float,
    log_machine_eps: float,
    fac: float = 1.01,
    p_eps: float = 1.0e-14,
) -> tuple[float, float, float]:
    phi_star_sq = np.sqrt(phi_star)
    phibar_star = (fac * phi_star) if phi_star > 0 else 0.01
    phibar_star_sq = np.sqrt(phibar_star)

    # search for fbar in [f_min, f_max]
    found = False
    f_min = 1
    f_max = 10
    f_tar = 5

    while not found:
        phi = phibar_star * t
        log_eps_t = log_eps / phi

        N: float = math.ceil(
            phi / np.pi * (1 - 3 * log_eps_t / 2 + math.sqrt(1 - 2 * log_eps_t))
        )
        A = np.pi * N / phi

        mu = phibar_star_sq * abs(4 - A) / abs(7 - math.sqrt(1 + 12 * A))
        fbar = ((phibar_star_sq - phi_star_sq) / mu) ** (-p)

        found = p < p_eps or f_min < fbar < f_max
        if not found:
            phibar_star_sq = f_tar ** (-1 / p) * mu + phi_star_sq
            phibar_star = phibar_star_sq**2

    mu = mu**2
    h = (-3 * A - 2 + 2 * math.sqrt(1 + 12 * A)) / (4 - A) / N

    # adjust integration parameters
    threshold = (log_eps - log_machine_eps) / t
    if mu > threshold:
        Q = 0.0 if abs(p) < p_eps else (f_tar ** (-1 / p) * math.sqrt(mu))
        phibar_star = (Q + math.sqrt(phi_star)) ** 2

        if phibar_star < threshold:
            w = math.sqrt(log_machine_eps / (log_machine_eps - log_eps))
            u = math.sqrt(-phibar_star * t / log_machine_eps)

            mu = threshold
            N = math.ceil(w * log_eps / (2 * np.pi * (u * w - 1)))
            h = w / N
        else:
            N = np.inf
            h = 0

    return mu, N, h


def _laplace_transform_inversion(
    t: float,
    z: complex,
    *,
    alpha: float,
    beta: float,
    eps: float,
    fac: float = 1.01,
) -> complex:
    if abs(z) < eps:
        return 1.0 / math.gamma(beta)

    # get machine precision and epsilon differences
    machine_eps = np.finfo(np.array(z).dtype).eps

    log_machine_eps = math.log(machine_eps)
    log_eps = math.log(eps)
    log_10 = math.log(10)
    d_log_eps = log_eps - log_machine_eps

    import cmath

    # evaluate relevant poles
    theta = cmath.phase(z)
    kmin = math.ceil(-alpha / 2 - theta / (2 * math.pi))
    kmax = math.floor(+alpha / 2 - theta / (2 * math.pi))
    k = np.arange(kmin, kmax + 1)
    s_star = abs(z) ** (1 / alpha) * np.exp(1j * (theta + 2 * np.pi * k) / alpha)

    # sort poles
    phi_star = (s_star.real + abs(s_star)) / 2
    s_star_index = np.argsort(phi_star)
    phi_star = phi_star[s_star_index]
    s_star = s_star[s_star_index]

    # filter out zero poles
    s_star_mask = phi_star > eps
    s_star = s_star[s_star_mask]
    phi_star = phi_star[s_star_mask]

    # add back the origin as a pole
    s_star = np.insert(s_star, 0, 0.0)
    phi_star = np.insert(phi_star, 0, 0.0)

    # strength of the singularities
    p = np.ones(s_star.shape, dtype=s_star.real.dtype)
    p[0] = max(0, -2 * (alpha - beta + 1))
    q = np.ones(s_star.shape, dtype=s_star.real.dtype)
    q[-1] = np.inf
    phi_star = np.insert(phi_star, phi_star.size, np.inf)

    # find admissible regions
    (region_index,) = np.nonzero(
        np.logical_and(
            phi_star[:-1] < d_log_eps / t,
            phi_star[:-1] < phi_star[1:],
        )
    )

    # evaluate parameters for LT inversion in each admissible region
    nregion = region_index[-1] + 1
    mu = np.full(nregion, np.inf, dtype=phi_star.dtype)
    N = np.full(nregion, np.inf, dtype=phi_star.dtype)
    h = np.full(nregion, np.inf, dtype=phi_star.dtype)

    found_region = False
    while not found_region:
        for j in region_index:
            if j < s_star.size - 1:
                mu[j], N[j], h[j] = _find_optimal_bounded_param(
                    t,
                    phi_star[j],
                    phi_star[j + 1],
                    p[j],
                    q[j],
                    log_eps=log_eps,
                    log_machine_eps=log_machine_eps,
                    fac=fac,
                )
            else:
                mu[j], N[j], h[j] = _find_optimal_unbounded_param(
                    t,
                    phi_star[j],
                    p[j],
                    log_eps=log_eps,
                    log_machine_eps=log_machine_eps,
                    fac=fac,
                )

        if np.min(N) > 200:
            log_eps += log_10
        else:
            found_region = True

        if log_eps >= 0.0:
            raise ValueError("Failed to find admissible region")

    # select region that contains the minimum number of nodes
    jmin = np.argmin(N)
    N_min = N[jmin]
    mu_min = mu[jmin]
    h_min = h[jmin]

    # evaluate inverse Laplace transform
    k = np.arange(-N_min, N_min + 1)
    hk = h_min * k
    zk = mu_min * (1j * hk + 1) ** 2
    zd = -2.0 * mu_min * hk + 2j * mu_min
    zexp = np.exp(zk * t)
    F = zk ** (alpha - beta) / (zk**alpha - z) * zd
    S = F * zexp

    integral = h_min * np.sum(S) / (2j * math.pi)

    # evaluate residues
    s_star_min = s_star[jmin + 1 :]
    residues = np.sum(1 / alpha * s_star_min ** (1 - beta) * np.exp(t * s_star_min))

    # sum up the results
    result = residues + integral

    return complex(result)


def _mittag_leffler_garrappa(
    z: complex, alpha: float, beta: float, *, eps: float | None = None
) -> complex:
    if eps is None:
        eps = 5 * float(np.finfo(np.array(z).dtype).eps)

    if abs(z) == 0:
        return complex(1 / math.gamma(beta), 0.0)

    return _laplace_transform_inversion(1.0, z, alpha=alpha, beta=beta, eps=eps)


def mittag_leffler_garrappa(
    z: Array, alpha: float, beta: float, *, eps: float | None = None
) -> Array:
    r"""Evaluate the Mittag-Leffler function :math:`E_{\alpha, \beta}(z)`.

    This is a pure Python implementation of the library. It should be completely
    equivalent, but much slower.
    """

    ml = np.vectorize(lambda zi: _mittag_leffler_garrappa(zi, alpha, beta))
    return np.array(ml(z))
