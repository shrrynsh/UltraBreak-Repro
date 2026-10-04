"""
stats.py — the statistics this study reports, implemented with numpy only.

Why not scipy/statsmodels: the project rule is that package versions inside the
`repro` venv are never changed, because that env produced every existing number
in the study. `repro` has no scipy/sklearn/statsmodels, so installing them to get
three closed-form quantities would risk silently invalidating completed results.
All four routines below are exact (not approximations of a library's output) and
`_selftest()` checks each against a hand-computed value.

Provides:
  wilson_ci          Wilson score 95% interval for a proportion
  mcnemar_exact      two-sided EXACT McNemar (binomial on discordant pairs)
  logistic_fit       IRLS logistic regression with Wald z/p
  normal_sf          standard-normal survival function (via math.erfc)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

Z95 = 1.959963984540054  # two-sided 95%


# ── normal tail ────────────────────────────────────────────────────────────────
def normal_sf(z: float) -> float:
    """P(Z > z) for standard normal. erfc is exact to double precision."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def two_sided_z_p(z: float) -> float:
    return 2.0 * normal_sf(abs(z))


# ── proportions ────────────────────────────────────────────────────────────────
@dataclass
class Prop:
    k: int
    n: int
    p: float
    lo: float
    hi: float

    def pct(self) -> str:
        """'83.49 [78.9, 87.3]' — the form used in the report tables."""
        if self.n == 0:
            return "n/a"
        return f"{100*self.p:.2f} [{100*self.lo:.1f}, {100*self.hi:.1f}]"


def wilson_ci(k: int, n: int, z: float = Z95) -> Prop:
    """
    Wilson score interval. Preferred over the normal approximation because our
    per-category cells are small (n=45) and several ASRs sit near 0 or 1, where
    the Wald interval leaves the unit interval.

        centre = (k + z^2/2) / (n + z^2)
        half   = z/(n + z^2) * sqrt(k(n-k)/n + z^2/4)
    """
    k, n = int(k), int(n)
    if n <= 0:
        return Prop(k, n, float("nan"), float("nan"), float("nan"))
    if not 0 <= k <= n:
        raise ValueError(f"k={k} out of range for n={n}")
    p = k / n
    denom = n + z * z
    centre = (k + z * z / 2.0) / denom
    half = (z / denom) * math.sqrt(k * (n - k) / n + z * z / 4.0)
    return Prop(k, n, p, max(0.0, centre - half), min(1.0, centre + half))


# ── paired comparison ──────────────────────────────────────────────────────────
@dataclass
class McNemar:
    b: int          # condition A success, condition B failure
    c: int          # condition A failure, condition B success
    n_pairs: int
    p_value: float
    note: str = ""

    def __str__(self) -> str:
        if self.b + self.c == 0:
            return f"b=0 c=0 (no discordant pairs) p=1.000"
        return f"b={self.b} c={self.c} p={self.p_value:.3g}"


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> McNemar:
    """
    Two-sided EXACT McNemar test on paired binary outcomes.

    `a` and `b` are boolean arrays of the SAME queries in the SAME order — the
    paired design the study requires. Only discordant pairs carry information:
    under H0 each discordant pair is a fair coin, so

        p = P(X <= min(b,c)) + P(X >= max(b,c)),  X ~ Binomial(b+c, 1/2)

    which for the symmetric case is the usual 2 * one-tail, capped at 1. The
    exact form is used rather than the chi-square approximation because several
    cells here have few discordant pairs.
    """
    a = np.asarray(a).astype(bool)
    b_ = np.asarray(b).astype(bool)
    if a.shape != b_.shape:
        raise ValueError(f"unpaired shapes {a.shape} vs {b_.shape}")
    n_b = int(np.sum(a & ~b_))
    n_c = int(np.sum(~a & b_))
    n = n_b + n_c
    if n == 0:
        return McNemar(0, 0, len(a), 1.0, "no discordant pairs")
    lo = min(n_b, n_c)
    # two-sided exact: both tails at or beyond the observed imbalance
    tail = sum(math.comb(n, i) for i in range(0, lo + 1)) / (2.0 ** n)
    p = min(1.0, 2.0 * tail)
    return McNemar(n_b, n_c, len(a), p)


# ── logistic regression ────────────────────────────────────────────────────────
@dataclass
class Logit:
    names: list
    coef: np.ndarray
    se: np.ndarray
    z: np.ndarray
    p: np.ndarray
    n: int
    iters: int
    converged: bool
    loglik: float
    separated: list = field(default_factory=list)

    def table(self) -> str:
        w = max(len(x) for x in self.names)
        out = [f"{'term':{w}s} {'coef':>9s} {'se':>8s} {'z':>7s} {'p':>10s}"]
        for i, nm in enumerate(self.names):
            out.append(
                f"{nm:{w}s} {self.coef[i]:>9.4f} {self.se[i]:>8.4f} "
                f"{self.z[i]:>7.2f} {self.p[i]:>10.3g}"
            )
        out.append(f"n={self.n}  loglik={self.loglik:.2f}  converged={self.converged}")
        return "\n".join(out)


def logistic_fit(X: np.ndarray, y: np.ndarray, names=None,
                 max_iter: int = 100, tol: float = 1e-9,
                 ridge: float = 1e-8) -> Logit:
    """
    Logistic regression by IRLS (Newton-Raphson). An intercept is NOT added
    automatically — pass it in X so the caller controls the design.

    Wald standard errors come from the inverse observed information
    (X' W X)^-1 at the optimum. `ridge` is a tiny Tikhonov term that only keeps
    the solve numerically stable; at 1e-8 it does not move reported coefficients.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n, k = X.shape
    if y.shape[0] != n:
        raise ValueError(f"X has {n} rows, y has {y.shape[0]}")
    if names is None:
        names = [f"x{i}" for i in range(k)]
    beta = np.zeros(k)
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        eta = X @ beta
        mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -500, 500)))
        w = np.clip(mu * (1.0 - mu), 1e-12, None)
        grad = X.T @ (y - mu)
        H = (X * w[:, None]).T @ X + ridge * np.eye(k)
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            break
        beta_new = beta + step
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            converged = True
            break
        beta = beta_new

    eta = X @ beta
    mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -500, 500)))
    w = np.clip(mu * (1.0 - mu), 1e-12, None)
    H = (X * w[:, None]).T @ X + ridge * np.eye(k)
    cov = np.linalg.inv(H)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, beta / se, 0.0)
    p = np.array([two_sided_z_p(zi) for zi in z])
    eps = 1e-12
    ll = float(np.sum(y * np.log(mu + eps) + (1 - y) * np.log(1 - mu + eps)))
    # flag quasi-separation: huge |coef| with huge se is not evidence
    sep = [names[i] for i in range(k) if abs(beta[i]) > 15 or se[i] > 15]
    return Logit(list(names), beta, se, z, p, n, it, converged, ll, sep)


# ── self-test ──────────────────────────────────────────────────────────────────
def _selftest() -> None:
    # Wilson, hand-computed: k=263, n=315 (job 809's SafeBench cell)
    w = wilson_ci(263, 315)
    assert abs(w.p - 263 / 315) < 1e-12
    assert 0.788 < w.lo < 0.792, w.lo          # 78.9%
    assert 0.870 < w.hi < 0.876, w.hi          # 87.3%
    # Wilson edge cases
    assert wilson_ci(0, 10).lo == 0.0
    assert wilson_ci(10, 10).hi == 1.0
    assert math.isnan(wilson_ci(0, 0).p)

    # Exact McNemar, hand-computed: b=10, c=2 -> 2*P(X<=2), X~Bin(12,.5)
    #   P = (C(12,0)+C(12,1)+C(12,2))/2^12 = (1+12+66)/4096 = 79/4096
    a = np.array([True] * 10 + [False] * 2 + [True] * 5 + [False] * 5)
    b = np.array([False] * 10 + [True] * 2 + [True] * 5 + [False] * 5)
    m = mcnemar_exact(a, b)
    assert (m.b, m.c) == (10, 2), (m.b, m.c)
    assert abs(m.p_value - 2 * 79 / 4096) < 1e-12, m.p_value
    # symmetric case
    s = mcnemar_exact(np.array([True, False]), np.array([False, True]))
    assert abs(s.p_value - 1.0) < 1e-12
    # no discordance
    nd = mcnemar_exact(np.array([True, False]), np.array([True, False]))
    assert nd.p_value == 1.0 and nd.b == 0 and nd.c == 0

    # normal tail: P(Z>1.96) ~ 0.025
    assert abs(normal_sf(1.959963984540054) - 0.025) < 1e-6

    # Logistic: perfectly recoverable signal, intercept + slope
    rng = np.random.RandomState(0)
    x = rng.normal(size=4000)
    pr = 1.0 / (1.0 + np.exp(-(-0.5 + 1.5 * x)))
    yy = (rng.uniform(size=4000) < pr).astype(float)
    X = np.column_stack([np.ones(4000), x])
    f = logistic_fit(X, yy, ["intercept", "x"])
    assert f.converged
    assert abs(f.coef[0] + 0.5) < 0.12, f.coef
    assert abs(f.coef[1] - 1.5) < 0.15, f.coef
    assert f.p[1] < 1e-50
    print("stats.py self-test OK")


if __name__ == "__main__":
    _selftest()
