"""Known-future covariates without training: regress, then forecast the residual with the frozen univariate model.

Per series, a ridge of the target on its covariates is fitted over the context; the model forecasts the target minus
the fitted effect, and the effect of the known future values is added back to every quantile. Linear and instantaneous:
only covariates whose values are known over the context AND the horizon are used. The gate keeps the covariate forecast
only if it beats the plain forecast by GATE_MARGIN on the last points of the context (nothing beyond the context is
read)."""
from __future__ import annotations

import numpy as np

MAX_ONE_HOT = 16                                   # categorical columns with more levels are encoded as ordinals
LAMBDAS = np.logspace(-3, 4, 15)                   # ridge penalties tried by GCV, times the number of points
GATE_MARGIN = 0.02   # the gate keeps the covariates only if they beat the plain forecast by 2 % (tuned on synthetic data)
MODES = ("off", "gate", "always")                 # when the covariate effect is applied


def encode_column(values) -> np.ndarray | None:
    """One covariate over context + horizon -> numeric [T + H, m]; None when it has no finite value.

    Numeric columns stay as they are; others are categorical: one-hot (first level as reference) when they have at most
    MAX_ONE_HOT levels, ordinal otherwise. Levels are read over context + horizon (the future is known, not a leak).
    Gaps: the last value is carried forward, then the mean fills what remains."""
    v = np.asarray(values, dtype=object).ravel()
    missing = np.array([e is None or (isinstance(e, (float, np.floating)) and np.isnan(e))
                        or str(e) in ("<NA>", "NaT", "nan") for e in v], bool)
    try:
        x = np.where(missing, np.nan, v).astype(np.float64)       # numeric, nullable pandas dtypes included
    except (TypeError, ValueError):
        levels = sorted({str(e) for e, m in zip(v, missing) if not m})
        index = {lv: i for i, lv in enumerate(levels)}
        codes = np.array([np.nan if m else index[str(e)] for e, m in zip(v, missing)])
        if len(levels) <= MAX_ONE_HOT:
            codes = _fill(codes)
            if codes is None:
                return None
            return np.eye(len(levels))[codes.astype(int)][:, 1:]
        x = codes
    x = _fill(x)
    return None if x is None else x[:, None]


def _fill(x: np.ndarray) -> np.ndarray | None:
    if not np.isfinite(x).any():
        return None
    fill = float(np.nanmean(np.where(np.isfinite(x), x, np.nan)))
    out = x.copy()
    for i in range(len(out)):
        if not np.isfinite(out[i]):
            out[i] = out[i - 1] if i and np.isfinite(out[i - 1]) else fill
    return out


def as_matrix(covariates, length: int, prediction_length: int, what: str) -> np.ndarray:
    """The covariates of one series (dict {name: values} or 2-D array [T + H, p]) -> numeric [T + H, m]."""
    total = length + prediction_length
    if isinstance(covariates, dict):
        columns = list(covariates.items())
    else:
        a = np.asarray(covariates, dtype=object)
        if a.ndim != 2:
            raise ValueError(f"{what} must be a dict {{name: values}} or a 2-D array [length + prediction_length, p], "
                             f"got shape {a.shape}")
        columns = [(j, a[:, j]) for j in range(a.shape[1])]
    encoded = []
    for name, values in columns:
        n = len(values)
        if n != total:
            past_only = " (past-only covariates are not supported: values over the horizon are needed)" \
                if n == length else ""
            raise ValueError(f"{what}, covariate {name!r}: {n} values, expected {total} "
                             f"(context length + prediction_length){past_only}")
        e = encode_column(values)
        if e is not None:
            encoded.append(e)
    return np.concatenate(encoded, 1) if encoded else np.zeros((total, 0))


def ridge_gcv(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Ridge coefficients of centred y on standardised X, the penalty chosen by generalised cross-validation (one SVD)."""
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    Uty = U.T @ y
    n = len(y)
    best, beta = np.inf, np.zeros(X.shape[1])
    for lam in LAMBDAS * n:
        d = s / (s ** 2 + lam)
        b = Vt.T @ (d * Uty)
        r = y - X @ b
        dof = np.sum(s ** 2 / (s ** 2 + lam))
        g = n * float(r @ r) / max(n - dof, 1.0) ** 2
        if g < best:
            best, beta = g, b
    return beta


def impact(y: np.ndarray, Xp: np.ndarray, Xf: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """(effect over the context [T], effect over the horizon [H]) of the covariates on y; None when not fitted.

    Columns are standardised with their context mean and deviation (constant ones dropped); horizon values are clipped
    to the range seen in the context, so the effect is never extrapolated. Not fitted with fewer than max(8, 2 p)
    observed targets."""
    if Xp.shape[1] == 0:
        return None
    mu, sd = Xp.mean(0), Xp.std(0)
    keep = sd > 1e-12
    if not keep.any():
        return None
    Zp, Zf = (Xp[:, keep] - mu[keep]) / sd[keep], (Xf[:, keep] - mu[keep]) / sd[keep]
    Zf = np.clip(Zf, Zp.min(0), Zp.max(0))
    ok = np.isfinite(y)
    if ok.sum() < max(8, 2 * Zp.shape[1]):
        return None
    b = ridge_gcv(Zp[ok], y[ok] - y[ok].mean())
    return Zp @ b, Zf @ b


# How the covariate effect is fitted: name -> impact(y, Xp, Xf) -> (effect over the context, effect over the horizon)
# or None. The gate and the residual forecast do not depend on it.
MODELS = {"ridge": impact}
