"""gta6-trophy-rarity-forecast-01: saturating logit-curve trophy rarity forecast.

Forecasts, for every GTA VI trophy, five quantiles of its Exophase earn rate
at the case's target, from the reads recorded before the case's checkpoint.

Model (per trophy, on the logit scale):
  - rates are smoothed ((E + 0.5) / (N + 1)) and logged; only reads whose
    (rate, players_tracked) pair changes from the previous point count as
    distinct points;
  - the median logit is a weighted ensemble of persistence, a 24h log-time
    linear fit, a weighted quadratic fit (refit linear if the x^2 coefficient
    is positive) and a saturating-exponential grid fit, each clamped to
    [logit(last) - 1.5, logit(last) + 3.0]; with fewer than 3 distinct points
    the median is persistence alone;
  - a refresh-due shift pulls the median down when the page has gone stale
    relative to its own refresh cadence (the launch-night page sat unchanged
    for hours, then refreshed with every rate but the 100% one lower);
  - the spread combines the quadratic fit's residual, the extrapolation
    distance in log time and the horizon, with a wider lower tail when a
    refresh is due.

Python 3.13, standard library only. Deterministic: no randomness, no clock.
Run: MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Hours are counted from the start of release day, 2026-11-19T00:00:00Z.
EPOCH0 = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Saturating-exponential grid (per hour).
SAT_EXP_KS = [0.002, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16]

# Ensemble weights: persistence, 24h linear, quadratic, saturating-exponential.
WEIGHTS = {"persist": 0.10, "lin24": 0.25, "quad": 0.30, "satexp": 0.35}


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct)
    return math.log(p / (100 - p))


def sigmoid(l: float) -> float:
    return 100 / (1 + math.exp(-l))


def round4(x: float):
    """x rounded to 4 decimals, half up, as JavaScript's Math.round does it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_epoch0(s: str) -> float:
    return (parse_iso(s) - EPOCH0).total_seconds() / 3600.0


def median(values):
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2


def ols_line(xs, ys, ws=None):
    """Least-squares line y ~ a + b*x (optionally weighted). Returns (a, b)."""
    if ws is None:
        ws = [1.0] * len(xs)
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    denom = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if denom <= 0:
        return my, 0.0
    b = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys)) / denom
    return my - b * mx, b


def solve3(a):
    """Solve a 3x3 linear system (augmented rows) by Gaussian elimination.

    Returns the solution list, or None if the matrix is singular.
    """
    m = [row[:] for row in a]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            return None
        m[col], m[piv] = m[piv], m[col]
        for r in range(3):
            if r == col:
                continue
            factor = m[r][col] / m[col][col]
            for c in range(col, 4):
                m[r][c] -= factor * m[col][c]
    return [m[i][3] / m[i][i] for i in range(3)]


def ols_quad(xs, ys, ws):
    """Weighted least-squares quadratic y ~ c0 + c1*x + c2*x^2.

    Returns (coeffs, sse) or None if the fit is degenerate.
    """
    # Normal equations: (X^T W X) c = X^T W y, basis [1, x, x^2].
    s = [[0.0] * 4 for _ in range(3)]
    for x, y, w in zip(xs, ys, ws):
        basis = [1.0, x, x * x]
        for i in range(3):
            for j in range(3):
                s[i][j] += w * basis[i] * basis[j]
            s[i][3] += w * basis[i] * y
    coeffs = solve3(s)
    if coeffs is None:
        return None
    sse = 0.0
    for x, y in zip(xs, ys):
        pred = coeffs[0] + coeffs[1] * x + coeffs[2] * x * x
        sse += (y - pred) ** 2
    return coeffs, sse


def forecast_trophy(points, x_target, t_target, horizon_days, at_hours):
    """points: list of (t_hours, x, logit) distinct points, oldest first.

    Returns the five quantiles (percent) for one trophy.
    """
    if not points:
        # No usable history at all: the practice mean rate ex-100%.
        med_logit = logit(0.89)
        sigma = 1.2
        stale_factor = 1.0
        return quantiles(med_logit, sigma, stale_factor, horizon_days)

    last_logit = points[-1][2]
    lo = last_logit - 1.5
    hi = last_logit + 3.0

    def clamp_member(v):
        return min(hi, max(lo, v))

    n = len(points)
    xs = [p[1] for p in points]
    ts = [p[0] for p in points]
    ys = [p[2] for p in points]
    x_last = xs[-1]

    # --- staleness / refresh cadence -------------------------------------
    if n < 2:
        stale_factor = 1.0
    else:
        gaps = [ts[i + 1] - ts[i] for i in range(n - 1)]
        med_gap = median(gaps)
        staleness = at_hours - ts[-1]
        if med_gap <= 0:
            stale_factor = 1.0 if staleness > 0 else 0.0
        elif staleness > 2 * med_gap:
            stale_factor = 1.0
        elif staleness > med_gap:
            stale_factor = 0.5
        else:
            stale_factor = 0.0

    # --- median -----------------------------------------------------------
    persist = last_logit
    if n < 3:
        med_logit = persist
        resid = 0.3
    else:
        members = {}
        members["persist"] = persist

        # 24h linear fit: points in the last ln(25) of x; fewer than 3
        # there, use all points.
        window = [i for i, x in enumerate(xs) if x >= x_last - math.log(25)]
        if len(window) < 3:
            window = list(range(n))
        a, b = ols_line([xs[i] for i in window], [ys[i] for i in window])
        members["lin24"] = clamp_member(a + b * x_target)

        # Weighted quadratic (weights ramp 0.5 -> 1.0 oldest to newest);
        # a positive x^2 coefficient means accelerating growth, which the
        # launch curve does not show, so refit linear instead.
        ws = [0.5 + 0.5 * i / (n - 1) for i in range(n)]
        quad = ols_quad(xs, ys, ws)
        if quad is None:
            resid = 0.3
        else:
            coeffs, sse = quad
            if coeffs[2] > 0:
                a2, b2 = ols_line(xs, ys, ws)
                members["quad"] = clamp_member(a2 + b2 * x_target)
            else:
                members["quad"] = clamp_member(
                    coeffs[0] + coeffs[1] * x_target + coeffs[2] * x_target * x_target
                )
            resid = math.sqrt(sse / max(1, n - 3)) if n > 3 else math.sqrt(sse / n)

        # Saturating exponential: logit = Linf - A * exp(-k * t), A >= 0,
        # grid over k, smallest SSE wins.
        best = None  # (sse, linf, a, k)
        for k in SAT_EXP_KS:
            us = [math.exp(-k * t) for t in ts]
            a_u, b_u = ols_line(us, ys)  # y ~ Linf + b*u, b = -A
            a_coef = -b_u
            if a_coef < 0:
                a_coef = 0.0
                linf = sum(ys) / n
            else:
                linf = a_u
            sse = 0.0
            for u, y in zip(us, ys):
                sse += (y - (linf - a_coef * u)) ** 2
            if best is None or sse < best[0]:
                best = (sse, linf, a_coef, k)
        _, linf, a_coef, k_best = best
        members["satexp"] = clamp_member(linf - a_coef * math.exp(-k_best * t_target))

        wsum = sum(WEIGHTS[k] for k in members)
        med_logit = sum(WEIGHTS[k] * members[k] for k in members) / wsum

    # Refresh-due shift: a page that has sat unchanged for longer than its
    # own refresh cadence is due a refresh, and a refresh drops the rates.
    shift = -0.7 * stale_factor * math.exp(-horizon_days / 0.5)
    med_logit = med_logit + shift

    # --- spread -----------------------------------------------------------
    dx = max(0.0, x_target - x_last)
    sigma = math.sqrt(0.35 ** 2 + resid ** 2 + (0.35 * dx) ** 2 + 0.10 * horizon_days)
    sigma = min(2.2, max(0.9, sigma))
    if n < 3:
        sigma = max(sigma, 1.2)

    return quantiles(med_logit, sigma, stale_factor, horizon_days)


def quantiles(med_logit, sigma, stale_factor, horizon_days):
    lower_mult = 1.0 + 0.4 * stale_factor * math.exp(-horizon_days)
    qs = []
    prev = None
    for z in Z:
        mult = lower_mult if z < 0 else 1.0
        q = round4(sigmoid(med_logit + z * sigma * mult))
        q = min(100.0, max(0.0, q))
        if prev is not None and q < prev:
            q = prev
        qs.append(q)
        prev = q
    return qs


def forecast_case(case, name):
    horizon_days = case["horizon_minutes"] / 1440
    target = case["target"]
    t_target = hours_since_epoch0(target)
    x_target = math.log1p(t_target)
    at_hours = hours_since_epoch0(case["at"])
    history = case.get("history") or []
    trophies = case["trophies"]

    rows = []
    for i, trophy in enumerate(trophies):
        points = []
        prev_key = None
        for read in history:
            rates = read.get("rates") or []
            if i >= len(rates):
                continue
            rate = rates[i]
            if rate is None:
                continue
            n_players = read.get("players_tracked")
            key = (rate, n_players)
            if key == prev_key:
                continue
            prev_key = key
            if n_players:
                implied = rate * n_players / 100.0
                smooth = (implied + 0.5) / (n_players + 1) * 100
            else:
                smooth = rate
            t = hours_since_epoch0(read["at"])
            points.append((t, math.log1p(t), logit(smooth)))
        qs = forecast_trophy(points, x_target, t_target, horizon_days, at_hours)
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
