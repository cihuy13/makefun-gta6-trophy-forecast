#!/usr/bin/env python3
"""Forecast GTA VI (PS5) trophy earn rates on Exophase.

Model: counts + dilution + saturating growth.

  * Per trophy, convert the page's rates to earner counts
    (earners = rate / 100 * players_tracked) and fit each trophy's
    earner count against log-time with a saturating curve. Fit
    players_tracked the same way. Project both to the target; the
    median rate is projected earners / projected players. Working in
    counts matters because the player base grows and dilutes rates:
    a rate can fall while its earners still grow.
  * Two candidate shapes per series, picked by least squares:
    a saturating exponential in log-time, A * (1 - exp(-k * x)),
    and an exponential in log-time, exp(a + b * x). Near-ties are
    broken toward the projection closest to the last observed value,
    so a flat series forecasts flat (persistence).
  * Earners and players are cumulative, so their projections never
    fall below the last observed value, and earners never exceed
    players.
  * Quantiles are logit-normal around the median. The spread combines
    the fit's residual scatter in logit space, a horizon term growing
    with sqrt(horizon in days), a small-count term (rare trophies are
    more volatile), and an extra term when the page never moved at
    all (a refresh is then overdue and the rates can jump).
  * The median is blended 50/50 in logit space with a last-24h
    velocity trend (the house curve baseline's fit: a least-squares
    line of logit(rate) against log-time over the last 24 hours of
    reads), so recent movement steers the forecast while the counts
    model anchors it; the trend fit's residuals join the spread's
    residual pool.

Only the Python standard library is used. No network, no randomness:
two runs on the same case produce byte-identical output.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

EPOCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Standard-normal quantiles for the levels the cases use.
Z_TABLE = {
    0.05: -1.6448536269514722,
    0.25: -0.6744897501960817,
    0.5: 0.0,
    0.75: 0.6744897501960817,
    0.95: 1.6448536269514722,
}


def norm_ppf(p):
    """Inverse standard-normal CDF (Acklam's approximation)."""
    if p in Z_TABLE:
        return Z_TABLE[p]
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285984469534e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def parse_time(text):
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def log_time(dt):
    """x = ln(1 + hours since 2026-11-19T00:00Z), floored at 0."""
    hours = (dt - EPOCH).total_seconds() / 3600.0
    return max(0.0, math.log(max(1.0 + hours, 0.05)))


def logit(p):
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def _sse(ys, preds):
    return sum((y - p) * (y - p) for y, p in zip(ys, preds))


def fit_saturating(xs, ys, x_target):
    """Least-squares A * (1 - exp(-k * x)) over a log-spaced k grid.

    Returns {"target", "sse", "predict"} or None. Near-ties in SSE are
    broken toward the projection closest to the last observed value.
    """
    grid = []
    k = 0.03
    while k <= 8.0:
        grid.append(k)
        k *= 1.09
    cands = []
    for kk in grid:
        fs = [1.0 - math.exp(-kk * x) for x in xs]
        denom = sum(f * f for f in fs)
        if denom <= 0:
            continue
        a = sum(y * f for y, f in zip(ys, fs)) / denom
        if a < 0:
            a = 0.0
        preds = [a * f for f in fs]
        sse = _sse(ys, preds)
        y_t = a * (1.0 - math.exp(-kk * x_target))
        cands.append((sse, abs(y_t - ys[-1]), y_t, a, kk))
    if not cands:
        return None
    best_sse = min(c[0] for c in cands)
    tol = best_sse * 0.01 + 1e-9
    near = [c for c in cands if c[0] <= best_sse + tol]
    near.sort(key=lambda c: c[1])
    _sse_v, _gap, y_t, a, kk = near[0]
    return {
        "target": y_t,
        "sse": near[0][0],
        "predict": lambda x, a=a, kk=kk: a * (1.0 - math.exp(-kk * x)),
    }


def fit_exponential(xs, ys, x_target):
    """Least-squares exp(a + b * x) fitted on log y (y > 0 only)."""
    pts = [(x, math.log(y)) for x, y in zip(xs, ys) if y > 0]
    if len(pts) < 2:
        return None
    n = float(len(pts))
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    if sxx <= 0:
        return None
    b = sum((p[0] - mx) * (p[1] - my) for p in pts) / sxx
    b = min(max(b, 0.0), 2.0)
    a = my - b * mx
    preds = [math.exp(a + b * x) for x in xs]
    sse = _sse(ys, preds)
    return {
        "target": math.exp(a + b * x_target),
        "sse": sse,
        "predict": lambda x, a=a, b=b: math.exp(a + b * x),
    }


def fit_series(xs, ys, x_target):
    """Project a cumulative series to x_target.

    Returns (y_target, predict_fn). The projection never falls below
    the last observed value and is capped at 10x it, so a runaway
    exponential cannot explode the forecast.
    """
    if not xs:
        return None, None
    cands = []
    sat = fit_saturating(xs, ys, x_target)
    if sat is not None:
        cands.append(sat)
    ex = fit_exponential(xs, ys, x_target)
    if ex is not None:
        cands.append(ex)
    if not cands:
        return ys[-1], lambda x: ys[-1]
    best_sse = min(c["sse"] for c in cands)
    tol = best_sse * 0.01 + 1e-9
    near = [c for c in cands if c["sse"] <= best_sse + tol]
    near.sort(key=lambda c: abs(c["target"] - ys[-1]))
    pick = near[0]
    y_t = pick["target"]
    y_t = max(y_t, ys[-1])
    y_t = min(y_t, max(10.0 * max(ys[-1], 1.0), ys[-1]))
    return y_t, pick["predict"]


def fit_velocity_trend(reads, x_target, hours_checkpoint):
    """House curve baseline's trend: least-squares line of logit(rate)
    against log-time over the last 24 hours of reads.

    Returns (median_percent, sse, n) at x_target, or None when fewer
    than 3 usable reads exist in the window.
    """
    pts = []
    for rd in reads:
        rate = rd["rate"]
        if rate is None or rate <= 0 or rate >= 100:
            continue
        if rd["hours"] < hours_checkpoint - 24.0:
            continue
        pts.append((rd["x"], logit(rate / 100.0)))
    if len(pts) < 3:
        return None
    n = float(len(pts))
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    if sxx <= 0:
        pred = my
        sse = sum((p[1] - my) ** 2 for p in pts)
        return (100.0 * sigmoid(pred), sse, len(pts))
    b = sum((p[0] - mx) * (p[1] - my) for p in pts) / sxx
    a = my - b * mx
    pred = a + b * x_target
    sse = sum((p[1] - (a + b * p[0])) ** 2 for p in pts)
    pred = clamp(pred, logit(0.0001), logit(0.9999))
    return (100.0 * sigmoid(pred), sse, len(pts))


def forecast_trophy(trophy, reads, players_fit, x_target, horizon_days,
                    hours_checkpoint, levels):
    """Forecast one trophy's five quantiles.

    reads: list of {"x", "hours", "players", "rate"} oldest first,
    rate None where the read lacked the trophy. players_fit is
    (p_target, p_predict) for the shared players series.
    """
    p_target, p_predict = players_fit
    tier = trophy.get("tier", "bronze")

    # Last observed rate and earner count.
    last_rate = None
    last_earners = 0.0
    for rd in reads:
        if rd["rate"] is not None:
            last_rate = rd["rate"]
            if rd["players"]:
                last_earners = rd["rate"] / 100.0 * rd["players"]
    if last_rate is None:
        # No read ever listed this trophy: tier prior, very wide.
        prior = {"platinum": 0.1, "gold": 0.5, "silver": 1.0}.get(tier, 1.0)
        return quantiles(prior, 1.0, levels)

    # A trophy everyone already has stays (almost) universal.
    if last_rate >= 99.5:
        return quantiles(99.99, 0.03, levels)

    # Earner-count series for this trophy: (x, earners, observed rate).
    pts = []
    for rd in reads:
        if rd["rate"] is None or not rd["players"]:
            continue
        pts.append((rd["x"], rd["rate"] / 100.0 * rd["players"], rd["rate"]))

    static = False
    if pts:
        es = [p[1] for p in pts]
        static = (max(es) - min(es) == 0.0)
    if len(pts) < 2:
        # Too little to fit: persistence with a modest spread.
        return quantiles(last_rate, 0.45, levels)

    xs = [p[0] for p in pts]
    es = [p[1] for p in pts]
    e_target, e_predict = fit_series(xs, es, x_target)
    if e_target is None or p_target is None or p_target <= 0:
        return quantiles(last_rate, 0.45, levels)
    e_target = min(e_target, p_target)
    median = 100.0 * e_target / p_target
    median = clamp(median, 0.01, 99.99)

    # Last-24h velocity trend, blended 50/50 in logit space with the
    # counts-model median (the house curve baseline's fit).
    trend = fit_velocity_trend(reads, x_target, hours_checkpoint)
    if trend is not None:
        t_median, t_sse, t_n = trend
        median = 100.0 * sigmoid(
            0.5 * logit(median / 100.0) + 0.5 * logit(t_median / 100.0))
        median = clamp(median, 0.01, 99.99)

    # Residual scatter of the rate model, in logit space: compare each
    # read's observed rate with the model rate at that read's time.
    sq = []
    for x, _e, obs in pts:
        if obs <= 0 or obs >= 100:
            continue
        p_pred = p_predict(x)
        if p_pred <= 0:
            continue
        m_rate = clamp(100.0 * e_predict(x) / p_pred, 0.01, 99.99)
        sq.append((logit(obs / 100.0) - logit(m_rate / 100.0)) ** 2)
    sq_sum = sum(sq)
    sq_n = len(sq)
    if trend is not None:
        sq_sum += t_sse
        sq_n += t_n
    s_fit = math.sqrt(sq_sum / sq_n) if sq_n >= 2 else 0.0

    # Spread: fit scatter + horizon random walk + small-count term
    # (+ a stale-page term when nothing ever moved). The stale-page
    # term is wide on purpose: the one refresh the house saw took
    # players 638 -> 3,868, a ~6x dilution, so a page that never moved
    # can jump about 1.7 logit units when it finally refreshes. It
    # only fires on a fully static history, so a case with real
    # movement (the final) is unaffected.
    s_h = 0.35 * math.sqrt(max(horizon_days, 0.0))
    s_rare = 0.35 / math.sqrt(1.0 + max(last_earners, 0.0) / 5.0)
    s_static = 1.0 if static else 0.0
    s = math.sqrt(s_fit * s_fit + s_h * s_h + s_rare * s_rare + s_static * s_static)
    s = clamp(s, 0.30, 1.10)
    return quantiles(median, s, levels)


def quantiles(median, spread, levels):
    """Logit-normal quantiles for `median` (percent) and `spread`."""
    m = clamp(median, 0.01, 99.99)
    base = logit(m / 100.0)
    out = []
    prev = None
    for tau in levels:
        q = 100.0 * sigmoid(base + norm_ppf(tau) * spread)
        q = round(clamp(q, 0.01, 99.99), 4)
        if prev is not None and q < prev:
            q = prev
        out.append(q)
        prev = q
    return out


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history", [])
    horizon_days = case.get("horizon_minutes", 0) / 1440.0
    levels = case.get("levels", [0.05, 0.25, 0.5, 0.75, 0.95])
    x_target = log_time(parse_time(case["target"]))
    hours_checkpoint = (parse_time(case["at"]) - EPOCH).total_seconds() / 3600.0

    # Shared players series (all reads with a player count).
    px, pv = [], []
    for rd in history:
        p = rd.get("players_tracked")
        if p is None or p <= 0:
            continue
        px.append(log_time(parse_time(rd["at"])))
        pv.append(float(p))
    if px:
        p_target, p_predict = fit_series(px, pv, x_target)
    else:
        p_target, p_predict = None, None
    players_fit = (p_target, p_predict)

    # Per-trophy read lists, oldest first.
    rows = []
    for j, trophy in enumerate(trophies):
        reads = []
        for rd in history:
            rates = rd.get("rates") or []
            rate = rates[j] if j < len(rates) else None
            reads.append({
                "x": log_time(parse_time(rd["at"])),
                "hours": (parse_time(rd["at"]) - EPOCH).total_seconds() / 3600.0,
                "players": rd.get("players_tracked"),
                "rate": rate,
            })
        q = forecast_trophy(trophy, reads, players_fit, x_target,
                            horizon_days, hours_checkpoint, levels)
        rows.append({"id": trophy["id"], "q": q})
    return rows


def main(argv):
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input/"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output/"
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    for name in names:
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as f:
            case = json.load(f)
        rows = forecast_case(case)
        stem = os.path.splitext(name)[0]
        with open(os.path.join(out_dir, stem + ".out"), "w",
                  encoding="utf-8") as f:
            f.write(json.dumps({"trophies": rows}))
            f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
