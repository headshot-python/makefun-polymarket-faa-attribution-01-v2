# Forecast for make.fun polymarket-faa-attribution-01.
#
# What it does: for every market in a case, predict the Yes price at each of
# the next 240 minutes as five quantiles (0.05 / 0.25 / 0.5 / 0.75 / 0.95).
#
# How it works (empirical quantiles around a persistent median):
#   * Prediction-market prices behave like martingales, so the median
#     forecast is simply the last observed price -- no trend is imposed.
#   * The width of the forecast band is learned from the market itself:
#     look at the last 8 hours of prices, take all h-minute price changes
#     for a coarse set of horizons h, and read off their empirical
#     quantiles. Those quantiles are interpolated in log(h) to every
#     horizon 1..240, then shifted so the 0.5 quantile sits exactly on the
#     last price. An 8-hour lookback (rather than a longer one) tracks the
#     market's current volatility regime instead of averaging over stale
#     history, which matters when news moves prices in bursts.
#   * The inner band (25%/75%) is narrowed by a factor of 0.5: prices
#     spend most minutes unmoved, so the honest interquartile range is
#     about half the raw empirical one; the outer 5%/95% quantiles keep
#     their full width so jumps stay inside the band. Both the lookback
#     and the shrink were chosen on the kit's practice cases (see README).
#   * A horizon with too few observed changes falls back to the house
#     baseline spread (0.0025 per square-root minute, normal quantiles).
#   * Every quantile is clamped to [0, 1], rounded to 6 decimals, and each
#     row is forced non-decreasing. No randomness anywhere: two runs of the
#     same input produce byte-identical output.
#
# Python 3.13, standard library only. Reads every *.in file in
# $MAKEFUN_INPUT_DIR (default /input) and writes one <id>.out per case to
# $MAKEFUN_OUTPUT_DIR (default /output).

import json
import math
import os
from pathlib import Path

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# How much recent history the empirical quantiles are measured on.
LOOKBACK_MINUTES = 480
# Shrink factor applied to the 25% and 75% quantiles only.
INNER_BAND_SHRINK = 0.50
# Coarse horizons (minutes) where empirical change quantiles are measured.
HORIZON_GRID = (1, 2, 3, 4, 6, 9, 13, 19, 27, 39, 56, 80, 115, 165, 240)
# House baseline spread, used when a horizon has too little history.
BASELINE_SIGMA = 0.0025
# Fewest h-minute changes accepted before falling back to the baseline.
MIN_CHANGES = 20
# Output size cap from the rules (bytes).
MAX_OUTPUT_BYTES = 1_048_576


def normal_quantile(p):
    """Inverse of the standard normal CDF (Acklam's rational fit).

    Accurate to about 1e-9, which is far finer than the 6-decimal rounding
    applied to the forecasts.
    """
    low = 0.02425
    high = 1.0 - low
    a = (-39.69683028665376, 220.9460984245205, -275.9285104469687,
         138.3577518672690, -30.66479806614716, 2.506628277459239)
    b = (-54.47609879822406, 161.5858368580409, -155.6989798598866,
         66.80131188771972, -13.28068155288572)
    c = (-0.007784894002430293, -0.3223964580411365, -2.400758277161838,
         -2.549732539343734, 4.374664141464968, 2.938163982698783)
    d = (0.007784695709041462, 0.3224671290700398, 2.445134137142996,
         3.754408661907416)
    if p < low:
        q = math.sqrt(-2.0 * math.log(p))
        num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        return num / den
    if p <= high:
        q = p - 0.5
        r = q * q
        num = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        den = (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        return num / den
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
    den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    return -num / den


def sample_quantile(sorted_values, tau):
    """tau-quantile of a sorted sample, linear interpolation between ranks."""
    n = len(sorted_values)
    pos = tau * (n - 1)
    lower = int(pos)
    upper = min(n - 1, lower + 1)
    frac = pos - lower
    return sorted_values[lower] * (1.0 - frac) + sorted_values[upper] * frac


def clamp_round(value):
    """Clamp to [0, 1] and round to 6 decimals; never returns -0.0."""
    rounded = round(min(1.0, max(0.0, value)) * 1_000_000) / 1_000_000
    return rounded + 0.0


def horizon_quantiles(prices, taus):
    """Empirical change quantiles at each grid horizon, keyed by horizon.

    Returns {h: {tau: change}} where change is measured from a price h
    minutes earlier. Horizons with too few changes get the baseline spread
    (normal quantiles times BASELINE_SIGMA * sqrt(h)) instead.
    """
    window = prices[-LOOKBACK_MINUTES:]
    count = len(window)
    wanted = set(taus) | {0.5}
    out = {}
    for h in HORIZON_GRID:
        if count > h and (count - h) >= MIN_CHANGES:
            changes = sorted(window[i + h] - window[i] for i in range(count - h))
            out[h] = {tau: sample_quantile(changes, tau) for tau in wanted}
        else:
            out[h] = {tau: normal_quantile(tau) * BASELINE_SIGMA * math.sqrt(h)
                      for tau in wanted}
    return out


def interpolate_quantiles(grid, horizon):
    """Interpolate the grid quantiles to one horizon, in log(h)."""
    points = sorted(grid)
    logs = [math.log(h) for h in points]
    target = math.log(horizon)
    if target <= logs[0]:
        return grid[points[0]]
    if target >= logs[-1]:
        return grid[points[-1]]
    idx = 0
    while logs[idx + 1] < target:
        idx += 1
    weight = (target - logs[idx]) / (logs[idx + 1] - logs[idx])
    lo, hi = grid[points[idx]], grid[points[idx + 1]]
    return {tau: lo[tau] * (1.0 - weight) + hi[tau] * weight for tau in lo}


def forecast_market(prices, taus, horizon):
    """One market's forecast: `horizon` rows of len(taus) quantiles."""
    last_price = prices[-1]
    grid = horizon_quantiles(prices, taus)
    rows = []
    for h in range(1, horizon + 1):
        at_h = interpolate_quantiles(grid, h)
        median_change = at_h[0.5]
        row = []
        for tau in taus:
            spread = at_h[tau] - median_change
            if tau != 0.5 and 0.2 <= tau <= 0.8:
                spread *= INNER_BAND_SHRINK
            row.append(clamp_round(last_price + spread))
        # Rows must be non-decreasing (equal values are fine).
        for j in range(1, len(row)):
            if row[j] < row[j - 1]:
                row[j] = row[j - 1]
        rows.append(row)
    return rows


def run_case(case):
    """Forecast every market of one case, in the case's market order."""
    taus = list(case["quantiles"])
    horizon = int(case["horizon_minutes"])
    forecasts = []
    for market in case["markets"]:
        prices = market["prices"]
        if not prices:
            raise ValueError("market %s has no prices" % market["id"])
        forecasts.append({
            "id": market["id"],
            "q": forecast_market(prices, taus, horizon),
        })
    return {"forecasts": forecasts}


def main():
    in_dir = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    case_files = sorted(p for p in in_dir.iterdir() if p.name.endswith(".in"))
    for case_file in case_files:
        case_id = case_file.name[: -len(".in")]
        case = json.loads(case_file.read_text(encoding="utf-8"))
        text = json.dumps(run_case(case), separators=(",", ":")) + "\n"
        if len(text.encode("utf-8")) > MAX_OUTPUT_BYTES:
            raise ValueError("forecast for %s exceeds 1 MiB" % case_id)
        (out_dir / (case_id + ".out")).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
