"""Minute-by-minute quantile forecaster for Polymarket's "Next CEO of NVIDIA?"
event (make.fun polymarket-nvidia-ceo-forecast-01).

Reads each case file from the input directory, and for every market writes
240 rows of five quantiles (0.05/0.25/0.5/0.75/0.95) of the Yes price at each
minute after the checkpoint.

Method (per market, from its own history only):
  * The predictive centre is persistence: the last observed price. Every
    centre variant tried (trailing means, EWMA, trend extrapolation,
    mean-reversion pulls) scored worse on the practice cases than simply
    staying at the last price.
  * The spread comes from the market's own recent past. For two lookback
    windows (240 and 720 minutes) and every horizon h = 1..240, take the
    empirical quantiles of the h-minute price changes observed in the
    window, average the two windows, and re-centre the quantiles on their
    own median so the spread is a pure dispersion estimate. This captures
    the step-like behaviour of these prices (most minutes unchanged, rare
    jumps) far better than a single fixed normal scale: quiet markets get
    narrow bands, jumpy markets wide ones, and the bands are asymmetric
    when the market's own moves are.
  * The empirical bands are blended with a fixed-scale normal band
    (sigma 0.0025 per sqrt-minute, the house baseline's scale): the inner
    quantiles keep 90% of the empirical band, the outer quantiles 70%.
    The blend is jump insurance: a short recent window can miss the next
    jump, and the normal band keeps the tails honest. The median is never
    blended; it stays at the last price.
  * Markets with fewer than 120 prices of history skip the empirical
    bands entirely and use the normal band alone: with that little data
    the empirical quantiles are noise.

Deterministic: a pure function of the input file. Standard library only,
no network, no clock, no randomness.
"""

import json
import math
import os
import sys

# Standard normal quantiles at the five required levels.
Z_SCORES = (-1.6448536269514722, -0.6744897501960817, 0.0,
            0.6744897501960817, 1.6448536269514722)
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)

BASE_SIGMA = 0.0025       # per sqrt-minute scale of the insurance band
WINDOWS = (240, 720)      # lookback windows for the empirical bands
MIN_SAMPLES = 8           # minimum h-step changes needed for a quantile
INNER_KEEP = 0.9          # empirical weight for the 25%/75% quantiles
OUTER_KEEP = 0.7          # empirical weight for the 5%/95% quantiles
SHORT_HISTORY = 120       # below this many prices, use the normal band only


def _quantile(ordered, tau):
    """Linear-interpolation quantile of an already sorted sequence."""
    n = len(ordered)
    if n == 0:
        return 0.0
    pos = tau * (n - 1)
    lo = int(pos)
    hi = lo + 1 if lo + 1 < n else lo
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _change_quantiles(series, horizon):
    """Quantiles of the h-minute changes inside one window.

    Returns None when the window holds too few h-step changes.
    """
    count = len(series) - horizon
    if count < MIN_SAMPLES:
        return None
    changes = sorted(series[i + horizon] - series[i] for i in range(count))
    return [_quantile(changes, tau) for tau in LEVELS]


def _market_rows(prices, horizon):
    """Forecast rows for one market: `horizon` rows of five quantiles."""
    last = prices[-1]
    if len(prices) < SHORT_HISTORY:
        # Too little history for empirical bands to mean anything (the
        # practice has markets with as few as 19 prices): fall back to
        # the fixed-scale normal band around the last price.
        rows = []
        for h in range(1, horizon + 1):
            root_h = math.sqrt(h)
            row = [min(1.0, max(0.0, last + z * BASE_SIGMA * root_h))
                   for z in Z_SCORES]
            rows.append([round(v, 6) for v in row])
        return rows
    # Empirical quantile table per window: table[w][h] is the five
    # quantiles of the h-minute changes, or None when unavailable.
    tables = []
    for width in WINDOWS:
        window = prices[-width:] if len(prices) > width else prices
        tables.append([_change_quantiles(window, h)
                       for h in range(1, horizon + 1)])
    rows = []
    for h in range(1, horizon + 1):
        available = [t[h - 1] for t in tables if t[h - 1] is not None]
        if available:
            merged = [sum(q[k] for q in available) / len(available)
                      for k in range(5)]
        else:
            merged = [0.0] * 5
        centre = merged[2]
        root_h = math.sqrt(h)
        row = []
        for k in range(5):
            empirical = last + (merged[k] - centre)
            normal = min(1.0, max(0.0, last + Z_SCORES[k] * BASE_SIGMA * root_h))
            if k == 2:
                value = last
            elif k in (0, 4):
                value = OUTER_KEEP * empirical + (1.0 - OUTER_KEEP) * normal
            else:
                value = INNER_KEEP * empirical + (1.0 - INNER_KEEP) * normal
            row.append(value)
        # Clamp to [0, 1] and enforce non-decreasing quantiles.
        row = [min(1.0, max(0.0, v)) for v in row]
        for k in range(1, 5):
            if row[k] < row[k - 1]:
                row[k] = row[k - 1]
        rows.append([round(v, 6) for v in row])
    return rows


def forecast_case(case):
    horizon = int(case.get("horizon_minutes", 240))
    forecasts = []
    for market in case["markets"]:
        forecasts.append({
            "id": market["id"],
            "q": _market_rows(market["prices"], horizon),
        })
    return {"forecasts": forecasts}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(output_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(input_dir) if n.endswith(".in"))
    for name in names:
        with open(os.path.join(input_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        result = forecast_case(case)
        out_name = name[:-3] + ".out"
        with open(os.path.join(output_dir, out_name), "w",
                  encoding="utf-8") as fh:
            json.dump(result, fh, separators=(",", ":"))
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
