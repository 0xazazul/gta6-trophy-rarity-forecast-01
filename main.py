"""GTA VI trophy-rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

Reads every /input/<id>.in (makefun/gta6-trophies-case/v1) and writes
/output/<id>.out: five quantiles (5/25/50/75/95%) of each trophy's Exophase
earn rate at the case's target. Python 3.13, standard library only, no
network, deterministic.

Model — refresh-aware dilution (see README.md for the full reasoning):

* Exophase does not update continuously: between its refreshes the page is
  frozen, and at a refresh the tracked-player count jumps and every rate
  moves toward the earn rate among the newly tracked players. So the only
  informative history is the sequence of refreshes, not the frozen reads.
* A refresh is detected from consecutive reads (players_tracked jumps, or
  any rate changes). For each refresh and trophy, the new players' earn
  fraction is recovered exactly from the counts:
      f = (r2*N2 - r1*N1) / (N2 - N1)
  and pooled across refreshes with recency weights.
* The median projects that dilution forward: over the expected number of
  further refreshes before the target (horizon / observed refresh cadence)
  the tracked base grows by the median per-refresh influx, and the rate is
  the player-weighted average of the current rate and f. As the base grows,
  each later refresh dilutes less, so the rate asymptotes to f instead of
  falling forever — the key structural fact a log-linear trend would miss.
* When the history holds no refresh (launch night), the median stays at the
  last rate (persistence) and the spread carries the refresh risk instead:
  a mixture over the number of refreshes before the target, where one
  refresh dilutes a non-universal trophy roughly threefold (new joiners at
  a launch refresh hold far fewer trophies than the earliest syncers; the
  trophy held by ~everyone stays ~100%).
* The spread is the mixture's own quantiles, computed on a fixed logit
  grid: a random-walk term (0.5*sqrt(days), the house scale), uncertainty
  in f, and per-refresh growth uncertainty, combined with Poisson refresh
  counts. Nothing is fitted to any practice answer — there are none in the
  kit — and no practice date, rate or value is hardcoded.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# --- priors / constants (all stated, none fitted to a practice answer) ---
SIGMA_RW_PER_SQRT_DAY = 0.5   # logit random-walk scale, the house scale
CADENCE_PRIOR_MIN = 300.0     # refresh cadence assumed with no refresh seen
                              # ("refreshes hours apart")
GROWTH_PRIOR = 4.0            # one unseen refresh ~quadruples tracked players
ALPHA_PRIOR = 0.10            # new joiners earn a non-universal trophy at
                              # ~10% of the tracked rate (launch influx)
F_HALF_LIFE_H = 24.0          # recency half-life for pooling f across refreshes
UNIVERSAL_AT = 99.0           # a trophy held by >=99% stays put (everyone earns it)
GRID_STEP = 0.05              # logit grid step for the mixture quantiles
MAX_REFRESHES = 48            # cap on Poisson components (bounds runtime)


def clamp_rate(x):
    return min(99.99, max(0.01, float(x)))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up (monotone, so ordering is preserved)."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def median(values):
    v = sorted(values)
    n = len(v)
    if n == 0:
        return None
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def norm_cdf(x, mu, sigma):
    return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2.0))))


def poisson_weights(lam, cap=MAX_REFRESHES):
    """Poisson pmf for k = 0..K, truncated once the tail is negligible."""
    if lam <= 0:
        return [1.0]
    weights = []
    total = 0.0
    p = math.exp(-lam)
    k = 0
    while True:
        weights.append(p)
        total += p
        if total >= 0.9995 or k >= cap:
            break
        k += 1
        p *= lam / k
    return [w / total for w in weights]


def mixture_quantiles(components, levels):
    """Quantiles of a mixture of normals (given as (weight, mu, sigma) in
    logit space), by inverting the CDF on a fixed grid. Deterministic."""
    lo = min(mu - 6.0 * sig for _, mu, sig in components)
    hi = max(mu + 6.0 * sig for _, mu, sig in components)
    n = int((hi - lo) / GRID_STEP) + 1
    xs = [lo + i * GRID_STEP for i in range(n + 1)]
    cdf = []
    for x in xs:
        c = 0.0
        for w, mu, sig in components:
            c += w * norm_cdf(x, mu, sig)
        cdf.append(c)
    out = []
    for level in levels:
        if level <= cdf[0]:
            out.append(xs[0])
            continue
        if level >= cdf[-1]:
            out.append(xs[-1])
            continue
        j = 0
        while cdf[j + 1] < level:
            j += 1
        c0, c1 = cdf[j], cdf[j + 1]
        frac = 0.0 if c1 == c0 else (level - c0) / (c1 - c0)
        out.append(xs[j] + frac * (xs[j + 1] - xs[j]))
    return out


def detect_refreshes(history):
    """Refreshes: consecutive reads where the tracked count or any rate moved.
    Returns [(time, n_before, n_after, rates_before, rates_after), ...]."""
    events = []
    for prev, cur in zip(history, history[1:]):
        n0, n1 = prev["players_tracked"], cur["players_tracked"]
        r0, r1 = prev["rates"], cur["rates"]
        changed = n0 != n1
        if not changed:
            for a, b in zip(r0, r1):
                if a is not None and b is not None and abs(b - a) > 0.005:
                    changed = True
                    break
        if changed:
            events.append((parse_time(cur["at"]), n0, n1, r0, r1))
    return events


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_min = float(case.get("horizon_minutes") or 0)
    horizon_days = max(horizon_min, 0.0) / 1440.0
    sigma_rw = SIGMA_RW_PER_SQRT_DAY * math.sqrt(horizon_days)

    # Last non-null rate per trophy (fall back to 1% if a trophy was never read).
    last_rate = []
    for i in range(len(trophies)):
        rate = None
        for read in reversed(history):
            if read["rates"][i] is not None:
                rate = read["rates"][i]
                break
        last_rate.append(1.0 if rate is None else float(rate))

    if not history:
        # No information at all: wide, centred forecast; never crash.
        rows = []
        for t in trophies:
            qs = [round4(inv_logit(logit(1.0) + z * 2.0)) for z in Z]
            rows.append({"id": t["id"], "q": qs})
        return {"trophies": rows}

    n_last = max(1, int(history[-1]["players_tracked"] or 1))
    t_first = parse_time(history[0]["at"])
    t_last = parse_time(history[-1]["at"])
    refreshes = detect_refreshes(history)

    # --- refresh cadence and per-refresh influx ---
    if len(refreshes) >= 2:
        gaps = [
            (refreshes[i + 1][0] - refreshes[i][0]).total_seconds() / 60.0
            for i in range(len(refreshes) - 1)
        ]
        cadence = max(30.0, median(gaps) or CADENCE_PRIOR_MIN)
    elif len(refreshes) == 1:
        cadence = max(30.0, (refreshes[0][0] - t_first).total_seconds() / 60.0)
    else:
        cadence = CADENCE_PRIOR_MIN

    lam = min(horizon_min / cadence, 40.0) if horizon_min > 0 else 0.0
    weights_k = poisson_weights(lam)

    if refreshes:
        step_n = median([ev[2] - ev[1] for ev in refreshes if ev[2] > ev[1]])
        step_n = float(step_n) if step_n else float(n_last)
        sigma_g = 0.15
    else:
        step_n = (GROWTH_PRIOR - 1.0) * n_last
        sigma_g = 0.45

    # --- per-trophy new-player earn fraction f ---
    total_weight = 0.0
    if refreshes:
        total_weight = sum(
            max(0, ev[2] - ev[1])
            * (0.5 ** ((t_last - ev[0]).total_seconds() / 3600.0 / F_HALF_LIFE_H))
            for ev in refreshes
        )
    # Uncertainty in f only matters if a refresh actually happens: the
    # k = 0 component (no further refresh) keeps just the random-walk term.
    sigma_f = 0.35 if not refreshes else 0.25 + 0.25 * 250.0 / (total_weight + 250.0)

    rows = []
    for i, trophy in enumerate(trophies):
        r = last_rate[i]
        if r >= UNIVERSAL_AT:
            f = r  # universal trophy: new joiners hold it too
        elif refreshes:
            num = 0.0
            den = 0.0
            for (t_ev, n0, n1, r0, r1) in refreshes:
                if n1 <= n0 or r0[i] is None or r1[i] is None:
                    continue
                f_j = (float(r1[i]) * n1 - float(r0[i]) * n0) / (n1 - n0)
                f_j = min(100.0, max(0.0, f_j))
                w = (n1 - n0) * (
                    0.5 ** ((t_last - t_ev).total_seconds() / 3600.0 / F_HALF_LIFE_H)
                )
                num += w * f_j
                den += w
            f = (num / den) if den > 0 else ALPHA_PRIOR * r
            f = min(99.99, max(0.01, f))
        else:
            f = ALPHA_PRIOR * r

        # --- mixture over refresh counts before the target ---
        components = []
        for k, w_k in enumerate(weights_k):
            n_k = min(n_last + k * step_n, 100.0 * n_last)
            if n_k <= n_last:
                r_k = r
            else:
                r_k = (r * n_last + f * (n_k - n_last)) / n_k
            mu_k = logit(r_k)
            if k == 0:
                var = sigma_rw * sigma_rw
            else:
                var = sigma_rw * sigma_rw + sigma_f * sigma_f + (sigma_g ** 2) * k
            components.append((w_k, mu_k, math.sqrt(var)))
        q_logits = mixture_quantiles(components, LEVELS)
        qs = [round4(min(100.0, max(0.0, inv_logit(lg)))) for lg in q_logits]
        for j in range(1, 5):  # enforce non-decreasing after rounding
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
