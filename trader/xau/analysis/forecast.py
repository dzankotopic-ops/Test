"""Four price models for gold from now to a horizon, each 10,000 paths on hourly steps, plus their equal-weight mix.

A  hour-of-day bootstrap, last 60 trading days, drift removed
B  hour-of-day bootstrap, last 250 trading days, drift removed
C  EWMA volatility with fat tails (Student-t, 4 df), hour-of-day volatility profile, no drift
D  regime analogs: whole stretches of hourly bars that followed past days like today (below the 50-day average, down over 20 days)
Usage: python3 forecast.py GC-2y1h.json technicals.json "2026-11-06 16:00" out.json
"""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from strategy_loop import load, daily_bars, NY
N, SEED = 10000, 19
h = load(sys.argv[1]); tech = json.load(open(sys.argv[2])); d = daily_bars(h)
end = pd.Timestamp(sys.argv[3], tz=NY)
h["r"] = np.log(h.c).diff(); h = h.iloc[1:]
h = h[h.index.to_series().diff().shift(-1).fillna(pd.Timedelta(hours=1)) >= pd.Timedelta(0)]
gaps = h.index.to_series().diff() <= pd.Timedelta(hours=2)                       # keep returns across the 17:00 pause, not weekends
hr = h[gaps]
start_t, start_p = h.index[-1], float(h.c.iloc[-1])
trade_hours = sorted(hr.index.hour.value_counts()[lambda s: s > s.max() * 0.5].index)
steps = [t for t in pd.date_range(start_t + pd.Timedelta(hours=1), end, freq="h") if t.weekday() < 5 and t.hour in trade_hours
         and not (t.weekday() == 4 and t.hour >= 17)]
step_h = np.array([t.hour for t in steps])
rng = np.random.default_rng(SEED)
days = sorted(set(hr.day)); 
def hod_bootstrap(n_days):
    pool = hr[hr.day.isin(days[-n_days:])]
    by = {k: (g.r.values - g.r.mean()) for k, g in pool.groupby(pool.index.hour)}
    return np.stack([rng.choice(by[k], N) for k in step_h], axis=1)
A = hod_bootstrap(60); B = hod_bootstrap(250)
# C: EWMA daily vol (lambda 0.94) spread over the hours by their share of variance
dr = np.log(d.c).diff().dropna(); var = dr.iloc[:20].var()
for x in dr.iloc[20:]: var = 0.94 * var + 0.06 * x * x
prof = hr[hr.day.isin(days[-250:])].groupby(hr[hr.day.isin(days[-250:])].index.hour).r.var()
share = prof / prof.sum()
sig = np.sqrt(var * share.reindex(step_h).values)
t4 = rng.standard_t(4, size=(N, len(steps))) / np.sqrt(2.0)                     # unit variance
C = t4 * sig
# D: analog days - same regime as now, then copy the next len(steps) hourly returns as one block
sma50 = d.c.rolling(50).mean(); r20 = d.c / d.c.shift(20) - 1
analog = d.index[(d.c < sma50) & (r20 < 0)][:-1]
hlist = hr.r.values; hidx = {day: i for i, day in enumerate(hr.day)}
starts = [hidx[x.date()] for x in analog if x.date() in hidx]
starts = [s for s in starts if s + len(steps) < len(hlist)]
pick = rng.choice(starts, N)
D_ = np.stack([hlist[s + 1:s + 1 + len(steps)] for s in pick])
models = {"A": A, "B": B, "C": C, "D": D_}
names = {"A": "Tageszeit-Bootstrap 60 Tage", "B": "Tageszeit-Bootstrap 250 Tage", "C": "EWMA-Volatilität mit dicken Rändern", "D": "Regime-Analogien (Tage wie heute)"}
paths = {k: start_p * np.exp(np.cumsum(v, axis=1)) for k, v in models.items()}
paths["ENS"] = np.concatenate([paths[k] for k in "ABCD"])
Q = (5, 25, 50, 75, 95)
lv = {"Widerstand 4.097 (5-Tage-Hoch, Ausstieg Short)": tech["exit5_high"], "Unterstützung 3.879 (52-Wochen-Tief)": tech["low_52w"], "4.000": 4000.0}
res = {"start_time": start_t.isoformat(), "start_price": round(start_p, 2), "end_time": steps[-1].isoformat(), "steps": len(steps),
       "times": [t.isoformat() for t in steps], "seed": SEED, "replays_per_model": N, "analog_days": len(starts), "names": names, "models": {}}
for k, P in paths.items():
    fin = P[:, -1]; mx, mn = P.max(axis=1), P.min(axis=1)
    hit_up = np.argmax(P >= tech["exit5_high"], axis=1); hit_up = np.where(mx >= tech["exit5_high"], hit_up, 10**9)
    hit_dn = np.argmax(P <= tech["low_52w"], axis=1); hit_dn = np.where(mn <= tech["low_52w"], hit_dn, 10**9)
    res["models"][k] = {
        "bands": np.round(np.percentile(P, Q, axis=0).T, 2).tolist(),
        "final": dict(zip(["p5", "p25", "median", "p75", "p95"], np.round(np.percentile(fin, Q), 2).tolist())),
        "mean_final": round(float(fin.mean()), 2),
        "p_below_start": round(float((fin < start_p).mean()), 4),
        "p_touch_4097": round(float((mx >= tech["exit5_high"]).mean()), 4),
        "p_touch_3879": round(float((mn <= tech["low_52w"]).mean()), 4),
        "p_touch_4000": round(float((mx >= 4000).mean()), 4),
        "p_3879_first": round(float((hit_dn < hit_up).mean()), 4), "p_4097_first": round(float((hit_up < hit_dn).mean()), 4),
        "short_avg_pct": round(float((start_p / fin - 1).mean() * 100), 3),
    }
    m = res["models"][k]; print(k, m["final"], "below now", m["p_below_start"], "3879 first", m["p_3879_first"], "4097 first", m["p_4097_first"])
json.dump(res, open(sys.argv[4], "w"), indent=1)
