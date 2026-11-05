"""Up or down from the live price: four models on 5-minute steps, several horizons, plus a check against history.

A  time-of-day bootstrap, 60 days, drift removed          (pure randomness, no opinion)
B  time-of-day bootstrap, 60 days, drift kept             (the recent trend continues)
C  EWMA volatility, Student-t tails, no drift              (current volatility level)
D  analog days: real 5-min sequences from the same clock time on past days that moved the same way as today so far
Usage: python3 nowcast.py GC-60d5m-full.json GC-2y1h.json PRICE "YYYY-MM-DD HH:MM" (Berlin time of PRICE) out.json
"""
import json, sys, zoneinfo
import numpy as np, pandas as pd
NY, DE = zoneinfo.ZoneInfo("America/New_York"), zoneinfo.ZoneInfo("Europe/Berlin")
N, SEED = 10000, 19
src5, src1h, price, at, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), pd.Timestamp(sys.argv[4], tz=DE), sys.argv[5]
r = json.load(open(src5))["chart"]["result"][0]; q = r["indicators"]["quote"][0]
b = pd.DataFrame({"c": q["close"]}, index=pd.to_datetime(r["timestamp"], unit="s", utc=True).tz_convert(NY)).dropna()
b = b[(b.index.second == 0) & (b.index.minute % 5 == 0)]
b["r"] = np.log(b.c).diff(); b["gap"] = b.index.to_series().diff()
b = b[b.gap <= pd.Timedelta(minutes=70)]                        # inside a session, or across the 17:00 pause
b["day"] = (b.index + pd.Timedelta(hours=6)).date; b["slot"] = b.index.hour * 12 + b.index.minute // 5
today = b.day.iloc[-1]; hist = b[b.day != today]
start = at.tz_convert(NY).floor("5min")
def close_on(n):                                                  # 16:00 New York, n weekdays after today
    day = start.normalize(); k = 0
    while k < n: day += pd.Timedelta(days=1); k += day.weekday() < 5
    return day.replace(hour=16, minute=0)
horizons = {"in 1 Stunde": start + pd.Timedelta(hours=1), "heute 22:00 Uhr": close_on(0), "morgen 22:00 Uhr": close_on(1)}
slots_ok = set(hist.slot.value_counts()[lambda s: s > s.max() * 0.5].index)
steps = [t for t in pd.date_range(start + pd.Timedelta(minutes=5), max(horizons.values()), freq="5min")
         if t.weekday() < 5 and (t.hour * 12 + t.minute // 5) in slots_ok]
slot_arr = np.array([t.hour * 12 + t.minute // 5 for t in steps]); hidx = {k: max(i for i, t in enumerate(steps) if t <= v) for k, v in horizons.items()}
rng = np.random.default_rng(SEED)
pools = {s: g.r.values for s, g in hist.groupby("slot")}
def near(s): return np.concatenate([pools.get((s + k) % 288, []) for k in range(-2, 3)])
P = {s: near(s) for s in set(slot_arr)}
mu = hist.r.mean()
A = np.stack([rng.choice(P[s] - mu, N) for s in slot_arr], axis=1)
B = np.stack([rng.choice(P[s], N) for s in slot_arr], axis=1)
dr = hist.groupby("day").r.sum(); var = dr.iloc[:10].var()
for x in dr.iloc[10:]: var = 0.94 * var + 0.06 * x * x
prof = hist.groupby("slot").r.var(); share = (prof / prof.sum()).reindex(slot_arr).values
C = rng.standard_t(4, size=(N, len(steps))) / np.sqrt(2.0) * np.sqrt(var * share)
# D: past days that were down at this clock time by a similar amount, then their real next sequence
prev_close = {d: g.c.iloc[-1] for d, g in hist.groupby("day")}; days = sorted(prev_close)
move_now = price / prev_close[days[-1]] - 1
seqs = []
rr = hist.r.values; ts = hist.index
pos_by_time = {t: i for i, t in enumerate(ts)}
for i, d in enumerate(days[1:], 1):
    t0 = pd.Timestamp(d).tz_localize(NY).replace(hour=start.hour, minute=start.minute)
    if start.hour >= 18: t0 -= pd.Timedelta(days=1)
    if t0 not in pos_by_time: continue
    k = pos_by_time[t0]; mv = hist.c.iloc[k] / prev_close[days[i - 1]] - 1
    if np.sign(mv) == np.sign(move_now) and k + len(steps) < len(rr): seqs.append(rr[k + 1:k + 1 + len(steps)])
D = np.stack([seqs[j] for j in rng.integers(0, len(seqs), N)])
models = {"A": A, "B": B, "C": C, "D": D}
names = {"A": "Reiner Zufall (Schwankung der letzten 60 Tage)", "B": "Trend läuft weiter (60 Tage inkl. Abwärtsdrift)",
         "C": "Aktuelle Volatilität mit dicken Rändern", "D": f"Ähnliche Tage (um diese Uhrzeit ebenfalls im {'Plus' if move_now > 0 else 'Minus'}, {len(seqs)} Tage)"}
paths = {k: price * np.exp(np.cumsum(v, axis=1)) for k, v in models.items()}
paths["ENS"] = np.concatenate([paths[k] for k in "ABCD"])
res = {"price": price, "at": at.isoformat(), "move_today_pct": round(move_now * 100, 2), "seed": SEED, "names": names,
       "times": [t.tz_convert(DE).isoformat() for t in steps], "horizons": {}, "touch": {}, "bands": {}}
for hz, i in hidx.items():
    res["horizons"][hz] = {k: {"p_up": round(float((p[:, i] > price).mean()), 4), "median": round(float(np.median(p[:, i])), 2),
                               "p5": round(float(np.percentile(p[:, i], 5)), 2), "p95": round(float(np.percentile(p[:, i], 95)), 2)} for k, p in paths.items()}
for dist in (20, 50):                                           # which comes first within the week: +dist or -dist
    for k, p in paths.items():
        up = np.where((p >= price + dist).any(1), np.argmax(p >= price + dist, 1), 10**9)
        dn = np.where((p <= price - dist).any(1), np.argmax(p <= price - dist, 1), 10**9)
        res["touch"].setdefault(str(dist), {})[k] = {"up_first": round(float((up < dn).mean()), 4), "down_first": round(float((dn < up).mean()), 4)}
for k, p in paths.items(): res["bands"][k] = np.round(np.percentile(p, (5, 25, 50, 75, 95), axis=0).T, 2).tolist()
# history check (2 years, hourly): after days like today (below 20- and 50-day average, RSI14 under 40), how often was gold higher?
sys.path.insert(0, __file__.rsplit("/", 1)[0]); from strategy_loop import load, daily_bars, rsi
d = daily_bars(load(src1h)); c = d.c
cond = (c < c.rolling(20).mean()) & (c < c.rolling(50).mean()) & (rsi(c, 14) < 40)
chk = {}
for n, lab in ((1, "1 Tag später"), (3, "3 Tage später")):
    fwd = c.shift(-n) / c - 1; sel = fwd[cond].dropna(); allf = fwd.dropna()
    chk[lab] = {"days_like_today": int(len(sel)), "p_up_like_today": round(float((sel > 0).mean()), 4), "p_up_all_days": round(float((allf > 0).mean()), 4),
                "avg_pct_like_today": round(float(sel.mean() * 100), 3)}
res["history_check"] = chk
json.dump(res, open(out, "w"), indent=1)
print("move today", res["move_today_pct"], "% | analog days", len(seqs))
for hz, v in res["horizons"].items(): print(f"{hz:18s}", {k: (x["p_up"], x["median"]) for k, x in v.items()})
for dist, v in res["touch"].items(): print("+-", dist, {k: (x["up_first"], x["down_first"]) for k, x in v.items()})
print(json.dumps(chk, indent=1))
