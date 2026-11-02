# Forward bootstrap: from the last closed 5-min bar to a horizon, drawing each future bar's
# close/high/low ratios from the same NY time-of-day slot (+-2) over the last 60 days.
import json, sys, random, datetime, zoneinfo, collections, statistics
raw, out, horizon = sys.argv[1], sys.argv[2], sys.argv[3]   # horizon "2026-11-03 16:00" NY
N, SEED, WIN = 10000, 19, 2
ny = zoneinfo.ZoneInfo("America/New_York")
r = json.load(open(raw))["chart"]["result"][0]; q = r["indicators"]["quote"][0]
bars = [(datetime.datetime.fromtimestamp(t, ny), o, h, l, c) for t, o, h, l, c in
        zip(r["timestamp"], q["open"], q["high"], q["low"], q["close"]) if None not in (o, h, l, c)]
bars = [b for b in bars if b[0].second == 0 and b[0].minute % 5 == 0]          # drop the live partial bar
last_ts = max(b[0] for b in bars)
now_ny = datetime.datetime.now(ny)
if last_ts + datetime.timedelta(minutes=5) > now_ny: bars = [b for b in bars if b[0] != last_ts]  # not closed yet
slot = lambda dt: dt.hour * 12 + dt.minute // 5
pool = collections.defaultdict(list)
for (t0, *_, c0), (t1, o, h, l, c) in zip(bars, bars[1:]):
    if (t1 - t0) <= datetime.timedelta(minutes=70):        # consecutive, or across the daily 17:00-18:00 pause
        pool[slot(t1)].append((h / c0, l / c0, c / c0))
start_t, start_p = bars[-1][0], bars[-1][4]
end_t = datetime.datetime.fromisoformat(horizon).replace(tzinfo=ny)
days = len({b[0].date() for b in bars})
steps, t = [], start_t + datetime.timedelta(minutes=5)
while t <= end_t:
    s = slot(t)
    if len(pool[s]) >= days // 3:                            # a slot gold actually trades on most days
        cand = [x for k in range(s - WIN, s + WIN + 1) for x in pool.get(k % 288, [])]
        steps.append((t, cand))
    t += datetime.timedelta(minutes=5)
rng = random.Random(SEED)
paths = []
hi_hit = collections.Counter()
for _ in range(N):
    p, path = start_p, []
    for _, cand in steps:
        h, l, c = rng.choice(cand); p = p * c; path.append(p)
    paths.append(path)
def pct(v, q): v = sorted(v); return v[min(len(v) - 1, int(q * len(v)))]
bands = [[round(pct([pp[i] for pp in paths], q), 2) for q in (.05, .25, .5, .75, .95)] for i in range(len(steps))]
finals = [pp[-1] for pp in paths]
res = {"symbol": "GC=F", "start_time_ny": start_t.isoformat(), "start_price": round(start_p, 2),
       "end_time_ny": steps[-1][0].isoformat(), "replays": N, "seed": SEED, "window": WIN, "pool_days": days,
       "times_ny": [s[0].isoformat() for s in steps], "bands": bands,
       "p_above_start": round(sum(f > start_p for f in finals) / N, 4),
       "final": {"p5": bands[-1][0], "p25": bands[-1][1], "median": bands[-1][2], "p75": bands[-1][3], "p95": bands[-1][4]},
       "avg_bar_drift_pct": round(statistics.mean(x[2] for v in pool.values() for x in v) * 100 - 100, 5),
       "sample_paths": [[round(x, 2) for x in pp[::3]] for pp in paths[:60]]}
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k not in ("times_ny", "bands", "sample_paths")}, indent=1), len(steps), "bars")
