"""Round 2: the same two setups, but only with the day's trend and only in active hours.
Day trend = price vs the previous day's close, at least `trend_pct` percent, trade only in that direction.
Hours (Berlin): all, London 09-13, New York 14:30-18:00, or both.
Picked on the 40 train days by avg R (at least 20 trades), judged once on the test days.
"""
import json, sys, itertools
import numpy as np, pandas as pd
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from intraday_lab import load, simulate, summary, s1, s2, TRAIN_DAYS
df = load(sys.argv[1]); days = sorted(df.day.unique()); split, today = str(days[TRAIN_DAYS]), str(days[-1])
prev_close = df.groupby("day").close.last().shift(); df["dchg"] = df.close / df.day.map(prev_close) - 1
hm = df.index.hour * 60 + df.index.minute
WIN = {"alle": (0, 1440), "London": (9 * 60, 13 * 60), "New York": (14 * 60 + 30, 18 * 60), "London+NY": None}
def in_win(i, w):
    if w == "London+NY": return in_win(i, "London") or in_win(i, "New York")
    a, b = WIN[w]; return a <= hm[i] < b
base = {"S1 Range-Ausbruch": [(s1, dict(n=n, k=k, target_r=t, stop_rule=s, filt=False)) for n, k, t, s in itertools.product([12, 18, 24], [0.6, 0.9, 1.2], [1.0, 1.5, 2.0], ["mid", "range"])],
        "S2 EMA-Pullback": [(s2, dict(target_r=t, buffer_atr=b, need_vwap=v)) for t, b, v in itertools.product([1.0, 1.5, 2.0, 3.0], [0.0, 0.25, 0.5], [False, True])]}
dch = df.dchg.values; res = {}
for fam, variants in base.items():
    rows = []
    for fn, p in variants:
        sig = fn(df, **p)
        for tp, w in itertools.product([0.0, 0.5, 1.0], WIN):
            ss = [s for s in sig if in_win(s[0], w) and (tp == 0 or (s[1] > 0 and dch[s[0]] >= tp / 100) or (s[1] < 0 and dch[s[0]] <= -tp / 100))]
            tr = simulate(df, ss); trn = [t for t in tr if t["day"] < split]; tst = [t for t in tr if split <= t["day"] < today]
            st = summary(trn)
            if st["trades"] >= 20: rows.append((st["avg_R"], {**p, "trend_pct": tp, "hours": w}, st, summary(tst), [t for t in tr if t["day"] == today]))
    rows.sort(key=lambda x: -x[0])
    res[fam] = [{"params": r[1], "train": r[2], "test": r[3], "today": r[4]} for r in rows[:5]]
    print(fam, "variants", len(rows))
    for r in rows[:5]: print("  ", r[1], "\n     train", r[2], "\n     test ", r[3])
json.dump(res, open(sys.argv[2], "w"), indent=1, default=str)
