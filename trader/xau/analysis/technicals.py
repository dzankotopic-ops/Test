"""Daily technicals and today's signal of the chosen strategies, from the hourly GC=F file."""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from strategy_loop import load, daily_bars, rsi, atr, donchian, sma_trend
h = load(sys.argv[1]); d = daily_bars(h); c = d.c
loop = json.load(open(sys.argv[2]))
out = {"last_close_day": str(d.index[-1].date()), "last": round(float(c.iloc[-1]), 2)}
for n in (20, 50, 100, 200): out[f"sma{n}"] = round(float(c.rolling(n).mean().iloc[-1]), 2)
out["rsi14"] = round(float(rsi(c, 14).iloc[-1]), 1); out["rsi2"] = round(float(rsi(c, 2).iloc[-1]), 1)
out["atr14"] = round(float(atr(d).iloc[-1]), 2); out["atr14_pct"] = round(float(atr(d).iloc[-1] / c.iloc[-1] * 100), 2)
y = d[d.index > d.index[-1] - pd.Timedelta(days=365)]
out["high_52w"] = round(float(y.h.max()), 2); out["high_52w_day"] = str(y.h.idxmax().date())
out["low_52w"] = round(float(y.l.min()), 2); out["low_52w_day"] = str(y.l.idxmin().date())
out["from_high_pct"] = round((c.iloc[-1] / y.h.max() - 1) * 100, 2)
out["ret_5d_pct"] = round(float(c.iloc[-1] / c.iloc[-6] - 1) * 100, 2); out["ret_20d_pct"] = round(float(c.iloc[-1] / c.iloc[-21] - 1) * 100, 2)
out["donchian20_high"] = round(float(d.h.iloc[-21:-1].max()), 2); out["donchian20_low"] = round(float(d.l.iloc[-21:-1].min()), 2)
out["exit5_high"] = round(float(d.h.iloc[-6:-1].max()), 2); out["exit5_low"] = round(float(d.l.iloc[-6:-1].min()), 2)
# swing levels: local extremes of the last 60 days (5-day window)
z = d.iloc[-60:]
sw_hi = [round(float(z.h.iloc[i]), 1) for i in range(2, len(z) - 2) if z.h.iloc[i] == z.h.iloc[i - 2:i + 3].max()]
sw_lo = [round(float(z.l.iloc[i]), 1) for i in range(2, len(z) - 2) if z.l.iloc[i] == z.l.iloc[i - 2:i + 3].min()]
last = c.iloc[-1]
out["resistance"] = sorted([x for x in sw_hi if x > last])[:3]; out["support"] = sorted([x for x in sw_lo if x < last], reverse=True)[:3]
# positions the strategies hold now (decided at the last close)
p = loop["rounds"][0]["result"]
pnl, _ = donchian(d, **p["donchian"]["params"])
def last_pos(fn, params):
    import strategy_loop as S
    src = S.from_positions
    held = {}
    def grab(d_, pos): held["pos"] = pd.Series(np.asarray(pos, dtype=float), d_.index); return src(d_, pos)
    S.from_positions = grab
    try: fn(d, **params)
    finally: S.from_positions = src
    return int(held["pos"].fillna(0).iloc[-1])
import strategy_loop as S
out["signal_donchian"] = last_pos(S.donchian, p["donchian"]["params"])
out["signal_sma_trend"] = last_pos(S.sma_trend, p["sma_trend"]["params"])
json.dump(out, open(sys.argv[3], "w"), indent=1); print(json.dumps(out, indent=1))
