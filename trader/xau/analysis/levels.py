"""Support and resistance zones for gold around the current price.

Sources of a level (all from GC=F bars):
  - daily: previous day's high/low/close, 20/50/100/200-day averages, swing highs/lows of the last 60 days
  - intraday: today's high/low so far, the overnight range
  - volume by price: price bands of the last 10 days where the most 5-minute volume traded (the market "spent time" there)
  - how often a zone stopped price: number of 5-minute candles over 60 days that turned within the zone
Usage: python3 levels.py GC-60d5m-full.json GC-2y1h.json out.json
"""
import json, sys, zoneinfo
import numpy as np, pandas as pd
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from intraday_lab import load
from strategy_loop import load as load_h, daily_bars, rsi, atr
DE = zoneinfo.ZoneInfo("Europe/Berlin")
m5 = load(sys.argv[1]); d = daily_bars(load_h(sys.argv[2])); price = float(m5.close.iloc[-1]); now = m5.index[-1]
today = m5.day.iloc[-1]; t = m5[m5.day == today]; y = m5[m5.day == sorted(m5.day.unique())[-2]]
lv = []
def add(p, name, kind): lv.append({"price": round(float(p), 1), "name": name, "kind": kind})
add(y.high.max(), "Hoch gestern", "Tag"); add(y.low.min(), "Tief gestern", "Tag"); add(y.close.iloc[-1], "Schluss gestern", "Tag")
add(t.high.max(), "Hoch heute", "heute"); add(t.low.min(), "Tief heute", "heute")
after_high = t[t.index > t.high.idxmax()]
if len(after_high): add(after_high.low.min(), "Rücksetzer-Tief heute", "heute")
add(t.vwap.iloc[-1], "VWAP heute", "Tag")
dc = d.c
for n in (20, 50, 100, 200): add(dc.rolling(n).mean().iloc[-1], f"{n}-Tage-Durchschnitt", "Durchschnitt")
z = d.iloc[-60:]
for i in range(2, len(z) - 2):
    if z.h.iloc[i] == z.h.iloc[i - 2:i + 3].max(): add(z.h.iloc[i], f"Swing-Hoch {z.index[i]:%d.%m.}", "Swing")
    if z.l.iloc[i] == z.l.iloc[i - 2:i + 3].min(): add(z.l.iloc[i], f"Swing-Tief {z.index[i]:%d.%m.}", "Swing")
for r in (4000, 4050, 4100, 4150, 4200, 4250): add(r, f"Rundes Niveau {r}", "rund")
# volume by price, last 10 days, 5-dollar bins
v = m5[m5.index >= now - pd.Timedelta(days=14)]
bins = np.arange(np.floor(v.low.min() / 5) * 5, np.ceil(v.high.max() / 5) * 5 + 5, 5)
mid = (v.high + v.low) / 2; vp = v.groupby(pd.cut(mid, bins)).volume.sum()
top = vp.sort_values(ascending=False).head(6)
for iv, vol in top.items(): add(iv.mid, f"Volumen-Knoten ({vol / vp.sum() * 100:.1f} % des Volumens)", "Volumen")
# reaction count: 5-min swing turns (3 bars each side) within +-3 $ of a level, last 60 days
H, L = m5.high.values, m5.low.values
turns_hi = [H[i] for i in range(3, len(H) - 3) if H[i] == H[i - 3:i + 4].max()]
turns_lo = [L[i] for i in range(3, len(L) - 3) if L[i] == L[i - 3:i + 4].min()]
for x in lv: x["turns"] = int(sum(abs(np.array(turns_hi + turns_lo) - x["price"]) <= 3))
# merge levels within 6 dollars into zones, keep those within +-150 of price
lv.sort(key=lambda x: x["price"]); zones = []
for x in lv:
    if abs(x["price"] - price) > 150: continue
    if zones and x["price"] - zones[-1]["hi"] <= 6: z_ = zones[-1]; z_["hi"] = x["price"]; z_["items"].append(x)
    else: zones.append({"lo": x["price"], "hi": x["price"], "items": [x]})
for z_ in zones:
    z_["score"] = len(z_["items"]) + sum(i["kind"] in ("Durchschnitt", "Tag", "Volumen") for i in z_["items"]) + max(i["turns"] for i in z_["items"]) / 10
    z_["side"] = "Widerstand" if z_["lo"] > price else ("Unterstützung" if z_["hi"] < price else "jetzt im Bereich")
out = {"price": price, "time": str(now), "today_high": float(t.high.max()), "today_low": float(t.low.min()),
       "y_high": float(y.high.max()), "y_low": float(y.low.min()), "y_close": float(y.close.iloc[-1]),
       "rsi14_daily": round(float(rsi(dc, 14).iloc[-1]), 1), "atr14_daily": round(float(atr(d).iloc[-1]), 1),
       "rsi14_5m": round(float(rsi(m5.close, 14).iloc[-1]), 1), "rsi14_1h": round(float(rsi(load_h(sys.argv[2]).c, 14).iloc[-1]), 1),
       "zones": zones}
json.dump(out, open(sys.argv[3], "w"), indent=1, default=str)
print({k: v for k, v in out.items() if k != "zones"})
for z_ in zones:
    print(f"{z_['side']:15s} {z_['lo']:7.1f}-{z_['hi']:7.1f} score {z_['score']:.1f}  " + "; ".join(f"{i['name']} {i['price']} ({i['turns']}x)" for i in z_["items"]))
