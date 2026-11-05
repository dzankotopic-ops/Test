"""5-minute candles with the support and resistance zones from levels.py."""
import json, sys
import pandas as pd
import matplotlib; matplotlib.use("Agg"); matplotlib.rcParams["text.parse_math"] = False
import matplotlib.pyplot as plt, matplotlib.dates as mdates
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from intraday_lab import load
df = load(sys.argv[1]); Z = json.load(open(sys.argv[2])); N = json.load(open(sys.argv[3])); DE = df.index.tz
t = df[df.index >= pd.Timestamp(sys.argv[5], tz=DE)]
INK, INK2, GRID, SURF, UP, DN, BLUE, ORANGE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#1baf7a", "#e34948", "#2a78d6", "#eb6834"
def eu(v, nd=0): return f"{v:,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")
fig, a = plt.subplots(figsize=(14, 7.4)); fig.patch.set_facecolor(SURF); a.set_facecolor(SURF); fig.subplots_adjust(top=.86)
for s in ("top", "right"): a.spines[s].set_visible(False)
for s in ("left", "bottom"): a.spines[s].set_color(GRID)
a.tick_params(colors=INK2, labelsize=9)
lo_y, hi_y = t.low.min() - 25, t.high.max() + 60
end = t.index[-1] + pd.Timedelta(minutes=30)
for z in Z["zones"]:
    if z["hi"] < lo_y or z["lo"] > hi_y or z["score"] < 2: continue
    col = DN if z["side"] == "Widerstand" else UP
    pad = 1.5; a.axhspan(z["lo"] - pad, z["hi"] + pad, color=col, alpha=min(.08 + z["score"] / 80, .28), lw=0)
    names = [i["name"].split(" (")[0] for i in z["items"]]
    rng = eu(z["lo"]) if z["lo"] == z["hi"] else f"{eu(z['lo'])}–{eu(z['hi'])}"
    a.text(end + pd.Timedelta(minutes=5), (z["lo"] + z["hi"]) / 2, f"{rng}  {z['side']}: " + ", ".join(dict.fromkeys(names))[:70],
           va="center", fontsize=8.3, color=INK)
w = pd.Timedelta(minutes=3.4)
for ts, r in t.iterrows():
    c = UP if r.close >= r.open else DN
    a.vlines(ts, r.low, r.high, color=c, lw=.9)
    a.add_patch(plt.Rectangle((mdates.date2num(ts - w / 2), min(r.open, r.close)), mdates.date2num(ts + w / 2) - mdates.date2num(ts - w / 2),
                              max(abs(r.close - r.open), .3), color=c, lw=0))
a.plot(t.index, t.ema21, color=ORANGE, lw=1.3, label="EMA 21 (5 Min)"); a.plot(t.index, t.vwap, color=INK2, lw=1.3, ls=(0, (2, 2)), label="VWAP (Tag)")
a.set_xlim(t.index[0] - pd.Timedelta(minutes=10), end + pd.Timedelta(hours=7.5)); a.set_ylim(lo_y, hi_y)
a.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 2), tz=DE)); a.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=DE))
a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: eu(v)))
a.set_ylabel("Gold-Future GC=F, US-Dollar je Feinunze", color=INK2, fontsize=9); a.set_xlabel("Deutsche Uhrzeit, 5-Minuten-Kerzen", color=INK2, fontsize=9)
a.legend(loc="upper left", frameon=False, fontsize=9)
h = N["horizons"]["heute 22:00 Uhr"]["ENS"]
fig.suptitle(sys.argv[6], x=.06, y=.975, ha="left", fontsize=13, color=INK, fontweight="bold")
fig.text(.06, .925, f"Grün = Unterstützung, Rot = Widerstand (je kräftiger, desto mehr Gründe)  ·  bis heute 22 Uhr: {eu(h['p_up'] * 100)} % höher, "
         f"Median {eu(h['median'])}, 90 %: {eu(h['p5'])}–{eu(h['p95'])}", fontsize=10, color=INK2)
fig.savefig(sys.argv[4], dpi=150, bbox_inches="tight", facecolor=SURF)
