"""5-minute candles of today with the range, EMAs, VWAP and both trigger plans."""
import json, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); matplotlib.rcParams["text.parse_math"] = False
import matplotlib.pyplot as plt, matplotlib.dates as mdates
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from intraday_lab import load
P = json.load(open(sys.argv[2])); df = load(sys.argv[1]); DE = df.index.tz
t = df[df.index >= pd.Timestamp(P["from"], tz=DE)]
INK, INK2, GRID, SURF, UP, DN, BLUE, ORANGE, AQUA = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#1baf7a", "#e34948", "#2a78d6", "#eb6834", "#1baf7a"
def eu(v, nd=0): return f"{v:,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")
fig, a = plt.subplots(figsize=(14, 7)); fig.patch.set_facecolor(SURF); a.set_facecolor(SURF); fig.subplots_adjust(top=.86)
a.grid(axis="y", color=GRID, lw=.8); a.set_axisbelow(True)
for s in ("top", "right"): a.spines[s].set_visible(False)
for s in ("left", "bottom"): a.spines[s].set_color(GRID)
a.tick_params(colors=INK2, labelsize=9)
w = pd.Timedelta(minutes=3.4)
for ts, r in t.iterrows():
    c = UP if r.close >= r.open else DN
    a.vlines(ts, r.low, r.high, color=c, lw=1)
    a.add_patch(plt.Rectangle((mdates.date2num(ts - w / 2), min(r.open, r.close)), mdates.date2num(ts + w / 2) - mdates.date2num(ts - w / 2),
                              max(abs(r.close - r.open), .3), color=c, lw=0))
a.plot(t.index, t.ema9, color=BLUE, lw=1.4, label="EMA 9"); a.plot(t.index, t.ema21, color=ORANGE, lw=1.4, label="EMA 21")
a.plot(t.index, t.vwap, color=INK2, lw=1.4, ls=(0, (2, 2)), label="VWAP (Tag)")
r0 = pd.Timestamp(P["range_from"], tz=DE); end = t.index[-1] + pd.Timedelta(minutes=95)
a.add_patch(plt.Rectangle((mdates.date2num(r0), P["range_low"]), mdates.date2num(t.index[-1]) - mdates.date2num(r0), P["range_high"] - P["range_low"],
                          fc=BLUE, alpha=.10, ec=BLUE, lw=1))
a.text(r0, P["range_high"] + 1.5, f"Range {eu(P['range_low'], 1)} – {eu(P['range_high'], 1)}", fontsize=9, color=INK)
x0 = t.index[-1] + pd.Timedelta(minutes=8); labels = {}
for side, col in (("long", UP), ("short", DN)):
    s = P[side]; nm = "Long" if side == "long" else "Short"
    for k, ls, lab in (("trigger", "-", nm + "-Einstieg (5-Min-Schluss " + ("darüber)" if side == "long" else "darunter)")), ("stop", (0, (4, 3)), nm + "-Stop"),
                       ("target1", (0, (1, 2)), nm + "-Ziel 1"), ("target2", (0, (1, 2)), nm + "-Ziel 2")):
        a.hlines(s[k], x0, end, color=col, lw=1.6 if k == "trigger" else 1.1, ls=ls)
        labels.setdefault(round(s[k], 1), []).append(lab)
for y, labs in labels.items():
    a.text(end + pd.Timedelta(minutes=2), y, f"{eu(y, 1)}: " + " / ".join(labs), va="center", fontsize=8.5, color=INK)
for y, lab in P.get("levels", []):
    a.axhline(y, color=GRID, lw=1.2); a.text(t.index[0], y + 1, lab, fontsize=8.5, color=INK2)
a.set_xlim(t.index[0] - pd.Timedelta(minutes=10), end + pd.Timedelta(minutes=150))
a.set_ylim(min(t.low.min(), P["short"]["target2"]) - 8, max(t.high.max(), P["long"]["target2"]) + 8)
a.xaxis.set_major_locator(mdates.HourLocator(tz=DE)); a.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=DE))
a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: eu(v)))
a.set_ylabel("Gold-Future GC=F, US-Dollar je Feinunze", color=INK2, fontsize=9); a.set_xlabel("Deutsche Uhrzeit, 5-Minuten-Kerzen", color=INK2, fontsize=9)
a.legend(loc="center left", frameon=False, fontsize=9)
fig.suptitle(P["title"], x=.06, y=.975, ha="left", fontsize=13, color=INK, fontweight="bold")
fig.text(.06, .925, P["subtitle"], fontsize=10, color=INK2)
fig.savefig(sys.argv[3], dpi=150, bbox_inches="tight", facecolor=SURF)
