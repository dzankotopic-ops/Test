import json, zoneinfo
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); matplotlib.rcParams["text.usetex"] = False; matplotlib.rcParams["text.parse_math"] = False
import matplotlib.pyplot as plt, matplotlib.dates as mdates
DE = zoneinfo.ZoneInfo("Europe/Berlin")
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COL = {"A": "#2a78d6", "B": "#eb6834", "C": "#eda100", "D": "#e87ba4", "ENS": "#0b0b0b"}
def eu(v, nd=0): return f"{v:,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")
R = json.load(open("analysis/nowcast.json")); price = R["price"]; at = pd.Timestamp(R["at"]).tz_convert(DE)
times = [at] + [pd.Timestamp(t).tz_convert(DE) for t in R["times"]]
r = json.load(open("GC-60d5m-full.json"))["chart"]["result"][0]
h = pd.Series(r["indicators"]["quote"][0]["close"], index=pd.to_datetime(r["timestamp"], unit="s", utc=True).tz_convert(DE)).dropna()
h = h[(h.index >= at - pd.Timedelta(hours=30)) & (h.index <= at)]
fig = plt.figure(figsize=(14, 6.6)); fig.patch.set_facecolor(SURF)
gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1], wspace=.18); fig.subplots_adjust(top=.84)
def style(a):
    a.set_facecolor(SURF); a.grid(axis="y", color=GRID, lw=.8); a.set_axisbelow(True)
    for s in ("top", "right"): a.spines[s].set_visible(False)
    for s in ("left", "bottom"): a.spines[s].set_color(GRID)
    a.tick_params(colors=INK2, labelsize=9)
a = fig.add_subplot(gs[0]); style(a)
b = np.vstack([[price] * 5, np.array(R["bands"]["ENS"])])
a.plot(h.index, h.values, color=INK2, lw=1.8, label="Tatsächlicher Verlauf")
a.fill_between(times, b[:, 0], b[:, 4], color=COL["A"], alpha=.15, lw=0, label="90 % der Simulationen")
a.fill_between(times, b[:, 1], b[:, 3], color=COL["A"], alpha=.30, lw=0, label="50 % der Simulationen")
a.plot(times, b[:, 2], color=COL["A"], lw=2.2, label="Gesamt-Median")
a.axhline(price, color=INK2, lw=1, ls=(0, (4, 3)))
a.plot([at], [price], "o", ms=8, color=INK, mec=SURF, mew=2)
a.annotate(f"Jetzt {at:%H:%M} Uhr: {eu(price, 2)}", (at, price), xytext=(-8, -26), textcoords="offset points", ha="right", fontsize=9, color=INK,
           bbox=dict(fc=SURF, ec="none", pad=1))
for k, lab in (("heute 22:00 Uhr", "heute 22 Uhr"), ("morgen 22:00 Uhr", "morgen 22 Uhr"), ("Freitag 22:00 Uhr", "Fr 22 Uhr")):
    x = {"heute 22:00 Uhr": "2026-11-03 22:00", "morgen 22:00 Uhr": "2026-11-04 22:00", "Freitag 22:00 Uhr": "2026-11-06 22:00"}[k]
    x = pd.Timestamp(x, tz=DE); a.axvline(x, color=GRID, lw=1.2)
    m = R["horizons"][k]["ENS"]["median"]; a.text(x, b[:, 4].max() + 8, f"{lab}\nMedian {eu(m)}", ha="center", fontsize=8.5, color=INK)
a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: eu(v)))
a.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12], tz=DE))
a.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: (lambda t: ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"][t.weekday()] + t.strftime(" %H:%M"))(mdates.num2date(v, tz=DE))))
a.set_ylim(b[:, 0].min() - 20, b[:, 4].max() + 45); a.legend(loc="lower left", frameon=False, fontsize=9)
a.set_ylabel("Gold, US-Dollar je Feinunze (GC=F)", color=INK2, fontsize=9); a.set_xlabel("Deutsche Uhrzeit", color=INK2, fontsize=9)
s = fig.add_subplot(gs[1]); style(s)
hz = list(R["horizons"]); y = np.arange(len(hz))[::-1]; w = .15
for i, k in enumerate(["A", "B", "C", "D", "ENS"]):
    vals = [R["horizons"][z][k]["p_up"] * 100 for z in hz]
    bars = s.barh(y + (2 - i) * w, vals, w * .9, color=COL[k], label=("Gesamt (alle Modelle)" if k == "ENS" else R["names"][k]))
    if k == "ENS":
        for bx, v in zip(bars, vals): s.text(v + 1, bx.get_y() + bx.get_height() / 2, f"{eu(v)} % hoch", va="center", fontsize=9, color=INK, fontweight="bold")
s.axvline(50, color=INK2, lw=1.2, ls=(0, (4, 3))); s.text(50, len(hz) - .45, " 50 % = Münzwurf", fontsize=8.5, color=INK2)
s.set_yticks(y, hz, fontsize=9.5, color=INK); s.set_xlim(0, 100)
s.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: eu(v) + " %"))
s.set_xlabel("Wahrscheinlichkeit, dass Gold dann HÖHER steht als jetzt", color=INK2, fontsize=9)
s.legend(loc="upper center", bbox_to_anchor=(.5, -.13), frameon=False, fontsize=8.5, ncol=1)
e = R["touch"]["50"]["ENS"]
fig.suptitle(f"Gold jetzt ({at:%d.%m. %H:%M} Uhr, {eu(price, 2)} $): eher hoch oder runter?", x=.06, y=.975, ha="left", fontsize=13, color=INK, fontweight="bold")
fig.text(.06, .915, f"4 Modelle × 10.000 Simulationen (Seed {R['seed']})  ·  heute {eu(abs(R["move_today_pct"]), 2)} % im Minus  ·  "
         f"bis Freitag zuerst −50 $: {eu(e['down_first'] * 100)} %, zuerst +50 $: {eu(e['up_first'] * 100)} %  ·  keine Vorhersage", fontsize=10, color=INK2)
fig.savefig("analysis/5-jetzt-hoch-oder-runter.png", dpi=150, bbox_inches="tight", facecolor=SURF)
