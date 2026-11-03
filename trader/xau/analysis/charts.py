import json, sys, datetime, zoneinfo
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as mdates
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from strategy_loop import load, daily_bars
DE = zoneinfo.ZoneInfo("Europe/Berlin")
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, RED, GREEN = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#e34948", "#008300"
def eu(v, nd=0): return f"{v:,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")
def style(a):
    a.set_facecolor(SURF); a.grid(axis="y", color=GRID, lw=.8); a.set_axisbelow(True)
    for s in ("top", "right"): a.spines[s].set_visible(False)
    for s in ("left", "bottom"): a.spines[s].set_color(GRID)
    a.tick_params(colors=INK2, labelsize=9)
    a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: eu(v)))
def title(fig, t, sub, y=None, gap=None):
    hgt = fig.get_size_inches()[1]; y = y or 1 - 0.25 / hgt; gap = 0.42 / hgt; fig.subplots_adjust(top=1 - 1.05 / hgt)
    fig.suptitle(t, x=.06, y=y, ha="left", fontsize=13, color=INK, fontweight="bold"); fig.text(.06, y - gap, sub, fontsize=10, color=INK2)
h = load("GC-2y1h.json"); d = daily_bars(h); T = json.load(open("analysis/technicals.json"))
L = json.load(open("analysis/strategy_loop.json")); F = json.load(open("analysis/forecast.json"))

# 1. overview: last 12 months, daily close, averages, Donchian channel, levels
z = d[d.index > d.index[-1] - pd.Timedelta(days=365)]
fig, a = plt.subplots(figsize=(13, 6.2)); fig.patch.set_facecolor(SURF); style(a)
a.plot(z.index, z.c, color=INK, lw=1.6, label="Schlusskurs (Gold-Future GC=F)")
for n, col in ((20, BLUE), (50, ORANGE), (200, AQUA)):
    a.plot(z.index, d.c.rolling(n).mean().reindex(z.index), color=col, lw=2, label=f"{n}-Tage-Durchschnitt")
hi20 = d.h.rolling(20).max().shift().reindex(z.index); lo20 = d.l.rolling(20).min().shift().reindex(z.index)
a.fill_between(z.index, lo20, hi20, color=BLUE, alpha=.08, lw=0, label="20-Tage-Kanal (Donchian)")
for y, lab in ((T["exit5_high"], "Widerstand / Ausstieg Short"), (T["low_52w"], "52-Wochen-Tief")):
    a.axhline(y, color=INK2, lw=1, ls=(0, (4, 3))); a.text(z.index[-1] - pd.Timedelta(days=150), y + 12, f"{lab} {eu(y, 1)}", va="bottom", fontsize=9, color=INK, bbox=dict(fc=SURF, ec="none", pad=1))
a.annotate(f"Jetzt {eu(T['last'], 1)}", (z.index[-1], T["last"]), xytext=(-8, -22), textcoords="offset points", ha="right", fontsize=9, color=INK)
a.xaxis.set_major_formatter(mdates.DateFormatter("%m/%y"))
a.set_ylabel("US-Dollar je Feinunze", color=INK2, fontsize=9); a.legend(loc="upper right", frameon=False, fontsize=9)
title(fig, "Gold: 12 Monate Überblick", f"RSI(14) {eu(T['rsi14'], 1)}  ·  ATR(14) {eu(T['atr14'], 0)} $ ({eu(T['atr14_pct'], 2)} % pro Tag)  ·  "
      f"{eu(T['from_high_pct'], 1)} % unter dem Hoch vom {pd.Timestamp(T['high_52w_day']):%d.%m.%Y}  ·  unter allen Durchschnitten")
fig.savefig("analysis/1-ueberblick.png", dpi=150, bbox_inches="tight", facecolor=SURF); plt.close(fig)

# 2. strategy loop: equity curves with the three windows
E = L["equity"]; idx = pd.to_datetime(E["dates"])
fig, a = plt.subplots(figsize=(13, 6.2)); fig.patch.set_facecolor(SURF); style(a)
a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: eu((v - 1) * 100) + " %"))
lab = {"buy_hold": "Kaufen und halten", "sma_trend": "Trend (Durchschnitte)", "donchian": "Ausbruch (Donchian) – gewählt",
       "rsi_revert": "Rückkehr (RSI)", "orb_hourly": "Eröffnungs-Ausbruch (stündlich)"}
for k, col, lw in (("buy_hold", INK2, 1.5), ("sma_trend", BLUE, 1.5), ("rsi_revert", YELLOW, 1.5), ("orb_hourly", AQUA, 1.5), ("donchian", ORANGE, 2.6)):
    a.plot(idx, E[k], color=col, lw=lw, label=lab[k])
w = L["windows"]
for (k, txt), shade in zip((("train", "Training"), ("valid", "Validierung"), ("test", "Test (ungesehen)")), (0, .05, .1)):
    a0, a1 = pd.Timestamp(w[k][0]), pd.Timestamp(w[k][1])
    if shade: a.axvspan(a0, a1, color=BLUE, alpha=shade, lw=0)
    a.text(a0 + (a1 - a0) / 2, a.get_ylim()[1], txt, ha="center", va="top", fontsize=9, color=INK2)
a.legend(loc="upper left", frameon=False, fontsize=9, bbox_to_anchor=(0, .93))
fs = L["final"]["test"]; bh = L["buy_hold"]["test"]
import itertools, strategy_loop as SL
nvar = sum(len(list(itertools.product(*g.values()))) for _, g in SL.GRIDS.values())
title(fig, f"Strategie-Loop: 4 Familien, {nvar} Varianten, 3 Runden", f"Gewählt ohne Blick auf den Test: Ausbruch 20/5 Tage, long+short.  Im Test: {eu(fs['return_pct'], 1)} % "
      f"bei {fs['trades']} Trades  ·  Kaufen und halten: {eu(bh['return_pct'], 1)} %  ·  Kosten eingerechnet")
fig.savefig("analysis/2-strategie-loop.png", dpi=150, bbox_inches="tight", facecolor=SURF); plt.close(fig)

# 3. ensemble forecast with the four models as small multiples
times = [pd.Timestamp(F["start_time"])] + [pd.Timestamp(t) for t in F["times"]]
times = [t.tz_convert(DE) for t in times]
def bands(k): b = np.array(F["models"][k]["bands"]); s = F["start_price"]; return np.vstack([[s] * 5, b])
hist = h[h.index >= h.index[-1] - pd.Timedelta(hours=72)]
fig = plt.figure(figsize=(13, 10)); fig.patch.set_facecolor(SURF)
gs = fig.add_gridspec(3, 4, height_ratios=[2.2, 1, 0.05], hspace=.45, wspace=.25)
a = fig.add_subplot(gs[0, :]); style(a)
b = bands("ENS")
a.plot([t.tz_convert(DE) for t in hist.index], hist.c, color=ORANGE, lw=2, label="Tatsächlicher Verlauf (72 Std.)")
a.fill_between(times, b[:, 0], b[:, 4], color=BLUE, alpha=.15, lw=0, label="90 % aller Simulationen")
a.fill_between(times, b[:, 1], b[:, 3], color=BLUE, alpha=.30, lw=0, label="50 % aller Simulationen")
a.plot(times, b[:, 2], color=BLUE, lw=2.4, label="Gesamt-Median (alle 4 Modelle)")
for k, col in zip("ABCD", (BLUE, AQUA, YELLOW, MAGENTA)):
    a.plot(times, bands(k)[:, 2], color=col, lw=1, ls=(0, (2, 2)), alpha=.9)
for y, txt in ((T["exit5_high"], "Widerstand 4.097"), (4000, "4.000"), (T["low_52w"], "52-Wochen-Tief 3.879")):
    a.axhline(y, color=INK2, lw=1, ls=(0, (4, 3))); a.text(times[0] - pd.Timedelta(hours=70), y + 3, txt, ha="left", va="bottom", fontsize=9, color=INK, bbox=dict(fc=SURF, ec="none", pad=1))
for txt, when in (("Wahllokale schließen", "2026-11-04 01:00"), ("US-Arbeitsmarkt 14:30", "2026-11-06 14:30")):
    x = pd.Timestamp(when, tz=DE); a.axvline(x, color=GRID, lw=1.2); a.text(x, a.get_ylim()[0], f" {txt}", rotation=90, va="bottom", ha="right", fontsize=8.5, color=INK2)
fe = F["models"]["ENS"]["final"]
for k, n in zip(("p95", "p75", "median", "p25", "p5"), ("95 %", "75 %", "Median", "25 %", "5 %")):
    a.text(times[-1] + pd.Timedelta(hours=2), fe[k] - (14 if abs(fe[k] - T["exit5_high"]) < 12 else 0), f"{n}: {eu(fe[k], 0)}", va="center", fontsize=9, color=INK)
a.set_xlim(times[0] - pd.Timedelta(hours=72), times[-1] + pd.Timedelta(hours=22))
a.set_ylim(3640, 4160)
a.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12], tz=DE))
a.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: (lambda t: ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"][t.weekday()] + t.strftime(" %H:%M"))(mdates.num2date(v, tz=DE))))
a.set_ylabel("US-Dollar je Feinunze", color=INK2, fontsize=9); a.legend(loc="lower left", frameon=False, fontsize=9)
for i, (k, col) in enumerate(zip("ABCD", (BLUE, AQUA, YELLOW, MAGENTA))):
    s = fig.add_subplot(gs[1, i], sharey=a); style(s); bb = bands(k)
    s.fill_between(times, bb[:, 0], bb[:, 4], color=col, alpha=.18, lw=0); s.fill_between(times, bb[:, 1], bb[:, 3], color=col, alpha=.35, lw=0)
    s.plot(times, bb[:, 2], color=col, lw=2); s.set_title(F["names"][k], fontsize=9.5, color=INK, loc="left")
    m = F["models"][k]["final"]; s.text(.02, .04, f"Median Fr {eu(m['median'])}\n90 %: {eu(m['p5'])} – {eu(m['p95'])}", transform=s.transAxes, fontsize=8.5, color=INK)
    s.xaxis.set_major_locator(mdates.DayLocator(tz=DE)); s.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m.", tz=DE))
    if i: plt.setp(s.get_yticklabels(), visible=False)
end = times[-1]
title(fig, f"Gold bis Freitag {end:%d.%m.} {end:%H:%M} Uhr: 4 Modelle × 10.000 Simulationen (Seed {F['seed']})",
      f"Start {eu(F['start_price'], 1)} $ um {times[0]:%H:%M} Uhr  ·  gestrichelt: Median je Modell  ·  deutsche Zeit  ·  keine Vorhersage, eine Bandbreite")
fig.savefig("analysis/3-prognose-ensemble.png", dpi=150, bbox_inches="tight", facecolor=SURF); plt.close(fig)

# 4. time machine for the current signal (short), odds per model
fig, a = plt.subplots(figsize=(13, 5.4)); fig.patch.set_facecolor(SURF); style(a)
a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: eu(v) + " %"))
rows = [("p_3879_first", "52-Wochen-Tief 3.879 zuerst"), ("p_4097_first", "Ausstieg 4.097 zuerst"), ("p_below_start", "Freitag unter Startkurs (Short im Plus)"), ("p_touch_4000", "4.000 wird berührt")]
keys = ["A", "B", "C", "D", "ENS"]; x = np.arange(len(rows)); wdt = .16
for i, (k, col) in enumerate(zip(keys, (BLUE, AQUA, YELLOW, MAGENTA, INK))):
    vals = [F["models"][k][r] * 100 for r, _ in rows]
    bars = a.bar(x + (i - 2) * wdt, vals, wdt * .9, color=col, label=(F["names"][k] if k != "ENS" else "Gesamt (alle Modelle)"))
    if k == "ENS":
        for bx, v in zip(bars, vals): a.text(bx.get_x() + bx.get_width() / 2, v + 1, eu(v, 0) + " %", ha="center", fontsize=9, color=INK, fontweight="bold")
a.set_xticks(x, [r[1] for r in rows], fontsize=9.5, color=INK); a.set_ylim(0, 100)
a.legend(loc="upper right", frameon=False, fontsize=9)
title(fig, "Time Machine für das aktuelle Signal: Short ab 3.915,7 bis Freitag", "Strategie ist short seit dem Bruch des 20-Tage-Tiefs; Ausstieg bei Schlusskurs über dem 5-Tage-Hoch (4.097)  ·  je Modell 10.000 Simulationen")
fig.savefig("analysis/4-time-machine-signal.png", dpi=150, bbox_inches="tight", facecolor=SURF); plt.close(fig)
print("ok")
