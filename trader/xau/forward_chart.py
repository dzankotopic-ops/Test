import json, sys, datetime, zoneinfo
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as mdates
fw, raw, out = sys.argv[1:4]
d = json.load(open(fw)); de = zoneinfo.ZoneInfo("Europe/Berlin"); ny = zoneinfo.ZoneInfo("America/New_York")
def eu(v, nd=2): return f"{v:,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")
start = datetime.datetime.fromisoformat(d["start_time_ny"]).astimezone(de)
T = [start] + [datetime.datetime.fromisoformat(t).astimezone(de) for t in d["times_ny"]]
B = [[d["start_price"]] * 5] + d["bands"]
col = lambda i: [b[i] for b in B]
r = json.load(open(raw))["chart"]["result"][0]; q = r["indicators"]["quote"][0]
hist = [(datetime.datetime.fromtimestamp(t, de), c) for t, c in zip(r["timestamp"], q["close"]) if c is not None]
hist = [(t, c) for t, c in hist if start - datetime.timedelta(hours=26) <= t <= start]
INK, INK2, GRID, BLUE, ORANGE, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834", "#fcfcfb"
fig, a = plt.subplots(figsize=(13, 6.2)); fig.patch.set_facecolor(SURF); a.set_facecolor(SURF)
a.grid(axis="y", color=GRID, lw=.8); a.set_axisbelow(True)
for s in ("top", "right"): a.spines[s].set_visible(False)
for s in ("left", "bottom"): a.spines[s].set_color(GRID)
a.tick_params(colors=INK2, labelsize=9)
a.plot([t for t, _ in hist], [c for _, c in hist], color=ORANGE, lw=2, label="Tatsächlicher Verlauf (letzte 26 Std.)")
a.fill_between(T, col(0), col(4), color=BLUE, alpha=.15, lw=0, label="90 % der Simulationen (5.–95. Perzentil)")
a.fill_between(T, col(1), col(3), color=BLUE, alpha=.30, lw=0, label="50 % der Simulationen (25.–75. Perzentil)")
a.plot(T, col(2), color=BLUE, lw=2, label="Median der Simulationen")
a.axvline(start, color=INK2, lw=1, ls=(0, (4, 3)))
a.text(start, a.get_ylim()[1], f"  Jetzt {start:%H:%M} Uhr: {eu(d['start_price'])}", va="top", fontsize=9, color=INK)
for hh, mm, lab in ((15, 30, "US-Börse öffnet 15:30"), (22, 0, "US-Börse schließt 22:00")):
    x = (start + datetime.timedelta(days=1)).replace(hour=hh, minute=mm)
    if x <= T[-1] + datetime.timedelta(minutes=5):
        a.axvline(x, color=GRID, lw=1.2); a.text(x, a.get_ylim()[0], f" {lab}", va="bottom", ha="right", fontsize=8.5, color=INK2, rotation=90)
names = ("5 %", "25 %", "Median", "75 %", "95 %")
for i, n in enumerate(names):
    a.text(T[-1] + datetime.timedelta(minutes=25), B[-1][i], f"{n}: {eu(B[-1][i])}", va="center", fontsize=9, color=INK)
a.set_xlim(hist[0][0] if hist else start, T[-1] + datetime.timedelta(hours=3.2))
a.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 3), tz=de))
a.xaxis.set_major_formatter(mdates.DateFormatter("%a %H:%M", tz=de))
fig.canvas.draw()
a.set_xticks(a.get_xticks(), [l.get_text().replace("Mon", "Mo").replace("Tue", "Di").replace("Wed", "Mi").replace("Sun", "So") for l in a.get_xticklabels()])
a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: eu(v, 0)))
a.set_ylabel("Gold-Future Dez. 2026 (GC=F), US-Dollar je Feinunze", color=INK2, fontsize=9)
a.set_xlabel("Deutsche Uhrzeit (MEZ)", color=INK2, fontsize=9)
a.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK)
end = T[-1]
fig.suptitle(f"Gold bis {end:%d.%m.%Y %H:%M} Uhr: 10.000 Simulationen ab jetzt (Seed {d['seed']})", x=.06, ha="left", fontsize=13, color=INK, fontweight="bold")
fig.text(.06, .915, f"Über dem jetzigen Kurs am Ende: {eu(d['p_above_start']*100,1)} % der Simulationen  ·  gebaut aus den Schwankungen der letzten {d['pool_days']} Tage  ·  keine Vorhersage", fontsize=10, color=INK2)
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURF)
