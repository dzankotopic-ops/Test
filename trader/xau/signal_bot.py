"""Gold signal bot: watches GC=F 5-minute candles and calls LONG or SHORT entries with stop and targets.

Two rules, checked on every newly closed 5-minute candle:
  1. Plan levels  - a call file (like analysis/call-2026-11-04.json): a close above long.trigger -> LONG,
                    a close below short.trigger -> SHORT, with the file's stop and targets.
  2. Range break  - after 18 quiet candles (range <= 1.2 x ATR x sqrt(18)), a close outside the range;
                    stop at the middle of the range, targets 1R and 2R.
Each rule calls once per range/level, then waits for a new setup.

Data: Yahoo's free chart API, about 10 minutes delayed. Good for practice and paper trading,
not for live fills. Nothing here places an order; it only prints (and, if set up, sends) a message.

  python3 signal_bot.py --plan analysis/call-2026-11-04.json          # run until stopped (checks every 60 s)
  python3 signal_bot.py --plan analysis/call-2026-11-04.json --once   # check once and exit

Optional Telegram: set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID as environment variables (never in the code).
Standard library only, Python 3.8+.
"""
import argparse, datetime, json, math, os, sys, time, urllib.parse, urllib.request
from zoneinfo import ZoneInfo

DE = ZoneInfo("Europe/Berlin")
URL = "https://query2.finance.yahoo.com/v8/finance/chart/GC%3DF?range=2d&interval=5m&includePrePost=true"
UA = "Mozilla/5.0 (compatible; gold-signal-bot/1.0)"
N_RANGE, K_RANGE = 18, 1.2

def fetch():
    req = urllib.request.Request(URL, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=20) as r:
        res = json.load(r)["chart"]["result"][0]
    q = res["indicators"]["quote"][0]; bars = []
    for t, o, h, l, c in zip(res["timestamp"], q["open"], q["high"], q["low"], q["close"]):
        if None in (o, h, l, c) or t % 300: continue
        bars.append({"t": datetime.datetime.fromtimestamp(t, DE), "o": o, "h": h, "l": l, "c": c})
    now = datetime.datetime.now(DE)
    return [b for b in bars if b["t"] + datetime.timedelta(minutes=5) <= now]   # closed candles only

def atr(bars, n=14):
    trs = [max(b["h"] - b["l"], abs(b["h"] - p["c"]), abs(b["l"] - p["c"])) for p, b in zip(bars, bars[1:])]
    return sum(trs[-n:]) / n

def send(msg, log):
    stamp = datetime.datetime.now(DE).strftime("%d.%m. %H:%M:%S")
    line = f"[{stamp}] {msg}"; print(line, flush=True)
    with open(log, "a", encoding="utf-8") as f: f.write(line + "\n")
    tok, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if tok and chat:
        data = urllib.parse.urlencode({"chat_id": chat, "text": msg}).encode()
        try: urllib.request.urlopen(f"https://api.telegram.org/bot{tok}/sendMessage", data=data, timeout=15)
        except OSError as e: print(f"  (Telegram failed: {e})", flush=True)

def fmt(v): return f"{v:,.1f}".replace(",", "X").replace(".", ",").replace("X", ".")

def call(side, bar, stop, targets, why):
    risk = abs(bar["c"] - stop)
    tg = "  ".join(f"Ziel {i + 1} {fmt(t)}" for i, t in enumerate(targets))
    return (f"{'LONG' if side > 0 else 'SHORT'} Gold @ {fmt(bar['c'])} (Kerze {bar['t']:%H:%M} geschlossen)  "
            f"Stop {fmt(stop)} (Risiko {fmt(risk)} $/oz)  {tg}  | {why}  | Daten ~10 Min verzögert, keine Anlageberatung")

def check(bars, plan, state):
    out, last = [], bars[-1]
    if plan:
        for side, key in ((1, "long"), (-1, "short")):
            p = plan[key]; tag = f"plan-{key}"
            if tag not in state and ((side > 0 and last["c"] > p["trigger"]) or (side < 0 and last["c"] < p["trigger"])):
                state.add(tag); out.append(call(side, last, p["stop"], [p["target1"], p["target2"]], f"Plan-Level {fmt(p['trigger'])} gebrochen"))
    win = bars[-N_RANGE - 1:-1]
    if len(win) == N_RANGE and len(bars) > 30:
        hi, lo = max(b["h"] for b in win), min(b["l"] for b in win); a = atr(bars[:-1])
        if hi - lo <= K_RANGE * a * math.sqrt(N_RANGE):
            mid = (hi + lo) / 2
            for side in (1, -1):
                tag = f"range-{side}-{round(hi, 1)}-{round(lo, 1)}"
                if tag not in state and ((side > 0 and last["c"] > hi) or (side < 0 and last["c"] < lo)):
                    state.add(tag); r = abs(last["c"] - mid)
                    out.append(call(side, last, mid, [last["c"] + side * r, last["c"] + side * 2 * r],
                                    f"Ausbruch aus 90-Min-Range {fmt(lo)}–{fmt(hi)}"))
    return out

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", help="call file with long/short trigger, stop, target1, target2")
    ap.add_argument("--once", action="store_true"); ap.add_argument("--every", type=int, default=60, help="seconds between checks")
    ap.add_argument("--log", default="signals.log")
    a = ap.parse_args()
    plan = json.load(open(a.plan)) if a.plan else None
    state, seen = set(), None
    send("Bot gestartet" + (f" mit Plan {os.path.basename(a.plan)}" if plan else "") + ". Ich melde mich nur bei einem Signal.", a.log)
    while True:
        try:
            bars = fetch()
            if bars and bars[-1]["t"] != seen:
                seen = bars[-1]["t"]; msgs = check(bars, plan, state)
                for m in msgs: send(m, a.log)
                if a.once and not msgs:
                    b = bars[-1]; send(f"Kein Signal. Letzte Kerze {b['t']:%H:%M}: Schluss {fmt(b['c'])}, Hoch {fmt(b['h'])}, Tief {fmt(b['l'])}", a.log)
        except (OSError, ValueError, KeyError) as e:
            print(f"Fehler beim Abruf: {e} - neuer Versuch in {a.every} s", file=sys.stderr, flush=True)
        if a.once: break
        time.sleep(a.every)

if __name__ == "__main__":
    main()
