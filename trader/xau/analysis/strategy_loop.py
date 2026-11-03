"""Strategy search for gold (GC=F) with an honest walk-forward split.

Train  : pick parameters (grid search, best by Sharpe, at least MIN_TRADES trades)
Valid  : choose between the train winners and decide whether each improvement round helps
Test   : touched once at the very end, never used to choose anything

Usage: python3 strategy_loop.py GC-2y1h.json out.json
"""
import json, sys, itertools, datetime, zoneinfo
import numpy as np, pandas as pd

COST = 0.000075            # per side, ~ $0.30/oz at $4,000: spread + slippage
MIN_TRADES = {"sma_trend": 6, "donchian": 6, "rsi_revert": 10, "orb_hourly": 30}   # in the train window
TRAIN_END, VALID_END = "2025-11-30", "2026-05-31"
NY = zoneinfo.ZoneInfo("America/New_York")

def load(path):
    r = json.load(open(path))["chart"]["result"][0]; q = r["indicators"]["quote"][0]
    df = pd.DataFrame({"o": q["open"], "h": q["high"], "l": q["low"], "c": q["close"]},
                      index=pd.to_datetime(r["timestamp"], unit="s", utc=True).tz_convert(NY)).dropna()
    df = df[df.index.minute == 0]                               # drop the live partial bar
    df["day"] = (df.index + pd.Timedelta(hours=6)).date           # futures day starts 18:00 NY
    return df

def daily_bars(h):
    d = h.groupby("day").agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"), n=("c", "size"))
    d = d[d.n >= 10]; d.index = pd.to_datetime(d.index); return d

def rsi(c, n):
    d = c.diff(); up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean(); dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)

def atr(d, n=14):
    tr = pd.concat([d.h - d.l, (d.h - d.c.shift()).abs(), (d.l - d.c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

# ---- strategies: each returns a daily P&L series (fraction of price) and a trade count ----
def from_positions(d, pos):
    pos = pos.fillna(0).shift(1).fillna(0)                        # decided at close, held next day
    ret = d.c.pct_change().fillna(0)
    pnl = pos * ret - COST * pos.diff().abs().fillna(0)
    pnl.attrs["entries"] = ((pos != 0) & (pos.shift().fillna(0) != pos)).astype(int)
    return pnl, None

def sma_trend(d, fast, slow, short):
    if fast >= slow: return None
    f, s = d.c.rolling(fast).mean(), d.c.rolling(slow).mean()
    return from_positions(d, np.where(f > s, 1.0, -1.0 if short else 0.0) * s.notna())

def donchian(d, n, m, short):
    if m >= n: return None
    hi, lo = d.h.rolling(n).max().shift(), d.l.rolling(n).min().shift()
    xhi, xlo = d.h.rolling(m).max().shift(), d.l.rolling(m).min().shift()
    pos, p = [], 0.0
    for c, a, b, xa, xb in zip(d.c, hi, lo, xhi, xlo):
        if p == 0:
            if c > a: p = 1.0
            elif short and c < b: p = -1.0
        elif p > 0 and c < xb: p = 0.0
        elif p < 0 and c > xa: p = 0.0
        pos.append(p)
    return from_positions(d, pd.Series(pos, d.index))

def rsi_revert(d, n, lo, hi, hold, short):
    r = rsi(d.c, n); pos, p, k = [], 0.0, 0
    for v in r:
        if p == 0:
            if v < lo: p, k = 1.0, 0
            elif short and v > 100 - lo: p, k = -1.0, 0
        else:
            k += 1
            if (p > 0 and v > hi) or (p < 0 and v < 100 - hi) or k >= hold: p = 0.0
        pos.append(p)
    return from_positions(d, pd.Series(pos, d.index))

def orb(h, d, rng_hour, target_r, stop_rule, side, trend):
    """Opening range = the NY hour bar at rng_hour; trade the first hourly close outside it, flat at 15:00 bar close."""
    sma = d.c.rolling(trend).mean().shift() if trend else None
    prevc = d.c.shift()
    pnl = pd.Series(0.0, d.index); n = 0
    for day, g in h.groupby("day"):
        day = pd.Timestamp(day)
        if day not in d.index: continue
        g = g[(g.index.hour >= rng_hour) & (g.index.hour <= 15) & (g.index.date == day.date())]
        if len(g) < 3 or g.index[0].hour != rng_hour: continue
        H, L = g.h.iloc[0], g.l.iloc[0]
        allow_long = side in ("long", "both") and (sma is None or prevc[day] > sma[day])
        allow_short = side in ("short", "both") and (sma is None or prevc[day] < sma[day])
        for i in range(1, len(g)):
            c = g.c.iloc[i]; direction = 0
            if allow_long and c > H: direction = 1
            elif allow_short and c < L: direction = -1
            if not direction: continue
            entry = c
            stop = (H + L) / 2 if stop_rule == "mid" else (L if direction > 0 else H)
            risk = abs(entry - stop)
            if risk <= 0: break
            tgt = entry + direction * target_r * risk; exitp = g.c.iloc[-1]
            for j in range(i + 1, len(g)):
                hi_, lo_ = g.h.iloc[j], g.l.iloc[j]
                if (direction > 0 and lo_ <= stop) or (direction < 0 and hi_ >= stop): exitp = stop; break
                if (direction > 0 and hi_ >= tgt) or (direction < 0 and lo_ <= tgt): exitp = tgt; break
            pnl[day] = direction * (exitp - entry) / entry - 2 * COST; n += 1
            break
    pnl.attrs["entries"] = (pnl != 0).astype(int)
    return pnl, n

def stats(pnl, trades=None):
    if len(pnl) == 0: return {}
    eq = (1 + pnl).cumprod(); dd = (eq / eq.cummax() - 1).min()
    sd = pnl.std()
    return {"return_pct": round(float(eq.iloc[-1] - 1) * 100, 2), "sharpe": round(float(pnl.mean() / sd * np.sqrt(252)), 2) if sd > 0 else 0.0,
            "max_dd_pct": round(float(dd) * 100, 2), "days": len(pnl), "trades": trades}

def window(s, a, b):
    w = s[(s.index > pd.Timestamp(a)) & (s.index <= pd.Timestamp(b))] if a else s[s.index <= pd.Timestamp(b)]
    w.attrs = s.attrs; return w

def count_trades(pnl):
    e = pnl.attrs.get("entries")
    return int(e.reindex(pnl.index).fillna(0).sum()) if e is not None else int((pnl != 0).sum())

GRIDS = {
    "sma_trend": (sma_trend, dict(fast=[5, 10, 20, 50], slow=[20, 50, 100, 200], short=[False, True])),
    "donchian": (donchian, dict(n=[10, 20, 55], m=[5, 10, 20], short=[False, True])),
    "rsi_revert": (rsi_revert, dict(n=[2, 3, 5, 14], lo=[10, 20, 30], hi=[50, 60, 70], hold=[3, 5, 10], short=[False, True])),
    "orb_hourly": (None, dict(rng_hour=[8, 9], target_r=[1.0, 1.5, 2.0, 3.0], stop_rule=["mid", "range"], side=["long", "short", "both"], trend=[0, 20, 50])),
}

def run(name, params, h, d):
    if name == "orb_hourly": return orb(h, d, **params)
    return GRIDS[name][0](d, **params)

def main(src, out):
    h = load(src); d = daily_bars(h)
    end = d.index[-1].strftime("%Y-%m-%d")
    W = {"train": (None, TRAIN_END), "valid": (TRAIN_END, VALID_END), "test": (VALID_END, end)}
    bh = d.c.pct_change().fillna(0)
    log = {"windows": {k: [str(window(bh, *v).index[0].date()), str(window(bh, *v).index[-1].date())] for k, v in W.items()},
           "buy_hold": {k: stats(window(bh, *v)) for k, v in W.items()}, "rounds": []}
    # Round 1: grid search per family on train, top 5 to validation, keep the best on validation
    r1, curves = {}, {}
    for name, (_, grid) in GRIDS.items():
        rows = []
        for combo in itertools.product(*grid.values()):
            p = dict(zip(grid.keys(), combo)); res = run(name, p, h, d)
            if res is None: continue
            pnl, _ = res; tr = window(pnl, *W["train"]); nt = count_trades(tr)
            if nt < MIN_TRADES[name]: continue
            rows.append((stats(tr, nt)["sharpe"], p, pnl))
        rows.sort(key=lambda x: -x[0])
        best = None
        for sh, p, pnl in rows[:10]:
            v = window(pnl, *W["valid"]); sv = stats(v, count_trades(v))
            if sv["trades"] < 2: continue                         # too few to judge on validation
            if best is None or sv["sharpe"] > best[2]["sharpe"]: best = (p, sh, sv, pnl)
        if best is None:                                          # nobody traded enough on validation: keep the train winner
            sh, p, pnl = rows[0]; v = window(pnl, *W["valid"]); best = (p, sh, stats(v, count_trades(v)), pnl)
        r1[name] = {"params": best[0], "train_sharpe": best[1], "valid": best[2], "tested_combos": len(rows)}
        curves[name] = best[3]
        print(f"R1 {name:11s} {best[0]} train Sharpe {best[1]:+.2f} -> valid {best[2]}")
    log["rounds"].append({"round": 1, "what": "grid search per family on train, top 5 re-ranked on validation", "result": r1})
    # Round 2: try to improve each winner with a volatility filter (skip days when ATR% is in the top third, judged on train)
    d["atrp"] = atr(d) / d.c
    cut = window(d.atrp.dropna(), *W["train"]).quantile(0.67)
    r2 = {}
    for name, pnl in list(curves.items()):
        calm = (d.atrp.shift() <= cut).reindex(pnl.index).fillna(False)
        filt = pnl.where(calm, 0.0); filt.attrs = {"entries": pnl.attrs["entries"].where(calm, 0)}
        v0, v1 = window(pnl, *W["valid"]), window(filt, *W["valid"])
        s0, s1 = stats(v0, count_trades(v0)), stats(v1, count_trades(v1))
        keep = s1["sharpe"] > s0["sharpe"] + 0.1
        r2[name] = {"valid_without": s0, "valid_with_filter": s1, "kept": keep}
        if keep: curves[name] = filt
        print(f"R2 {name:11s} vol filter valid Sharpe {s0['sharpe']:+.2f} -> {s1['sharpe']:+.2f} {'KEPT' if keep else 'dropped'}")
    log["rounds"].append({"round": 2, "what": "volatility filter (skip high-ATR days), kept only if validation Sharpe improves by > 0.1", "result": r2})
    # Round 3: equal-weight blend of the families with positive validation Sharpe
    good = [n for n, p in curves.items() if stats(window(p, *W["valid"]))["sharpe"] > 0]
    blend = sum(curves[n] for n in good) / max(len(good), 1) if good else pd.Series(0.0, d.index)
    blend.attrs = {"entries": sum(curves[n].attrs["entries"] for n in good)} if good else {}
    vb = window(blend, *W["valid"]); sb = stats(vb, count_trades(vb))
    best_single = max(curves, key=lambda n: stats(window(curves[n], *W["valid"]))["sharpe"])
    bs = stats(window(curves[best_single], *W["valid"]), count_trades(window(curves[best_single], *W["valid"])))
    final_name = "blend" if good and sb["sharpe"] >= bs["sharpe"] else best_single
    final = blend if final_name == "blend" else curves[best_single]
    log["rounds"].append({"round": 3, "what": "equal-weight blend of families with positive validation Sharpe vs best single", "blend_of": good,
                          "blend_valid": sb, "best_single": best_single, "best_single_valid": bs, "chosen": final_name})
    print(f"R3 blend of {good}: valid {sb} vs best single {best_single} {bs} -> {final_name}")
    # Final: the untouched test window, once
    ft = window(final, *W["test"])
    log["final"] = {"chosen": final_name, "members": good if final_name == "blend" else [best_single],
                    "test": stats(ft, count_trades(ft)), **{k: stats(window(final, *W[k]), count_trades(window(final, *W[k]))) for k in ("train", "valid")}}
    bt = window(blend, *W["test"]); log["blend_test_for_reference"] = stats(bt, count_trades(bt))
    log["families_test"] = {n: stats(window(p, *W["test"]), count_trades(window(p, *W["test"]))) for n, p in curves.items()}
    print("FINAL", log["final"]); print("BUY&HOLD", log["buy_hold"]); print("BLEND TEST (reference only)", log["blend_test_for_reference"])
    for n, v in log["families_test"].items(): print("TEST (reference only)", n, v)
    log["equity"] = {"dates": [str(x.date()) for x in d.index],
                     "buy_hold": list(np.round((1 + bh).cumprod().values, 5)),
                     "final": list(np.round((1 + final.reindex(d.index).fillna(0)).cumprod().values, 5)),
                     **{n: list(np.round((1 + p.reindex(d.index).fillna(0)).cumprod().values, 5)) for n, p in curves.items()}}
    # what does the final strategy want today?
    log["today"] = today_signals(h, d, r1, r2, good if final_name == "blend" else [best_single])
    json.dump(log, open(out, "w"), indent=1, default=str)

def today_signals(h, d, r1, r2, members):
    out = {}
    for n in members:
        p = r1[n]["params"]
        if n == "sma_trend":
            f, s = d.c.rolling(p["fast"]).mean().iloc[-1], d.c.rolling(p["slow"]).mean().iloc[-1]
            out[n] = {"position": 1 if f > s else (-1 if p["short"] else 0), "fast": round(f, 1), "slow": round(s, 1)}
        elif n == "donchian":
            pnl, _ = donchian(d, **p); out[n] = {"note": "position follows breakout state", "last_day_pnl": float(pnl.iloc[-1])}
        elif n == "rsi_revert":
            out[n] = {"rsi": round(float(rsi(d.c, p["n"]).iloc[-1]), 1), "buy_below": p["lo"], "sell_above": 100 - p["lo"] if p["short"] else None}
        elif n == "orb_hourly":
            out[n] = {"rule": p, "note": "decides intraday after the range hour"}
    return out

if __name__ == "__main__":
    main(*sys.argv[1:3])
