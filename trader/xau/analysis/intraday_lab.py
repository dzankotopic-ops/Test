"""5-minute setups for gold, long and short, tested on 60 days of GC=F bars.
Parameters are picked on the first 40 trading days (train) and judged once on the rest (test).

S1 range breakout : after N quiet bars (range height < k x ATR), trade the first close outside the range
S2 EMA pullback   : in a trend (EMA21 vs EMA50 and price vs VWAP), buy/sell the first touch of EMA21 that closes back with the trend
Every trade: stop and target fixed at entry, exit after at most 36 bars (3 hours), cost 0.30 $ per side.
Usage: python3 intraday_lab.py GC-60d5m-full.json out.json
"""
import json, sys, itertools, zoneinfo
import numpy as np, pandas as pd
DE = zoneinfo.ZoneInfo("Europe/Berlin"); COST = 0.30; MAXBARS = 36; TRAIN_DAYS = 40

def load(path):
    r = json.load(open(path))["chart"]["result"][0]; q = r["indicators"]["quote"][0]
    df = pd.DataFrame({k: q[k] for k in ("open", "high", "low", "close", "volume")},
                      index=pd.to_datetime(r["timestamp"], unit="s", utc=True).tz_convert(DE)).dropna()
    df = df[(df.index.second == 0) & (df.index.minute % 5 == 0)]
    df["day"] = (df.index.tz_convert("America/New_York") + pd.Timedelta(hours=6)).date   # futures day starts 18:00 NY = 00:00 Berlin
    tr = pd.concat([df.high - df.low, (df.high - df.close.shift()).abs(), (df.low - df.close.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    df["ema9"] = df.close.ewm(span=9).mean(); df["ema21"] = df.close.ewm(span=21).mean(); df["ema50"] = df.close.ewm(span=50).mean()
    pv = (df.close * df.volume).groupby(df.day).cumsum(); vv = df.volume.groupby(df.day).cumsum()
    df["vwap"] = (pv / vv.replace(0, np.nan)).ffill()
    return df

def simulate(df, signals):
    """signals: list of (i, direction, stop, target). One trade at a time; returns list of trade dicts."""
    H, L, C = df.high.values, df.low.values, df.close.values; out, busy = [], -1
    for i, d, stop, tgt in signals:
        if i <= busy or i + 1 >= len(C): continue
        entry = C[i]; risk = abs(entry - stop)
        if risk <= 0: continue
        exitp, j = C[min(i + MAXBARS, len(C) - 1)], min(i + MAXBARS, len(C) - 1)
        for k in range(i + 1, min(i + MAXBARS, len(C) - 1) + 1):
            if (d > 0 and L[k] <= stop) or (d < 0 and H[k] >= stop): exitp, j = stop, k; break
            if (d > 0 and H[k] >= tgt) or (d < 0 and L[k] <= tgt): exitp, j = tgt, k; break
        pnl = d * (exitp - entry) - 2 * COST
        out.append({"i": i, "time": str(df.index[i]), "day": str(df.day.iloc[i]), "dir": d, "entry": entry, "stop": stop, "target": tgt,
                    "exit": exitp, "R": pnl / risk, "usd": pnl}); busy = j
    return out

def s1(df, n, k, target_r, stop_rule, filt):
    H, L, C, A = df.high.values, df.low.values, df.close.values, df.atr.values; vw, e21, e50 = df.vwap.values, df.ema21.values, df.ema50.values
    sig = []
    for i in range(n + 1, len(C)):
        hi, lo = H[i - n:i].max(), L[i - n:i].min()
        if np.isnan(A[i - 1]) or hi - lo > k * A[i - 1] * np.sqrt(n): continue
        for d in (1, -1):
            if (d > 0 and C[i] > hi) or (d < 0 and C[i] < lo):
                if filt and not ((d > 0 and C[i] > vw[i] and e21[i] > e50[i]) or (d < 0 and C[i] < vw[i] and e21[i] < e50[i])): continue
                stop = (hi + lo) / 2 if stop_rule == "mid" else (lo if d > 0 else hi)
                sig.append((i, d, stop, C[i] + d * target_r * abs(C[i] - stop)))
    return sig

def s2(df, target_r, buffer_atr, need_vwap):
    O, H, L, C, A = df.open.values, df.high.values, df.low.values, df.close.values, df.atr.values
    e9, e21, e50, vw = df.ema9.values, df.ema21.values, df.ema50.values, df.vwap.values
    sig = []
    for i in range(60, len(C)):
        if np.isnan(A[i]): continue
        up = e21[i] > e50[i] and (not need_vwap or C[i] > vw[i]); dn = e21[i] < e50[i] and (not need_vwap or C[i] < vw[i])
        if up and L[i] <= e21[i] and C[i] > e21[i] and C[i] > O[i] and min(L[i - 3:i]) > e21[i - 1]:
            stop = min(L[i - 2:i + 1]) - buffer_atr * A[i]; sig.append((i, 1, stop, C[i] + target_r * (C[i] - stop)))
        if dn and H[i] >= e21[i] and C[i] < e21[i] and C[i] < O[i] and max(H[i - 3:i]) < e21[i - 1]:
            stop = max(H[i - 2:i + 1]) + buffer_atr * A[i]; sig.append((i, -1, stop, C[i] - target_r * (stop - C[i])))
    return sig

GRIDS = {"S1 Range-Ausbruch": (s1, dict(n=[12, 18, 24], k=[0.6, 0.9, 1.2], target_r=[1.0, 1.5, 2.0], stop_rule=["mid", "range"], filt=[False, True])),
         "S2 EMA-Pullback": (s2, dict(target_r=[1.0, 1.5, 2.0, 3.0], buffer_atr=[0.0, 0.25, 0.5], need_vwap=[False, True]))}

def summary(tr):
    if not tr: return {"trades": 0}
    R = np.array([t["R"] for t in tr]); usd = np.array([t["usd"] for t in tr])
    return {"trades": len(tr), "win_rate": round(float((R > 0).mean()), 3), "avg_R": round(float(R.mean()), 3), "sum_R": round(float(R.sum()), 1),
            "usd_per_oz": round(float(usd.sum()), 1), "longs": sum(t["dir"] > 0 for t in tr), "shorts": sum(t["dir"] < 0 for t in tr)}

def main(src, out):
    df = load(src); days = sorted(df.day.unique()); split = days[TRAIN_DAYS]; today = days[-1]
    res = {"train_days": [str(days[0]), str(days[TRAIN_DAYS - 1])], "test_days": [str(split), str(days[-2])], "families": {}}
    for name, (fn, grid) in GRIDS.items():
        rows = []
        for combo in itertools.product(*grid.values()):
            p = dict(zip(grid, combo)); tr = simulate(df, fn(df, **p))
            trn = [t for t in tr if t["day"] < str(split)]; tst = [t for t in tr if str(split) <= t["day"] < str(today)]
            s = summary(trn)
            if s["trades"] >= 30: rows.append((s["avg_R"], p, s, summary(tst), tr))
        rows.sort(key=lambda x: -x[0]); best = rows[0]
        res["families"][name] = {"params": best[1], "train": best[2], "test": best[3], "variants": len(rows),
                                 "today": [t for t in best[4] if t["day"] == str(today)],
                                 "top5_test": [{"params": r[1], "train_avgR": r[0], "test": r[3]} for r in rows[:5]]}
        print(name, best[1]); print("  train", best[2]); print("  test ", best[3])
        for r in rows[1:5]: print("   also", r[1], "train", r[0], "test avgR", r[3].get("avg_R"), "n", r[3]["trades"])
    json.dump(res, open(out, "w"), indent=1, default=str)

if __name__ == "__main__":
    main(*sys.argv[1:3])
