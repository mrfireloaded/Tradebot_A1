# signal_bot_sniper_ultra.py
# FINAL SNIPER — Ultra (SMC + FVG Strength + Volume Filter + Live Logs + Heartbeat)
# Paste into Pydroid 3 and run. Requirements: requests, numpy, matplotlib

import requests, time, io, datetime, math, sys
import numpy as np
import matplotlib.pyplot as plt

# ---------------- USER CONFIG (YOUR CONFIRMED KEYS) ----------------
API_KEY   = "ea8f686ae37d43abab62b54a4eb2cf87"   # TwelveData
BOT_TOKEN = "8583973969:AAFGKPQ66bOJi6uNB2oskpfVWZHmIlhRP6k"
CHAT_ID   = "-1003319570833"  # Telegram channel numeric ID

# ---------------- PER-PAIR CONFIG ----------------
PAIRS_CFG = {
    "XAU/USD": {
        "symbol":"XAU/USD",
        "analysis_tfs":["1day","4h","2h","1h","30min","15min","5min"],
        "signal_tf":"1min",
        "min_score":8,        # XAU stricter
        "retest_bars":8,
        "cooldown_s":60*20
    },
    "GBP/USD": {
        "symbol":"GBP/USD",
        "analysis_tfs":["1day","4h","2h","1h","30min"],
        "signal_tf":"15min",
        "min_score":7,
        "retest_bars":6,
        "cooldown_s":60*12
    },
    "EUR/USD": {
        "symbol":"EUR/USD",
        "analysis_tfs":["1day","4h","2h","1h","30min"],
        "signal_tf":"15min",
        "min_score":7,
        "retest_bars":6,
        "cooldown_s":60*12
    }
}

# Global tuning (you can change later)
LOOP_INTERVAL = 60             # seconds between full scans
CANDLES_PER_TF = 400
ATR_PERIOD = 14
MIN_TF_ALIGNMENT = 5           # require >=5 TF agreeing (HTF vote)
HEARTBEAT_MINUTES = 10
LOG_VERBOSE = True             # prints logs to Pydroid console
FVG_MIN_REL_SIZE = 0.0025      # minimum FVG gap relative to price (0.25%)
VOLUME_SPIKE_MULT = 1.8        # last volume > average * this multiplier

# ---------------- TELEGRAM HELPERS ----------------
def tg_send_text(text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        r = requests.post(url, data={"chat_id": CHAT_ID, "text": text, "parse_mode":"HTML"}, timeout=12)
        if LOG_VERBOSE: print(f"[TG] text {r.status_code}")
        return r.ok
    except Exception as e:
        print("[TG] text error", e); return False

def tg_send_photo_bytes(buf, caption=""):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    files = {"photo": ("chart.png", buf.getvalue())}
    data = {"chat_id": CHAT_ID, "caption": caption, "parse_mode":"HTML"}
    try:
        r = requests.post(url, files=files, data=data, timeout=30)
        if LOG_VERBOSE: print(f"[TG] photo {r.status_code}")
        return r.ok
    except Exception as e:
        print("[TG] photo error", e); return False

# ---------------- FETCH DATA (TwelveData) ----------------
def fetch_td(symbol, interval, output=CANDLES_PER_TF):
    """
    Fetch candles from TwelveData. Tries both symbol formats (with and without slash).
    Returns list of candles ordered old -> new.
    Each candle dict: {"t","o","h","l","c","v"(optional)}
    """
    def _req(sym):
        url = f"https://api.twelvedata.com/time_series?symbol={sym}&interval={interval}&outputsize={output}&apikey={API_KEY}"
        r = requests.get(url, timeout=12).json()
        return r
    try:
        r = _req(symbol)
        if "values" in r:
            vals = r["values"][::-1]
            candles = []
            for v in vals:
                c = {"t":v.get("datetime"), "o":float(v["open"]), "h":float(v["high"]), "l":float(v["low"]), "c":float(v["close"])}
                if "volume" in v: 
                    try: c["v"] = float(v.get("volume", 0))
                    except: c["v"] = None
                candles.append(c)
            return candles
        # fallback try without slash
        alt = symbol.replace("/","")
        r2 = _req(alt)
        if "values" in r2:
            vals = r2["values"][::-1]
            candles = []
            for v in vals:
                c = {"t":v.get("datetime"), "o":float(v["open"]), "h":float(v["high"]), "l":float(v["low"]), "c":float(v["close"])}
                if "volume" in v:
                    try: c["v"] = float(v.get("volume", 0))
                    except: c["v"] = None
                candles.append(c)
            return candles
        if LOG_VERBOSE: print("[TD] no values:", r.get("message") if isinstance(r, dict) else r)
        return None
    except Exception as e:
        print("[TD] error:", e); return None

# ---------------- INDICATORS ----------------
def ema_val(prices, period):
    if len(prices) < period: return None
    arr = np.array(prices, dtype=float)
    k = 2/(period+1)
    ema = arr[0]
    for p in arr[1:]:
        ema = p * k + ema * (1-k)
    return float(ema)

def rsi_val(prices, period=14):
    if len(prices) < period+1: return None
    deltas = np.diff(np.array(prices))
    ups = np.where(deltas>0, deltas, 0); downs = np.where(deltas<0, -deltas, 0)
    avg_up = np.mean(ups[-period:]); avg_down = np.mean(downs[-period:]) if np.mean(downs[-period:])>0 else 1e-9
    rs = avg_up/avg_down
    return float(100 - (100/(1+rs)))

def macd_hist_val(prices, fast=12, slow=26, signal=9):
    if len(prices) < slow + signal: return None
    fast_ema = ema_val(prices, fast); slow_ema = ema_val(prices, slow)
    if fast_ema is None or slow_ema is None: return None
    macd = fast_ema - slow_ema
    macd_series = []
    for i in range(slow, len(prices)):
        sub = prices[:i+1]
        ef = ema_val(sub, fast); es = ema_val(sub, slow)
        if ef is None or es is None: continue
        macd_series.append(ef - es)
    if not macd_series: return None
    sig = np.mean(macd_series[-signal:]) if len(macd_series)>=1 else 0
    return float(macd - sig)

def atr_from_candles(candles, period=ATR_PERIOD):
    highs = np.array([c["h"] for c in candles]); lows = np.array([c["l"] for c in candles]); closes = np.array([c["c"] for c in candles])
    tr1 = highs - lows
    tr2 = np.abs(highs - np.roll(closes,1))
    tr3 = np.abs(lows - np.roll(closes,1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)[1:]
    if len(tr) < period: return None
    return float(np.mean(tr[-period:]))

# ---------------- SMC helpers (BOS, CHoCH, FVG, OB) ----------------
def find_last_swings(candles, lookback=120):
    highs = [c["h"] for c in candles]; lows = [c["l"] for c in candles]
    n=len(candles); last_high=None; last_low=None
    for i in range(n-4, max(4, n-lookback), -1):
        if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
            last_high = highs[i]; break
    for i in range(n-4, max(4, n-lookback), -1):
        if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
            last_low = lows[i]; break
    return last_high, last_low

def detect_bos_ch(candles):
    last_high, last_low = find_last_swings(candles, lookback=120)
    close = candles[-1]["c"]
    if last_high and close > last_high: return "BOS_UP", last_high, last_low
    if last_low and close < last_low: return "BOS_DOWN", last_high, last_low
    return None, last_high, last_low

def detect_choch(candles, lookback=40, threshold_rel=0.0025):
    n = len(candles)
    if n < 20: return None
    closes = [c["c"] for c in candles]; highs=[c["h"] for c in candles]; lows=[c["l"] for c in candles]
    pivot_idx=None; pivot_type=None
    for i in range(n-4, max(4, n-lookback), -1):
        if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
            pivot_idx=i; pivot_type="high"; break
        if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
            pivot_idx=i; pivot_type="low"; break
    if pivot_idx is None: return None
    pre_avg = np.mean(closes[max(0,pivot_idx-6):pivot_idx])
    post_avg = np.mean(closes[pivot_idx+1: min(n, pivot_idx+7)])
    if pre_avg == 0: return None
    rel = (post_avg - pre_avg) / pre_avg
    if pivot_type=="high" and rel < -threshold_rel: return "CHOCH_DOWN"
    if pivot_type=="low" and rel > threshold_rel: return "CHOCH_UP"
    return None

def detect_fvg(candles):
    # conservative FVG detection between last 3 candles
    if len(candles) < 6: return None
    # bullish FVG if current low > candle[-3] high (gap up)
    if candles[-1]["l"] > candles[-3]["h"]:
        return ("FVG_BULL", candles[-3]["h"], candles[-1]["l"])
    if candles[-1]["h"] < candles[-3]["l"]:
        return ("FVG_BEAR", candles[-3]["l"], candles[-1]["h"])
    return None

def detect_order_block(candles):
    n=len(candles)
    for i in range(max(0, n-10), n-2):
        a=candles[i]; b=candles[i+1]
        if a["c"] < a["o"] and b["c"] > b["o"] and b["c"] > a["o"] and b["o"] < a["c"]:
            return ("OB_BULL", b["h"], b["l"], i)
        if a["c"] > a["o"] and b["c"] < b["o"] and b["c"] < a["o"] and b["o"] > a["c"]:
            return ("OB_BEAR", b["h"], b["l"], i)
    return None

def premium_discount_filter(candles):
    lh,ll = find_last_swings(candles, lookback=300)
    if not lh or not ll: return None
    mid = (lh + ll)/2; price = candles[-1]["c"]
    return "DISCOUNT" if price < mid else "PREMIUM"

# ---------------- FVG Strength & Volume Filters ----------------
def fvg_strength(candles, atr=None):
    """
    Return a strength score for last FVG (0 = none / weak, higher = stronger).
    Uses relative gap size vs ATR and vs price.
    """
    fvg = detect_fvg(candles)
    if not fvg: return 0, None
    kind, top, bot = fvg
    current = candles[-1]["c"]
    gap_size = abs(top - bot)
    rel = gap_size / current if current else 0
    # prefer if gap >= threshold or gap >= 0.5 * ATR
    score = 0
    if rel >= FVG_MIN_REL_SIZE: score += 2
    if atr and gap_size >= (0.5 * atr): score += 2
    # check how many candles since gap formed (fresh is stronger)
    # if the preceding impulsive candle size is big, add score
    prev_body = abs(candles[-3]["c"] - candles[-3]["o"])
    if prev_body > (0.5 * gap_size): score += 1
    return score, fvg

def volume_spike_ok(candles, lookback=30):
    """
    Return True if last candle's volume shows smart-money spike:
    last_vol > avg_vol * VOLUME_SPIKE_MULT OR last_vol > previous_vol * some_mult
    """
    vols = [c.get("v") for c in candles if c.get("v") is not None]
    if len(vols) < 10: 
        return False, 0.0
    last_v = candles[-1].get("v") or 0.0
    avg = float(np.mean(vols[-lookback:]))
    if avg <= 0: return False, 0.0
    mult = last_v / avg
    ok = mult >= VOLUME_SPIKE_MULT
    return ok, mult

# ---------------- Analyze TF ----------------
def analyze_tf(symbol, tf):
    candles = fetch_td(symbol, tf, output=CANDLES_PER_TF)
    if not candles or len(candles) < 40: return None
    closes = [c["c"] for c in candles]
    ema50 = ema_val(closes, 50)
    rsi = rsi_val(closes, 14)
    macd = macd_hist_val(closes)
    bos, lh, ll = detect_bos_ch(candles)
    choch = detect_choch(candles)
    fvg = detect_fvg(candles)
    ob = detect_order_block(candles)
    pd = premium_discount_filter(candles)
    atr = atr_from_candles(candles, period=ATR_PERIOD)
    vol_ok, vol_mult = volume_spike_ok(candles)
    return {"candles":candles,"ema50":ema50,"rsi":rsi,"macdh":macd,"bos":bos,"choch":choch,"fvg":fvg,"ob":ob,"pd":pd,"atr":atr,"last_high":lh,"last_low":ll,"vol_ok":vol_ok,"vol_mult":vol_mult}

# ---------------- HTF SR zones ----------------
def get_htf_swing_zones(results):
    resistances=[]; supports=[]
    for tf in ("1day","4h","2h","1h"):
        r = results.get(tf)
        if not r: continue
        if r.get("last_high"): resistances.append((float(r["last_high"]), tf))
        if r.get("last_low"): supports.append((float(r["last_low"]), tf))
    sample = next(iter(results.values()))
    current = sample["candles"][-1]["c"]
    resistances.sort(key=lambda x: abs(x[0]-current))
    supports.sort(key=lambda x: abs(x[0]-current))
    return {"resistances":resistances, "supports":supports}

# ---------------- Entry confirmation (strict + FVG+Volume) ----------------
def entry_confirmation(low_tf, low_candles, final_dir, zones, results, require_retest_bars):
    closes=[c["c"] for c in low_candles]; highs=[c["h"] for c in low_candles]; lows=[c["l"] for c in low_candles]
    last_close = closes[-1]
    # require at least one HTF CHoCH matching direction
    htf_choch_ok = False
    for tf in ("1h","2h","4h","1day"):
        r = results.get(tf)
        if not r: continue
        if r.get("choch"):
            if final_dir=="BUY" and "UP" in r["choch"]: htf_choch_ok=True; break
            if final_dir=="SELL" and "DOWN" in r["choch"]: htf_choch_ok=True; break
    if not htf_choch_ok:
        if LOG_VERBOSE: print("[ENTRY] Rejected: missing HTF CHoCH")
        return False
    # Volume confirmation on low TF and at least one HTF vol_ok
    low_vol_ok, low_mult = volume_spike_ok(low_candles)
    any_htf_vol = any((results.get(tf) and results.get(tf).get("vol_ok")) for tf in results.keys())
    if not (low_vol_ok or any_htf_vol):
        if LOG_VERBOSE: print("[ENTRY] Rejected: no volume confirmation (low_vol_ok:", low_vol_ok, "htf_any:", any_htf_vol,")")
        return False
    # FVG strength check on low TF and some HTFs
    low_atr = atr_from_candles(low_candles, period=ATR_PERIOD)
    fvg_score, fvg_data = fvg_strength(low_candles, atr=low_atr)
    if fvg_score < 2:
        if LOG_VERBOSE: print("[ENTRY] Rejected: FVG too weak (score:", fvg_score, ")")
        return False
    # HTF SR break + retest + rejection (same as before)
    if final_dir=="BUY" and zones["supports"]:
        s_price = zones["supports"][0][0]
        broke = any(c > s_price for c in closes[-(require_retest_bars*3):])
        if broke:
            buffer = max((max(highs[-6:]) - min(lows[-6:]))*0.6, last_close*0.0006)
            for i in range(-require_retest_bars,0):
                if lows[i] <= s_price + buffer:
                    if low_candles[i]["c"] > low_candles[i]["o"] or (highs[i]-low_candles[i]["c"]) > (low_candles[i]["c"]-lows[i]):
                        return True
    if final_dir=="SELL" and zones["resistances"]:
        r_price = zones["resistances"][0][0]
        broke = any(c < r_price for c in closes[-(require_retest_bars*3):])
        if broke:
            buffer = max((max(highs[-6:]) - min(lows[-6:]))*0.6, last_close*0.0006)
            for i in range(-require_retest_bars,0):
                if highs[i] >= r_price - buffer:
                    if low_candles[i]["c"] < low_candles[i]["o"] or (low_candles[i]["c"]-lows[i]) > (highs[i]-low_candles[i]["c"]):
                        return True
    # FVG / OB retest on low TF
    if fvg_data:
        # require price currently inside or near FVG zone (retest)
        top, bot = fvg_data[1], fvg_data[2]
        inside = any((c < top and c > bot) for c in closes[-require_retest_bars:])
        if inside:
            if LOG_VERBOSE: print("[ENTRY] FVG retest OK")
            return True
    if low_res := results.get(low_tf):
        if low_res.get("ob"):
            ob = low_res["ob"]; top,bot = ob[1],ob[2]; entered = any((c < top and c > bot) for c in closes[-require_retest_bars:])
            if entered:
                return True
    # CHoCH confirmation on any HTF + directional candle on low TF
    for tf,r in results.items():
        if r.get("choch"):
            if ("UP" in r["choch"] and final_dir=="BUY") or ("DOWN" in r["choch"] and final_dir=="SELL"):
                if final_dir=="BUY" and closes[-1] > closes[-2]: return True
                if final_dir=="SELL" and closes[-1] < closes[-2]: return True
    return False

# ---------------- Evaluate pair (SNIPER strict + new filters) ----------------
def evaluate_pair(pair_key):
    cfg = PAIRS_CFG[pair_key]; sym = cfg["symbol"]
    results={}
    for tf in cfg["analysis_tfs"]:
        res = analyze_tf(sym, tf)
        if not res:
            if LOG_VERBOSE: print(f"[{pair_key}] {tf}: no data"); return None
        results[tf]=res
    # voting
    votes=[]
    for tf,res in results.items():
        if res["ema50"] is None or res["rsi"] is None or res["macdh"] is None:
            votes.append(None); continue
        bullish = (res["rsi"]>50 and res["macdh"]>0 and res["candles"][-1]["c"]>res["ema50"])
        bearish = (res["rsi"]<50 and res["macdh"]<0 and res["candles"][-1]["c"]<res["ema50"])
        if res["bos"]=="BOS_UP" or res["choch"]=="CHOCH_UP" or (res["fvg"] and res["fvg"][0]=="FVG_BULL"): bullish=True
        if res["bos"]=="BOS_DOWN" or res["choch"]=="CHOCH_DOWN" or (res["fvg"] and res["fvg"][0]=="FVG_BEAR"): bearish=True
        if bullish and not bearish: votes.append("BUY")
        elif bearish and not bullish: votes.append("SELL")
        else: votes.append(None)
    nonnull=[v for v in votes if v]; 
    if not nonnull: 
        if LOG_VERBOSE: print(f"[{pair_key}] No non-null votes"); return None
    buy_count = nonnull.count("BUY"); sell_count = nonnull.count("SELL")
    if buy_count >= MIN_TF_ALIGNMENT and buy_count > sell_count: final_dir="BUY"
    elif sell_count >= MIN_TF_ALIGNMENT and sell_count > buy_count: final_dir="SELL"
    else: 
        if LOG_VERBOSE: print(f"[{pair_key}] TF alignment insufficient (buy:{buy_count}, sell:{sell_count})"); return None
    # SMC score (HTF weighted)
    score=0
    htfs = ["1day","4h","2h","1h"]
    for tf in htfs:
        r = results.get(tf)
        if not r: continue
        if r["bos"] and ("UP" in r["bos"] and final_dir=="BUY" or "DOWN" in r["bos"] and final_dir=="SELL"): score += 2
        if r["choch"] and ("UP" in r["choch"] and final_dir=="BUY" or "DOWN" in r["choch"] and final_dir=="SELL"): score += 2
        if r["fvg"] and ((r["fvg"][0]=="FVG_BULL" and final_dir=="BUY") or (r["fvg"][0]=="FVG_BEAR" and final_dir=="SELL")): score += 1
        if r["ob"] and ((r["ob"][0]=="OB_BULL" and final_dir=="BUY") or (r["ob"][0]=="OB_BEAR" and final_dir=="SELL")): score += 1
        if r["pd"] and ((r["pd"]=="DISCOUNT" and final_dir=="BUY") or (r["pd"]=="PREMIUM" and final_dir=="SELL")): score += 1
    # require minimum per-pair sniper score
    if score < cfg["min_score"]:
        if LOG_VERBOSE: print(f"[{pair_key}] SMC score too low: {score} < {cfg['min_score']}"); return None
    # HTF SR zones + low TF confirmation
    zones = get_htf_swing_zones(results)
    low_tf = cfg["signal_tf"]; low_res = analyze_tf(sym, low_tf)
    if not low_res: 
        if LOG_VERBOSE: print(f"[{pair_key}] low tf {low_tf} missing"); return None
    low_candles = low_res["candles"]; low_close = low_candles[-1]["c"]
    confirmed = entry_confirmation(low_tf, low_candles, final_dir, zones, results, require_retest_bars=cfg["retest_bars"])
    if not confirmed:
        if LOG_VERBOSE: print(f"[{pair_key}] Entry not confirmed for {final_dir}"); return None
    # ATR sizing prefer HTF 1h/4h then low TF fallback
    size_atr = None
    for prefer in ("1h","4h","2h","1day","30min"):
        if prefer in results and results[prefer].get("atr"):
            size_atr = results[prefer]["atr"]; break
    if size_atr is None: size_atr = low_res.get("atr") or (0.001 * low_close)
    buffer = max(size_atr * 0.6, low_close*0.0006)
    # SL anchored to HTF swing zone
    if final_dir=="BUY":
        if zones["supports"]:
            s_price = zones["supports"][0][0]; sl = float(s_price) - buffer
            if sl >= low_close: sl = low_close - (size_atr*1.0)
        else:
            sl = low_close - (size_atr * 1.2)
        tp1 = low_close + (size_atr * (1.0 + score/6.0))
        tp2 = low_close + (size_atr * (2.0 + score/5.0))
        tp3 = low_close + (size_atr * (3.5 + score/4.0))
    else:
        if zones["resistances"]:
            r_price = zones["resistances"][0][0]; sl = float(r_price) + buffer
            if sl <= low_close: sl = low_close + (size_atr*1.0)
        else:
            sl = low_close + (size_atr * 1.2)
        tp1 = low_close - (size_atr * (1.0 + score/6.0))
        tp2 = low_close - (size_atr * (2.0 + score/5.0))
        tp3 = low_close - (size_atr * (3.5 + score/4.0))
    # final sanity checks
    if final_dir=="BUY" and not (tp1 > low_close and tp2 > tp1 and tp3 > tp2):
        tp1 = low_close + size_atr*1.0; tp2 = low_close + size_atr*2.0; tp3 = low_close + size_atr*3.0
    if final_dir=="SELL" and not (tp1 < low_close and tp2 < tp1 and tp3 < tp2):
        tp1 = low_close - size_atr*1.0; tp2 = low_close - size_atr*2.0; tp3 = low_close - size_atr*3.0
    return {"symbol":pair_key,"side":final_dir,"entry":low_close,"sl":sl,"tp1":tp1,"tp2":tp2,"tp3":tp3,"score":score,"atr":size_atr,"mtf":results,"low_candles":low_candles,"low_tf":low_tf}

# ---------------- Plotting ----------------
def plot_chart_and_bytes(candles, entry=None, sl=None, tp1=None, tp2=None, tp3=None, fvg=None, ob=None, title="Chart"):
    closes=[c["c"] for c in candles]; times=list(range(len(closes)))
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(times, closes, lw=1.3, label="Price")
    emav = [ema_val(closes[:i+1],50) if i+1>=50 else np.nan for i in range(len(closes))]
    ax.plot(times, emav, lw=1, linestyle='--', label="EMA50")
    if entry: ax.axhline(entry, color='green', linestyle='-', linewidth=1.3, label='Entry')
    if sl: ax.axhline(sl, color='red', linestyle='--', linewidth=1.1, label='SL')
    if tp1: ax.axhline(tp1, color='blue', linestyle='--', linewidth=1.0, label='TP1')
    if tp2: ax.axhline(tp2, color='cyan', linestyle='--', linewidth=1.0, label='TP2')
    if tp3: ax.axhline(tp3, color='magenta', linestyle='--', linewidth=1.0, label='TP3')
    if fvg: top,bot=fvg[1],fvg[2]; ax.fill_between(times[-20:], [top]*20, [bot]*20, color='orange', alpha=0.15)
    if ob: top,bot=ob[1],ob[2]; ax.fill_between(times[-10:], [top]*10, [bot]*10, color='purple', alpha=0.12)
    ax.set_title(title); ax.set_xlabel("bars (old → new)"); ax.set_ylabel("price"); ax.grid(alpha=0.25)
    ax.legend(loc='upper left', fontsize='small')
    buf = io.BytesIO(); plt.tight_layout(); plt.savefig(buf, format='png', dpi=120); plt.close(fig); buf.seek(0)
    return buf

# ---------------- MAIN LOOP + HEARTBEAT (no auto-trade) ----------------
def main():
    if not API_KEY or not BOT_TOKEN or not CHAT_ID:
        print("ERROR: API_KEY, BOT_TOKEN or CHAT_ID missing. Edit top of script.")
        return
    print("Starting FINAL SNIPER Ultra — strict SMC + FVG strength + Volume filter.")
    tg_send_text("✅ FINAL SNIPER Ultra started — strict SMC + FVG strength + Volume filter. Demo-test first.")
    last_sent = {}
    last_heartbeat = 0
    while True:
        try:
            nowt = time.time()
            # Heartbeat to telegram every HEARTBEAT_MINUTES
            if nowt - last_heartbeat > HEARTBEAT_MINUTES * 60:
                heartbeat_text = f"🫀 Heartbeat: SNIPER Ultra running — {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"
                tg_send_text(heartbeat_text)
                last_heartbeat = nowt
            # For each pair, evaluate
            for pair in PAIRS_CFG.keys():
                cfg = PAIRS_CFG[pair]
                if LOG_VERBOSE: print(f"[{datetime.datetime.utcnow().isoformat()}] START CHECK {pair}")
                res = evaluate_pair(pair)
                if res:
                    key = f"{pair}_{res['side']}"
                    now = time.time()
                    if key in last_sent and now - last_sent[key] < cfg["cooldown_s"]:
                        if LOG_VERBOSE: print(f"[{pair}] Cooldown active for {key} (skip)")
                        continue
                    # prepare MTF update lines (human readable)
                    mtf = res["mtf"]; order = PAIRS_CFG[pair]["analysis_tfs"]; lines=[]
                    for tf in order:
                        r = mtf.get(tf)
                        if not r: lines.append(f"{tf}: No data"); continue
                        direction = "Neutral"
                        if r["rsi"] and r["macdh"] is not None:
                            if r["rsi"]>50 and r["macdh"]>0: direction="Bullish"
                            elif r["rsi"]<50 and r["macdh"]<0: direction="Bearish"
                        smc_tags=[]
                        if r["bos"]: smc_tags.append(r["bos"])
                        if r["choch"]: smc_tags.append(r["choch"])
                        if r["fvg"]: smc_tags.append(r["fvg"][0])
                        pd = r.get("pd")
                        vol_tag = f"VolM:{r.get('vol_mult'):.2f}" if r.get("vol_mult") else ("VolOK" if r.get("vol_ok") else "VolN")
                        lines.append(f"{tf}: {direction} | SMC:{','.join(smc_tags) if smc_tags else 'N'} | PD:{pd or 'N'} | {vol_tag}")
                    update_txt = "<b>📊 " + pair + " MTF Update</b>\n" + "\n".join(lines)
                    tg_send_text(update_txt)
                    # signal text
                    txt = (f"🔔 <b>{pair} SIGNAL (SNIPER ULTRA)</b>\n"
                           f"<b>Side:</b> {res['side']}\n"
                           f"<b>Entry:</b> {res['entry']:.5f}\n"
                           f"<b>SL:</b> {res['sl']:.5f}\n"
                           f"<b>TP1:</b> {res['tp1']:.5f}\n"
                           f"<b>TP2:</b> {res['tp2']:.5f}\n"
                           f"<b>TP3:</b> {res['tp3']:.5f}\n"
                           f"<b>SMC Score:</b> {res['score']}\n"
                           f"<b>ATR (sizing):</b> {res['atr']:.6f}\n"
                           f"<b>Model:</b> SNIPER ULTRA (Strict SMC + FVG strength + Volume)\n"
                           f"<b>Time (UTC):</b> {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
                    tg_send_text(txt)
                    # chart
                    low_candles = res["low_candles"]; fvg=None; ob=None
                    for tf in reversed(list(res["mtf"].keys())):
                        r = res["mtf"][tf]
                        if r.get("fvg") and not fvg: fvg = r["fvg"]
                        if r.get("ob") and not ob: ob = r["ob"]
                    buf = plot_chart_and_bytes(low_candles, entry=res["entry"], sl=res["sl"], tp1=res["tp1"], tp2=res["tp2"], tp3=res["tp3"], fvg=fvg, ob=ob, title=f"{pair} {res['side']} (Score {res['score']})")
                    caption = f"{pair} • {res['side']} • Entry {res['entry']:.5f} • TP1 {res['tp1']:.5f} • TP2 {res['tp2']:.5f} • TP3 {res['tp3']:.5f}"
                    tg_send_photo_bytes(buf, caption)
                    last_sent[key] = now
                    if LOG_VERBOSE: print(f"[{pair}] Signal sent for {res['side']} score:{res['score']}")
                else:
                    if LOG_VERBOSE: print(f"[{pair}] No high-probability signal")
                time.sleep(2)
            if LOG_VERBOSE: print(f"[{datetime.datetime.utcnow().isoformat()}] SCAN COMPLETE. Sleeping {LOOP_INTERVAL}s")
            time.sleep(LOOP_INTERVAL)
        except Exception as e:
            print("[MAIN] error:", e)
            time.sleep(5)

if __name__=="__main__":
    main()