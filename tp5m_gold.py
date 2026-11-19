import os
import json
import requests
import pandas as pd
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

# ───── Parámetros (tu configuración de TradingView) ─────
SYMBOL = "GOLD"                     # epic del oro en Capital.com
BASE_URL = "https://demo-api-capital.backend-capital.com"
RESOLUTION = "MINUTE_15"            # velas de 15 minutos
TF_MIN = 15                         # minutos por vela
EMA_F, EMA_M, EMA_S = 21, 50, 200
TOUCH_BARS = 8                      # velas para tocar la EMA rápida
RSI_LEN, RSI_MIN, RSI_MAX = 14, 25, 75
RR = 2.0                            # ratio R:B
MIN_GAP = 4                         # velas mínimas entre señales
ATR_LEN = 14
SWING_LEN = 10
SWING_BUF = 0.2
MIN_SL, MAX_SL = 0.8, 2.0
SESS_START, SESS_END = 8, 20        # hora de Roma
USE_VOL = True                      # filtro volumen > media 20
STATE_FILE = "state_gold.json"


def fetch_candles():
    # 1) Abrir sesión
    s = requests.post(
        f"{BASE_URL}/api/v1/session",
        headers={"X-CAP-API-KEY": os.environ["CAPITAL_API_KEY"],
                 "Content-Type": "application/json"},
        json={"identifier": os.environ["CAPITAL_LOGIN"],
              "password": os.environ["CAPITAL_API_PASSWORD"]},
        timeout=20,
    )
    s.raise_for_status()
    auth = {"CST": s.headers["CST"],
            "X-SECURITY-TOKEN": s.headers["X-SECURITY-TOKEN"]}

    # 2) Descargar velas
    r = requests.get(
        f"{BASE_URL}/api/v1/prices/{SYMBOL}",
        headers=auth,
        params={"resolution": RESOLUTION, "max": 500},
        timeout=20,
    )
    r.raise_for_status()
    rows = []
    for p in r.json()["prices"]:
        mid = lambda k: (p[k]["bid"] + p[k]["ask"]) / 2
        rows.append({
            "open_time": pd.to_datetime(p["snapshotTimeUTC"], utc=True),
            "open": mid("openPrice"), "high": mid("highPrice"),
            "low": mid("lowPrice"), "close": mid("closePrice"),
            "volume": float(p.get("lastTradedVolume", 0)),
        })
    df = pd.DataFrame(rows).sort_values("open_time").reset_index(drop=True)

    # 3) Solo velas cerradas
    now = pd.Timestamp(datetime.now(timezone.utc))
    df = df[df["open_time"] + pd.Timedelta(minutes=TF_MIN) <= now].reset_index(drop=True)
    return df


def add_indicators(df):
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    df["ef"] = c.ewm(span=EMA_F, adjust=False).mean()
    df["em"] = c.ewm(span=EMA_M, adjust=False).mean()
    df["es"] = c.ewm(span=EMA_S, adjust=False).mean()

    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / RSI_LEN, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / RSI_LEN, adjust=False).mean()
    df["rsi"] = 100 - 100 / (1 + up / dn)

    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / ATR_LEN, adjust=False).mean()
    df["vma"] = v.rolling(20).mean()
    df["swl"] = l.rolling(SWING_LEN).min()
    df["swh"] = h.rolling(SWING_LEN).max()
    return df


def clamp(d, atr):
    return min(max(d, atr * MIN_SL), atr * MAX_SL)


def check_signal(df):
    r = df.iloc[-1]
    tail = df.iloc[-TOUCH_BARS:]

    local = r["open_time"].tz_convert(ZoneInfo("Europe/Rome"))
    if not (SESS_START <= local.hour < SESS_END):
        return None
    if USE_VOL and not r["volume"] > r["vma"]:
        return None

    trend_up = r["close"] > r["es"] and r["ef"] > r["em"] and r["em"] > r["es"]
    trend_dn = r["close"] < r["es"] and r["ef"] < r["em"] and r["em"] < r["es"]
    touch_up = (tail["low"] <= tail["ef"]).any()
    touch_dn = (tail["high"] >= tail["ef"]).any()
    conf_up = r["close"] > r["ef"] and r["close"] > r["open"]
    conf_dn = r["close"] < r["ef"] and r["close"] < r["open"]
    rsi_up = RSI_MIN < r["rsi"] < RSI_MAX
    rsi_dn = (100 - RSI_MAX) < r["rsi"] < (100 - RSI_MIN)

    c, atr = r["close"], r["atr"]
    if trend_up and touch_up and conf_up and rsi_up:
        d = clamp(c - (r["swl"] - atr * SWING_BUF), atr)
        return ("LONG", c, c - d, c + d * RR)
    if trend_dn and touch_dn and conf_dn and rsi_dn:
        d = clamp((r["swh"] + atr * SWING_BUF) - c, atr)
        return ("SHORT", c, c + d, c - d * RR)
    return None


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


def send_telegram(msg):
    r = requests.post(
        f"https://api.telegram.org/bot{os.environ['TELEGRAM_TOKEN']}/sendMessage",
        data={"chat_id": os.environ["TELEGRAM_CHAT_ID"], "text": msg},
        timeout=20,
    )
    r.raise_for_status()


def main():
    df = add_indicators(fetch_candles())

    # Modo prueba: manda la última vela para comprobar que todo funciona
    if os.environ.get("TEST_MODE") == "true":
        r = df.iloc[-1]
        hora = r["open_time"].tz_convert(ZoneInfo("Europe/Rome")).strftime("%d/%m %H:%M")
        send_telegram(f"✅ Prueba OK (15m)\nÚltima vela {hora} Roma\n"
                      f"Cierre {r['close']:.2f} | Volumen {r['volume']:.0f} | RSI {r['rsi']:.1f}")
        return

    state = load_state()
    sig = check_signal(df)
    if sig:
        side, entry, sl, tp = sig
        t = df["open_time"].iloc[-1]
        last = state.get("last")
        if last is None or (t - pd.Timestamp(last)) >= pd.Timedelta(minutes=TF_MIN * MIN_GAP):
            hora = t.tz_convert(ZoneInfo("Europe/Rome")).strftime("%d/%m %H:%M")
            send_telegram(f"🟡 ORO {side}  (vela 15m {hora} Roma)\n"
                          f"Entrada {entry:.2f}\nSL {sl:.2f}\nTP {tp:.2f}  (1:{RR:g})")
            state["last"] = t.isoformat()
    save_state(state)


if __name__ == "__main__":
    main()
