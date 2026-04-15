import yfinance as yf
import pandas as pd
import numpy as np
import os
import requests
from dotenv import load_dotenv
from datetime import datetime
import time

load_dotenv()
TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

def stuur_telegram(bericht):
    if not TOKEN or not CHAT_ID: return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": CHAT_ID, "text": bericht, "parse_mode": "Markdown"}, timeout=20)
    except: pass

def bereken_indicatoren(df, s, t, use_trend_filter, is_hyper):
    p = df['Close'].ffill()
    h = df['High'].ffill()
    l = df['Low'].ffill()
    v = df['Volume'].ffill()

    f_line = p.rolling(window=s).mean() if s >= 20 else p.ewm(span=s, adjust=False).mean()
    s_line = p.rolling(window=t).mean() if t >= 50 else p.ewm(span=t, adjust=False).mean()
    ema200 = p.ewm(span=200, adjust=False).mean()
    vol_ma = v.rolling(window=20).mean()

    delta = p.diff()
    if is_hyper:
        rsi3_gain = (delta.where(delta > 0, 0)).rolling(3).mean()
        rsi3_loss = (-delta.where(delta < 0, 0)).rolling(3).mean()
        rsi3 = 100 - (100 / (1 + (rsi3_gain / (rsi3_loss + 1e-10))))
        
        change = np.sign(delta)
        streak = change.groupby((change != change.shift()).cumsum()).cumsum()
        s_delta = streak.diff()
        s_gain = (s_delta.where(s_delta > 0, 0)).rolling(2).mean()
        s_loss = (-s_delta.where(s_delta < 0, 0)).rolling(2).mean()
        streak_rsi = 100 - (100 / (1 + (s_gain / (s_loss + 1e-10))))
        
        p_rank = delta.rolling(100).apply(lambda x: (x < x.iloc[-1]).sum() / 99.0 * 100, raw=False)
        rsi_val = (rsi3 + streak_rsi + p_rank) / 3
    else:
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi_val = 100 - (100 / (1 + (gain / (loss + 1e-10))))

    tr = pd.concat([h-l, abs(h-p.shift()), abs(l-p.shift())], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    
    return p, f_line, s_line, ema200, vol_ma, rsi_val, atr, v

def voer_lijst_uit(tickers, naam_sector):
    nu = datetime.now().strftime("%d/%m/%Y %H:%M")
    inzet = 2500.0
    res = {"T": 0, "S": 0, "HT": 0, "HS": 0}
    sig = {"T": [], "S": [], "HT": [], "HS": []}

    for t in tickers:
        try:
            data = yf.download(t, period="5y", progress=False)
            if len(data) < 200: continue

            for key, (s, t_per, use_tr, is_hyp) in [
                ("T", (50, 200, True, False)), 
                ("S", (20, 50, True, False)), 
                ("HT", (9, 21, True, True)), 
                ("HS", (9, 21, False, True))
            ]:
                p, f, s_line, e200, v_ma, rsi, atr, vol = bereken_indicatoren(data, s, t_per, use_tr, is_hyp)
                
                p_bt, f_bt, s_bt, e_bt = p.iloc[-252:], f.iloc[-252:], s_line.iloc[-252:], e200.iloc[-252:]
                v_bt, v_ma_bt, atr_bt = vol.iloc[-252:], v_ma.iloc[-252:], atr.iloc[-252:]
                
                profit, pos, instap, sl_val = 0, False, 0, 0
                kosten = 15.0 + (inzet * 0.0035)

                for i in range(1, len(p_bt)):
                    cp = p_bt.iloc[i]
                    if not pos:
                        if f_bt.iloc[i] > s_bt.iloc[i] and f_bt.iloc[i-1] <= s_bt.iloc[i-1]:
                            if v_bt.iloc[i] > (v_ma_bt.iloc[i] * 0.6):
                                if not use_tr or cp > e_bt.iloc[i]:
                                    instap, sl_val, pos = cp, cp - (2 * atr_bt.iloc[i]), True
                                    profit -= kosten
                    else:
                        if cp < sl_val or f_bt.iloc[i] < s_bt.iloc[i]:
                            profit += (inzet * (cp / instap) - inzet) - kosten
                            pos = False
                
                res[key] += profit

                cp_act, f_act, s_act, e_act = p.iloc[-1], f.iloc[-1], s_line.iloc[-1], e200.iloc[-1]
                if f_act > s_act and f.iloc[-2] <= s_line.iloc[-2]:
                    if not use_tr or cp_act > e_act:
                        sig[key].append(f"`{t}`: 🟢 KOOP (€{cp_act:.2f})")
                elif f_act < s_act and f.iloc[-2] >= s_line.iloc[-2]:
                    sig[key].append(f"`{t}`: 🔴 VERKOOP (€{cp_act:.2f})")

        except: continue

    rapport = [
        f"📊 *{naam_sector} RAPPORT*",
        f"_{nu}_",
        "----------------------------------",
        f"🐢 *Traag (50/200):* €{100000 + res['T']:,.0f}",
        f"⚡ *Snel (20/50):* €{100000 + res['S']:,.0f}",
        f"🚀 *Hyper Trend:* €{100000 + res['HT']:,.0f}",
        f"🔥 *Hyper Scalp:* €{100000 + res['HS']:,.0f}",
        "",
        "*SIGNALEN:*",
        f"T: {', '.join(sig['T']) if sig['T'] else 'None'}",
        f"S: {', '.join(sig['S']) if sig['S'] else 'None'}"
    ]
    stuur_telegram("\n".join(rapport))

if __name__ == "__main__":
    benelux_30 = ["ASML.AS", "ADYEN.AS", "WKL.AS", "LOTB.BR", "ARGX.BR", "REN.AS", "DSFIR.AS", "IMCD.AS", "AZE.BR", "MELE.BR", "SOF.BR", "ACKB.BR", "KINE.BR", "UCB.BR", "DIE.BR", "BFIT.AS", "VGP.BR", "WDP.BR", "AD.AS", "HEIA.AS", "BESI.AS", "ALFEN.AS", "GLPG.AS", "EURN.BR", "ELI.BR", "BAR.BR", "ENX.AS", "NN.AS", "AGS.BR", "RAND.AS"]
    voer_lijst_uit(benelux_30, "BENELUX")
