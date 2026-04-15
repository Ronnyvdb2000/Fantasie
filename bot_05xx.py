import yfinance as yf
import pandas as pd
import os
import requests
import numpy as np
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
        requests.post(url, data={"chat_id": CHAT_ID, "text": bericht, "parse_mode": "Markdown", "disable_web_page_preview": True}, timeout=20)
        time.sleep(1) 
    except: pass

def check_munger_kwaliteit(ticker_obj):
    """
    Fundamentele filter: ROIC > 10%, Debt/Equity < 1.0, Positieve winst.
    Dit scheidt de 'kwaliteit' van de 'speculatie'.
    """
    try:
        info = ticker_obj.info
        # Probeer verschillende bronnen voor ROIC/ROE
        roic = info.get('returnOnCapitalEmployed') or info.get('returnOnAssets', 0)
        debt_to_equity = info.get('debtToEquity', 100) / 100.0
        profit_margin = info.get('profitMargins', 0)
        
        if roic > 0.10 and debt_to_equity < 1.0 and profit_margin > 0:
            return True, roic, debt_to_equity
    except: pass
    return False, 0, 0

def bereken_indicatoren_vectorized(df, s, t, use_trend_filter, is_hyper):
    """Berekent alle technische indicatoren"""
    p = df['Close'].ffill()
    h = df['High'].ffill()
    l = df['Low'].ffill()
    v = df['Volume'].ffill()

    f_line = p.rolling(window=s).mean() if s >= 20 else p.ewm(span=s, adjust=False).mean()
    s_line = p.rolling(window=t).mean() if t >= 50 else p.ewm(span=t, adjust=False).mean()
    ema200 = p.ewm(span=200, adjust=False).mean()
    vol_ma = v.rolling(window=20).mean()

    # RSI Logica
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

    # ATR & ADX
    tr = pd.concat([h-l, abs(h-p.shift()), abs(l-p.shift())], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    up, down = h.diff().clip(lower=0), (-l.diff()).clip(lower=0)
    tr14 = tr.rolling(14).sum()
    plus_di = 100 * (up.rolling(14).sum() / (tr14 + 1e-10))
    minus_di = 100 * (down.rolling(14).sum() / (tr14 + 1e-10))
    adx = (100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)).rolling(14).mean()

    return p, f_line, s_line, ema200, vol_ma, rsi_val, atr, adx, v

def voer_lijst_uit(tickers, naam_sector):
    nu = datetime.now().strftime("%d/%m/%Y %H:%M")
    
    inzet = 2500.0
    res = {"T": 0, "S": 0, "HT": 0, "HS": 0}
    sig = {"T": [], "S": [], "HT": [], "HS": []}

    for t in tickers:
        try:
            t_obj = yf.Ticker(t)
            # STAP 1: Kwaliteitscheck (Munger Filter)
            is_ok, roic, debt = check_munger_kwaliteit(t_obj)
            if not is_ok: continue

            # STAP 2: Data download
            data = t_obj.history(period="5y")
            if len(data) < 200: continue

            # STAP 3: Verwerking per strategie
            for strat_key, (s_per, t_per, use_tr, is_hyp) in [
                ("T",(50,200,True,False)), ("S",(20,50,True,False)), 
                ("HT",(9,21,True,True)), ("HS",(9,21,False,True))
            ]:
                p, f, s_line, e200, v_ma, rsi, atr, adx, vol = bereken_indicatoren_vectorized(data, s_per, t_per, use_tr, is_hyp)
                
                # Backtest laatste jaar
                p_bt, f_bt, s_bt, e_bt = p.iloc[-252:], f.iloc[-252:], s_line.iloc[-252:], e200.iloc[-252:]
                v_bt, v_ma_bt, atr_bt, adx_bt = vol.iloc[-252:], v_ma.iloc[-252:], atr.iloc[-252:], adx.iloc[-252:]
                
                profit, pos, instap, high_p, sl_val = 0, False, 0, 0, 0
                kosten = 15.0 + (inzet * 0.0035)

                for i in range(1, len(p_bt)):
                    cp_i = p_bt.iloc[i]
                    if not pos:
                        if f_bt.iloc[i] > s_bt.iloc[i] and f_bt.iloc[i-1] <= s_bt.iloc[i-1]:
                            if adx_bt.iloc[i] > 15 and v_bt.iloc[i] > (v_ma_bt.iloc[i] * 0.6):
                                if not use_tr or cp_i > e_bt.iloc[i]:
                                    instap, high_p, sl_val, pos = cp_i, cp_i, cp_i - (2 * atr_bt.iloc[i]), True
                                    profit -= kosten
                    else:
                        high_p = max(high_p, cp_i)
                        sl_val = max(sl_val, high_p - (2 * atr_bt.iloc[i]))
                        if cp_i < sl_val or f_bt.iloc[i] < s_bt.iloc[i]:
                            profit += (inzet * (cp_i / instap) - inzet) - kosten
                            pos = False
                
                res[strat_key] += profit

                # Actueel Signaal
                cp, catr, crsi = p.iloc[-1], atr.iloc[-1], rsi.iloc[-1]
                l_rsi = "💎 CRSI" if is_hyp else "📊 RSI"
                y_link = f"[Link](https://finance.yahoo.com/quote/{t})"
                
                if f.iloc[-1] > s_line.iloc[-1] and f.iloc[-2] <= s_line.iloc[-2]:
                    if adx.iloc[-1] > 15 and cp > e200.iloc[-1]:
                        sig[strat_key].append(f"• `{t}`: 🟢 *KOOP* | €{cp:.2f} | {l_rsi}: {crsi:.1f} | {y_link}")
                elif f.iloc[-1] < s_line.iloc[-1] and f.iloc[-2] >= s_line.iloc[-2]:
                    sig[strat_key].append(f"• `{t}`: 🔴 *VERKOOP* | €{cp:.2f} | {y_link}")

        except: continue

    # Rapportage
    def get_s(lst): return "\n".join(lst) if lst else "Geen actie"
    rapport = [
        f"🏆 *{naam_sector} QUALITY REPORT*", f"_{nu}_", "----------------------------------",
        f"🐢 *Traag (50/200):* €{100000 + res['T']:,.0f}",
        f"⚡ *Snel (20/50):* €{100000 + res['S']:,.0f}",
        f"🚀 *Hyper Trend:* €{100000 + res['HT']:,.0f}",
        f"🔥 *Hyper Scalp:* €{100000 + res['HS']:,.0f}",
        "", "🛡️ *SIGNALEN TRAAG:*", get_s(sig["T"]),
        "", "🎯 *SIGNALEN SNEL:*", get_s(sig["S"]),
        "", "📈 *HYPER TREND:*", get_s(sig["HT"]),
        "", "⚡ *HYPER SCALP:*", get_s(sig["HS"])
    ]
    stuur_telegram("\n".join(rapport))

def main():
    benelux_tickers = [
        "ASML.AS", "ADYEN.AS", "WKL.AS", "LOTB.BR", "ARGX.BR", "REN.AS", "DSFIR.AS", 
        "IMCD.AS", "AZE.BR", "MELE.BR", "SOF.BR", "ACKB.BR", "KINE.BR", "UCB.BR", 
        "DIE.BR", "BFIT.AS", "VGP.BR", "WDP.BR", "AD.AS", "HEIA.AS", "BESI.AS", 
        "ALFEN.AS", "GLPG.AS", "EURN.BR", "ELI.BR", "BAR.BR", "ENX.AS", "NN.AS", 
        "AGS.BR", "RAND.AS"
    ]
    voer_lijst_uit(benelux_tickers, "BENELUX")

if __name__ == "__main__":
    main()
