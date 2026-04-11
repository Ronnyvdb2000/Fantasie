import yfinance as yf
import pandas as pd
import os
import requests
import numpy as np
from dotenv import load_dotenv
from datetime import datetime
import time

# --- SETUP ---
load_dotenv()
TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

def stuur_telegram(bericht):
    if not TOKEN or not CHAT_ID: return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        # Splitsen indien bericht te lang is voor Telegram (max 4096 tekens)
        if len(bericht) > 4000:
            for i in range(0, len(bericht), 4000):
                requests.post(url, data={"chat_id": CHAT_ID, "text": bericht[i:i+4000], "parse_mode": "Markdown"}, timeout=20)
        else:
            requests.post(url, data={"chat_id": CHAT_ID, "text": bericht, "parse_mode": "Markdown"}, timeout=20)
    except: pass

def bereken_indicatoren(df, s, t):
    p = df['Close'].astype(float)
    h = df['High'].astype(float)
    l = df['Low'].astype(float)
    
    # MA's
    f_line = p.rolling(window=s).mean() if s >= 20 else p.ewm(span=s, adjust=False).mean()
    s_line = p.rolling(window=t).mean() if t >= 50 else p.ewm(span=t, adjust=False).mean()
    ema200 = p.ewm(span=200, adjust=False).mean()
    vol_ma = df['Volume'].rolling(window=20).mean()
    
    # ATR
    tr = pd.concat([h-l, abs(h-p.shift()), abs(l-p.shift())], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    
    # ADX
    up, down = h.diff().clip(lower=0), (-l.diff()).clip(lower=0)
    tr14 = tr.rolling(14).sum()
    plus_di = 100 * (up.rolling(14).sum() / (tr14 + 1e-10))
    minus_di = 100 * (down.rolling(14).sum() / (tr14 + 1e-10))
    adx = (100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)).rolling(14).mean()
    
    return f_line, s_line, ema200, vol_ma, atr, adx

def voer_backtest(ticker, s, t, use_trend, inzet=2500):
    try:
        df = yf.download(ticker, period="3y", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        if df.empty or len(df) < 260: return 0, []

        f_line, s_line, ema200, vol_ma, atr, adx = bereken_indicatoren(df, s, t)
        
        # Test over laatste jaar
        p_bt = df['Close'].iloc[-252:]
        f_bt, s_bt, e_bt = f_line.iloc[-252:], s_line.iloc[-252:], ema200.iloc[-252:]
        v_bt, vm_bt, a_bt, adx_bt = df['Volume'].iloc[-252:], vol_ma.iloc[-252:], atr.iloc[-252:], adx.iloc[-252:]
        
        profit_totaal, pos, instap, high_p, sl_val = 0, False, 0, 0, 0
        kosten = 15.0 + (inzet * 0.0035)
        trades = []

        for i in range(1, len(p_bt)):
            cp = float(p_bt.iloc[i])
            # Aangepast naar dd/mm/yy formaat
            datum = p_bt.index[i].strftime('%d/%m/%y')
            
            if not pos:
                # KOOP CONDITIES
                if f_bt.iloc[i] > s_bt.iloc[i] and f_bt.iloc[i-1] <= s_bt.iloc[i-1]:
                    if adx_bt.iloc[i] > 15 and v_bt.iloc[i] > (vm_bt.iloc[i] * 0.6):
                        if not use_trend or cp > e_bt.iloc[i]:
                            instap, high_p, pos = cp, cp, True
                            sl_val = cp - (2 * a_bt.iloc[i])
                            trades.append(f"🔵 `{ticker}`: {datum} KOOP €{cp:.2f}")
            else:
                high_p = max(high_p, cp)
                sl_val = max(sl_val, high_p - (2 * a_bt.iloc[i]))
                
                # VERKOOP CONDITIES
                if cp < sl_val or f_bt.iloc[i] < s_bt.iloc[i]:
                    winst = (inzet * (cp / instap) - inzet) - (2 * kosten)
                    # 10% tax op winst
                    if winst > 0: winst *= 0.9
                    profit_totaal += winst
                    trades.append(f"🔴 `{ticker}`: {datum} VERK €{cp:.2f} | Netto: €{winst:.2f}")
                    pos = False
        
        return profit_totaal, trades
    except: return 0, []

def main():
    print(f"[{datetime.now()}] Start Megatest...")
    inzet = 2500.0
    start_kapitaal = 50000.0
    
    # Tickers inladen
    tickers = []
    for i in range(1, 10):
        fname = f"tickers_0{i}.txt"
        if os.path.exists(fname):
            with open(fname, 'r') as f:
                content = f.read().replace('\n', ',')
                tickers.extend([t.strip().upper() for t in content.split(',') if t.strip()])
    tickers = sorted(list(set(tickers)))

    strategieen = [
        ("T", "TURTLE TRAAG (50/200)", 50, 200, True),
        ("S", "SWING SNEL (20/50)", 20, 50, True),
        ("HT", "HYPER TREND (9/21)", 9, 21, True),
        ("HS", "HYPER SCALP (9/21)", 9, 21, False)
    ]

    for code, naam, s, t, trend in strategieen:
        print(f"Backtesting {naam}...")
        totaal_winst = 0
        alle_trades = []
        
        for ticker in tickers:
            winst, trade_logs = voer_backtest(ticker, s, t, trend, inzet)
            totaal_winst += winst
            alle_trades.extend(trade_logs)

        # Rapport samenstellen
        eindstand = start_kapitaal + totaal_winst
        rendement = (totaal_winst / start_kapitaal) * 100
        
        rapport = [
            f"📊 *BACKTEST: {naam}*",
            f"💰 Start: €{start_kapitaal:,.0f} | Inzet: €{inzet:,.0f}",
            f"🏁 Eindstand: €{eindstand:,.2f}",
            f"📈 Rendement: {rendement:.2f}%",
            "----------------------------------",
            "*TRADE LOG:*",
            "\n".join(alle_trades) if alle_trades else "Geen trades"
        ]
        
        stuur_telegram("\n".join(rapport))
        time.sleep(2)

    print(f"[{datetime.now()}] Alle backtests voltooid.")

if __name__ == "__main__":
    main()
