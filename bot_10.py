import yfinance as yf
import pandas as pd
import os
import requests
from dotenv import load_dotenv
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
load_dotenv()

TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

def bereken_rsi(data, window=14):
    delta = data.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=window-1, adjust=False).mean()
    ema_down = down.ewm(com=window-1, adjust=False).mean()
    rs = ema_up / (ema_down + 1e-10)
    return 100 - (100 / (1 + rs))

def stuur_telegram(bericht):
    if not TOKEN or not CHAT_ID: return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": CHAT_ID, "text": bericht, "parse_mode": "Markdown"})

def check_live_signaal(df, s, t, ticker, is_ema=False):
    df = df.copy()
    # Gebruik EWM voor Hyper bot, anders Rolling Mean
    if is_ema:
        df['F'] = df['Close'].ewm(span=s, adjust=False).mean()
        df['S'] = df['Close'].ewm(span=t, adjust=False).mean()
    else:
        df['F'] = df['Close'].rolling(window=s).mean()
        df['S'] = df['Close'].rolling(window=t).mean()
    
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    df['RSI'] = bereken_rsi(df['Close'])
    
    rsi_nu = round(df['RSI'].iloc[-1], 1)
    prijs_nu = round(df['Close'].iloc[-1], 2)
    ema200_nu = round(df['EMA200'].iloc[-1], 2)
    link = f"https://finance.yahoo.com/quote/{ticker}"
    
    # Status labels
    status = "🔥 OVERVERHIT" if rsi_nu > 72 else "✅ TREND GEZOND"
    if rsi_nu < 35: status = "💎 ONDERGEWAARDEERD"
    elif rsi_nu > 60 and rsi_nu <= 72: status = "⚡ STERK MOMENTUM"

    info = f"Prijs: €{prijs_nu} | RSI: {rsi_nu} | EMA200: €{ema200_nu} ({status}) [Grafiek]({link})"

    if df['F'].iloc[-1] > df['S'].iloc[-1] and df['F'].iloc[-2] <= df['S'].iloc[-2]:
        return f"🚀 *KOOP* | {info}"
    elif df['F'].iloc[-1] < df['S'].iloc[-1] and df['F'].iloc[-2] >= df['S'].iloc[-2]:
        return f"💀 *VERKOOP* | {info}"
    return None

def bereken_bt(df, inzet, s, t, is_ema=False):
    VASTE_KOST = 15.00 
    BEURSTAKS = 0.0035 
    
    data = df.iloc[-252:].copy()
    if is_ema:
        data['F'] = data['Close'].ewm(span=s, adjust=False).mean()
        data['S'] = data['Close'].ewm(span=t, adjust=False).mean()
    else:
        data['F'] = data['Close'].rolling(s).mean()
        data['S'] = data['Close'].rolling(t).mean()
    
    pos, k, saldo = False, 0, inzet
    for i in range(1, len(data)):
        f_nu, f_oud = data['F'].iloc[i], data['F'].iloc[i-1]
        s_nu, s_oud = data['S'].iloc[i], data['S'].iloc[i-1]
        prijs = float(data['Close'].iloc[i])

        if not pos and f_nu > s_nu and f_oud <= s_oud:
            k, pos = prijs, True
            saldo -= (VASTE_KOST + (inzet * BEURSTAKS))
        elif pos and f_nu < s_nu and f_oud >= s_oud:
            bruto = inzet * (prijs / k)
            saldo += (bruto - inzet) - (VASTE_KOST + (bruto * BEURSTAKS))
            pos = False
    return saldo

def main():
    stuur_telegram("🛡️ *HOOGLAND SCANNER V2: HYPER-MODUS ACTIEF*")
    
    if not os.path.exists('tickers_01.txt'):
        stuur_telegram("❌ Fout: `tickers_01.txt` niet gevonden.")
        return

    with open('tickers_01.txt', 'r') as f:
        tickers = [t.strip() for t in f.read().split(',') if t.strip()]

    start_kap, inzet = 100000, 2500
    b_traag, b_snel, b_hyper = start_kap, start_kap, start_kap
    live_traag, live_snel, live_hyper = [], [], []

    for t in tickers:
        try:
            df = yf.download(t, period="2y", progress=False)
            if df.empty or len(df) < 200: continue
            
            # Rendementen
            b_traag += (bereken_bt(df, inzet, 50, 200) - inzet)
            b_snel += (bereken_bt(df, inzet, 20, 50) - inzet)
            b_hyper += (bereken_bt(df, inzet, 9, 21, is_ema=True) - inzet)
            
            # Signalen
            s_t = check_live_signaal(df, 50, 200, t)
            s_s = check_live_signaal(df, 20, 50, t)
            s_h = check_live_signaal(df, 9, 21, t, is_ema=True)
            
            if s_t: live_traag.append(f"• `{t}`: {s_t}")
            if s_s: live_snel.append(f"• `{t}`: {s_s}")
            if s_h: live_hyper.append(f"• `{t}`: {s_h}")
        except Exception as e:
            print(f"Fout bij ticker {t}: {e}")

    # Rapportage opbouwen
    rapport = f"📊 *RENDEMENTSRAPPORT Hoogland*\n----------------------------------\n"
    rapport += f"🐢 *Bot Traag (50/200):* €{b_traag:,.0f}\n"
    rapport += f"⚡ *Bot Snel (20/50):* €{b_snel:,.0f}\n"
    rapport += f"🚀 *Bot Hyper (9/21 EMA):* €{b_hyper:,.0f}\n"
    
    rapport += "\n🎯 *LIVE SIGNALEN HYPER (9/21):*\n" + ("\n".join(live_hyper) if live_hyper else "😴 Geen actie.")
    rapport += "\n\n🎯 *LIVE SIGNALEN SNEL (20/50):*\n" + ("\n".join(live_snel) if live_snel else "😴 Geen actie.")
    rapport += "\n\n🎯 *LIVE SIGNALEN TRAAG (50/200):*\n" + ("\n".join(live_traag) if live_traag else "😴 Geen actie.")
    
    stuur_telegram(rapport)

if __name__ == "__main__":
    main()
