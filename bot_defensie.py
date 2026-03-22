import yfinance as yf
import pandas as pd
import os
import requests
from dotenv import load_dotenv
import warnings
from datetime import datetime

# Onderdruk waarschuwingen
warnings.simplefilter(action='ignore', category=FutureWarning)
load_dotenv()

TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

def stuur_telegram(bericht):
    if not TOKEN or not CHAT_ID: return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": CHAT_ID, "text": bericht, "parse_mode": "Markdown"}, timeout=10)
    except:
        print("Telegram verzenden mislukt.")

def check_live_signaal(df, snelle_ma, trage_ma):
    """Checkt of er VANDAAG een kruising is"""
    df['Fast'] = df['Close'].rolling(window=snelle_ma).mean()
    df['Slow'] = df['Close'].rolling(window=trage_ma).mean()
    
    nu_f, oud_f = df['Fast'].iloc[-1], df['Fast'].iloc[-2]
    nu_s, oud_s = df['Slow'].iloc[-1], df['Slow'].iloc[-2]
    
    if nu_f > nu_s and oud_f <= oud_s:
        return "🚀 GOLDEN CROSS (KOOP)"
    elif nu_f < nu_s and oud_f >= oud_s:
        return "💀 DEATH CROSS (VERKOOP)"
    return None

def bereken_backtest(df, inzet, snelle_ma, trage_ma):
    """Berekent rendement over het laatste jaar (inclusief kosten)"""
    VASTE_KOST = 15.00
    BEURSTAKS_PCT = 0.0035
    
    test_data = df.iloc[-252:].copy()
    test_data['Fast'] = test_data['Close'].rolling(window=snelle_ma).mean()
    test_data['Slow'] = test_data['Close'].rolling(window=trage_ma).mean()
    
    positie = False
    koop_prijs = 0
    huidig_saldo = inzet

    for i in range(1, len(test_data)):
        f_nu, f_oud = test_data['Fast'].iloc[i], test_data['Fast'].iloc[i-1]
        s_nu, s_oud = test_data['Slow'].iloc[i], test_data['Slow'].iloc[i-1]
        prijs = float(test_data['Close'].iloc[i])

        if not positie and f_nu > s_nu and f_oud <= s_oud:
            koop_prijs, positie = prijs, True
        elif positie and f_nu < s_nu and f_oud >= s_oud:
            bruto = inzet * (prijs / koop_prijs)
            kosten = (VASTE_KOST * 2) + (inzet * BEURSTAKS_PCT) + (bruto * BEURSTAKS_PCT)
            huidig_saldo = bruto - kosten
            positie = False

    if positie:
        huidig_saldo = (inzet * (float(test_data['Close'].iloc[-1]) / koop_prijs)) - 35
    return huidig_saldo

def main():
    # 1. Startmelding (Zelfde stijl als Parijs/Benelux)
    stuur_telegram("🛡️ *START ANALYSE: GLOBALE DEFENSIE*\n_Bezig met scannen van 25+ tickers (VS, EU, CA, IL)..._")
    
    start_kapitaal = 100000
    inzet_per_aandeel = 2500
    
    with open('tickers_defensie.txt', 'r') as f:
        tickers = [t.strip() for t in f.read().split(',') if t.strip()]

    bot1_totaal = start_kapitaal - (len(tickers) * inzet_per_aandeel)
    bot2_totaal = start_kapitaal - (len(tickers) * inzet_per_aandeel)
    live_meldingen = []

    for t in tickers:
        print(f"Analyseert Defensie: {t}")
        try:
            df = yf.download(t, period="2y", progress=False)
            if df.empty or len(df) < 200: continue
            
            # Backtest
            bot1_totaal += bereken_backtest(df, inzet_per_aandeel, 50, 200)
            bot2_totaal += bereken_backtest(df, inzet_per_aandeel, 20, 50)
            
            # Live Signalen
            sig1 = check_live_signaal(df, 50, 200)
            sig2 = check_live_signaal(df, 20, 50)
            if sig1: live_meldingen.append(f"• `{t}` (B1): {sig1}")
            if sig2: live_meldingen.append(f"• `{t}` (B2): {sig2}")
        except:
            print(f"Fout bij {t}")

    # 2. Eindrapport (Zelfde layout als de rest)
    winnaar = "BOT 1 (50/200)" if bot1_totaal > bot2_totaal else "BOT 2 (20/50)"
    vandaag = datetime.now().strftime('%d-%m-%Y %H:%M')

    rapport = f"🛡️ *EINDRAPPORT GLOBALE DEFENSIE*\n"
    rapport += f"📅 Datum: {vandaag}\n"
    rapport += f"💰 Startkapitaal: €{start_kapitaal:,.2f}\n"
    rapport += f"----------------------------------\n\n"
    rapport += f"🤖 *BOT 1 (Trend: 50/200 SMA)*\n"
    rapport += f"   • Eindstand: €{bot1_totaal:,.2f}\n\n"
    rapport += f"🤖 *BOT 2 (Actief: 20/50 SMA)*\n"
    rapport += f"   • Eindstand: €{bot2_totaal:,.2f}\n\n"
    rapport += f"🏆 *Beste strategie:* {winnaar}\n\n"
    
    if live_meldingen:
        rapport += "🎯 *LIVE SIGNALEN NU:*\n" + "\n".join(live_meldingen)
    else:
        rapport += "😴 *Geen nieuwe signalen op dit moment.*"

    stuur_telegram(rapport)

if __name__ == "__main__":
    main()
