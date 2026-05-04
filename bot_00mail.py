import yfinance as yf
import pandas as pd
import os
import requests
import smtplib
import numpy as np
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
from datetime import datetime
import time
import logging

# ---------------------------------------------------------------------------
# CONFIG & LOGGING
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()
TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
EMAIL_USER = os.getenv('EMAIL_USER')
EMAIL_PASS = os.getenv('EMAIL_PASS')
EMAIL_RECEIVER = os.getenv('EMAIL_RECEIVER')

def stuur_telegram(bericht: str) -> bool:
    if not TOKEN or not CHAT_ID: return False
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        r = requests.post(url, data={"chat_id": CHAT_ID, "text": bericht, "parse_mode": "Markdown", "disable_web_page_preview": True}, timeout=30)
        r.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"Telegram fout: {e}")
        return False

def stuur_mail(onderwerp, inhoud_tekst):
    """Verstuurt het volledige verzamelrapport aan het einde van de run."""
    if not EMAIL_USER or not EMAIL_PASS or not EMAIL_RECEIVER:
        logger.warning("E-mail niet volledig geconfigureerd in omgevingsvariabelen.")
        return
    
    msg = MIMEMultipart()
    msg['From'] = EMAIL_USER
    msg['To'] = EMAIL_RECEIVER
    msg['Subject'] = onderwerp

    # Markdown opschonen voor e-mail leesbaarheid
    schone_inhoud = inhoud_tekst.replace('*', '').replace('`', '').replace('•', '-')
    msg.attach(MIMEText(schone_inhoud, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASS)
        server.send_message(msg)
        server.quit()
        logger.info(f"E-mail succesvol verzonden naar {EMAIL_RECEIVER}")
    except Exception as e:
        logger.error(f"E-mail fout: {e}")

# ---------------------------------------------------------------------------
# INDICATOREN (Vectorized Bot 00 Logica)
# ---------------------------------------------------------------------------
def bereken_indicatoren_vectorized(df, s, t, use_trend_filter, is_hyper):
    p = df['Close'].ffill()
    h = df['High'].ffill()
    l = df['Low'].ffill()
    v = df['Volume'].ffill()

    f_line = p.rolling(window=s).mean() if s >= 20 else p.ewm(span=s, adjust=False).mean()
    s_line = p.rolling(window=t).mean() if t >= 50 else p.ewm(span=t, adjust=False).mean()
    ema200 = p.ewm(span=200, adjust=False).mean()
    vol_ma = v.rolling(window=20).mean()

    delta = p.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi_val = 100 - (100 / (1 + (gain / (loss + 1e-10))))

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

    tr = pd.concat([h-l, abs(h-p.shift()), abs(l-p.shift())], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    up, down = h.diff().clip(lower=0), (-l.diff()).clip(lower=0)
    tr14 = tr.rolling(14).sum()
    plus_di = 100 * (up.rolling(14).sum() / (tr14 + 1e-10))
    minus_di = 100 * (down.rolling(14).sum() / (tr14 + 1e-10))
    adx = (100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)).rolling(14).mean()

    return p, f_line, s_line, ema200, vol_ma, rsi_val, atr, adx, v

# ---------------------------------------------------------------------------
# CORE ENGINE
# ---------------------------------------------------------------------------
def voer_lijst_uit(bestandsnaam, label, naam_sector):
    if not os.path.exists(bestandsnaam): return ""
    nu = datetime.now().strftime("%d/%m/%Y %H:%M")
    
    with open(bestandsnaam, 'r') as f:
        content = f.read().replace('\n', ',').replace('$', '')
        tickers = sorted(list(set([t.strip().upper() for t in content.split(',') if t.strip()])))

    if not tickers: return ""

    raw_df = yf.download(tickers, period="5y", progress=False, auto_adjust=True)
    
    inzet = 2500.0
    res = {"T": 0, "S": 0, "HT": 0, "HS": 0}
    num_trades = {"T": 0, "S": 0, "HT": 0, "HS": 0}
    sig = {"T": [], "S": [], "HT": [], "HS": []}

    for strat_key, (s_per, t_per, use_tr, is_hyp) in [
        ("T",(50,200,True,False)), ("S",(20,50,True,False)), 
        ("HT",(9,21,True,True)), ("HS",(9,21,False,True))
    ]:
        for t in tickers:
            try:
                if len(tickers) > 1:
                    t_data = raw_df.xs(t, axis=1, level=1)
                else:
                    t_data = raw_df

                p, f, s_line, e200, v_ma, rsi, atr, adx, vol = bereken_indicatoren_vectorized(t_data, s_per, t_per, use_tr, is_hyp)
                
                p_bt, f_bt, s_bt = p.iloc[-252:], f.iloc[-252:], s_line.iloc[-252:]
                e_bt, v_bt, v_ma_bt = e200.iloc[-252:], vol.iloc[-252:], v_ma.iloc[-252:]
                atr_bt, adx_bt = atr.iloc[-252:], adx.iloc[-252:]
                
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
                                    num_trades[strat_key] += 1
                    else:
                        high_p = max(high_p, cp_i)
                        sl_val = max(sl_val, high_p - (2 * atr_bt.iloc[i]))
                        if cp_i < sl_val or f_bt.iloc[i] < s_bt.iloc[i]:
                            profit += (inzet * (cp_i / instap) - inzet) - kosten
                            pos = False
                
                res[strat_key] += profit

                cp, catr, crsi = p.iloc[-1], atr.iloc[-1], rsi.iloc[-1]
                l_rsi = "💎 CRSI" if is_hyp else "📊 RSI"
                y_l = f"[Grafiek](https://finance.yahoo.com/quote/{t})"

                if f.iloc[-1] > s_line.iloc[-1] and f.iloc[-2] <= s_line.iloc[-2]:
                    if adx.iloc[-1] > 15 and vol.iloc[-1] > (v_ma.iloc[-1] * 0.6):
                        if not use_tr or cp > e200.iloc[-1]:
                            sig[strat_key].append(f"• `{t}`: 🟢 *KOOP* | €{cp:.2f} | ⚡ ATR: {catr:.2f} | {l_rsi}: {crsi:.1f} | 🛡️ SL: €{cp-(2*catr):.2f} | {y_l}")
                elif f.iloc[-1] < s_line.iloc[-1] and f.iloc[-2] >= s_line.iloc[-2]:
                    sig[strat_key].append(f"• `{t}`: 🔴 *VERKOOP* | €{cp:.2f} | ⚡ ATR: {catr:.2f} | {l_rsi}: {crsi:.1f} | 🛡️ SL: €{cp-(2*catr):.2f} | {y_l}")

            except: continue

    def get_s(lst): return "\n".join(lst) if lst else "Geen actie"
    rapport_lijst = [
        f"📊 *{label} {naam_sector} RAPPORT mail*", f"_{nu}_", "----------------------------------",
        f"🐢 *Traag (50/200):* €{100000 + res['T']:,.0f} ({num_trades['T']} trades)",
        f"⚡ *Snel (20/50):* €{100000 + res['S']:,.0f} ({num_trades['S']} trades)",
        f"🚀 *Hyper Trend:* €{100000 + res['HT']:,.0f} ({num_trades['HT']} trades)",
        f"🔥 *Hyper Scalp:* €{100000 + res['HS']:,.0f} ({num_trades['HS']} trades)",
        "", "🛡️ *SIGNALEN TRAAG (RSI):*", get_s(sig["T"]),
        "", "🎯 *SIGNALEN SNEL (RSI):*", get_s(sig["S"]),
        "", "📈 *SIGNALEN HYPER TREND (CRSI):*", get_s(sig["HT"]),
        "", "⚡ *SIGNALEN HYPER SCALP (CRSI):*", get_s(sig["HS"]),
        "", "💡 _ATR %: <2% laag, >5% hoog. RSI: >70 overbought, <30 oversold. CRSI: >90 overbought, <10 oversold_"
    ]
    volledige_tekst = "\n".join(rapport_lijst)
    stuur_telegram(volledige_tekst)
    return volledige_tekst + "\n\n" + "="*40 + "\n\n"

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    sectoren = {
        "01":"Hoogland", "02":"Macrotrends", "03":"Beursbrink",
        "04":"Benelux", "05":"Parijs", "06":"Power & AI",
        "07":"Metalen", "08":"Defensie", "09":"Varia"
    }

    verzamel_rapport = "🚀 *FULL GLOBAL SCAN COLLECTIVE REPORT (BOT 00)*\n" + "="*30 + "\n\n"
    
    logger.info("Start globale scan...")
    
    for nr, naam in sectoren.items():
        try: 
            logger.info(f"Verwerken sector: {naam}")
            sector_bericht = voer_lijst_uit(f"tickers_{nr}.txt", nr, naam)
            verzamel_rapport += sector_bericht
        except Exception as e:
            logger.error(f"Fout in sector {naam}: {e}")
        time.sleep(2)

    # Verzend de verzamelde data per e-mail
    datum_vandaag = datetime.now().strftime("%d-%m-%Y")
    stuur_mail(f"Trading Rapport Bot 00: {datum_vandaag}", verzamel_rapport)
    
    logger.info("Scan voltooid. Telegrams en E-mail verwerkt.")

if __name__ == "__main__":
    main()
