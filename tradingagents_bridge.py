"""
tradingagents_bridge.py

Koppelt TradingAgents (multi-agent LLM-analyse) aan bot_00vcp.

Werking:
  1. bot_00vcp.py schrijft zijn kandidaat-tickers weg naar vcp_signals.json
     (zie de kleine aanpassing onderaan dit bestand / in bot_00vcp_patch.md)
  2. Dit script leest dat bestand, laat TradingAgents per ticker een besluit
     genereren, en stuurt het resultaat naar Telegram en/of Gmail.

Blijft gratis door:
  - Groq als LLM-provider (gratis rate-limited tier, geen lokale GPU nodig,
    en werkt dus prima binnen de gratis GitHub Actions-runner)
  - max_debate_rounds=1 om binnen de gratis Groq rate limits te blijven
  - GitHub Actions blijft gratis voor public/private repos binnen de
    standaard minuten-quota; dit stapje kost enkele API-calls, geen compute

Benodigde secrets (naast de vijf die je al gebruikt):
  GROQ_API_KEY   -> gratis aan te maken op https://console.groq.com
"""

import os
import sys
import json
import time
import argparse
import smtplib
import datetime
from email.mime.text import MIMEText

import requests

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG


def build_config():
    """Gratis configuratie: Groq-inference i.p.v. betaalde OpenAI/Anthropic calls."""
    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = "groq"
    config["backend_url"] = "https://api.groq.com/openai/v1"
    # Beide modellen zijn gratis-tier Groq-modellen
    config["deep_think_llm"] = "llama-3.3-70b-versatile"
    config["quick_think_llm"] = "llama-3.1-8b-instant"
    # Kleinere debat-diepte = minder calls = ruim binnen gratis rate limits
    config["max_debate_rounds"] = 1
    config["max_risk_discuss_rounds"] = 1
    # Uitgezet: online_tools haalt live nieuws/social data op en maakt de
    # prompt vaak 10-60k tokens groot — dat past niet binnen de gratis
    # Groq TPM-limiet (6000 tokens/min voor llama-3.1-8b-instant).
    # Met online_tools=False gebruikt TradingAgents alleen prijs/volume-data,
    # wat de prompt klein genoeg houdt om gratis te blijven werken.
    config["online_tools"] = False
    return config


def analyze_ticker(ticker: str, trade_date: str) -> str:
    config = build_config()
    ta = TradingAgentsGraph(debug=False, config=config)
    _, decision = ta.propagate(ticker, trade_date)
    return decision


def send_telegram(message: str) -> None:
    token = os.environ["TELEGRAM_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    resp = requests.post(
        url,
        data={"chat_id": chat_id, "text": message, "parse_mode": "HTML"},
        timeout=30,
    )
    resp.raise_for_status()


def send_email(subject: str, message: str) -> None:
    user = os.environ["EMAIL_USER"]
    password = os.environ["EMAIL_PASS"]
    receiver = os.environ["EMAIL_RECEIVER"]

    msg = MIMEText(message)
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = receiver

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(user, password)
        server.send_message(msg)


def main():
    parser = argparse.ArgumentParser(description="TradingAgents-bridge voor bot_00vcp")
    parser.add_argument(
        "--signals-file",
        default="vcp_signals.json",
        help="JSON-bestand met kandidaat-tickers van bot_00vcp (default: vcp_signals.json)",
    )
    parser.add_argument("--date", default=None, help="Handelsdatum YYYY-MM-DD (default: vandaag)")
    parser.add_argument(
        "--channel",
        choices=["telegram", "email", "both"],
        default="telegram",
        help="Waarheen het resultaat sturen (default: telegram)",
    )
    args = parser.parse_args()

    trade_date = args.date or datetime.date.today().isoformat()

    if not os.path.exists(args.signals_file):
        print(f"Geen signalenbestand gevonden: {args.signals_file} — niets te doen.")
        return

    with open(args.signals_file) as f:
        signals = json.load(f)  # verwacht formaat: {"tickers": ["ABI.BR", "KBC.BR"]}

    tickers = signals.get("tickers", [])
    if not tickers:
        print("Geen VCP-kandidaten vandaag — TradingAgents wordt overgeslagen.")
        return

    blocks = []
    for ticker in tickers:
        print(f"Analyseren: {ticker} ...")
        try:
            decision = analyze_ticker(ticker, trade_date)
            blocks.append(f"📊 <b>{ticker}</b>\n{decision}\n")
        except Exception as e:
            blocks.append(f"⚠️ {ticker}: analyse mislukt ({e})")
        # Kleine pauze tussen tickers om de gratis TPM-limiet niet te bursten
        time.sleep(5)

    message = f"TradingAgents-analyse VCP-kandidaten ({trade_date}):\n\n" + "\n".join(blocks)

    if args.channel in ("telegram", "both"):
        send_telegram(message)
        print("Verstuurd naar Telegram.")
    if args.channel in ("email", "both"):
        send_email(f"TradingAgents VCP-analyse {trade_date}", message)
        print("Verstuurd naar Gmail.")


if __name__ == "__main__":
    main()
