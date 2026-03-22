import os
import requests
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

print(f"Gebruikte Chat ID: {CHAT_ID}")
print(f"Token aanwezig: {'Ja' if TOKEN else 'Nee'}")

url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
res = requests.post(url, data={"chat_id": CHAT_ID, "text": "Hallo! De verbinding werkt! 🚀"})
print(f"Resultaat van Telegram: {res.text}")
