import os
import requests

def test_bot():
    token = os.getenv('TELEGRAM_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    print(f"--- Telegram Test ---")
    print(f"Chat ID gevonden: {'Ja' if chat_id else 'Nee'}")
    print(f"Token gevonden: {'Ja' if token else 'Nee'}")
    
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": "✅ GitHub-verbinding werkt! Ronny's bot is online."}
    
    try:
        response = requests.post(url, json=payload)
        result = response.json()
        if result.get("ok"):
            print("SUCCES: Bericht is verzonden naar Telegram!")
        else:
            print(f"FOUTMELDING VAN TELEGRAM: {result.get('description')}")
            print(f"Foutcode: {result.get('error_code')}")
    except Exception as e:
        print(f"ER ging iets mis bij het verbinden: {e}")

if __name__ == "__main__":
    test_bot()
