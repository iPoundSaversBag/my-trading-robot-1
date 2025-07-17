# ==============================================================================
#
#                            NOTIFICATION MODULE
#
# ==============================================================================
#
# FILE: notifications.py
#
# PURPOSE:
#   This script provides a centralized place for sending notifications to the
#   user. It is designed to be easily integrated with various notification
#   services like Telegram, Discord, or email.
#
# HOW TO USE:
#   To enable a notification service, you will need to:
#   1.  Install the required library (e.g., `pip install python-telegram-bot`).
#   2.  Fill in your API keys and user/chat IDs in the placeholder variables.
#   3.  Uncomment the code in the relevant function.
#
# ==============================================================================

import os
import requests

# --- Placeholder for Telegram ---
# To enable, get your Bot Token from the BotFather on Telegram and your Chat ID from a bot like @userinfobot
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID_HERE")

def send_notification(message):
    """
    Sends a general notification.
    """
    print(f"NOTIFICATION: {message}")
    # --- Telegram Implementation (Example) ---
    # if TELEGRAM_BOT_TOKEN != "YOUR_BOT_TOKEN_HERE" and TELEGRAM_CHAT_ID != "YOUR_CHAT_ID_HERE":
    #     try:
    #         url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    #         payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    #         response = requests.post(url, json=payload)
    #         response.raise_for_status()
    #     except Exception as e:
    #         print(f"Failed to send Telegram notification: {e}")


def send_error_alert(message):
    """
    Sends an urgent error alert.
    """
    # Prepend the message with an alert emoji for visibility
    error_message = f"🚨 CRITICAL ERROR 🚨\n\n{message}"
    print(error_message)
    send_notification(error_message)

def send_trade_alert(message):
    """
    Sends an alert for a new trade.
    """
    # Prepend the message with a trade emoji for visibility
    trade_message = f"📈 NEW TRADE 📉\n\n{message}"
    print(trade_message)
    send_notification(trade_message)
