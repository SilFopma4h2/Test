📈 XAU/USD Trading Signal Bot
This is a Python-based trading signal bot that monitors the price of XAU/USD (Gold) using Twelve Data API and sends real-time buy/sell signals to Telegram and Discord based on technical indicators like EMA, RSI, and Bollinger Bands.

🚀 Features
Fetches real-time price data using Twelve Data API

Calculates:

Exponential Moving Averages (EMA9 & EMA21)

Relative Strength Index (RSI)

Bollinger Bands

Detects Buy/Sell signals based on indicator conditions

Sends alerts to:

Telegram (Bot & Chat)

Discord (Webhook)

Runs continuously every 60 seconds

🔧 Setup & Requirements
✅ Requirements
Install required libraries:

pip install requests pandas
🔑 API Configuration
Before running the bot, you must fill in your API keys and tokens in the config section of the script.

1. Twelve Data API
Sign up at: https://twelvedata.com

Get your free API key

Add it to the script:



TD_API_KEY = "your_twelve_data_api_key"
2. Telegram Bot
Create a bot with @BotFather on Telegram.

Save your bot token.

Get your chat ID using:

Start a chat with your bot

Go to: https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates

Look for "chat":{"id":YOUR_CHAT_ID,...} in the JSON

Add to the script:



TELEGRAM_BOT_TOKEN = "your_bot_token"
TELEGRAM_CHAT_ID = "your_chat_id"
3. Discord Webhook
Go to your Discord channel → Edit Channel → Integrations → Webhooks

Create a new webhook and copy the URL

Add to the script:



DISCORD_WEBHOOK_URL = "your_discord_webhook_url"
⚙️ How It Works
The bot checks for:

Buy Signal when:

EMA9 crosses above EMA21

RSI < 70

Price is below upper Bollinger Band

Sell Signal when:

EMA9 crosses below EMA21

RSI > 30

Price is above lower Bollinger Band

Every time a signal is detected, the bot sends a formatted message to both Telegram and Discord.

🖥️ Running the Bot

python your_script_name.py
The bot will start and print logs to the console.

📦 Example Telegram Alert

🟢 BUY Signal Detected
⏰ 2025-07-10 16:00:00
💰 Entry Price: 2350.45
🎯 TP1: 2365.45, TP2: 2380.45, TP3: 2395.45
🛡️ SL: 2327.95

⚠️ Notes
This script does not place real trades. It only detects signals.

You can integrate it with a broker API (like OANDA, Binance, Alpaca) if desired.

📄 License
This project is open for educational or personal use. No warranty is provided. Use at your own risk.
