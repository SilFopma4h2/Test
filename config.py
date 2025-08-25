"""
Configuration file for XAU/USD Trading Bot
=========================================

IMPORTANT: Update the API keys and webhook URL below with your own credentials
"""

import os

# === API CONFIGURATION ===
CONFIG = {
    # Twelve Data API Key (Get free API key from https://twelvedata.com)
    'TD_API_KEY': os.getenv('TD_API_KEY', "ffd6038d213c4136b6573fb22efbe00a"),
    
    # Trading Symbol
    'SYMBOL': "XAU/USD",
    
    # Data intervals
    'INTERVALS': {
        'minute': '1min',
        'daily': '1day'
    },
    
    # Discord Webhook URL (Optional - for notifications)
    'DISCORD_WEBHOOK_URL': os.getenv('DISCORD_WEBHOOK_URL', "https://discord.com/api/webhooks/1377567990882635846/hSKzqlyooHmOMXCxTl_yJm7yT63WjzO40geC9HyWQIqdigDYTB2DGlxLgecKj4Lakm00"),
    
    # === FILE STORAGE ===
    'LOG_FILE': "data/trading_log.json",
    'MODEL_FILE': "models/lstm_model.pt",
    'SCALER_PARAMS_FILE': "models/scaler_params.pt",
    
    # === TRADING PARAMETERS ===
    'CONFIDENCE_THRESHOLD': 0.6,  # Minimum confidence for trade execution (60%)
    'INITIAL_BALANCE': 10000.0,   # Starting balance in USD
    'RISK_PER_TRADE': 0.02,       # Risk 2% of balance per trade
    'MIN_SL_DISTANCE': 0.001,     # Minimum stop loss distance (0.1%)
    'MAX_SL_DISTANCE': 0.05,      # Maximum stop loss distance (5%)
    'TP_RISK_RATIO': 2.0,         # Take profit to risk ratio (2:1)
    'MAX_POSITIONS': 3,            # Maximum concurrent positions
    
    # === MODEL PARAMETERS ===
    'SEQUENCE_LENGTH': 60,         # Number of time steps to analyze
    'LEARNING_RATE': 0.001,        # Learning rate for model training
    'HIDDEN_SIZE': 128,            # LSTM hidden layer size
    'NUM_LAYERS': 3,               # Number of LSTM layers
    'DROPOUT': 0.2,                # Dropout rate for regularization
    'GRADIENT_CLIP': 2.0,          # Gradient clipping value
}

# === ENVIRONMENT VARIABLE SETUP ===
def setup_environment():
    """Setup required directories"""
    import os
    
    # Create required directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    print("📁 Created required directories")

def validate_config():
    """Validate configuration settings"""
    issues = []
    
    # Check API key
    if not CONFIG['TD_API_KEY'] or CONFIG['TD_API_KEY'] == "your_twelve_data_api_key":
        issues.append("❌ TD_API_KEY not set. Get your free API key from https://twelvedata.com")
    
    # Check Discord webhook (optional)
    if not CONFIG['DISCORD_WEBHOOK_URL'] or CONFIG['DISCORD_WEBHOOK_URL'] == "your_discord_webhook_url":
        print("⚠️  Discord webhook not configured. Notifications will be disabled.")
    
    # Check trading parameters
    if CONFIG['RISK_PER_TRADE'] > 0.1:
        issues.append("⚠️  Risk per trade is very high (>10%). Consider reducing it.")
    
    if CONFIG['CONFIDENCE_THRESHOLD'] < 0.5:
        issues.append("⚠️  Confidence threshold is low (<50%). This may result in many trades.")
    
    if issues:
        print("\n🔍 Configuration Issues:")
        for issue in issues:
            print(f"  {issue}")
        print()
    
    return len([i for i in issues if i.startswith("❌")]) == 0

if __name__ == "__main__":
    setup_environment()
    if validate_config():
        print("✅ Configuration validated successfully!")
    else:
        print("❌ Please fix configuration issues before running the bot.")