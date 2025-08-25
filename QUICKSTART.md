# 🚀 Quick Start Guide

## Instant Setup (30 seconds)

### 1. Clone & Setup
```bash
git clone https://github.com/SilFopma4h2/Test.git
cd Test
chmod +x install.sh
./install.sh
```

### 2. Configure API Keys
Edit `config.py`:
```python
'TD_API_KEY': "your_twelve_data_api_key",  # Get from https://twelvedata.com
'DISCORD_WEBHOOK_URL': "your_discord_webhook_url",  # Optional
```

### 3. Run the Bot
```bash
python run.py    # Easy launcher menu
```

Or directly:
```bash
python simple_bot.py    # Lightweight version (works immediately)
python main.py          # Advanced ML version (requires dependencies)
```

## 📊 What Happens Next

1. **Simple Bot**: Works immediately with basic momentum strategy
2. **Advanced Bot**: Requires `pip install -r requirements.txt` for ML features
3. **Both versions** include:
   - Real-time XAU/USD price monitoring
   - Automatic position management
   - Stop loss & take profit execution
   - Discord notifications
   - Complete trade logging
   - GUI dashboard (when available)

## 🎯 Key Features

- ✅ **Ready to Use**: Simple version works instantly
- ✅ **Professional**: Advanced ML-powered trading
- ✅ **Safe**: Paper trading only (no real money)
- ✅ **Educational**: Learn algorithmic trading
- ✅ **Customizable**: Easy configuration

## 💡 Tips

- Start with the simple bot to understand the interface
- Update API keys for live data
- Monitor performance through the GUI or logs
- Adjust risk settings in `config.py`

---

**⚡ Ready to trade smarter? Start now!**