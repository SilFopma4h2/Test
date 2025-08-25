# 🚀 Enhanced XAU/USD Deep Learning Trading Bot v2.0

A comprehensive, production-ready trading bot that uses advanced LSTM neural networks and technical analysis to generate intelligent trading signals for XAU/USD (Gold) with real-time position management.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## ✨ Key Features

### 🧠 Advanced AI & Machine Learning
- **LSTM Neural Network**: Deep learning model with attention mechanism
- **Multi-head Attention**: Enhanced pattern recognition
- **Dynamic Feature Engineering**: 12+ technical indicators
- **Adaptive Learning**: Continuous model improvement

### 📊 Technical Analysis
- **Moving Averages**: EMA9, EMA21, EMA50
- **Momentum Indicators**: RSI, MACD with signal line
- **Volatility Analysis**: Bollinger Bands, ATR
- **Volume Analysis**: Volume ratio and patterns

### 🎯 Smart Position Management
- **Automated Stop Loss**: AI-predicted optimal levels
- **Take Profit Targets**: Dynamic risk-reward ratios
- **Position Sizing**: Kelly criterion based sizing
- **Risk Management**: Configurable risk per trade

### 📱 Real-time Monitoring
- **GUI Dashboard**: Beautiful tkinter interface
- **Discord Notifications**: Real-time trade alerts
- **Live Statistics**: Performance tracking
- **Console Mode**: Terminal-based operation

### 💼 Professional Features
- **Paper Trading**: Safe backtesting environment
- **Trade Logging**: Complete transaction history
- **Model Persistence**: Save/load trained models
- **Error Handling**: Robust error recovery

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/SilFopma4h2/Test.git
cd Test
pip install -r requirements.txt
```

### 2. Configure API Keys
Edit `config.py` and add your API credentials:

```python
CONFIG = {
    'TD_API_KEY': "your_twelve_data_api_key",  # Get from https://twelvedata.com
    'DISCORD_WEBHOOK_URL': "your_discord_webhook_url",  # Optional
    # ... other settings
}
```

### 3. Run the Bot
```bash
python main.py
```

Choose your preferred mode:
- **Console Mode (1)**: Text-based interface
- **GUI Dashboard (2)**: Graphical interface

## 🔧 Configuration

### API Setup

#### 1. Twelve Data API (Required)
1. Sign up at [TwelveData](https://twelvedata.com)
2. Get your free API key (500 requests/day)
3. Add to `config.py`: `TD_API_KEY = "your_key"`

#### 2. Discord Webhook (Optional)
1. Go to Discord channel → Settings → Integrations → Webhooks
2. Create webhook and copy URL
3. Add to `config.py`: `DISCORD_WEBHOOK_URL = "your_url"`

### Trading Parameters

```python
CONFIG = {
    'CONFIDENCE_THRESHOLD': 0.6,    # Minimum confidence (60%)
    'INITIAL_BALANCE': 10000.0,     # Starting balance
    'RISK_PER_TRADE': 0.02,         # Risk 2% per trade
    'MIN_SL_DISTANCE': 0.001,       # Min stop loss (0.1%)
    'MAX_SL_DISTANCE': 0.05,        # Max stop loss (5%)
    'TP_RISK_RATIO': 2.0,           # 2:1 risk-reward ratio
}
```

## 📈 How It Works

### 1. Data Collection
- Fetches real-time XAU/USD price data
- Calculates technical indicators
- Normalizes features for AI model

### 2. AI Prediction
- LSTM processes 60-step price sequences
- Predicts direction, confidence, SL, and TP
- Uses attention mechanism for better accuracy

### 3. Signal Generation
- Generates BUY/SELL signals when confidence > threshold
- Calculates optimal entry, stop loss, and take profit
- Applies risk management rules

### 4. Position Management
- Opens positions with proper sizing
- Monitors for SL/TP triggers
- Automatically closes positions
- Logs all transactions

## 🖥️ GUI Dashboard

The enhanced GUI provides:

- **📊 Statistics Tab**: Performance metrics, win rate, ROI
- **🎯 Positions Tab**: Active positions monitoring
- **📋 Trades Tab**: Complete trade history
- **⚙️ Controls**: Start/stop bot, save model

## 📊 Example Alerts

### Discord Notification
```
🚀 New BUY Position Opened!
Symbol: XAU/USD
Entry: $2350.45
SL: $2327.95 (0.96%)
TP: $2395.45 (1.92%)
Confidence: 78.5%
Size: 1.25
Balance: $9,875.00
```

## ⚙️ Model Architecture

### LSTM Network
- **Input Features**: 12 technical indicators
- **Sequence Length**: 60 time steps
- **Hidden Layers**: 3 LSTM layers (128 units each)
- **Attention Heads**: 8-head multi-attention
- **Output**: Direction, Stop Loss, Take Profit

### Technical Indicators
1. **Price Normalization**: Rolling mean/std
2. **EMA Differences**: 9/21 and 21/50 crossovers
3. **RSI**: Normalized relative strength
4. **MACD**: Signal line and histogram
5. **ATR**: Average true range
6. **Bollinger Bands**: Price position
7. **Returns & Volatility**: Price momentum
8. **Volume Ratio**: Volume analysis

## 🎯 Performance Tracking

The bot tracks comprehensive statistics:
- **Total Trades**: Number of completed trades
- **Win Rate**: Percentage of profitable trades  
- **ROI**: Return on investment
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest loss period
- **Average Trade Duration**: Holding time analysis

## 🛡️ Risk Management

### Position Sizing
- **Kelly Criterion**: Optimal position sizing
- **Risk Per Trade**: Configurable percentage
- **Maximum Positions**: Limited concurrent trades

### Stop Loss & Take Profit
- **AI-Predicted Levels**: Model determines optimal SL/TP
- **Dynamic Ratios**: Adaptive risk-reward ratios
- **Minimum Distances**: Prevents over-tight stops

## 📝 Logging & Persistence

### Trade Logging
- **JSON Format**: Complete trade records
- **Real-time Updates**: Instant logging
- **Historical Analysis**: Performance review

### Model Persistence
- **Checkpoint Saving**: Model state preservation
- **Feature Normalization**: Scaler parameters saved
- **Continuous Learning**: Model improvements retained

## 🔍 Troubleshooting

### Common Issues

**API Errors**
```
Solution: Check API key validity and request limits
```

**Model Loading Errors**
```
Solution: Delete models/ directory to retrain
```

**Discord Webhook Failures**
```
Solution: Verify webhook URL or disable notifications
```

## ⚠️ Important Disclaimers

- **Paper Trading Only**: This bot does not place real trades
- **Educational Purpose**: For learning and research only
- **No Financial Advice**: Not investment advice
- **Risk Warning**: Trading involves significant risk
- **No Warranty**: Use at your own risk

## 🛠️ Development

### Project Structure
```
├── main.py              # Main application
├── config.py            # Configuration settings
├── requirements.txt     # Dependencies
├── data/               # Trading logs
├── models/             # Saved AI models
└── README.md           # This file
```

### Adding Features
- Modify `main.py` for new functionality
- Update `config.py` for new parameters
- Add dependencies to `requirements.txt`

## 📊 System Requirements

- **Python**: 3.8 or higher
- **Memory**: 4GB RAM minimum
- **Storage**: 1GB free space
- **Internet**: Stable connection required
- **OS**: Windows, macOS, or Linux

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Check the troubleshooting section
- Review configuration settings

---

**⚡ Ready to trade smarter with AI? Get started now!** 

*Remember: Always test with paper trading before considering real money.*
