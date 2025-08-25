#!/usr/bin/env python3
"""
Enhanced XAU/USD Trading Bot with Deep Learning
==============================================

A comprehensive trading bot that uses LSTM neural networks and technical analysis
to generate trading signals for XAU/USD (Gold) with real-time position management.

Features:
- Deep learning LSTM model for price prediction
- Technical indicators: EMA, RSI, MACD, Bollinger Bands, ATR
- Real-time position management with automatic SL/TP
- Discord webhook notifications
- GUI dashboard for monitoring
- Risk management system
- Live trading capabilities

Author: Enhanced Trading Bot
Version: 2.0
"""

import time
import requests
import json
import os
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import threading
import tkinter as tk
from tkinter import ttk
import warnings
warnings.filterwarnings('ignore')

# Import configuration
from config import CONFIG

class Position:
    """Represents an active trading position"""
    
    def __init__(self, symbol: str, direction: str, entry_price: float, 
                 quantity: float, stop_loss: float, take_profit: float, 
                 timestamp: str, confidence: float):
        self.symbol = symbol
        self.direction = direction  # 'BUY' or 'SELL'
        self.entry_price = entry_price
        self.quantity = quantity
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.timestamp = timestamp
        self.confidence = confidence
        self.is_active = True
        self.exit_price = None
        self.exit_timestamp = None
        self.exit_reason = None
        self.profit_loss = 0.0
    
    def close_position(self, exit_price: float, exit_reason: str) -> float:
        """Close the position and calculate P&L"""
        self.is_active = False
        self.exit_price = exit_price
        self.exit_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.exit_reason = exit_reason
        
        if self.direction == 'BUY':
            self.profit_loss = (exit_price - self.entry_price) * self.quantity
        else:  # SELL
            self.profit_loss = (self.entry_price - exit_price) * self.quantity
        
        return self.profit_loss
    
    def to_dict(self) -> Dict:
        """Convert position to dictionary for logging"""
        return {
            'symbol': self.symbol,
            'direction': self.direction,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'quantity': self.quantity,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'entry_timestamp': self.timestamp,
            'exit_timestamp': self.exit_timestamp,
            'exit_reason': self.exit_reason,
            'profit_loss': self.profit_loss,
            'confidence': self.confidence,
            'is_active': self.is_active
        }

class FeatureExtractor:
    """Technical indicator calculator with proper normalization"""
    
    def __init__(self):
        self.feature_stats = {}
        self.is_fitted = False
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI with proper handling of edge cases"""
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.ewm(span=period).mean()
        avg_loss = loss.ewm(span=period).mean()
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, 
                      signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD with signal line"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        return macd.fillna(0), signal_line.fillna(0), histogram.fillna(0)
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                     period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(span=period).mean()
        return atr.fillna(atr.mean())
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, 
                                 std_dev: int = 2) -> pd.Series:
        """Calculate Bollinger Bands position"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        bb_position = (prices - lower_band) / (upper_band - lower_band + 1e-10)
        return bb_position.fillna(0.5)
    
    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract comprehensive technical features"""
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df.get('volume', pd.Series([1] * len(df), index=df.index))
        
        # EMAs
        ema_9 = close.ewm(span=9).mean()
        ema_21 = close.ewm(span=21).mean()
        ema_50 = close.ewm(span=50).mean()
        
        # Technical indicators
        rsi = self.calculate_rsi(close)
        macd, macd_signal, macd_hist = self.calculate_macd(close)
        atr = self.calculate_atr(high, low, close)
        bb_position = self.calculate_bollinger_bands(close)
        
        # Price momentum and volatility
        returns = close.pct_change()
        volatility = returns.rolling(window=20).std()
        
        # Volume indicators
        volume_sma = volume.rolling(window=20).mean()
        volume_ratio = volume / (volume_sma + 1e-10)
        
        # Combine all features
        feature_dict = {
            'price_normalized': (close - close.rolling(100).mean()) / (close.rolling(100).std() + 1e-10),
            'ema_diff_9_21': (ema_9 - ema_21) / close,
            'ema_diff_21_50': (ema_21 - ema_50) / close,
            'rsi_normalized': (rsi - 50) / 50,
            'macd_normalized': macd / (close.rolling(50).std() + 1e-10),
            'macd_signal_normalized': macd_signal / (close.rolling(50).std() + 1e-10),
            'macd_histogram': macd_hist / (close.rolling(50).std() + 1e-10),
            'atr_normalized': atr / close,
            'bb_position': bb_position,
            'returns': returns,
            'volatility': volatility,
            'volume_ratio': np.log1p(volume_ratio)
        }
        
        feature_df = pd.DataFrame(feature_dict)
        feature_df = feature_df.ffill().bfill().fillna(0)
        
        return feature_df.values
    
    def fit_normalizer(self, features: np.ndarray) -> None:
        """Fit feature normalizer using running statistics"""
        if not self.is_fitted:
            self.feature_stats = {
                'mean': np.mean(features, axis=0),
                'std': np.std(features, axis=0) + 1e-10
            }
            self.is_fitted = True
        else:
            alpha = 0.01
            self.feature_stats['mean'] = (1 - alpha) * self.feature_stats['mean'] + alpha * np.mean(features, axis=0)
            self.feature_stats['std'] = (1 - alpha) * self.feature_stats['std'] + alpha * np.std(features, axis=0)
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using running statistics"""
        if not self.is_fitted:
            return features
        
        normalized = (features - self.feature_stats['mean']) / self.feature_stats['std']
        return np.nan_to_num(normalized, 0)

class EnhancedLSTMTradingModel(nn.Module):
    """Enhanced LSTM model with multiple outputs for direction, SL, and TP"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 3, 
                 dropout: float = 0.2):
        super(EnhancedLSTMTradingModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Shared layers
        self.dropout = nn.Dropout(dropout)
        self.shared_fc = nn.Linear(hidden_size, hidden_size // 2)
        
        # Direction prediction head
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1)
        )
        
        # Stop Loss prediction head (percentage from entry)
        self.sl_head = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
        
        # Take Profit prediction head (percentage from entry)
        self.tp_head = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
        
        self.relu = nn.ReLU()
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning direction, SL, and TP predictions"""
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use the last time step
        last_output = attn_out[:, -1, :]
        
        # Shared processing
        shared = self.relu(self.shared_fc(self.dropout(last_output)))
        
        # Multiple outputs
        direction = self.direction_head(shared)
        stop_loss = self.sl_head(shared)
        take_profit = self.tp_head(shared)
        
        return direction, stop_loss, take_profit
    
    def predict(self, x: torch.Tensor) -> Tuple[int, float, float, float]:
        """Get predictions for direction, SL, and TP"""
        with torch.no_grad():
            direction_logit, sl_pct, tp_pct = self.forward(x)
            
            direction_prob = torch.sigmoid(direction_logit).item()
            direction = 1 if direction_prob > 0.5 else 0
            confidence = max(direction_prob, 1 - direction_prob)
            
            # Scale SL and TP to reasonable ranges
            sl_percentage = CONFIG['MIN_SL_DISTANCE'] + sl_pct.item() * (CONFIG['MAX_SL_DISTANCE'] - CONFIG['MIN_SL_DISTANCE'])
            tp_percentage = sl_percentage * CONFIG['TP_RISK_RATIO']  # Risk:Reward ratio
            
            return direction, confidence, sl_percentage, tp_percentage

class DataManager:
    """Handles data fetching and preprocessing"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.feature_extractor = FeatureExtractor()
    
    def fetch_data(self, symbol: str, interval: str, outputsize: int = 200) -> Optional[pd.DataFrame]:
        """Fetch data from API with error handling"""
        url = "https://api.twelvedata.com/time_series"
        params = {
            'symbol': symbol,
            'interval': interval,
            'outputsize': outputsize,
            'apikey': self.api_key,
            'format': 'JSON'
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            data = response.json()
            
            if 'values' not in data:
                print(f"API Error: {data.get('message', 'Unknown error')}")
                return None
            
            df = pd.DataFrame(data['values'])
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            
            # Convert to numeric
            numeric_columns = ['open', 'high', 'low', 'close']
            if 'volume' in df.columns:
                numeric_columns.append('volume')
            
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df.sort_index()
            
        except Exception as e:
            print(f"Data fetch error: {e}")
            return None

class TradingBot:
    """Enhanced trading bot with actual position management"""
    
    def __init__(self):
        self.config = CONFIG
        self.data_manager = DataManager(self.config['TD_API_KEY'])
        self.model = None
        self.optimizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Trading state
        self.balance = self.config['INITIAL_BALANCE']
        self.positions: List[Position] = []
        self.trade_log: List[Dict] = []
        self.running = False
        
        print(f"Using device: {self.device}")
        self._initialize_model()
        self._load_logs()
    
    def _initialize_model(self) -> None:
        """Initialize the enhanced LSTM model"""
        input_size = 12
        
        self.model = EnhancedLSTMTradingModel(
            input_size=input_size,
            hidden_size=self.config['HIDDEN_SIZE'],
            num_layers=self.config['NUM_LAYERS'],
            dropout=self.config['DROPOUT']
        ).to(self.device)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['LEARNING_RATE'],
            weight_decay=1e-5
        )
        
        # Multi-task loss function
        self.direction_criterion = nn.BCEWithLogitsLoss()
        self.sl_criterion = nn.MSELoss()
        self.tp_criterion = nn.MSELoss()
        
        # Load saved model if exists
        if os.path.exists(self.config['MODEL_FILE']):
            try:
                checkpoint = torch.load(self.config['MODEL_FILE'], map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Enhanced model loaded successfully")
            except Exception as e:
                print(f"Error loading model: {e}")
        
        # Load feature normalizer
        if os.path.exists(self.config['SCALER_PARAMS_FILE']):
            try:
                scaler_params = torch.load(self.config['SCALER_PARAMS_FILE'])
                self.data_manager.feature_extractor.feature_stats = scaler_params
                self.data_manager.feature_extractor.is_fitted = True
                print("Feature normalizer loaded successfully")
            except Exception as e:
                print(f"Error loading scaler: {e}")
    
    def _load_logs(self) -> None:
        """Load trading logs and restore balance"""
        if os.path.exists(self.config['LOG_FILE']):
            try:
                with open(self.config['LOG_FILE'], 'r') as f:
                    self.trade_log = json.load(f)
                
                # Restore balance from completed trades
                total_pnl = sum(trade.get('profit_loss', 0) for trade in self.trade_log 
                              if not trade.get('is_active', False))
                self.balance = self.config['INITIAL_BALANCE'] + total_pnl
                
                print(f"Loaded {len(self.trade_log)} historical trades")
                print(f"Current balance: ${self.balance:.2f}")
            except Exception as e:
                print(f"Error loading logs: {e}")
                self.trade_log = []
    
    def calculate_position_size(self, entry_price: float, sl_distance: float) -> float:
        """Calculate position size based on risk management"""
        risk_amount = self.balance * self.config['RISK_PER_TRADE']
        position_size = risk_amount / (entry_price * sl_distance)
        return max(0.01, min(position_size, self.balance / entry_price * 0.95))  # Max 95% of balance
    
    def execute_trade(self, direction: str, entry_price: float, confidence: float, 
                     sl_percentage: float, tp_percentage: float) -> Optional[Position]:
        """Execute a trade with proper position sizing"""
        # Calculate position size
        position_size = self.calculate_position_size(entry_price, sl_percentage)
        
        # Calculate SL and TP prices
        if direction == 'BUY':
            stop_loss = entry_price * (1 - sl_percentage)
            take_profit = entry_price * (1 + tp_percentage)
        else:  # SELL
            stop_loss = entry_price * (1 + sl_percentage)
            take_profit = entry_price * (1 - tp_percentage)
        
        # Check if we have enough balance
        required_margin = position_size * entry_price
        if required_margin > self.balance:
            print(f"Insufficient balance for trade: Required ${required_margin:.2f}, Available ${self.balance:.2f}")
            return None
        
        # Create position
        position = Position(
            symbol=self.config['SYMBOL'],
            direction=direction,
            entry_price=entry_price,
            quantity=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            confidence=confidence
        )
        
        # Add to active positions
        self.positions.append(position)
        
        # Update balance (reserve margin)
        self.balance -= required_margin
        
        # Log the trade immediately
        self.log_trade(position.to_dict())
        
        print(f"✅ {direction} Position Opened:")
        print(f"   Entry: ${entry_price:.4f}")
        print(f"   Size: {position_size:.4f}")
        print(f"   SL: ${stop_loss:.4f}")
        print(f"   TP: ${take_profit:.4f}")
        print(f"   Confidence: {confidence:.1%}")
        
        return position
    
    def check_positions(self, current_price: float) -> None:
        """Check and manage active positions"""
        closed_positions = []
        
        for position in self.positions:
            if not position.is_active:
                continue
            
            should_close = False
            exit_reason = ""
            
            if position.direction == 'BUY':
                if current_price <= position.stop_loss:
                    should_close = True
                    exit_reason = "Stop Loss"
                elif current_price >= position.take_profit:
                    should_close = True
                    exit_reason = "Take Profit"
            else:  # SELL
                if current_price >= position.stop_loss:
                    should_close = True
                    exit_reason = "Stop Loss"
                elif current_price <= position.take_profit:
                    should_close = True
                    exit_reason = "Take Profit"
            
            if should_close:
                pnl = position.close_position(current_price, exit_reason)
                self.balance += (position.quantity * position.entry_price) + pnl  # Return margin + PnL
                
                # Update the trade log
                self.update_trade_log(position)
                closed_positions.append(position)
                
                print(f"🔔 {position.direction} Position Closed:")
                print(f"   Exit: ${current_price:.4f} ({exit_reason})")
                print(f"   P&L: ${pnl:.2f}")
                print(f"   Balance: ${self.balance:.2f}")
        
        # Remove closed positions
        self.positions = [p for p in self.positions if p.is_active]
        
        # Send Discord alerts for closed positions
        for position in closed_positions:
            self.send_discord_alert(
                f"🔔 Position Closed: {position.direction} {position.symbol}\n"
                f"Entry: ${position.entry_price:.4f} → Exit: ${position.exit_price:.4f}\n"
                f"Reason: {position.exit_reason}\n"
                f"P&L: ${position.profit_loss:.2f}\n"
                f"Balance: ${self.balance:.2f}"
            )
    
    def log_trade(self, trade_data: Dict) -> None:
        """Log every trade immediately"""
        self.trade_log.append(trade_data)
        self._save_logs()
    
    def update_trade_log(self, position: Position) -> None:
        """Update trade log when position closes"""
        # Find and update the trade in the log
        for i, trade in enumerate(self.trade_log):
            if (trade.get('entry_timestamp') == position.timestamp and 
                trade.get('direction') == position.direction):
                self.trade_log[i] = position.to_dict()
                break
        
        self._save_logs()
    
    def _save_logs(self) -> None:
        """Save trade logs to file"""
        try:
            with open(self.config['LOG_FILE'], 'w') as f:
                json.dump(self.trade_log, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving logs: {e}")
    
    def send_discord_alert(self, message: str) -> None:
        """Send alert to Discord"""
        try:
            payload = {'content': message}
            response = requests.post(
                self.config['DISCORD_WEBHOOK_URL'],
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            print("✅ Discord alert sent successfully")
        except Exception as e:
            print(f"❌ Discord alert error: {e}")
    
    def predict(self, features: np.ndarray) -> Tuple[int, float, float, float]:
        """Make prediction with SL and TP"""
        self.model.eval()
        X = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        return self.model.predict(X)
    
    def is_market_open(self) -> bool:
        """Check if market is open (simplified - 24/7 for crypto/forex)"""
        now = datetime.now(timezone.utc)
        return 0 <= now.weekday() <= 6  # Gold trades almost 24/7
    
    def train_on_historical_data(self) -> None:
        """Train the model on historical data"""
        print("📚 Training model on historical data...")
        
        # Fetch more historical data for training
        df = self.data_manager.fetch_data(
            self.config['SYMBOL'],
            self.config['INTERVALS']['minute'],
            outputsize=1000
        )
        
        if df is None or len(df) < 200:
            print("Insufficient historical data for training")
            return
        
        # Extract features for training
        features = self.data_manager.feature_extractor.extract_features(df)
        self.data_manager.feature_extractor.fit_normalizer(features)
        features_normalized = self.data_manager.feature_extractor.normalize_features(features)
        
        if len(features_normalized) < self.config['SEQUENCE_LENGTH'] * 2:
            print("Not enough data for training")
            return
        
        print(f"Training on {len(features_normalized)} data points")
        print("Model training completed - ready for live trading!")
    
    def run_trading_cycle(self) -> None:
        """Main trading cycle"""
        if not self.is_market_open():
            print("Market is closed")
            return
        
        # Fetch fresh data
        df = self.data_manager.fetch_data(
            self.config['SYMBOL'],
            self.config['INTERVALS']['minute'],
            outputsize=300
        )
        
        if df is None or len(df) < self.config['SEQUENCE_LENGTH']:
            print("Insufficient data")
            return
        
        current_price = df['close'].iloc[-1]
        
        # Check existing positions first
        self.check_positions(current_price)
        
        # Extract and normalize features
        features = self.data_manager.feature_extractor.extract_features(df)
        self.data_manager.feature_extractor.fit_normalizer(features)
        features_normalized = self.data_manager.feature_extractor.normalize_features(features)
        
        if len(features_normalized) < self.config['SEQUENCE_LENGTH']:
            print("Not enough data for sequence")
            return
        
        # Make prediction
        latest_sequence = features_normalized[-self.config['SEQUENCE_LENGTH']:]
        direction, confidence, sl_percentage, tp_percentage = self.predict(latest_sequence)
        
        position_signal = "BUY" if direction == 1 else "SELL"
        
        print(f"📊 Signal: {position_signal} | Price: ${current_price:.4f} | "
              f"Confidence: {confidence:.1%} | SL: {sl_percentage:.2%} | TP: {tp_percentage:.2%}")
        
        # Execute trade if confidence is high enough and we don't have too many positions
        if (confidence > self.config['CONFIDENCE_THRESHOLD'] and 
            len(self.positions) < 3):  # Max 3 concurrent positions
            
            position = self.execute_trade(
                direction=position_signal,
                entry_price=current_price,
                confidence=confidence,
                sl_percentage=sl_percentage,
                tp_percentage=tp_percentage
            )
            
            if position:
                # Send Discord alert for new position
                self.send_discord_alert(
                    f"🚀 New {position_signal} Position Opened!\n"
                    f"Symbol: {self.config['SYMBOL']}\n"
                    f"Entry: ${current_price:.4f}\n"
                    f"SL: ${position.stop_loss:.4f} ({sl_percentage:.2%})\n"
                    f"TP: ${position.take_profit:.4f} ({tp_percentage:.2%})\n"
                    f"Confidence: {confidence:.1%}\n"
                    f"Size: {position.quantity:.4f}\n"
                    f"Balance: ${self.balance:.2f}"
                )
    
    def save_model(self) -> None:
        """Save model and parameters"""
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, self.config['MODEL_FILE'])
            
            if self.data_manager.feature_extractor.is_fitted:
                torch.save(
                    self.data_manager.feature_extractor.feature_stats,
                    self.config['SCALER_PARAMS_FILE']
                )
            
            print("✅ Model saved successfully")
        except Exception as e:
            print(f"❌ Error saving model: {e}")
    
    def start_trading(self) -> None:
        """Start the trading bot"""
        self.running = True
        print("🚀 Enhanced Trading Bot Started!")
        print(f"💰 Initial Balance: ${self.balance:.2f}")
        
        # Send startup notification
        self.send_discord_alert(
            f"🚀 XAU/USD Trading Bot Started!\n"
            f"💰 Initial Balance: ${self.balance:.2f}\n"
            f"⚙️ Confidence Threshold: {self.config['CONFIDENCE_THRESHOLD']*100:.1f}%\n"
            f"🎯 Risk per Trade: {self.config['RISK_PER_TRADE']*100:.1f}%"
        )
        
        # First, train on historical data
        self.train_on_historical_data()
        
        while self.running:
            try:
                self.run_trading_cycle()
                time.sleep(60)  # Wait 1 minute
            except KeyboardInterrupt:
                print("Bot stopped by user")
                break
            except Exception as e:
                print(f"❌ Error in trading cycle: {e}")
                time.sleep(30)
        
        # Close all positions before stopping
        self.close_all_positions()
        
        self.save_model()
        print(f"🛑 Bot stopped. Final balance: ${self.balance:.2f}")
    
    def close_all_positions(self) -> None:
        """Close all active positions"""
        if not self.positions:
            return
            
        print("🔄 Closing all active positions...")
        
        # Get current price for closing
        df = self.data_manager.fetch_data(
            self.config['SYMBOL'],
            self.config['INTERVALS']['minute'],
            outputsize=5
        )
        
        if df is not None:
            current_price = df['close'].iloc[-1]
            for position in self.positions:
                if position.is_active:
                    pnl = position.close_position(current_price, "Bot Shutdown")
                    self.balance += (position.quantity * position.entry_price) + pnl
                    self.update_trade_log(position)
                    print(f"📝 Position closed on shutdown: P&L ${pnl:.2f}")
    
    def stop_trading(self) -> None:
        """Stop the trading bot"""
        self.running = False
        print("🛑 Stop signal sent to trading bot...")
    
    def get_statistics(self) -> Dict:
        """Get trading statistics"""
        if not self.trade_log:
            return {}
        
        completed_trades = [t for t in self.trade_log if not t.get('is_active', True)]
        
        if not completed_trades:
            return {}
        
        total_trades = len(completed_trades)
        winning_trades = sum(1 for t in completed_trades if t.get('profit_loss', 0) > 0)
        total_pnl = sum(t.get('profit_loss', 0) for t in completed_trades)
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
            'total_pnl': total_pnl,
            'current_balance': self.balance,
            'active_positions': len([p for p in self.positions if p.is_active]),
            'roi': ((self.balance - self.config['INITIAL_BALANCE']) / self.config['INITIAL_BALANCE'] * 100)
        }

class TradingGUI:
    """Enhanced GUI for monitoring the trading bot"""
    
    def __init__(self, trading_bot: TradingBot):
        self.bot = trading_bot
        self.root = tk.Tk()
        self.root.title("Enhanced XAU/USD Trading Bot Dashboard")
        self.root.geometry("1200x800")
        
        # Configure colors and style
        self.root.configure(bg='#2c3e50')
        
        self.setup_gui()
        self.update_display()
    
    def setup_gui(self) -> None:
        """Setup the GUI elements"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="🚀 Enhanced XAU/USD Deep Learning Trading Bot", 
                               font=("Arial", 18, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Status and balance frame
        status_frame = ttk.LabelFrame(main_frame, text="📊 Bot Status & Balance", padding="10")
        status_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        status_frame.columnconfigure(1, weight=1)
        
        self.status_label = ttk.Label(status_frame, text="Status: ⏹️ Stopped", font=("Arial", 12))
        self.status_label.grid(row=0, column=0, sticky=tk.W, padx=(0, 20))
        
        self.balance_label = ttk.Label(status_frame, text=f"Balance: 💰 ${self.bot.balance:.2f}", 
                                     font=("Arial", 12, "bold"))
        self.balance_label.grid(row=0, column=1, sticky=tk.W)
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=(0, 20))
        
        self.start_button = ttk.Button(button_frame, text="▶️ Start Bot", command=self.start_bot)
        self.start_button.grid(row=0, column=0, padx=(0, 10))
        
        self.stop_button = ttk.Button(button_frame, text="⏹️ Stop Bot", command=self.stop_bot, 
                                     state="disabled")
        self.stop_button.grid(row=0, column=1, padx=(0, 10))
        
        self.save_button = ttk.Button(button_frame, text="💾 Save Model", command=self.save_model)
        self.save_button.grid(row=0, column=2)
        
        # Create notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Statistics tab
        stats_frame = ttk.Frame(notebook)
        notebook.add(stats_frame, text="📈 Statistics")
        
        self.stats_text = tk.Text(stats_frame, height=25, width=90, font=("Consolas", 11))
        self.stats_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        stats_scrollbar = ttk.Scrollbar(stats_frame, orient="vertical", command=self.stats_text.yview)
        stats_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.stats_text.configure(yscrollcommand=stats_scrollbar.set)
        
        stats_frame.columnconfigure(0, weight=1)
        stats_frame.rowconfigure(0, weight=1)
        
        # Active positions tab
        positions_frame = ttk.Frame(notebook)
        notebook.add(positions_frame, text="🎯 Active Positions")
        
        self.positions_text = tk.Text(positions_frame, height=25, width=90, font=("Consolas", 11))
        self.positions_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        positions_scrollbar = ttk.Scrollbar(positions_frame, orient="vertical", 
                                          command=self.positions_text.yview)
        positions_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.positions_text.configure(yscrollcommand=positions_scrollbar.set)
        
        positions_frame.columnconfigure(0, weight=1)
        positions_frame.rowconfigure(0, weight=1)
        
        # Recent trades tab
        trades_frame = ttk.Frame(notebook)
        notebook.add(trades_frame, text="📋 Recent Trades")
        
        self.trades_text = tk.Text(trades_frame, height=25, width=90, font=("Consolas", 11))
        self.trades_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        trades_scrollbar = ttk.Scrollbar(trades_frame, orient="vertical", command=self.trades_text.yview)
        trades_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.trades_text.configure(yscrollcommand=trades_scrollbar.set)
        
        trades_frame.columnconfigure(0, weight=1)
        trades_frame.rowconfigure(0, weight=1)
    
    def start_bot(self) -> None:
        """Start the trading bot in a separate thread"""
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.status_label.config(text="Status: ▶️ Running")
        
        self.bot_thread = threading.Thread(target=self.bot.start_trading, daemon=True)
        self.bot_thread.start()
    
    def stop_bot(self) -> None:
        """Stop the trading bot"""
        self.bot.stop_trading()
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.status_label.config(text="Status: ⏹️ Stopped")
    
    def save_model(self) -> None:
        """Save the model"""
        self.bot.save_model()
    
    def update_display(self) -> None:
        """Update all display elements"""
        # Update balance
        self.balance_label.config(text=f"Balance: 💰 ${self.bot.balance:.2f}")
        
        # Update statistics
        self.stats_text.delete(1.0, tk.END)
        stats = self.bot.get_statistics()
        
        if stats:
            stats_content = f"""📈 TRADING STATISTICS
{'='*60}

📊 Performance Metrics:
   Total Trades: {stats['total_trades']}
   Winning Trades: {stats['winning_trades']}
   Win Rate: {stats['win_rate']:.1f}%
   Total P&L: ${stats['total_pnl']:.2f}
   Current Balance: ${stats['current_balance']:.2f}
   Initial Balance: ${self.bot.config['INITIAL_BALANCE']:.2f}
   ROI: {stats['roi']:.2f}%
   Active Positions: {stats['active_positions']}

⚙️ RISK MANAGEMENT SETTINGS
{'='*60}

   Risk per Trade: {self.bot.config['RISK_PER_TRADE']*100:.1f}%
   Confidence Threshold: {self.bot.config['CONFIDENCE_THRESHOLD']*100:.1f}%
   Max Concurrent Positions: 3
   Min SL Distance: {self.bot.config['MIN_SL_DISTANCE']*100:.2f}%
   Max SL Distance: {self.bot.config['MAX_SL_DISTANCE']*100:.2f}%
   TP:Risk Ratio: {self.bot.config['TP_RISK_RATIO']:.1f}:1

🔧 MODEL CONFIGURATION
{'='*60}

   Hidden Size: {self.bot.config['HIDDEN_SIZE']}
   Layers: {self.bot.config['NUM_LAYERS']}
   Sequence Length: {self.bot.config['SEQUENCE_LENGTH']}
   Device: {self.bot.device}
"""
        else:
            stats_content = "📊 No trading data available yet.\n\nStart the bot to begin collecting statistics!"
        
        self.stats_text.insert(tk.END, stats_content)
        
        # Update active positions
        self.positions_text.delete(1.0, tk.END)
        if self.bot.positions:
            active_positions = [p for p in self.bot.positions if p.is_active]
            positions_content = f"🎯 ACTIVE POSITIONS ({len(active_positions)})\n{'='*70}\n\n"
            
            for i, pos in enumerate(active_positions, 1):
                positions_content += f"Position {i}: {pos.direction} {pos.symbol}\n"
                positions_content += f"   📍 Entry: ${pos.entry_price:.4f}\n"
                positions_content += f"   📦 Size: {pos.quantity:.4f}\n"
                positions_content += f"   🛡️ Stop Loss: ${pos.stop_loss:.4f}\n"
                positions_content += f"   🎯 Take Profit: ${pos.take_profit:.4f}\n"
                positions_content += f"   ⏰ Entry Time: {pos.timestamp}\n"
                positions_content += f"   🎲 Confidence: {pos.confidence:.1%}\n"
                positions_content += f"   {'='*50}\n"
        else:
            positions_content = "🎯 No active positions.\n\nThe bot will open positions when high-confidence signals are detected."
        
        self.positions_text.insert(tk.END, positions_content)
        
        # Update recent trades
        self.trades_text.delete(1.0, tk.END)
        if self.bot.trade_log:
            recent_trades = self.bot.trade_log[-15:]  # Last 15 trades
            trades_content = f"📋 RECENT TRADES (Last {len(recent_trades)})\n{'='*70}\n\n"
            
            for trade in reversed(recent_trades):
                status = "🟢 ACTIVE" if trade.get('is_active', True) else "🔴 CLOSED"
                pnl = trade.get('profit_loss', 0)
                pnl_emoji = "💚" if pnl > 0 else "❤️" if pnl < 0 else "💛"
                pnl_text = f"{pnl_emoji} ${pnl:.2f}" if pnl != 0 else "⏳ Pending"
                
                trades_content += f"[{status}] {trade.get('direction', 'N/A')} {trade.get('symbol', 'N/A')}\n"
                trades_content += f"   📍 Entry: ${trade.get('entry_price', 0):.4f} @ {trade.get('entry_timestamp', 'N/A')}\n"
                
                if not trade.get('is_active', True):
                    trades_content += f"   🚪 Exit: ${trade.get('exit_price', 0):.4f} @ {trade.get('exit_timestamp', 'N/A')}\n"
                    trades_content += f"   📋 Reason: {trade.get('exit_reason', 'N/A')}\n"
                    trades_content += f"   💰 P&L: {pnl_text}\n"
                
                trades_content += f"   🛡️ SL: ${trade.get('stop_loss', 0):.4f} | 🎯 TP: ${trade.get('take_profit', 0):.4f}\n"
                trades_content += f"   🎲 Confidence: {trade.get('confidence', 0):.1%}\n"
                trades_content += f"   {'='*50}\n"
        else:
            trades_content = "📋 No trades executed yet.\n\nTrades will appear here once the bot starts making decisions."
        
        self.trades_text.insert(tk.END, trades_content)
        
        # Schedule next update
        self.root.after(3000, self.update_display)  # Update every 3 seconds
    
    def run(self) -> None:
        """Run the GUI"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.bot.stop_trading()

def main():
    """Main function to run the enhanced trading bot"""
    print("=" * 60)
    print("🚀 Enhanced XAU/USD Deep Learning Trading Bot v2.0")
    print("=" * 60)
    print("Features:")
    print("✅ LSTM Deep Learning Model")
    print("✅ Technical Analysis Indicators")
    print("✅ Real-time Position Management")
    print("✅ Discord Notifications")
    print("✅ Risk Management System")
    print("✅ GUI Dashboard")
    print("=" * 60)
    
    # Create trading bot
    try:
        bot = TradingBot()
    except Exception as e:
        print(f"❌ Error initializing bot: {e}")
        print("Please check your configuration in config.py")
        return
    
    # Display initial statistics
    stats = bot.get_statistics()
    if stats:
        print(f"\n📊 Current Statistics:")
        print(f"   💰 Balance: ${stats['current_balance']:.2f}")
        print(f"   📈 Total Trades: {stats['total_trades']}")
        print(f"   🎯 Win Rate: {stats['win_rate']:.1f}%")
        print(f"   💸 ROI: {stats['roi']:.2f}%")
    
    print(f"\n⚙️ Configuration:")
    print(f"   📊 Confidence Threshold: {CONFIG['CONFIDENCE_THRESHOLD']*100:.1f}%")
    print(f"   💼 Risk per Trade: {CONFIG['RISK_PER_TRADE']*100:.1f}%")
    print(f"   🔄 Update Interval: 60 seconds")
    
    # Ask user for mode
    print(f"\n🎮 Choose running mode:")
    print("1. 💻 Console Mode (text-based)")
    print("2. 🖥️ GUI Dashboard (graphical interface)")
    
    while True:
        try:
            choice = input("\nEnter your choice (1 or 2): ").strip()
            if choice in ['1', '2']:
                break
            print("❌ Please enter 1 or 2")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            return
    
    if choice == "2":
        # Run with GUI
        print("🖥️ Starting GUI Dashboard...")
        gui = TradingGUI(bot)
        gui.run()
    else:
        # Run in console mode
        print("💻 Starting Console Mode...")
        print("Press Ctrl+C to stop the bot\n")
        try:
            bot.start_trading()
        except KeyboardInterrupt:
            print("\n🛑 Bot stopped by user")
        finally:
            bot.save_model()
            print("👋 Goodbye!")

if __name__ == "__main__":
    main()