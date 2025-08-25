#!/usr/bin/env python3
"""
Lightweight XAU/USD Trading Bot Demo
====================================

A simplified version that demonstrates the bot's functionality
without heavy ML dependencies. Perfect for immediate testing.
"""

import time
import requests
import json
import os
import random
from datetime import datetime, timezone
from typing import Dict, List, Optional
import threading
try:
    import tkinter as tk
    from tkinter import ttk
    GUI_AVAILABLE = True
except ImportError:
    print("GUI not available - running in console mode only")
    GUI_AVAILABLE = False

# Import configuration
from config import CONFIG

class SimplePosition:
    """Simple position management"""
    
    def __init__(self, symbol: str, direction: str, entry_price: float, 
                 quantity: float, stop_loss: float, take_profit: float, 
                 timestamp: str, confidence: float):
        self.symbol = symbol
        self.direction = direction
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
        """Convert to dictionary"""
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

class SimpleDataManager:
    """Simplified data fetching"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.last_price = None
        self.price_history = []
    
    def fetch_current_price(self) -> Optional[float]:
        """Fetch current price from API"""
        url = "https://api.twelvedata.com/price"
        params = {
            'symbol': 'XAU/USD',
            'apikey': self.api_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if 'price' in data:
                price = float(data['price'])
                self.last_price = price
                self.price_history.append((datetime.now(), price))
                # Keep only last 100 prices
                if len(self.price_history) > 100:
                    self.price_history = self.price_history[-100:]
                return price
            else:
                print(f"API Error: {data.get('message', 'Unknown error')}")
                return None
                
        except Exception as e:
            print(f"Data fetch error: {e}")
            return None
    
    def get_simple_signal(self) -> tuple:
        """Generate simple trading signal based on price movement"""
        if len(self.price_history) < 10:
            return None, 0.5
        
        # Simple momentum strategy
        recent_prices = [p[1] for p in self.price_history[-10:]]
        price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        
        # Add some randomness to simulate ML prediction
        confidence = min(0.9, 0.5 + abs(price_change) * 10 + random.uniform(0, 0.3))
        
        if price_change > 0.001:  # Rising trend
            return 'BUY', confidence
        elif price_change < -0.001:  # Falling trend
            return 'SELL', confidence
        else:
            return None, 0.5

class SimpleTradingBot:
    """Simplified trading bot for demonstration"""
    
    def __init__(self):
        self.config = CONFIG
        self.data_manager = SimpleDataManager(self.config['TD_API_KEY'])
        self.balance = self.config['INITIAL_BALANCE']
        self.positions: List[SimplePosition] = []
        self.trade_log: List[Dict] = []
        self.running = False
        
        self._load_logs()
        print("✅ Simple Trading Bot initialized")
    
    def _load_logs(self) -> None:
        """Load existing logs"""
        if os.path.exists(self.config['LOG_FILE']):
            try:
                with open(self.config['LOG_FILE'], 'r') as f:
                    self.trade_log = json.load(f)
                
                # Restore balance
                total_pnl = sum(trade.get('profit_loss', 0) for trade in self.trade_log 
                              if not trade.get('is_active', False))
                self.balance = self.config['INITIAL_BALANCE'] + total_pnl
                
                print(f"📊 Loaded {len(self.trade_log)} historical trades")
                print(f"💰 Current balance: ${self.balance:.2f}")
            except Exception as e:
                print(f"⚠️  Error loading logs: {e}")
                self.trade_log = []
    
    def calculate_position_size(self, entry_price: float, sl_distance: float) -> float:
        """Calculate position size"""
        risk_amount = self.balance * self.config['RISK_PER_TRADE']
        position_size = risk_amount / (entry_price * sl_distance)
        return max(0.01, min(position_size, self.balance / entry_price * 0.9))
    
    def execute_trade(self, direction: str, entry_price: float, confidence: float) -> Optional[SimplePosition]:
        """Execute a simple trade"""
        # Calculate SL and TP (simplified)
        sl_percentage = 0.02  # 2% stop loss
        tp_percentage = 0.04  # 4% take profit (2:1 ratio)
        
        if direction == 'BUY':
            stop_loss = entry_price * (1 - sl_percentage)
            take_profit = entry_price * (1 + tp_percentage)
        else:
            stop_loss = entry_price * (1 + sl_percentage)
            take_profit = entry_price * (1 - tp_percentage)
        
        position_size = self.calculate_position_size(entry_price, sl_percentage)
        required_margin = position_size * entry_price
        
        if required_margin > self.balance:
            print(f"❌ Insufficient balance: ${required_margin:.2f} required, ${self.balance:.2f} available")
            return None
        
        position = SimplePosition(
            symbol=self.config['SYMBOL'],
            direction=direction,
            entry_price=entry_price,
            quantity=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            confidence=confidence
        )
        
        self.positions.append(position)
        self.balance -= required_margin
        self.log_trade(position.to_dict())
        
        print(f"🚀 {direction} Position Opened:")
        print(f"   📍 Entry: ${entry_price:.2f}")
        print(f"   📦 Size: {position_size:.4f}")
        print(f"   🛡️  SL: ${stop_loss:.2f}")
        print(f"   🎯 TP: ${take_profit:.2f}")
        print(f"   🎲 Confidence: {confidence:.1%}")
        
        # Send Discord notification
        self.send_discord_alert(
            f"🚀 New {direction} Position!\n"
            f"Symbol: {self.config['SYMBOL']}\n"
            f"Entry: ${entry_price:.2f}\n"
            f"SL: ${stop_loss:.2f} | TP: ${take_profit:.2f}\n"
            f"Confidence: {confidence:.1%}\n"
            f"Balance: ${self.balance:.2f}"
        )
        
        return position
    
    def check_positions(self, current_price: float) -> None:
        """Check and manage positions"""
        for position in self.positions[:]:
            if not position.is_active:
                continue
                
            should_close = False
            exit_reason = ""
            
            if position.direction == 'BUY':
                if current_price <= position.stop_loss:
                    should_close, exit_reason = True, "Stop Loss"
                elif current_price >= position.take_profit:
                    should_close, exit_reason = True, "Take Profit"
            else:
                if current_price >= position.stop_loss:
                    should_close, exit_reason = True, "Stop Loss"
                elif current_price <= position.take_profit:
                    should_close, exit_reason = True, "Take Profit"
            
            if should_close:
                pnl = position.close_position(current_price, exit_reason)
                self.balance += (position.quantity * position.entry_price) + pnl
                self.update_trade_log(position)
                self.positions.remove(position)
                
                print(f"📝 {position.direction} Position Closed:")
                print(f"   🚪 Exit: ${current_price:.2f} ({exit_reason})")
                print(f"   💰 P&L: ${pnl:.2f}")
                print(f"   💳 Balance: ${self.balance:.2f}")
                
                # Send Discord notification
                pnl_emoji = "💚" if pnl > 0 else "❤️"
                self.send_discord_alert(
                    f"📝 Position Closed: {position.direction}\n"
                    f"Exit: ${current_price:.2f} ({exit_reason})\n"
                    f"P&L: {pnl_emoji} ${pnl:.2f}\n"
                    f"Balance: ${self.balance:.2f}"
                )
    
    def log_trade(self, trade_data: Dict) -> None:
        """Log trade"""
        self.trade_log.append(trade_data)
        self._save_logs()
    
    def update_trade_log(self, position: SimplePosition) -> None:
        """Update trade log"""
        for i, trade in enumerate(self.trade_log):
            if (trade.get('entry_timestamp') == position.timestamp and 
                trade.get('direction') == position.direction):
                self.trade_log[i] = position.to_dict()
                break
        self._save_logs()
    
    def _save_logs(self) -> None:
        """Save logs"""
        try:
            os.makedirs(os.path.dirname(self.config['LOG_FILE']), exist_ok=True)
            with open(self.config['LOG_FILE'], 'w') as f:
                json.dump(self.trade_log, f, indent=2, default=str)
        except Exception as e:
            print(f"⚠️  Error saving logs: {e}")
    
    def send_discord_alert(self, message: str) -> None:
        """Send Discord alert"""
        if not self.config.get('DISCORD_WEBHOOK_URL') or 'your_discord' in self.config['DISCORD_WEBHOOK_URL']:
            return
            
        try:
            payload = {'content': message}
            response = requests.post(
                self.config['DISCORD_WEBHOOK_URL'],
                json=payload,
                timeout=10
            )
            if response.status_code == 200:
                print("✅ Discord alert sent")
            else:
                print("⚠️  Discord alert failed")
        except Exception as e:
            print(f"⚠️  Discord error: {e}")
    
    def run_trading_cycle(self) -> None:
        """Main trading cycle"""
        # Fetch current price
        current_price = self.data_manager.fetch_current_price()
        
        if current_price is None:
            print("❌ Failed to fetch price data")
            return
        
        print(f"📊 Current XAU/USD Price: ${current_price:.2f}")
        
        # Check existing positions
        self.check_positions(current_price)
        
        # Get trading signal
        signal, confidence = self.data_manager.get_simple_signal()
        
        if signal and confidence > self.config['CONFIDENCE_THRESHOLD'] and len(self.positions) < 2:
            print(f"🎯 Signal: {signal} | Confidence: {confidence:.1%}")
            self.execute_trade(signal, current_price, confidence)
        else:
            if signal:
                print(f"⚠️  Signal {signal} (confidence: {confidence:.1%}) - Below threshold")
            else:
                print("📊 No signal generated")
    
    def start_trading(self) -> None:
        """Start trading bot"""
        self.running = True
        print("🚀 Simple Trading Bot Started!")
        print(f"💰 Initial Balance: ${self.balance:.2f}")
        print(f"⚙️  Confidence Threshold: {self.config['CONFIDENCE_THRESHOLD']*100:.1f}%")
        print("=" * 50)
        
        # Send startup notification
        self.send_discord_alert(
            f"🚀 XAU/USD Trading Bot Started!\n"
            f"💰 Balance: ${self.balance:.2f}\n"
            f"⚙️ Threshold: {self.config['CONFIDENCE_THRESHOLD']*100:.1f}%"
        )
        
        cycle_count = 0
        while self.running:
            try:
                cycle_count += 1
                print(f"\n🔄 Trading Cycle #{cycle_count}")
                self.run_trading_cycle()
                
                # Display current status
                active_positions = len([p for p in self.positions if p.is_active])
                print(f"📈 Active Positions: {active_positions}")
                print(f"💰 Current Balance: ${self.balance:.2f}")
                
                time.sleep(30)  # Check every 30 seconds for demo
                
            except KeyboardInterrupt:
                print("\n🛑 Bot stopped by user")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                time.sleep(10)
        
        # Close all positions
        if self.positions:
            print("🔄 Closing all positions...")
            current_price = self.data_manager.fetch_current_price()
            if current_price:
                for position in self.positions:
                    if position.is_active:
                        pnl = position.close_position(current_price, "Bot Shutdown")
                        self.balance += (position.quantity * position.entry_price) + pnl
                        self.update_trade_log(position)
                        print(f"📝 Position closed: P&L ${pnl:.2f}")
        
        print(f"🏁 Final Balance: ${self.balance:.2f}")
        roi = ((self.balance - self.config['INITIAL_BALANCE']) / self.config['INITIAL_BALANCE']) * 100
        print(f"📊 ROI: {roi:.2f}%")
    
    def stop_trading(self) -> None:
        """Stop the bot"""
        self.running = False
    
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

class SimpleGUI:
    """Simple GUI for the trading bot"""
    
    def __init__(self, bot: SimpleTradingBot):
        if not GUI_AVAILABLE:
            print("❌ GUI not available")
            return
            
        self.bot = bot
        self.root = tk.Tk()
        self.root.title("🚀 XAU/USD Trading Bot")
        self.root.geometry("800x600")
        
        self.setup_gui()
        self.update_display()
    
    def setup_gui(self):
        """Setup GUI elements"""
        # Title
        title = ttk.Label(self.root, text="🚀 XAU/USD Trading Bot", 
                         font=("Arial", 16, "bold"))
        title.pack(pady=10)
        
        # Status frame
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.status_label = ttk.Label(status_frame, text="Status: ⏹️ Stopped")
        self.status_label.pack(side=tk.LEFT)
        
        self.balance_label = ttk.Label(status_frame, 
                                     text=f"Balance: 💰 ${self.bot.balance:.2f}")
        self.balance_label.pack(side=tk.RIGHT)
        
        # Buttons
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=10)
        
        self.start_button = ttk.Button(button_frame, text="▶️ Start", 
                                     command=self.start_bot)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="⏹️ Stop", 
                                    command=self.stop_bot, state="disabled")
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Info display
        self.info_text = tk.Text(self.root, height=25, width=80)
        self.info_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        scrollbar = ttk.Scrollbar(self.root, command=self.info_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.info_text.config(yscrollcommand=scrollbar.set)
    
    def start_bot(self):
        """Start the bot"""
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.status_label.config(text="Status: ▶️ Running")
        
        self.bot_thread = threading.Thread(target=self.bot.start_trading, daemon=True)
        self.bot_thread.start()
    
    def stop_bot(self):
        """Stop the bot"""
        self.bot.stop_trading()
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.status_label.config(text="Status: ⏹️ Stopped")
    
    def update_display(self):
        """Update display"""
        self.balance_label.config(text=f"Balance: 💰 ${self.bot.balance:.2f}")
        
        # Update info
        self.info_text.delete(1.0, tk.END)
        
        stats = self.bot.get_statistics()
        info = f"📊 TRADING STATISTICS\n{'='*50}\n\n"
        
        if stats:
            info += f"Total Trades: {stats['total_trades']}\n"
            info += f"Winning Trades: {stats['winning_trades']}\n"
            info += f"Win Rate: {stats['win_rate']:.1f}%\n"
            info += f"Total P&L: ${stats['total_pnl']:.2f}\n"
            info += f"ROI: {stats['roi']:.2f}%\n"
            info += f"Active Positions: {stats['active_positions']}\n\n"
        
        info += f"🎯 ACTIVE POSITIONS\n{'='*50}\n"
        active_positions = [p for p in self.bot.positions if p.is_active]
        
        if active_positions:
            for i, pos in enumerate(active_positions, 1):
                info += f"\nPosition {i}: {pos.direction} {pos.symbol}\n"
                info += f"  Entry: ${pos.entry_price:.2f}\n"
                info += f"  SL: ${pos.stop_loss:.2f} | TP: ${pos.take_profit:.2f}\n"
                info += f"  Confidence: {pos.confidence:.1%}\n"
        else:
            info += "\nNo active positions\n"
        
        info += f"\n📋 RECENT TRADES\n{'='*50}\n"
        recent_trades = self.bot.trade_log[-5:] if self.bot.trade_log else []
        
        for trade in reversed(recent_trades):
            status = "ACTIVE" if trade.get('is_active', True) else "CLOSED"
            pnl = trade.get('profit_loss', 0)
            info += f"\n[{status}] {trade.get('direction')} @ ${trade.get('entry_price', 0):.2f}\n"
            if not trade.get('is_active', True):
                info += f"  P&L: ${pnl:.2f}\n"
        
        self.info_text.insert(tk.END, info)
        
        # Schedule update
        self.root.after(2000, self.update_display)
    
    def run(self):
        """Run GUI"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.bot.stop_trading()

def main():
    """Main function"""
    print("=" * 60)
    print("🚀 XAU/USD Trading Bot - Lightweight Demo Version")
    print("=" * 60)
    print("Features:")
    print("✅ Real-time price fetching")
    print("✅ Simple momentum strategy")
    print("✅ Position management")
    print("✅ Discord notifications")
    print("✅ GUI dashboard")
    print("=" * 60)
    
    # Create bot
    try:
        bot = SimpleTradingBot()
    except Exception as e:
        print(f"❌ Error creating bot: {e}")
        return
    
    # Show stats
    stats = bot.get_statistics()
    if stats:
        print(f"📊 Current Stats:")
        print(f"   Balance: ${stats['current_balance']:.2f}")
        print(f"   Total Trades: {stats['total_trades']}")
        print(f"   Win Rate: {stats['win_rate']:.1f}%")
    
    # Choose mode
    if GUI_AVAILABLE:
        print("\n🎮 Choose mode:")
        print("1. 💻 Console Mode")
        print("2. 🖥️ GUI Mode")
        
        choice = input("\nEnter choice (1 or 2): ").strip()
        
        if choice == "2":
            gui = SimpleGUI(bot)
            gui.run()
            return
    
    # Console mode
    print("\n💻 Starting Console Mode...")
    print("Press Ctrl+C to stop\n")
    
    try:
        bot.start_trading()
    except KeyboardInterrupt:
        print("👋 Goodbye!")

if __name__ == "__main__":
    main()