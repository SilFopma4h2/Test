#!/usr/bin/env python3
"""
XAU/USD Trading Bot Launcher
============================

Easy launcher for the trading bot with automatic setup
"""

import os
import sys

def main():
    print("🚀 XAU/USD Trading Bot Launcher")
    print("=" * 40)
    
    print("Available versions:")
    print("1. 🖥️  Simple Bot (lightweight, works immediately)")
    print("2. 🧠 Advanced Bot (ML-powered, requires dependencies)")
    print("3. ⚙️  Setup & Configuration")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == "1":
                print("🚀 Starting Simple Bot...")
                os.system("python simple_bot.py")
                break
            elif choice == "2":
                print("🧠 Starting Advanced Bot...")
                try:
                    os.system("python main.py")
                except Exception as e:
                    print(f"❌ Error: {e}")
                    print("💡 Try running: pip install -r requirements.txt")
                break
            elif choice == "3":
                print("⚙️ Running setup...")
                os.system("python setup.py")
                break
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            sys.exit(0)

if __name__ == "__main__":
    main()