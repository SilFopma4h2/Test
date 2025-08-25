#!/bin/bash
# XAU/USD Trading Bot - Quick Install Script

echo "🚀 XAU/USD Trading Bot - Quick Installation"
echo "============================================="

# Check Python version
python3 --version
if [ $? -ne 0 ]; then
    echo "❌ Python 3 not found. Please install Python 3.8+"
    exit 1
fi

echo "✅ Python is available"

# Create directories
echo "📁 Creating directories..."
mkdir -p data models logs
echo "✅ Directories created"

# Install basic requirements (optional)
echo "📦 Installing basic requirements..."
python3 -m pip install requests --user --quiet
echo "✅ Basic requirements installed"

# Run setup
echo "⚙️ Running setup validation..."
python3 setup.py

echo ""
echo "🎉 Installation completed!"
echo ""
echo "🚀 Quick Start:"
echo "   python3 run.py          # Launch menu"
echo "   python3 simple_bot.py   # Run simple bot"
echo ""
echo "💡 Tips:"
echo "   • Update config.py with your API keys"
echo "   • The simple bot works without ML dependencies"
echo "   • Use run.py for an easy menu interface"
echo ""
echo "📖 Documentation: README.md"