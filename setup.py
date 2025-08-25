#!/usr/bin/env python3
"""
Setup and validation script for XAU/USD Trading Bot
Ensures all dependencies and configurations are ready
"""

import sys
import subprocess
import importlib.util
import os

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
    return True

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        ('requests', 'requests'),
        ('pandas', 'pandas'), 
        ('numpy', 'numpy'),
        ('torch', 'torch'),
        ('tkinter', 'tkinter')
    ]
    
    missing = []
    print("\n🔍 Checking dependencies...")
    
    for package_name, import_name in required_packages:
        try:
            if import_name == 'tkinter':
                import tkinter
            else:
                importlib.import_module(import_name)
            print(f"✅ {package_name}")
        except ImportError:
            print(f"❌ {package_name} - Missing")
            missing.append(package_name)
    
    if missing:
        print(f"\n📦 Installing missing packages: {', '.join(missing)}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
            print("✅ All packages installed successfully!")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install packages. Please run: pip install -r requirements.txt")
            return False
    
    return True

def setup_directories():
    """Create required directories"""
    directories = ['data', 'models', 'logs']
    
    print("\n📁 Setting up directories...")
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ {directory}/")
        except Exception as e:
            print(f"❌ Failed to create {directory}/: {e}")
            return False
    return True

def validate_config():
    """Import and validate configuration"""
    print("\n⚙️ Validating configuration...")
    try:
        from config import CONFIG, validate_config, setup_environment
        setup_environment()
        
        if validate_config():
            print("✅ Configuration valid")
            return True
        else:
            print("⚠️  Configuration has issues but bot can still run")
            return True
            
    except ImportError as e:
        print(f"❌ Failed to import config: {e}")
        return False
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def test_imports():
    """Test importing main bot components"""
    print("\n🧪 Testing bot imports...")
    try:
        from main import TradingBot, TradingGUI
        print("✅ Main bot classes imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing imports: {e}")
        return False

def main():
    """Main setup function"""
    print("=" * 60)
    print("🚀 XAU/USD Trading Bot - Setup & Validation")
    print("=" * 60)
    
    steps = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Directories", setup_directories), 
        ("Configuration", validate_config),
        ("Bot Components", test_imports)
    ]
    
    success_count = 0
    for step_name, step_function in steps:
        print(f"\n🔄 {step_name}...")
        if step_function():
            success_count += 1
        else:
            print(f"❌ {step_name} failed")
    
    print("\n" + "=" * 60)
    print(f"📊 Setup Results: {success_count}/{len(steps)} steps completed")
    
    if success_count == len(steps):
        print("🎉 Setup completed successfully!")
        print("\n🚀 Ready to start trading bot:")
        print("   python main.py")
        print("\n💡 Tips:")
        print("   • Update config.py with your API keys")
        print("   • Start with paper trading mode")
        print("   • Monitor performance regularly")
        
    else:
        print("⚠️  Setup incomplete. Please fix the issues above.")
        print("\n🔧 Common fixes:")
        print("   • Install Python 3.8+")
        print("   • Run: pip install -r requirements.txt")
        print("   • Check config.py settings")
    
    print("=" * 60)

if __name__ == "__main__":
    main()