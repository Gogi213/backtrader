#!/usr/bin/env python3
"""
Dependency installer for Bollinger Bands Backtester
Automatically installs PyQt6 and other required dependencies
"""
import subprocess
import sys
import os


def run_command(command):
    """Run command and return success status"""
    try:
        print(f"Running: {' '.join(command)}")
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("✅ Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False


def install_requirements():
    """Install all dependencies from requirements.txt"""
    print("="*60)
    print("🚀 Installing Bollinger Bands Backtester Dependencies")
    print("="*60)

    # Check if requirements.txt exists
    if not os.path.exists('requirements.txt'):
        print("❌ requirements.txt not found!")
        return False

    print("📋 Installing from requirements.txt...")
    success = run_command([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])

    if success:
        print("\n🎉 All dependencies installed successfully!")
        print("\n📝 Installed packages:")
        print("  • PyQt6 >= 6.4.0 (Modern GUI framework)")
        print("  • numpy >= 1.21.0 (Numerical computing)")
        print("  • pandas >= 1.3.0 (Data analysis)")
        print("  • matplotlib >= 3.4.0 (Plotting)")
        print("  • plotly >= 5.0.0 (Interactive charts)")

        print("\n🔧 Testing installation...")
        return test_imports()
    else:
        print("\n❌ Installation failed!")
        print("Try manual installation:")
        print("pip install PyQt6 numpy pandas matplotlib plotly")
        return False


def test_imports():
    """Test if critical modules can be imported"""
    critical_modules = [
        ('PyQt6.QtWidgets', 'PyQt6 GUI framework'),
        ('numpy', 'NumPy numerical library'),
        ('pandas', 'Pandas data analysis'),
        ('matplotlib.pyplot', 'Matplotlib plotting'),
        ('src.data.vectorized_klines_handler', 'Custom data handler')
    ]

    all_good = True

    print("\n🔍 Testing imports...")
    for module_name, description in critical_modules:
        try:
            __import__(module_name)
            print(f"✅ {module_name} - OK")
        except ImportError as e:
            print(f"❌ {module_name} - FAILED: {e}")
            all_good = False

    if all_good:
        print("\n✅ All imports successful! Ready to run backtester.")
        print("\n🚀 You can now run:")
        print("   python main.py           # Start GUI")
        print("   python run_app.py        # Alternative entry point")
    else:
        print("\n❌ Some imports failed. Check error messages above.")

    return all_good


def main():
    """Main installation function"""
    print("Bollinger Bands Backtester - Dependency Installer")

    # Check Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7+ required")
        return 1

    print(f"✅ Python {sys.version.split()[0]} detected")

    # Install dependencies
    success = install_requirements()

    if success:
        print("\n" + "="*60)
        print("🎉 INSTALLATION COMPLETE!")
        print("="*60)
        print("You can now start the backtester with:")
        print("  python main.py")
        return 0
    else:
        print("\n" + "="*60)
        print("❌ INSTALLATION FAILED")
        print("="*60)
        print("Please check error messages above and try manual installation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())