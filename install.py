#!/usr/bin/env python3
"""
Interactive Installation Script for Multi-Omics Data Fusion Pipeline

This script provides a user-friendly installation experience with:
- Interactive installation options
- Automatic dependency management
- Installation verification
- Environment setup guidance

Usage:
    python install.py
"""

import sys
import subprocess
import os
from pathlib import Path

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_status(message, status="INFO"):
    """Print colored status messages."""
    if status == "SUCCESS":
        print(f"{Colors.GREEN}✅ {message}{Colors.END}")
    elif status == "ERROR":
        print(f"{Colors.RED}❌ {message}{Colors.END}")
    elif status == "WARNING":
        print(f"{Colors.YELLOW}⚠️  {message}{Colors.END}")
    elif status == "INFO":
        print(f"{Colors.BLUE}ℹ️  {message}{Colors.END}")
    else:
        print(f"{Colors.BOLD}{message}{Colors.END}")

def print_header():
    """Print welcome header."""
    print(f"{Colors.BOLD}{Colors.BLUE}")
    print("=" * 70)
    print("  Multi-Omics Data Fusion Pipeline - Interactive Installer")
    print("=" * 70)
    print(f"{Colors.END}")
    print()
    print("This script will guide you through the installation process.")
    print("You can choose from different installation options based on your needs.")
    print()

def check_python_version():
    """Check Python version compatibility."""
    version = sys.version_info
    if version >= (3, 8):
        print_status(f"Python {version.major}.{version.minor}.{version.micro} detected - Compatible ", "SUCCESS")
        return True
    else:
        print_status(f"Python {version.major}.{version.minor}.{version.micro} detected - Incompatible", "ERROR")
        print_status("This pipeline requires Python 3.8 or higher.", "ERROR")
        print_status("Please upgrade your Python installation and try again.", "INFO")
        return False

def get_installation_choice():
    """Get user's installation preference."""
    print(f"{Colors.BOLD}Installation Options:{Colors.END}")
    print()
    print("1.  Basic Installation (Core dependencies only)")
    print("   - Essential packages for running the pipeline")
    print("   - Recommended for production use")
    print()
    print("2. 📊 Visualization Installation (Core + enhanced plotting)")
    print("   - Includes scikit-posthocs for MAD analysis diagrams")
    print("   - Recommended for research and analysis")
    print()
    print("3. 🛠️  Development Installation (Core + dev tools)")
    print("   - Includes testing, formatting, and linting tools")
    print("   - Recommended for contributors and developers")
    print()
    print("4. 🚀 Full Installation (All dependencies)")
    print("   - Everything included")
    print("   - Recommended for comprehensive usage")
    print()
    
    while True:
        try:
            choice = input(f"{Colors.YELLOW}Choose installation type (1-4): {Colors.END}").strip()
            if choice in ['1', '2', '3', '4']:
                return int(choice)
            else:
                print_status("Please enter a number between 1 and 4.", "WARNING")
        except KeyboardInterrupt:
            print("\n")
            print_status("Installation cancelled by user.", "INFO")
            sys.exit(0)

def run_pip_install(package_spec, description):
    """Run pip install with error handling."""
    print_status(f"Installing {description}...", "INFO")
    
    try:
        # Use setup.py directly to avoid pyproject.toml path issues
        setup_dir = Path(__file__).parent / "setup_and_info"
        setup_file = setup_dir / "setup.py"
        
        if package_spec == ".":
            # Basic installation
            cmd = [sys.executable, str(setup_file), "install"]
        else:
            # Installation with extras - use pip with setup.py
            cmd = [sys.executable, "-m", "pip", "install", "-e", f"{setup_dir}[{package_spec.split('[')[1].split(']')[0]}]"]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print_status(f"{description} installed successfully!", "SUCCESS")
            return True
        else:
            print_status(f"Failed to install {description}", "ERROR")
            print(f"Error: {result.stderr}")
            return False
            
    except Exception as e:
        print_status(f"Installation error: {str(e)}", "ERROR")
        return False

def install_dependencies(choice):
    """Install dependencies based on user choice."""
    print(f"\n{Colors.BOLD}Installing Dependencies:{Colors.END}")
    
    # First install core dependencies using requirements.txt
    setup_dir = Path(__file__).parent / "setup_and_info"
    requirements_file = setup_dir / "requirements.txt"
    
    print_status("Installing core dependencies...", "INFO")
    try:
        cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print_status("Failed to install core dependencies", "ERROR")
            print(f"Error: {result.stderr}")
            return False
        else:
            print_status("Core dependencies installed successfully!", "SUCCESS")
    except Exception as e:
        print_status(f"Error installing core dependencies: {str(e)}", "ERROR")
        return False
    
    # Install optional dependencies based on choice
    if choice == 2:  # Visualization
        print_status("Installing visualization dependencies...", "INFO")
        try:
            cmd = [sys.executable, "-m", "pip", "install", "scikit-posthocs>=0.6.0"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print_status("Visualization dependencies installed!", "SUCCESS")
            else:
                print_status("Warning: Failed to install visualization dependencies", "WARNING")
        except Exception as e:
            print_status(f"Warning: {str(e)}", "WARNING")
    
    elif choice == 3:  # Development
        print_status("Installing development dependencies...", "INFO")
        try:
            dev_requirements = setup_dir / "requirements-dev.txt"
            cmd = [sys.executable, "-m", "pip", "install", "-r", str(dev_requirements)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print_status("Development dependencies installed!", "SUCCESS")
            else:
                print_status("Warning: Failed to install development dependencies", "WARNING")
        except Exception as e:
            print_status(f"Warning: {str(e)}", "WARNING")
    
    elif choice == 4:  # Full
        print_status("Installing all optional dependencies...", "INFO")
        try:
            dev_requirements = setup_dir / "requirements-dev.txt"
            cmd = [sys.executable, "-m", "pip", "install", "-r", str(dev_requirements)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print_status("All dependencies installed!", "SUCCESS")
            else:
                print_status("Warning: Failed to install some optional dependencies", "WARNING")
        except Exception as e:
            print_status(f"Warning: {str(e)}", "WARNING")
    
    return True

def run_verification():
    """Run installation verification."""
    print(f"\n{Colors.BOLD}Running Installation Verification:{Colors.END}")
    
    try:
        setup_dir = Path(__file__).parent / "setup_and_info"
        test_script = setup_dir / "test_installation.py"
        
        if test_script.exists():
            result = subprocess.run([sys.executable, str(test_script)], 
                                  capture_output=False, text=True)
            return result.returncode == 0
        else:
            print_status("Verification script not found, skipping...", "WARNING")
            return True
            
    except Exception as e:
        print_status(f"Verification failed: {str(e)}", "ERROR")
        return False

def show_next_steps():
    """Show next steps after installation."""
    print(f"\n{Colors.BOLD}Next Steps:{Colors.END}")
    print()
    print("🎉 Installation completed successfully!")
    print()
    print("To get started:")
    print("1. Ensure your data is in the correct format (see README.md)")
    print("2. Run the pipeline:")
    print(f"   {Colors.GREEN}python main.py{Colors.END}")
    print()
    print("For help and options:")
    print(f"   {Colors.GREEN}python main.py --help{Colors.END}")
    print()
    print("For MAD analysis only:")
    print(f"   {Colors.GREEN}python main.py --mad-only{Colors.END}")
    print()
    print("📖 For detailed documentation, see README.md")
    print()

def main():
    """Main installation workflow."""
    print_header()
    
    # Check Python version
    if not check_python_version():
        return 1
    
    print()
    
    # Get installation choice
    choice = get_installation_choice()
    
    # Install dependencies
    if not install_dependencies(choice):
        print_status("Installation failed. Please check the error messages above.", "ERROR")
        return 1
    
    # Ask about verification
    print()
    run_tests = input(f"{Colors.YELLOW}Run installation verification tests? (y/n): {Colors.END}").strip().lower()
    
    if run_tests in ['y', 'yes', '']:
        if not run_verification():
            print_status("Some verification tests failed. Check the output above.", "WARNING")
            print_status("The installation may still work, but some features might be limited.", "INFO")
    
    # Show next steps
    show_next_steps()
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n")
        print_status("Installation cancelled by user.", "INFO")
        sys.exit(0) 