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
        print_status(f"Python {version.major}.{version.minor}.{version.micro} detected - Compatible ✓", "SUCCESS")
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
    print("1. 🔧 Basic Installation (Core dependencies only)")
    print("   - Essential packages for running the pipeline")
    print("   - scikit-learn, pandas, numpy, xgboost, lightgbm")
    print("   - Recommended for production use")
    print()
    print("2. 📊 Visualization Installation (Core + enhanced plotting)")
    print("   - Includes scikit-posthocs for MAD analysis diagrams")
    print("   - Enhanced plotting capabilities for analysis")
    print("   - Recommended for research and analysis")
    print()
    print("3. 🛠️  Development Installation (Core + dev tools)")
    print("   - Includes testing, formatting, and linting tools")
    print("   - pytest, black, flake8, mypy")
    print("   - Recommended for contributors and developers")
    print()
    print("4. 🚀 Advanced Installation (Core + optional fusion libraries)")
    print("   - Includes experimental fusion methods")
    print("   - SNF (Similarity Network Fusion), MKL (Multiple-Kernel Learning)")
    print("   - oct2py for advanced computational methods")
    print("   - Recommended for advanced research")
    print()
    print("5. 🌟 Full Installation (All dependencies)")
    print("   - Everything included: core + visualization + dev + advanced")
    print("   - Recommended for comprehensive usage and development")
    print()
    
    while True:
        try:
            choice = input(f"{Colors.YELLOW}Choose installation type (1-5): {Colors.END}").strip()
            if choice in ['1', '2', '3', '4', '5']:
                return int(choice)
            else:
                print_status("Please enter a number between 1 and 5.", "WARNING")
        except KeyboardInterrupt:
            print("\n")
            print_status("Installation cancelled by user.", "INFO")
            sys.exit(0)

def run_pip_install(packages, description):
    """Run pip install with error handling."""
    if isinstance(packages, str):
        packages = [packages]
        
    print_status(f"Installing {description}...", "INFO")
    
    try:
        cmd = [sys.executable, "-m", "pip", "install", "--upgrade"] + packages
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

def install_core_dependencies():
    """Install core dependencies required for basic functionality."""
    print(f"\n{Colors.BOLD}Installing Core Dependencies:{Colors.END}")
    
    core_packages = [
        # Numerical computing and data manipulation
        "numpy>=1.21.0",
        "pandas>=1.3.0", 
        "scipy>=1.7.0",
        
        # Machine learning libraries
        "scikit-learn>=1.0.0",
        "xgboost>=1.6.0",
        "lightgbm>=3.3.0",
        
        # Visualization
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        
        # Performance and monitoring
        "joblib>=1.1.0",
        "threadpoolctl>=3.0.0",
        "psutil>=5.8.0",
        
        # Feature selection
        "boruta>=0.3.0",
        
        # Hyperparameter optimization
        "scikit-optimize>=0.9.0",
    ]
    
    success = True
    for package in core_packages:
        if not run_pip_install(package, package.split(">=")[0]):
            success = False
    
    # Try to install optional but recommended packages
    optional_packages = [
        ("imbalanced-learn>=0.8.0", "imbalanced-learn (for handling class imbalance)"),
    ]
    
    for package, desc in optional_packages:
        if not run_pip_install(package, desc):
            print_status(f"Optional package {desc} failed to install - continuing", "WARNING")
    
    return success

def install_visualization_dependencies():
    """Install visualization dependencies."""
    print_status("Installing visualization dependencies...", "INFO")
    
    viz_packages = [
        "scikit-posthocs>=0.6.0",  # Critical difference diagrams for MAD analysis
    ]
    
    success = True
    for package in viz_packages:
        if not run_pip_install(package, package.split(">=")[0]):
            success = False
    
    return success

def install_development_dependencies():
    """Install development dependencies."""
    print_status("Installing development dependencies...", "INFO")
    
    dev_packages = [
        "pytest>=6.0.0",
        "pytest-cov>=2.12.0",
        "black>=21.0.0",
        "flake8>=3.9.0",
        "mypy>=0.910",
    ]
    
    success = True
    for package in dev_packages:
        if not run_pip_install(package, package.split(">=")[0]):
            success = False
    
    return success

def install_advanced_dependencies():
    """Install advanced fusion dependencies."""
    print_status("Installing advanced fusion dependencies...", "INFO")
    
    advanced_packages = [
        ("snfpy>=0.2.2", "SNF (Similarity Network Fusion)"),
        ("mklaren>=1.2", "MKL (Multiple-Kernel Learning)"),
        ("oct2py>=5.0.0", "oct2py (Octave bridge for advanced computations)"),
    ]
    
    success_count = 0
    for package, desc in advanced_packages:
        if run_pip_install(package, desc):
            success_count += 1
        else:
            print_status(f"Optional advanced package {desc} failed to install", "WARNING")
            print_status("This may limit some advanced fusion methods", "INFO")
    
    if success_count > 0:
        print_status(f"Successfully installed {success_count}/{len(advanced_packages)} advanced packages", "SUCCESS")
    
    # Special handling for oct2py - requires Octave to be installed
    if success_count < len(advanced_packages):
        print_status("Note: Some advanced fusion methods require additional system dependencies:", "INFO")
        print("  - oct2py requires GNU Octave to be installed separately")
        print("  - SNF and MKL are optional for advanced research use cases")
    
    return True  # Don't fail installation if advanced packages fail

def check_octave_installation():
    """Check if Octave is installed for oct2py."""
    try:
        result = subprocess.run(["octave", "--version"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print_status("GNU Octave found - oct2py will work properly", "SUCCESS")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    print_status("GNU Octave not found in PATH", "WARNING")
    print_status("Some advanced fusion methods may not work without Octave", "INFO")
    print_status("You can install Octave from: https://octave.org/download", "INFO")
    return False

def install_dependencies(choice):
    """Install dependencies based on user choice."""
    success = True
    
    # Always install core dependencies
    if not install_core_dependencies():
        success = False
    
    # Install additional dependencies based on choice
    if choice >= 2:  # Visualization
        if not install_visualization_dependencies():
            success = False
    
    if choice == 3 or choice == 5:  # Development or Full
        if not install_development_dependencies():
            success = False
    
    if choice == 4 or choice == 5:  # Advanced or Full
        if not install_advanced_dependencies():
            success = False
        check_octave_installation()
    
    return success

def run_verification():
    """Run installation verification."""
    print(f"\n{Colors.BOLD}Running Installation Verification:{Colors.END}")
    
    # Test core imports
    test_imports = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("sklearn", "scikit-learn"),
        ("xgboost", "XGBoost"),
        ("lightgbm", "LightGBM"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("joblib", "joblib"),
        ("psutil", "psutil"),
        ("boruta", "Boruta"),
    ]
    
    failed_imports = []
    for module, name in test_imports:
        try:
            __import__(module)
            print_status(f"{name} imported successfully", "SUCCESS")
        except ImportError as e:
            print_status(f"Failed to import {name}: {e}", "ERROR")
            failed_imports.append(name)
    
    # Test optional imports
    optional_imports = [
        ("skposthocs", "scikit-posthocs"),
        ("snf", "SNF"),
        ("mklaren", "MKL"),
        ("oct2py", "oct2py"),
        ("imblearn", "imbalanced-learn"),
    ]
    
    for module, name in optional_imports:
        try:
            __import__(module)
            print_status(f"{name} (optional) imported successfully", "SUCCESS")
        except ImportError:
            print_status(f"{name} (optional) not available", "INFO")
    
    # Test if main pipeline components can be imported
    try:
        # Try to import key pipeline modules
        current_dir = Path(__file__).parent
        sys.path.insert(0, str(current_dir))
        
        import config
        import models
        import fusion
        import preprocessing
        print_status("Pipeline modules imported successfully", "SUCCESS")
        
    except ImportError as e:
        print_status(f"Warning: Pipeline module import failed: {e}", "WARNING")
        print_status("This may indicate missing dependencies or configuration issues", "INFO")
    
    if failed_imports:
        print_status(f"Installation verification completed with {len(failed_imports)} failures", "WARNING")
        return False
    else:
        print_status("Installation verification completed successfully!", "SUCCESS")
        return True

def setup_environment():
    """Setup environment variables and configurations."""
    print(f"\n{Colors.BOLD}Setting up Environment:{Colors.END}")
    
    try:
        # Set environment variable to enable resource logging
        os.environ["DEBUG_RESOURCES"] = "1"
        print_status("Resource debugging enabled", "SUCCESS")
        
        # Check for Octave and set path if found
        octave_paths = [
            r"C:\Program Files\GNU Octave\Octave-*\mingw64\bin\octave-cli.exe",
            r"C:\Users\*\AppData\Local\Programs\GNU Octave\Octave-*\mingw64\bin\octave-cli.exe",
            "/usr/bin/octave",
            "/usr/local/bin/octave",
        ]
        
        octave_found = False
        for pattern in octave_paths:
            if "*" in pattern:
                # Handle wildcard patterns
                from glob import glob
                matches = glob(pattern)
                if matches:
                    octave_path = matches[0]
                    if os.path.exists(octave_path):
                        os.environ["OCTAVE_EXECUTABLE"] = octave_path
                        print_status(f"Octave executable configured: {octave_path}", "SUCCESS")
                        octave_found = True
                        break
            else:
                if os.path.exists(pattern):
                    os.environ["OCTAVE_EXECUTABLE"] = pattern
                    print_status(f"Octave executable configured: {pattern}", "SUCCESS")
                    octave_found = True
                    break
        
        if not octave_found:
            print_status("Octave executable not found in common locations", "WARNING")
            print_status("Advanced fusion methods may require manual Octave installation", "INFO")
        
        return True
    except Exception as e:
        print_status(f"Environment setup failed: {e}", "ERROR")
        return False

def show_next_steps():
    """Show next steps after installation."""
    print(f"\n{Colors.BOLD}🎉 Installation Completed Successfully!{Colors.END}")
    print()
    print(f"{Colors.BOLD}Next Steps:{Colors.END}")
    print()
    print("1. 📁 Prepare your data:")
    print("   - Ensure data is in CSV format with proper structure")
    print("   - See README.md for data format requirements")
    print()
    print("2. 🚀 Run the pipeline:")
    print(f"   {Colors.GREEN}python main.py{Colors.END}")
    print()
    print("3. 📊 For MAD analysis only:")
    print(f"   {Colors.GREEN}python main.py --mad-only{Colors.END}")
    print()
    print("4. ❓ For help and all available options:")
    print(f"   {Colors.GREEN}python main.py --help{Colors.END}")
    print()
    print("5. 🧪 Test the installation:")
    print(f"   {Colors.GREEN}python test_data/test_pipeline.py{Colors.END}")
    print()
    print("📖 For detailed documentation and examples, see README.md")
    print()
    print(f"{Colors.BOLD}Troubleshooting:{Colors.END}")
    print("- If you encounter issues, check that all dependencies are properly installed")
    print("- For advanced fusion methods, ensure Octave is installed separately")
    print("- Check the logs in the pipeline output for detailed error information")
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
    
    # Setup environment first
    setup_environment()
    
    # Install dependencies
    print(f"\n{Colors.BOLD}Starting Installation Process...{Colors.END}")
    if not install_dependencies(choice):
        print_status("Installation completed with some errors. Please check the messages above.", "WARNING")
        print_status("The pipeline may still work, but some features might be limited.", "INFO")
    else:
        print_status("All dependencies installed successfully!", "SUCCESS")
    
    # Ask about verification
    print()
    run_tests = input(f"{Colors.YELLOW}Run installation verification tests? (y/n) [y]: {Colors.END}").strip().lower()
    
    if run_tests in ['y', 'yes', '']:
        if not run_verification():
            print_status("Some verification tests failed. Check the output above.", "WARNING")
            print_status("The installation may still work, but some features might be limited.", "INFO")
        else:
            print_status("Installation verification passed!", "SUCCESS")
    
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