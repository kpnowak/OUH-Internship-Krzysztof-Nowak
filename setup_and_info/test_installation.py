#!/usr/bin/env python3
"""
Installation Verification Script for Multi-Omics Data Fusion Pipeline

This script verifies that all dependencies are correctly installed and
the pipeline is ready to use. It performs comprehensive checks including:
- Python version compatibility
- Core dependency verification
- Module import tests
- Basic functionality tests
- Command-line interface tests
- Optional dependency checks

Usage:
    python test_installation.py
"""

import sys
import subprocess
import importlib
import warnings
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

def check_python_version():
    """Check if Python version is compatible."""
    print_status("Checking Python version compatibility...", "INFO")
    
    version = sys.version_info
    if version >= (3, 8):
        print_status(f"Python {version.major}.{version.minor}.{version.micro} - Compatible", "SUCCESS")
        return True
    else:
        print_status(f"Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+", "ERROR")
        return False

def check_core_dependencies():
    """Check if all core dependencies are installed."""
    print_status("Checking core dependencies...", "INFO")
    
    core_deps = {
        'numpy': '1.21.0',
        'pandas': '1.3.0',
        'scipy': '1.7.0',
        'sklearn': '1.0.0',
        'matplotlib': '3.5.0',
        'seaborn': '0.11.0',
        'joblib': '1.1.0',
        'threadpoolctl': '3.0.0',
        'psutil': '5.8.0',
        'boruta': '0.3.0'
    }
    
    all_installed = True
    
    for package, min_version in core_deps.items():
        try:
            module = importlib.import_module(package)
            if hasattr(module, '__version__'):
                version = module.__version__
                print_status(f"{package} {version} - Installed", "SUCCESS")
            else:
                print_status(f"{package} - Installed (version unknown)", "SUCCESS")
        except ImportError:
            print_status(f"{package} - Missing (required >= {min_version})", "ERROR")
            all_installed = False
    
    return all_installed

def check_optional_dependencies():
    """Check optional dependencies and warn if missing."""
    print_status("Checking optional dependencies...", "INFO")
    
    optional_deps = {
        'scikit_posthocs': 'Enhanced visualization (MAD analysis critical difference diagrams)',
        'pytest': 'Testing framework',
        'black': 'Code formatting',
        'flake8': 'Code linting',
        'mypy': 'Type checking'
    }
    
    for package, description in optional_deps.items():
        try:
            importlib.import_module(package)
            print_status(f"{package} - Available ({description})", "SUCCESS")
        except ImportError:
            print_status(f"{package} - Not installed ({description})", "WARNING")

def test_module_imports():
    """Test importing main pipeline modules."""
    print_status("Testing module imports...", "INFO")
    
    # Get the parent directory (project root)
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    modules_to_test = [
        'config',
        'data_io',
        'preprocessing',
        'models',
        'fusion',
        'cv',
        'plots',
        'utils',
        'mad_analysis'
    ]
    
    all_imported = True
    
    for module_name in modules_to_test:
        try:
            importlib.import_module(module_name)
            print_status(f"{module_name} - Import successful", "SUCCESS")
        except ImportError as e:
            print_status(f"{module_name} - Import failed: {str(e)}", "ERROR")
            all_imported = False
        except Exception as e:
            print_status(f"{module_name} - Import error: {str(e)}", "WARNING")
    
    return all_imported

def test_basic_functionality():
    """Test basic pipeline functionality."""
    print_status("Testing basic functionality...", "INFO")
    
    try:
        # Test numpy operations
        import numpy as np
        test_array = np.random.rand(10, 5)
        assert test_array.shape == (10, 5)
        print_status("NumPy operations - Working", "SUCCESS")
        
        # Test pandas operations
        import pandas as pd
        test_df = pd.DataFrame(test_array)
        assert len(test_df) == 10
        print_status("Pandas operations - Working", "SUCCESS")
        
        # Test scikit-learn
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        transformed = pca.fit_transform(test_array)
        assert transformed.shape == (10, 2)
        print_status("Scikit-learn operations - Working", "SUCCESS")
        
        # Test matplotlib
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        plt.close(fig)
        print_status("Matplotlib operations - Working", "SUCCESS")
        
        return True
        
    except Exception as e:
        print_status(f"Basic functionality test failed: {str(e)}", "ERROR")
        return False

def test_cli_availability():
    """Test if command-line interface is available."""
    print_status("Testing command-line interface...", "INFO")
    
    project_root = Path(__file__).parent.parent
    main_script = project_root / "main.py"
    
    if main_script.exists():
        print_status("main.py - Available", "SUCCESS")
        
        # Test if the script can be executed (just check syntax)
        try:
            result = subprocess.run([
                sys.executable, "-m", "py_compile", str(main_script)
            ], capture_output=True, text=True, cwd=project_root)
            
            if result.returncode == 0:
                print_status("main.py syntax - Valid", "SUCCESS")
                return True
            else:
                print_status(f"main.py syntax error: {result.stderr}", "ERROR")
                return False
        except Exception as e:
            print_status(f"CLI test failed: {str(e)}", "ERROR")
            return False
    else:
        print_status("main.py - Not found", "ERROR")
        return False

def check_data_directory():
    """Check if data directory structure exists."""
    print_status("Checking data directory structure...", "INFO")
    
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    
    if data_dir.exists():
        print_status("data/ directory - Found", "SUCCESS")
        
        # Check for expected subdirectories
        expected_dirs = ['aml', 'breast', 'clinical', 'colon', 'kidney', 
                        'liver', 'lung', 'melanoma', 'ovarian', 'sarcoma']
        
        found_dirs = [d.name for d in data_dir.iterdir() if d.is_dir()]
        
        for expected_dir in expected_dirs:
            if expected_dir in found_dirs:
                print_status(f"data/{expected_dir}/ - Found", "SUCCESS")
            else:
                print_status(f"data/{expected_dir}/ - Missing", "WARNING")
        
        return True
    else:
        print_status("data/ directory - Not found", "WARNING")
        print_status("Note: Data directory is required for running the pipeline", "INFO")
        return False

def main():
    """Run all installation verification tests."""
    print_status("Multi-Omics Data Fusion Pipeline - Installation Verification", "HEADER")
    print("=" * 70)
    
    tests = [
        ("Python Version", check_python_version),
        ("Core Dependencies", check_core_dependencies),
        ("Module Imports", test_module_imports),
        ("Basic Functionality", test_basic_functionality),
        ("Command-Line Interface", test_cli_availability),
        ("Data Directory", check_data_directory),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{Colors.BOLD}Testing {test_name}:{Colors.END}")
        results[test_name] = test_func()
    
    # Check optional dependencies separately
    print(f"\n{Colors.BOLD}Optional Dependencies:{Colors.END}")
    check_optional_dependencies()
    
    # Summary
    print("\n" + "=" * 70)
    print_status("Installation Verification Summary", "HEADER")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "SUCCESS" if result else "ERROR"
        print_status(f"{test_name}: {'PASSED' if result else 'FAILED'}", status)
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print_status("Installation verification completed successfully!", "SUCCESS")
        print_status("The pipeline is ready to use.", "INFO")
        print_status("Run 'python main.py --help' to see usage options.", "INFO")
        return 0
    else:
        print_status("Installation verification failed.", "ERROR")
        print_status("Please install missing dependencies and fix any issues.", "INFO")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 