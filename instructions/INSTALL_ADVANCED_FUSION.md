# Advanced Fusion Dependencies Installation Guide

## Overview

This guide explains how to install the optional dependencies required for advanced fusion strategies in the Multi-Omics Data Fusion Pipeline.

## Required Dependencies

### 1. SNFpy (Similarity Network Fusion)

**Installation:**
```bash
pip install snfpy
```

**Important Note:** The package is installed as `snfpy` but must be imported as `snf`:
```python
import snf  # Correct import
# NOT: import snfpy  # This will fail
```

**Verification:**
```bash
python -c "import snf; print('SNF available:', hasattr(snf, 'snf'))"
```

### 2. Mklaren (Multiple-Kernel Learning)

**Installation:**
```bash
pip install mklaren
```

**Verification:**
```bash
python -c "from mklaren.kernel.kernel import exponential_kernel; print('Mklaren available')"
```

## Installation Issues and Solutions

### Issue 1: "SNFpy library not available"

**Cause:** Incorrect import statement in code
**Solution:**  **FIXED** - The code has been updated to import `snf` instead of `snfpy`

### Issue 2: Oct2Py Import Interference

**Cause:** Oct2Py's lazy import checks can interfere with other packages
**Solution:**  **FIXED** - Import SNF before any oct2py-related modules (implemented in the code)

### Issue 3: "Mklaren not available" despite installation

**Cause:** Mklaren has optional dependencies that may not be available
**Solution:**  **FIXED** - The code now uses a fallback implementation when full Mklaren functionality is not available

**What was fixed:**
- Mklaren tries to import an `align` module that may not be compatible
- The code now gracefully handles missing optional dependencies
- Falls back to sklearn-based kernel implementations when needed
- Provides basic MKL functionality even without full Mklaren support

## Complete Installation Command

To install both advanced fusion dependencies:

```bash
# Install both packages
pip install snfpy mklaren

# Verify installation
python -c "from fusion import SNF_AVAILABLE, MKL_AVAILABLE; print(f'SNF Available: {SNF_AVAILABLE}'); print(f'MKL Available: {MKL_AVAILABLE}')"
```

**Expected Output:**
```
SNF Available: True
MKL Available: True
```

**Note:** You may see warnings about "octave not found" - this is normal and doesn't affect functionality.

## Testing the Installation

Run the fusion test script to verify everything works:

```bash
python tests/test_fusion_enhancements.py
```

## Troubleshooting

### If you still get import errors:

1. **Check your Python environment:**
   ```bash
   python -c "import sys; print(sys.path)"
   pip list | findstr /i "snf mkl"
   ```

2. **Try importing in the correct order:**
   ```python
   import snf  # Import SNF first
   from oct2py import Oct2Py  # Then oct2py
   ```

3. **Reinstall in a clean environment:**
   ```bash
   pip uninstall snfpy mklaren
   pip install snfpy mklaren
   ```

## Alternative: Using Without Advanced Fusion

If you cannot install these dependencies, the pipeline will automatically fall back to basic fusion strategies:
- `weighted_concat` (for 0% missing data)
- `early_fusion_pca` (for data with missing values)

The pipeline will log warnings but continue to work with reduced functionality. 