# Z_alg Pipeline Verbosity Control

The Z_alg pipeline now supports multiple verbosity levels to control the amount of output displayed during execution.

## Verbosity Levels

### 1. Quiet Mode (Default)
**Command**: `python -m Z_alg.main [options]`

- **Level**: WARNING and above only
- **Output**: Minimal output, only errors and important warnings
- **Use case**: Production runs, automated scripts, when you want clean output

### 2. Verbose Mode
**Command**: `python -m Z_alg.main --verbose [options]`

- **Level**: INFO and above
- **Output**: Detailed progress information, dataset loading, model training progress
- **Use case**: When you want to see what's happening but not overwhelmed with details

### 3. Debug Mode
**Command**: `python -m Z_alg.main --debug [options]`

- **Level**: DEBUG and above (everything)
- **Output**: Very detailed output including memory usage, cache operations, feature selection details
- **Use case**: Troubleshooting, development, detailed analysis

## Environment Variables

You can also control verbosity using environment variables:

```bash
# Verbose mode
export Z_ALG_VERBOSE=1
python -m Z_alg.main

# Debug mode  
export Z_ALG_DEBUG=1
python -m Z_alg.main

# Quiet mode (errors only)
export Z_ALG_QUIET=1
python -m Z_alg.main
```

## Examples

```bash
# Run with minimal output (default)
python -m Z_alg.main --dataset AML

# Run with detailed progress information
python -m Z_alg.main --verbose --dataset AML

# Run with full debugging information
python -m Z_alg.main --debug --dataset AML

# Run all regression datasets quietly
python -m Z_alg.main --regression-only

# Run with verbose output for classification only
python -m Z_alg.main --verbose --classification-only
```

## Log Files

Regardless of console verbosity, detailed logs are always written to `debug.log` in the current directory. This file contains:

- All DEBUG level messages
- Timestamps for all operations
- Error tracebacks
- Memory usage patterns
- Performance metrics

## Recommended Usage

- **For regular use**: Default quiet mode
- **For monitoring progress**: `--verbose` flag
- **For troubleshooting**: `--debug` flag
- **For automated scripts**: Default mode with log file monitoring

The enhanced logging system provides comprehensive monitoring while keeping the console output manageable by default. 