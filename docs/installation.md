# Installation Guide

This guide covers all installation methods for permeabledt, from basic usage to full development setup.

## Table of Contents

- [Installation Methods](#installation-methods)
- [Optional Dependencies](#optional-dependencies)
- [Virtual Environment Setup](#virtual-environment-setup)
- [Development Installation](#development-installation)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

## Installation Methods

### 1. Basic Installation (PyPI)

For most users, the simplest installation method:

```bash
pip install permeabledt
```

This installs permeabledt with core dependencies only (pandas, numpy).

### 2. Feature-Specific Installation

Install only the features you need:

```bash
# Calibration support (genetic algorithms)
pip install permeabledt[calib]

# Particle filtering support
pip install permeabledt[pf]

# Weather data integration
pip install permeabledt[weather]

# Plotting capabilities
pip install permeabledt[plots]

# Multiple features
pip install permeabledt[calib,pf,plots]

# All features
pip install permeabledt[all]
```

### 3. Development Installation

For contributors and advanced users:

```bash
# Clone the repository
git clone https://github.com/yourusername/permeabledt.git
cd permeabledt

# Install in editable mode with all dependencies
pip install -e .[all,dev]
```

## Optional Dependencies

permeabledt uses a modular dependency system. Here's what each group includes:

### Core Dependencies (Always Installed)
- **pandas** (≥1.3.0): Data manipulation and analysis
- **numpy** (≥1.20.0): Numerical computing
- **configparser**: Configuration file parsing

### Optional Feature Groups

#### `calib` - Calibration Support
```bash
pip install permeabledt[calib]
```
- **deap** (≥1.3.0): Genetic algorithms
- **scikit-learn** (≥1.0.0): Machine learning metrics

#### `pf` - Particle Filtering
```bash
pip install permeabledt[pf]
```
- **pypfilt** (≥0.6.0): Particle filtering framework
- **scipy** (≥1.7.0): Scientific computing
- **tomlkit** (≥0.11.0): TOML file handling

#### `plots` - Visualization
```bash
pip install permeabledt[plots]
```
- **matplotlib** (≥3.5.0): Plotting and visualization

#### `weather` - Weather Data
```bash
pip install permeabledt[weather]
```
- **herbie-data** (≥2023.4.0): HRRR data access
- **xarray** (≥0.20.0): N-dimensional arrays
- **pytz** (≥2021.3): Timezone handling

#### `dev` - Development Tools
```bash
pip install permeabledt[dev]
```
- **pytest** (≥6.0.0): Testing framework
- **pytest-cov** (≥2.12.0): Coverage testing
- **black** (≥21.0.0): Code formatting
- **flake8** (≥3.9.0): Code linting
- **mypy** (≥0.910): Type checking

## Virtual Environment Setup

### Why Use Virtual Environments?

Virtual environments prevent dependency conflicts and keep your system Python clean. **Highly recommended** for all users.

### Using venv (Built-in)

```bash
# Create virtual environment
python -m venv permeabledt_env

# Activate environment
# Windows:
permeabledt_env\Scripts\activate
# macOS/Linux:
source permeabledt_env/bin/activate

# Install permeabledt
pip install permeabledt[all]

# Deactivate when done
deactivate
```

### Using conda

```bash
# Create conda environment
conda create -n permeabledt python=3.9
conda activate permeabledt

# Install permeabledt
pip install permeabledt[all]

# Deactivate when done
conda deactivate
```

### Using virtualenv

```bash
# Install virtualenv if needed
pip install virtualenv

# Create environment
virtualenv permeabledt_env

# Activate and install
# Windows:
permeabledt_env\Scripts\activate
# macOS/Linux:
source permeabledt_env/bin/activate

pip install permeabledt[all]
```

## Development Installation

For developers who want to contribute or modify permeabledt:

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/permeabledt.git
cd permeabledt
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv dev_env
# Windows:
dev_env\Scripts\activate
# macOS/Linux:
source dev_env/bin/activate

# Install in editable mode with all dependencies
pip install -e .[all,dev]
```

### 3. Verify Development Setup

```bash
# Run tests
pytest

# Check code formatting
black --check permeabledt/

# Run linting
flake8 permeabledt/

# Type checking
mypy permeabledt/
```

## Verification

After installation, verify that permeabledt is working correctly:

### Basic Import Test

```python
import permeabledt as pdt
print(f"permeabledt version: {pdt.__version__}")
print("Available modules:", [name for name in dir(pdt) if not name.startswith('_')])
```

### Feature Availability Test

```python
import permeabledt as pdt

# Test core functionality
print("✓ Core water flow:", hasattr(pdt, 'run_simulation'))

# Test optional features
print("✓ Calibration:", hasattr(pdt, 'run_calibration'))
print("✓ Particle filter:", pdt.PavementModel is not None)
print("✓ Weather data:", pdt.HRRRAccumulatedPrecipitationDownloader is not None)
print("✓ Plotting:", pdt.plots is not None)
```

### Simple Functionality Test

```python
import permeabledt as pdt
import tempfile
import os

# Create a simple test setup file
test_setup = """[GENERAL]
Kc = 0.0
Df = 0.05
Dtl = 0.10
Dg = 0.20
nf = 0.32
nt = 0.4
ng = 0.35

[PONDING_ZONE]
Ap = 195
Hover = 0.5
Kweir = 1.3
wWeir = 5.0
expWeir = 2.5
Cs = 0
Pp = 0
flagp = 1

[UNSATURATED_ZONE]
A = 195
husz = 0.05
nusz = 0.32
Ks = 0.0005
sh = 0.01
sw = 0.02
sfc = 0.08
ss = 0.09
gama = 5.0
Kf = 0

[SATURATED_ZONE]
Psz = 0
hpipe = 0.025
flagsz = 1
dpipe = 152.4
Cd = 0.13
eta = 0.23

[TIMESTEP]
dt = 60
"""

# Test basic functionality
with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
    f.write(test_setup)
    setup_file = f.name

try:
    # Test parameter loading
    setup = pdt.read_setup_file(setup_file)
    params = pdt.initialize_parameters(setup)
    print("✓ Parameter loading successful")

    # Test basic simulation
    import numpy as np
    qin = np.zeros(10)
    qrain = np.ones(10) * 0.001  # 1mm/timestep
    emax = np.zeros(10)

    data, wb = pdt.run_simulation(params, qin, qrain, emax)
    print("✓ Basic simulation successful")
    print(f"  Peak outflow: {max(data['Qpipe']):.6f} m³/s")

finally:
    os.unlink(setup_file)
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Permission Errors

**Problem**: `PermissionError` during installation
```bash
ERROR: Could not install packages due to an EnvironmentError: [Errno 13] Permission denied
```

**Solution**: Use user installation or virtual environment
```bash
# Option 1: User installation
pip install --user permeabledt[all]

# Option 2: Virtual environment (recommended)
python -m venv permeabledt_env
permeabledt_env\Scripts\activate  # Windows
pip install permeabledt[all]
```

#### 2. Dependency Conflicts

**Problem**: Conflicting package versions
```bash
ERROR: permeabledt 0.1.0 has requirement pandas>=1.3.0, but you have pandas 1.2.0
```

**Solution**: Upgrade conflicting packages or use fresh environment
```bash
# Option 1: Upgrade packages
pip install --upgrade pandas numpy

# Option 2: Fresh virtual environment
python -m venv fresh_env
fresh_env\Scripts\activate
pip install permeabledt[all]
```

#### 3. Missing C Compiler (Windows)

**Problem**: Error building packages with C extensions
```bash
Microsoft Visual C++ 14.0 is required
```

**Solution**: Install Visual Studio Build Tools
1. Download [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2019)
2. Install with C++ build tools
3. Retry installation

#### 4. Git Not Found

**Problem**: Git not available for development installation
```bash
'git' is not recognized as an internal or external command
```

**Solution**: Install Git
- Windows: Download from [git-scm.com](https://git-scm.com/download/win)
- macOS: `xcode-select --install`
- Linux: `sudo apt install git` (Ubuntu/Debian)

#### 5. Python Version Issues

**Problem**: Python version too old
```bash
ERROR: permeabledt requires Python '>=3.8' but the running Python is 3.7.0
```

**Solution**: Update Python
- Use [python.org](https://python.org) for latest version
- Or use conda: `conda install python=3.9`

#### 6. Import Errors After Installation

**Problem**: `ModuleNotFoundError` even after installation
```python
ImportError: No module named 'permeabledt'
```

**Solutions**:
```bash
# Check if installed
pip list | grep permeabledt

# Reinstall if missing
pip install permeabledt[all]

# Check virtual environment activation
which python  # Should point to your virtual environment
```

#### 7. Optional Dependencies Missing

**Problem**: Feature not available despite installation
```python
AttributeError: 'NoneType' object has no attribute '...'
```

**Solution**: Install specific feature dependencies
```bash
# Check what's installed
pip list | grep -E "(deap|pypfilt|SALib|herbie)"

# Install missing features
pip install permeabledt[calib,pf,weather]
```

### Getting Help

If you encounter issues not covered here:

1. **Check the logs**: Look for specific error messages
2. **Search existing issues**: [GitHub Issues](https://github.com/yourusername/permeabledt/issues)
3. **Create a new issue**: Include:
   - Operating system and version
   - Python version (`python --version`)
   - Installation method used
   - Complete error message
   - Steps to reproduce

### Platform-Specific Notes

#### Windows
- Use Command Prompt or PowerShell
- Backslashes in paths: `C:\path\to\file`
- Virtual environment activation: `venv\Scripts\activate`

#### macOS
- May need Xcode command line tools: `xcode-select --install`
- Use Terminal
- Virtual environment activation: `source venv/bin/activate`

#### Linux
- May need build tools: `sudo apt install build-essential` (Ubuntu/Debian)
- Use terminal
- Virtual environment activation: `source venv/bin/activate`

## Next Steps

After successful installation:

1. **Read the [Quick Start Guide](quickstart.md)** for your first simulation
2. **Check out [Basic Examples](examples/basic_examples.md)** for common use cases
3. **Review [Data Preparation](data_preparation.md)** to format your data
4. **Explore advanced features** in the user guides