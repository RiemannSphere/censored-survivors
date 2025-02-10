"""Test the installation and basic functionality of the censored-survivors package."""

import pytest
import warnings
from pathlib import Path

# Suppress specific scipy deprecation warning
warnings.filterwarnings("ignore", category=DeprecationWarning, 
                       message="scipy.misc is deprecated")

def test_package_structure():
    """Test if the basic package structure exists."""
    # Check main package directory
    assert Path("censored_survivors").is_dir(), "Main package directory not found"
    
    # Check core modules
    required_modules = [
        "weibull_rnn",
        "cox_proportional_hazards",
        "kaplan_meier",
        "shared"
    ]
    
    for module in required_modules:
        assert Path(f"censored_survivors/{module}").is_dir(), f"{module} module directory not found"

def test_package_importable():
    """Test if the package can be imported."""
    try:
        import censored_survivors
    except ImportError as e:
        pytest.fail(f"Failed to import censored_survivors: {e}")

def test_dependencies():
    """Test if key dependencies are installed."""
    required_packages = [
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('torch', 'torch'),
        ('pytest', 'pytest'),
        ('ruff', 'ruff'),
        ('sklearn', 'scikit-learn'),  # Note: import name differs from package name
        ('lifelines', 'lifelines'),
        ('pydantic', 'pydantic')
    ]
    
    missing_packages = []
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        pytest.fail(f"Missing required packages: {', '.join(missing_packages)}")

def test_basic_imports():
    """Test that core modules can be imported."""
    try:
        from censored_survivors.shared import distributions
        from censored_survivors.cox_proportional_hazards import run
        from censored_survivors.weibull_rnn import run
        from censored_survivors.kaplan_meier import run
    except ImportError as e:
        pytest.fail(f"Failed to import core modules: {e}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 