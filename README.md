# Censored Survivors

A project for exploring and experimenting with survival analysis, focusing on customer churn prediction using proxy indicators when exact time-to-event data is unavailable.

## Project Overview

This project implements survival analysis techniques to predict customer churn using various proxy indicators such as:
- Days since last activity
- Changes in platform engagement
- Other behavioral patterns

## Installation

### Local Development Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/censored-survivors.git
cd censored-survivors
```

2. Create and activate a virtual environment:

macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

3. Install in editable mode with development dependencies:
```bash
pip install -e ".[dev]"
```

### Installing in Another Project

To use this package in another project, install it directly from the local directory:
```bash
pip install /path/to/censored-survivors
```

## Usage

```python
from censored_survivors.kaplan_meier import KaplanMeierModel
from censored_survivors.weibull_rnn import WeibullRNNModel
from censored_survivors.cox_proportional_hazards import CoxModel

# Example usage
model = KaplanMeierModel()
# ... etc
```

## Development

Run tests:
```bash
pytest
# skip the slowest test
pytest -v -k "not weibull_rnn" 
```

Format code:
```bash
ruff format .
```

Type checking:
```bash
mypy .
```
