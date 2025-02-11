# Censored Survivors

A Python library for survival analysis focused on customer churn prediction using proxy indicators when exact time-to-event data is unavailable.

## 📚 Project Overview

This project implements advanced survival analysis techniques to predict customer churn using various proxy indicators such as:
- Days since last activity
- Changes in platform engagement
- Other behavioral patterns

The library provides three main models:

1. **Kaplan-Meier Estimator** (`censored_survivors.kaplan_meier`)
   - Non-parametric survival curve estimation
   - Handles right-censored data
   - Provides survival probability estimates over time

2. **Cox Proportional Hazards** (`censored_survivors.cox_proportional_hazards`)
   - Semi-parametric model for survival analysis
   - Incorporates multiple covariates (features)
   - Estimates hazard ratios and baseline hazard

3. **Weibull Time-To-Event RNN** (`censored_survivors.weibull_rnn`)
   - Deep learning approach to survival analysis
   - Processes sequential customer behavior data
   - Predicts time-to-churn with uncertainty estimates

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Virtual environment (recommended)

### Installation Options

Choose the installation option that best fits your needs:

1. **Basic Installation** (core functionality only):
```bash
pip install -e .
```

2. **Development Installation** (includes testing tools):
```bash
pip install -e ".[dev]"
```

3. **Documentation Tools** (for building docs):
```bash
pip install -e ".[docs]"
```

4. **Full Installation** (everything included):
```bash
pip install -e ".[all]"
```

### Local Development Setup

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

3. Install in development mode with all dependencies:
```bash
pip install -e ".[all]"
```

## 📖 Documentation

The documentation is automatically generated from source code docstrings. To generate and view the documentation:

1. Install documentation dependencies:
```bash
pip install -e ".[docs]"  # If not already installed
```

2. Generate the docs:
```bash
./scripts/generate_docs.py
```

3. View in your browser:
   - Open `docs/generated/censored_survivors.html`
   - Or serve using Python's HTTP server:
     ```bash
     python -m http.server --directory docs/generated
     ```

## 📖 Usage Examples

### Kaplan-Meier Analysis

```python
from censored_survivors.kaplan_meier.models import KaplanMeierInput, KaplanMeierOutput
from censored_survivors.kaplan_meier.run import KaplanMeierModel

# Prepare your data
data = [
    KaplanMeierInput(customer_id=1, time_since_signup=120, event_status=1),
    KaplanMeierInput(customer_id=2, time_since_signup=200, event_status=0),
]

# Create and run model
model = KaplanMeierModel()
results: KaplanMeierOutput = model.fit_predict(data)
```

### Cox Proportional Hazards

```python
from censored_survivors.cox_proportional_hazards.models import CoxInput
from censored_survivors.cox_proportional_hazards.run import CoxModel

# Prepare your data with covariates
data = [
    CoxInput(
        customer_id=1,
        time_since_signup=120,
        event_status=1,
        logins_per_week=2,
        posts_per_week=5
    ),
    # ... more data
]

model = CoxModel()
results = model.fit_predict(data)
```

### Weibull RNN

```python
from censored_survivors.weibull_rnn.models import WeibullRNNInput
from censored_survivors.weibull_rnn.run import WeibullRNNModel

# Sequential data
data = [
    WeibullRNNInput(
        customer_id=1,
        week=[1, 2, 3, 4],
        logins=[5, 3, 2, 1],
        posts=[7, 5, 3, 1]
    ),
    # ... more data
]

model = WeibullRNNModel()
predictions = model.fit_predict(data)
```

## 📁 Project Structure

```
censored_survivors/
├── kaplan_meier/           # Kaplan-Meier implementation
│   ├── models.py           # Input/Output data models
│   ├── run.py             # Model implementation
│   └── generate.py        # Data generation utilities
├── cox_proportional_hazards/
│   ├── models.py
│   ├── run.py
│   └── generate.py
└── weibull_rnn/
    ├── models.py
    ├── run.py
    └── generate.py
```

## 🧪 Development

Run tests:
```bash
pytest                          # Run all tests
pytest -v -k "not weibull_rnn" # Skip slow tests
```

Format code:
```bash
ruff format .
```

Type checking:
```bash
mypy .
```

## 🤝 Contributing

Contributions are welcome! Please ensure your code:
1. Has comprehensive docstrings (Google style)
2. Includes type hints
3. Passes all tests
4. Is formatted with ruff

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
