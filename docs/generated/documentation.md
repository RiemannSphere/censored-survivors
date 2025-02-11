# Censored Survivors

## Overview

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


# API Reference


This section describes the main components and how to use them in your code.

## Kaplan Meier

### KaplanMeierData

Container for multiple Kaplan-Meier input records.

Attributes:
    data: List of KaplanMeierInput records

**Fields:**

- `data`: `List`


### KaplanMeierInput

Single data point for Kaplan-Meier analysis input.

Attributes:
    entity_id: Unique identifier for the entity (e.g., customer ID)
    event_time: Time until event or censoring occurred
    event_status: Binary indicator (1 = event occurred, 0 = censored)

**Fields:**

- `entity_id`: `int`
- `event_time`: `int`
- `event_status`: `int`


### KaplanMeierOutput

Single data point for Kaplan-Meier analysis output.

Attributes:
    time: Time point at which survival probability is calculated
    probability: Estimated survival probability at the given time point

**Fields:**

- `time`: `int`
- `probability`: `float`


### KaplanMeierResult

Container for Kaplan-Meier analysis results.

Attributes:
    estimates: List of survival probability estimates at different time points

**Fields:**

- `estimates`: `List`



## Cox Proportional Hazards

### CoxData

Container for multiple Cox Proportional Hazards input records.

Attributes:
    data: List of CoxInput records

**Fields:**

- `data`: `List`


### CoxInput

Single data point for Cox Proportional Hazards analysis input.

Attributes:
    entity_id: Unique identifier for the entity (e.g., customer ID)
    event_time: Time until event or censoring occurred
    event_status: Binary indicator (1 = event occurred, 0 = censored)
    covariates: Dictionary of covariate names and their values

**Fields:**

- `entity_id`: `int`
- `event_time`: `int`
- `event_status`: `int`
- `covariates`: `Dict`


### CoxOutput

Output for Cox Proportional Hazards analysis.

Attributes:
    covariate_coefficients: Dictionary of covariate names and their coefficients
    baseline_hazard: Baseline hazard rate at time t
    log_likelihood: Log likelihood of the fitted model
    concordance_index: C-index measuring model discrimination

**Fields:**

- `covariate_coefficients`: `Dict`
- `baseline_hazard`: `float`
- `log_likelihood`: `Optional`
- `concordance_index`: `Optional`


### CoxPrediction

Prediction output for a single entity.

Attributes:
    entity_id: Identifier of the entity
    survival_probability: Predicted survival probability at specified time
    hazard_ratio: Hazard ratio for this entity relative to baseline

**Fields:**

- `entity_id`: `int`
- `survival_probability`: `float`
- `hazard_ratio`: `float`


### CoxResult

Container for Cox Proportional Hazards analysis results.

Attributes:
    model: Fitted model parameters and metrics
    predictions: Optional list of predictions for entities

**Fields:**

- `model`: `CoxOutput`
- `predictions`: `Optional`


### CoxModel

Implementation of Cox Proportional Hazards model.

This class provides methods for fitting the Cox model, calculating hazard ratios,
and making predictions. It uses both a custom implementation and lifelines library
for validation.

#### `fit`

Fit the Cox Proportional Hazards model.

Args:
    data: Training data in CoxData format

Returns:
    CoxResult containing fitted model parameters and metrics

#### `predict`

Make predictions for new data.

Args:
    data: Test data in CoxData format

Returns:
    CoxResult containing model parameters and predictions

## Weibull Rnn

### WeibullParameters

Weibull distribution parameters for each time step.

Attributes:
    shape: Shape parameter (k) of the Weibull distribution
    scale: Scale parameter (λ) of the Weibull distribution

**Fields:**

- `shape`: `float`
- `scale`: `float`


### WeibullRNNData

Container for multiple Weibull Time-To-Event RNN input records.

Attributes:
    data: List of WeibullRNNInput records

**Fields:**

- `data`: `List`


### WeibullRNNInput

Single data point for Weibull Time-To-Event RNN analysis input.

Attributes:
    entity_id: Unique identifier for the entity (e.g., customer ID)
    event_time: Time until event or censoring occurred
    event_status: Binary indicator (1 = event occurred, 0 = censored)
    sequence_data: Dictionary of feature names and their time series values

**Fields:**

- `entity_id`: `int`
- `event_time`: `int`
- `event_status`: `int`
- `sequence_data`: `Dict`


### WeibullRNNOutput

Output for Weibull Time-To-Event RNN analysis.

Attributes:
    weibull_params: Predicted Weibull parameters
    predicted_time: Expected time-to-event
    survival_probability: Survival probability at predicted time
    log_likelihood: Log likelihood of the prediction

**Fields:**

- `weibull_params`: `WeibullParameters`
- `predicted_time`: `float`
- `survival_probability`: `float`
- `log_likelihood`: `Optional`


### WeibullRNNPrediction

Prediction output for a single entity.

Attributes:
    entity_id: Identifier of the entity
    predicted_time: Predicted time-to-event
    survival_probability: Survival probability at predicted time
    weibull_params: Predicted Weibull parameters

**Fields:**

- `entity_id`: `int`
- `predicted_time`: `float`
- `survival_probability`: `float`
- `weibull_params`: `WeibullParameters`


### WeibullRNNResult

Container for Weibull Time-To-Event RNN analysis results.

Attributes:
    model_metrics: Overall model metrics
    predictions: Optional list of predictions for entities

**Fields:**

- `model_metrics`: `WeibullRNNOutput`
- `predictions`: `Optional`


### WeibullLoss

Custom loss function for Weibull Time-To-Event model.

### WeibullRNN

Base class for all neural network modules.

Your models should also subclass this class.

Modules can also contain other Modules, allowing them to be nested in
a tree structure. You can assign the submodules as regular attributes::

    import torch.nn as nn
    import torch.nn.functional as F

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(1, 20, 5)
            self.conv2 = nn.Conv2d(20, 20, 5)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            return F.relu(self.conv2(x))

Submodules assigned in this way will be registered, and will also have their
parameters converted when you call :meth:`to`, etc.

.. note::
    As per the example above, an ``__init__()`` call to the parent class
    must be made before assignment on the child.

:ivar training: Boolean represents whether this module is in training or
                evaluation mode.
:vartype training: bool

### WeibullRNNDataset

PyTorch Dataset for Weibull RNN data.

### WeibullRNNModel

Implementation of Weibull Time-To-Event RNN model.

#### `fit`

Fit the model to training data with early stopping and learning rate scheduling.

#### `predict`

Make predictions for new data.

Args:
    data: Test data

Returns:
    Predictions
