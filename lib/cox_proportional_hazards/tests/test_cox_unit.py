import pytest
import numpy as np
from typing import Dict, List
import pandas as pd
from lifelines import CoxPHFitter
import os
import sys

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, project_root)

from lib.cox_proportional_hazards.generate import CoxCohortConfig, generate_cox_data
from lib.cox_proportional_hazards.run import CoxModel
from lib.shared.distributions import (
    WeibullParams,
    ExponentialParams,
    GammaParams,
    LogNormalParams,
    DistributionType,
    CovariateConfig
)


@pytest.fixture
def sample_config() -> CoxCohortConfig:
    """Create a sample configuration for testing."""
    return CoxCohortConfig(
        name="Test Cohort",
        survival_distribution=WeibullParams(
            distribution=DistributionType.WEIBULL,
            sample_size=2000,  # Increased sample size
            censoring_rate=0.2,  # Reduced censoring
            shape=2.0,  # Increased shape
            scale=15.0  # Increased scale
        ),
        covariates=[
            CovariateConfig(
                name="logins_per_week",
                distribution="normal",
                params={"mean": 5, "std": 2},
                effect_size=-0.5
            ),
            CovariateConfig(
                name="posts_per_week",
                distribution="normal",
                params={"mean": 10, "std": 3},
                effect_size=-0.3
            ),
            CovariateConfig(
                name="native_posts_ratio",
                distribution="uniform",
                params={"low": 0, "high": 1},
                effect_size=0.7
            )
        ]
    )


@pytest.fixture
def sample_data(sample_config: CoxCohortConfig):
    """Generate sample data for testing."""
    return generate_cox_data(sample_config)


@pytest.fixture
def fitted_model(sample_data):
    """Create and fit a model with sample data."""
    model = CoxModel()
    model.fit(sample_data)
    return model


def test_model_initialization():
    """Test model initialization."""
    model = CoxModel()
    assert model.coef_ is None
    assert model.baseline_hazard_ is None
    assert model.log_likelihood_ is None
    assert model.concordance_index_ is None
    assert isinstance(model._lifelines_model, CoxPHFitter)


def test_data_generation(sample_data):
    """Test data generation."""
    assert len(sample_data.data) > 0
    first_record = sample_data.data[0]
    assert first_record.entity_id > 0
    assert first_record.event_time >= 0
    assert first_record.event_status in [0, 1]
    assert len(first_record.covariates) == 3
    assert all(isinstance(v, float) for v in first_record.covariates.values())


def test_model_fitting(sample_config, sample_data, fitted_model):
    """Test model fitting."""
    # Check if coefficients are estimated
    assert fitted_model.coef_ is not None
    assert len(fitted_model.coef_) == len(sample_config.covariates)
    
    # Check if baseline hazard is estimated
    assert fitted_model.baseline_hazard_ is not None
    assert fitted_model.baseline_hazard_ > 0
    
    # Check if log likelihood is calculated
    assert fitted_model.log_likelihood_ is not None
    
    # Check if concordance index is calculated
    assert fitted_model.concordance_index_ is not None
    assert 0 <= fitted_model.concordance_index_ <= 1


def test_coefficient_accuracy(sample_config, fitted_model):
    """Test accuracy of coefficient estimation."""
    # Scale factor used in data generation
    scale_factor = 0.5
    
    # Get true coefficients (scaled)
    true_coefficients = {
        cov.name: scale_factor * cov.effect_size
        for cov in sample_config.covariates
    }
    
    estimated_coefficients = fitted_model.coef_
    
    # Check if coefficients have correct signs
    for name, true_coef in true_coefficients.items():
        estimated_coef = estimated_coefficients[name]
        assert np.sign(true_coef) == np.sign(estimated_coef)
    
    # Convert coefficients to arrays for correlation calculation
    true_coefs = np.array([true_coefficients[name] for name in true_coefficients])
    est_coefs = np.array([estimated_coefficients[name] for name in true_coefficients])
    
    # Calculate correlation between true and estimated coefficients
    correlation = np.corrcoef(true_coefs, est_coefs)[0, 1]
    assert correlation > 0.8  # Strong correlation threshold
    
    # Check relative ordering of coefficients
    true_order = np.argsort(np.abs(true_coefs))
    est_order = np.argsort(np.abs(est_coefs))
    assert np.array_equal(true_order, est_order)  # Same ordering of importance


def test_predictions(sample_data, fitted_model):
    """Test model predictions."""
    predictions = fitted_model.predict(sample_data)
    
    # Check if predictions are returned
    assert predictions.predictions is not None
    assert len(predictions.predictions) == len(sample_data.data)
    
    # Check prediction values
    for pred in predictions.predictions:
        assert 0 <= pred.survival_probability <= 1
        assert pred.hazard_ratio > 0


def test_concordance_with_lifelines(sample_data, fitted_model):
    """Test concordance with lifelines implementation."""
    # Prepare data for lifelines
    df = pd.DataFrame([
        {
            'duration': d.event_time,
            'event': d.event_status,
            **d.covariates
        }
        for d in sample_data.data
    ])
    
    # Fit lifelines model
    lifelines_model = CoxPHFitter()
    lifelines_model.fit(df, 'duration', 'event')
    
    # Compare concordance indices
    lifelines_concordance = lifelines_model.score(df, scoring_method="concordance_index")
    custom_concordance = fitted_model.concordance_index_
    
    assert abs(lifelines_concordance - custom_concordance) < 0.1


@pytest.mark.parametrize("distribution_params", [
    WeibullParams(
        distribution=DistributionType.WEIBULL,
        sample_size=2000,
        censoring_rate=0.2,
        shape=2.0,
        scale=15.0
    ),
    ExponentialParams(
        distribution=DistributionType.EXPONENTIAL,
        sample_size=2000,
        censoring_rate=0.2,
        scale=15.0
    ),
    GammaParams(
        distribution=DistributionType.GAMMA,
        sample_size=2000,
        censoring_rate=0.2,
        shape=2.0,
        scale=7.5
    ),
    LogNormalParams(
        distribution=DistributionType.LOG_NORMAL,
        sample_size=2000,
        censoring_rate=0.2,
        mu=2.0,
        sigma=0.5
    )
])
def test_different_distributions(distribution_params):
    """Test model performance with different survival time distributions."""
    config = CoxCohortConfig(
        name="Distribution Test",
        survival_distribution=distribution_params,
        covariates=[
            CovariateConfig(
                name="feature1",
                distribution="normal",
                params={"mean": 0, "std": 1},
                effect_size=1.0  # Using a simpler effect size
            )
        ]
    )
    
    data = generate_cox_data(config)
    model = CoxModel()
    result = model.fit(data)
    
    # Check if model converges
    assert result.model.covariate_coefficients is not None
    assert result.model.concordance_index > 0.52  # Slightly above random


def test_error_handling():
    """Test error handling in the model."""
    model = CoxModel()
    
    # Test prediction without fitting
    with pytest.raises(ValueError):
        config = CoxCohortConfig(
            name="Error Test",
            survival_distribution=WeibullParams(
                distribution=DistributionType.WEIBULL,
                sample_size=10,
                censoring_rate=0.3,
                shape=1.5,
                scale=10
            ),
            covariates=[
                CovariateConfig(
                    name="feature1",
                    distribution="normal",
                    params={"mean": 0, "std": 1},
                    effect_size=0.5
                )
            ]
        )
        data = generate_cox_data(config)
        model.predict(data)


if __name__ == "__main__":
    pytest.main([__file__]) 