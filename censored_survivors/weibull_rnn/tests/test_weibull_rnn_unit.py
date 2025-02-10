import pytest
import numpy as np
import torch
from typing import Dict, List
import pandas as pd
import os
import sys

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, project_root)

from lib.weibull_rnn.generate import (
    WeibullRNNConfig,
    SequenceConfig,
    generate_weibull_rnn_data,
    generate_sequence
)
from lib.weibull_rnn.run import WeibullRNNModel, WeibullLoss
from lib.shared.distributions import (
    WeibullParams,
    ExponentialParams,
    GammaParams,
    LogNormalParams,
    DistributionType
)


@pytest.fixture
def sample_config() -> WeibullRNNConfig:
    """Create a sample configuration for testing."""
    return WeibullRNNConfig(
        name="Test Config",
        survival_distribution=WeibullParams(
            distribution=DistributionType.WEIBULL,
            sample_size=1000,
            censoring_rate=0.2,
            shape=2.0,
            scale=100.0
        ),
        sequence_length=52,
        sequences=[
            SequenceConfig(
                name="logins_per_week",
                base_mean=5,
                base_std=1,
                trend_coefficient=-0.02,
                seasonality_amplitude=1,
                seasonality_period=52,
                noise_std=0.5,
                effect_on_survival=-0.5
            ),
            SequenceConfig(
                name="posts_per_week",
                base_mean=10,
                base_std=2,
                trend_coefficient=-0.05,
                seasonality_amplitude=2,
                seasonality_period=26,
                noise_std=1.0,
                effect_on_survival=-0.3
            )
        ]
    )


@pytest.fixture
def sample_data(sample_config: WeibullRNNConfig):
    """Generate sample data for testing."""
    return generate_weibull_rnn_data(sample_config)


@pytest.fixture
def fitted_model(sample_data):
    """Create and fit a model with sample data."""
    model = WeibullRNNModel(
        hidden_size=64,
        num_layers=2,
        dropout=0.1,
        num_epochs=10  # Reduced for testing
    )
    model.fit(sample_data)
    return model


def test_sequence_generation():
    """Test sequence generation with trend and seasonality."""
    config = WeibullRNNConfig(
        name="Test Config",
        survival_distribution=WeibullParams(
            distribution=DistributionType.WEIBULL,
            sample_size=500,
            censoring_rate=0.2,
            shape=2.0,
            scale=100.0
        ),
        sequence_length=52,
        sequences=[
            SequenceConfig(
                name="feature1",
                base_mean=5,
                base_std=1,
                trend_coefficient=-0.02,
                seasonality_amplitude=1,
                seasonality_period=52,
                noise_std=0.5,
                effect_on_survival=-0.5
            )
        ]
    )
    
    data = generate_weibull_rnn_data(config)
    
    # Check sequence properties
    sequence = data.data[0].sequence_data["feature1"]
    assert len(sequence) == config.sequence_length
    assert abs(np.mean(sequence) - config.sequences[0].base_mean) < 1.0


def test_data_generation(sample_config, sample_data):
    """Test data generation."""
    assert len(sample_data.data) == sample_config.survival_distribution.sample_size
    
    first_record = sample_data.data[0]
    assert first_record.entity_id > 0
    assert first_record.event_time >= 0
    assert first_record.event_status in [0, 1]
    
    # Check sequence data
    assert len(first_record.sequence_data) == len(sample_config.sequences)
    for name in first_record.sequence_data:
        assert len(first_record.sequence_data[name]) == sample_config.sequence_length


def test_model_initialization():
    """Test model initialization with default parameters."""
    model = WeibullRNNModel(input_size=2)
    assert model.hidden_size == 64
    assert model.num_layers == 2
    assert model.bidirectional == True


def test_model_architecture():
    """Test model architecture and forward pass."""
    model = WeibullRNNModel(input_size=2)
    x = torch.randn(32, 10, 2)  # batch_size=32, seq_len=10, features=2
    shape, scale = model.model(x)
    
    assert shape.shape == (32,)
    assert scale.shape == (32,)
    assert torch.all(shape > 1.0)
    assert torch.all(scale > 0)


def test_loss_function():
    """Test Weibull loss function."""
    criterion = WeibullLoss()
    
    # Test with simple values
    shape = torch.tensor([[1.0], [2.0]])
    scale = torch.tensor([[10.0], [20.0]])
    time = torch.tensor([[5.0], [15.0]])
    event = torch.tensor([[1.0], [0.0]])
    
    loss = criterion(shape, scale, time, event)
    assert isinstance(loss.item(), float)
    assert not torch.isnan(loss)


def test_model_fitting(sample_data, fitted_model):
    """Test model fitting."""
    # Check if model is fitted
    assert fitted_model.model is not None
    assert fitted_model.feature_names is not None
    
    # Make predictions
    result = fitted_model.predict(sample_data)
    
    # Check predictions
    assert result.predictions is not None
    assert len(result.predictions) == len(sample_data.data)
    
    first_pred = result.predictions[0]
    assert first_pred.entity_id > 0
    assert first_pred.predicted_time > 0
    assert 0 <= first_pred.survival_probability <= 1
    assert first_pred.weibull_params.shape > 0
    assert first_pred.weibull_params.scale > 0


def test_prediction_accuracy(sample_data, fitted_model):
    """Test prediction accuracy."""
    predictions = fitted_model.predict(sample_data)
    
    # Extract true and predicted times
    true_times = np.array([d.event_time for d in sample_data.data])
    predicted_times = np.array([p.predicted_time for p in predictions.predictions])
    
    # Calculate metrics
    mae = np.mean(np.abs(true_times - predicted_times))
    r2 = 1 - np.sum((true_times - predicted_times)**2) / np.sum((true_times - np.mean(true_times))**2)
    
    # Check if metrics are reasonable
    assert mae < np.mean(true_times)  # MAE should be less than mean time
    assert r2 > 0  # Model should explain some variance


@pytest.mark.parametrize("distribution_params", [
    WeibullParams(
        distribution=DistributionType.WEIBULL,
        sample_size=500,
        censoring_rate=0.2,
        shape=2.0,
        scale=100.0
    ),
    ExponentialParams(
        distribution=DistributionType.EXPONENTIAL,
        sample_size=500,
        censoring_rate=0.2,
        scale=100.0
    ),
    GammaParams(
        distribution=DistributionType.GAMMA,
        sample_size=500,
        censoring_rate=0.2,
        shape=2.0,
        scale=50.0
    ),
    LogNormalParams(
        distribution=DistributionType.LOG_NORMAL,
        sample_size=500,
        censoring_rate=0.2,
        mu=4.0,
        sigma=0.5
    )
])
def test_different_distributions(distribution_params):
    """Test model performance with different survival time distributions."""
    config = WeibullRNNConfig(
        name="Distribution Test",
        survival_distribution=distribution_params,
        sequence_length=52,
        sequences=[
            SequenceConfig(
                name="feature1",
                base_mean=5,
                base_std=1,
                trend_coefficient=-0.02,
                seasonality_amplitude=1,
                seasonality_period=52,
                noise_std=0.5,
                effect_on_survival=-0.5
            )
        ]
    )
    
    # Generate data and fit model
    data = generate_weibull_rnn_data(config)
    model = WeibullRNNModel(num_epochs=10)  # Reduced epochs for testing
    result = model.fit(data)
    
    # Basic checks on output
    assert result.model_metrics.weibull_params.shape > 0
    assert result.model_metrics.weibull_params.scale > 0
    assert result.model_metrics.predicted_time > 0
    assert 0 <= result.model_metrics.survival_probability <= 1


def test_error_handling():
    """Test error handling in the model."""
    model = WeibullRNNModel()
    
    # Test prediction without fitting
    config = WeibullRNNConfig(
        name="Error Test",
        survival_distribution=WeibullParams(
            distribution=DistributionType.WEIBULL,
            sample_size=10,
            censoring_rate=0.2,
            shape=2.0,
            scale=100.0
        ),
        sequence_length=52,
        sequences=[
            SequenceConfig(
                name="feature1",
                base_mean=5,
                base_std=1,
                trend_coefficient=-0.02,
                seasonality_amplitude=1,
                seasonality_period=52,
                noise_std=0.5,
                effect_on_survival=-0.5
            )
        ]
    )
    data = generate_weibull_rnn_data(config)
    
    with pytest.raises(ValueError):
        model.predict(data)


if __name__ == "__main__":
    pytest.main([__file__]) 