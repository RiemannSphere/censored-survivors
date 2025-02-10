import pytest
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import pandas as pd
import torch
import os
import sys

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, project_root)

from censored_survivors.weibull_rnn.generate import (
    WeibullRNNConfig,
    generate_weibull_rnn_data,
    SequenceConfig
)
from censored_survivors.weibull_rnn.run import WeibullRNNModel
from censored_survivors.shared.distributions import (
    WeibullParams,
    ExponentialParams,
    GammaParams,
    LogNormalParams,
    DistributionType
)


def plot_sequence_predictions(
    true_times: np.ndarray,
    predicted_times: np.ndarray,
    sequences: Dict[str, List[List[float]]],
    title: Optional[str] = None
) -> None:
    """Plot comparison of true vs predicted times along with sequences.
    
    Args:
        true_times: Array of true event times
        predicted_times: Array of predicted event times
        sequences: Dictionary of feature names and their sequences
        title: Optional title for the plot
    """
    n_sequences = len(sequences)
    fig, axes = plt.subplots(n_sequences + 1, 1, figsize=(12, 4 * (n_sequences + 1)))
    
    # Plot 1: True vs Predicted times
    axes[0].scatter(true_times, predicted_times, alpha=0.5)
    max_time = max(np.max(true_times), np.max(predicted_times))
    axes[0].plot([0, max_time], [0, max_time], 'r--')  # Perfect prediction line
    axes[0].set_xlabel("True Time")
    axes[0].set_ylabel("Predicted Time")
    axes[0].set_title("True vs Predicted Times")
    
    # Plot sequences for a few examples
    n_examples = 5
    for i, (name, sequence_list) in enumerate(sequences.items()):
        ax = axes[i + 1]
        
        # Plot sequences with color based on prediction accuracy
        errors = np.abs(true_times - predicted_times)
        sorted_indices = np.argsort(errors)
        
        # Plot best and worst predictions
        for idx in sorted_indices[:n_examples]:  # Best predictions
            ax.plot(sequence_list[idx], 'g-', alpha=0.5, label='Good' if idx == sorted_indices[0] else None)
        
        for idx in sorted_indices[-n_examples:]:  # Worst predictions
            ax.plot(sequence_list[idx], 'r-', alpha=0.5, label='Poor' if idx == sorted_indices[-1] else None)
        
        ax.set_title(f"Sequences for {name}")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Value")
        if i == 0:  # Add legend to first sequence plot
            ax.legend()
    
    if title:
        fig.suptitle(title, fontsize=14)
    
    plt.tight_layout()
    plt.show()


def get_base_config(distribution_params) -> WeibullRNNConfig:
    """Get base configuration with common parameters."""
    return WeibullRNNConfig(
        name=f"{distribution_params.distribution} Test",
        survival_distribution=distribution_params,
        sequence_length=52,  # Weekly data instead of daily
        sequences=[
            SequenceConfig(
                name="logins_per_week",
                base_mean=3.0,  # Average 3 logins per week
                base_std=1.0,
                trend_coefficient=-0.005,
                seasonality_amplitude=0.5,
                seasonality_period=4,  # Monthly seasonality
                noise_std=0.3,
                effect_on_survival=-0.8
            ),
            SequenceConfig(
                name="posts_per_week",
                base_mean=1.5,
                base_std=0.5,
                trend_coefficient=-0.003,
                seasonality_amplitude=0.3,
                seasonality_period=4,
                noise_std=0.2,
                effect_on_survival=-0.6
            ),
            SequenceConfig(
                name="native_posts_ratio",
                base_mean=0.3,
                base_std=0.1,
                trend_coefficient=0.001,
                seasonality_amplitude=0.05,
                seasonality_period=12,
                noise_std=0.02,
                effect_on_survival=0.7
            )
        ]  # Reduced number of features
    )


def run_distribution_test(config: WeibullRNNConfig) -> None:
    """Run test for a single distribution configuration."""
    print(f"\nTesting {config.name}:")
    
    # Generate data
    data = generate_weibull_rnn_data(config)
    
    # Create and fit model with improved parameters
    model = WeibullRNNModel(
        hidden_size=256,      # Increased hidden size for more capacity
        num_layers=3,         # Added one more layer
        dropout=0.1,          # Reduced dropout
        learning_rate=0.001,  # Increased learning rate
        batch_size=32,        # Smaller batch size for better generalization
        num_epochs=200,       # More epochs
        patience=20,          # More patience
        bidirectional=True    # Keep bidirectional
    )
    result = model.fit(data)
    
    # Make predictions
    predictions = model.predict(data)
    
    # Extract true and predicted times
    true_times = np.array([d.event_time for d in data.data])
    predicted_times = np.array([p.predicted_time for p in predictions.predictions])
    
    # Extract sequences
    sequences = {
        name: [d.sequence_data[name] for d in data.data]
        for name in data.data[0].sequence_data.keys()
    }
    
    # Plot results
    plot_sequence_predictions(
        true_times,
        predicted_times,
        sequences,
        f"{config.name} Results"
    )
    
    # Print metrics
    print(f"Shape Parameter: {result.model_metrics.weibull_params.shape:.4f}")
    print(f"Scale Parameter: {result.model_metrics.weibull_params.scale:.4f}")
    print(f"Mean Absolute Error: {np.mean(np.abs(true_times - predicted_times)):.4f}")
    print(f"R² Score: {1 - np.sum((true_times - predicted_times)**2) / np.sum((true_times - np.mean(true_times))**2):.4f}")
    if result.model_metrics.log_likelihood is not None:
        print(f"Log Likelihood: {result.model_metrics.log_likelihood:.4f}")


def test_weibull_distribution() -> None:
    """Test Weibull distribution."""
    config = get_base_config(
        WeibullParams(
            distribution=DistributionType.WEIBULL,
            sample_size=500,    # Reduced from 2000
            censoring_rate=0.2,
            shape=2.0,
            scale=180.0  # Roughly 6 months average survival
        )
    )
    run_distribution_test(config)


def test_exponential_distribution() -> None:
    """Test Exponential distribution."""
    config = get_base_config(
        ExponentialParams(
            distribution=DistributionType.EXPONENTIAL,
            sample_size=2000,
            censoring_rate=0.2,
            scale=180.0  # Match Weibull scale
        )
    )
    run_distribution_test(config)


def test_gamma_distribution() -> None:
    """Test Gamma distribution."""
    config = get_base_config(
        GammaParams(
            distribution=DistributionType.GAMMA,
            sample_size=2000,
            censoring_rate=0.2,
            shape=2.0,
            scale=90.0  # Half of Weibull scale to match mean
        )
    )
    run_distribution_test(config)


def test_lognormal_distribution() -> None:
    """Test LogNormal distribution."""
    config = get_base_config(
        LogNormalParams(
            distribution=DistributionType.LOG_NORMAL,
            sample_size=2000,
            censoring_rate=0.2,
            mu=5.0,  # ln(180) ≈ 5.2
            sigma=0.5
        )
    )
    run_distribution_test(config)


if __name__ == "__main__":
    import sys
    
    # Get distribution to test from command line argument
    if len(sys.argv) != 2:
        print("Usage: python test_weibull_rnn_visual.py <distribution>")
        print("Available distributions: weibull, exponential, gamma, lognormal")
        sys.exit(1)
    
    distribution = sys.argv[1].lower()
    
    if distribution == "weibull":
        test_weibull_distribution()
    elif distribution == "exponential":
        test_exponential_distribution()
    elif distribution == "gamma":
        test_gamma_distribution()
    elif distribution == "lognormal":
        test_lognormal_distribution()
    else:
        print(f"Unknown distribution: {distribution}")
        print("Available distributions: weibull, exponential, gamma, lognormal")
        sys.exit(1) 