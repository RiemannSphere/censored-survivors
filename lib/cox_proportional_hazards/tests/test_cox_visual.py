import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import pandas as pd
from scipy import special
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


def plot_coefficient_comparison(
    true_coefficients: Dict[str, float],
    estimated_coefficients: Dict[str, float],
    title: Optional[str] = None
) -> None:
    """Plot comparison between true and estimated coefficients.
    
    Args:
        true_coefficients: Dictionary of true coefficient values
        estimated_coefficients: Dictionary of estimated coefficient values
        title: Optional title for the plot
    """
    features = list(true_coefficients.keys())
    true_values = [true_coefficients[f] for f in features]
    estimated_values = [estimated_coefficients[f] for f in features]
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(features))
    width = 0.35
    
    plt.bar(x - width/2, true_values, width, label='True')
    plt.bar(x + width/2, estimated_values, width, label='Estimated')
    
    plt.xlabel('Features')
    plt.ylabel('Coefficient Value')
    plt.title(title or 'Comparison of True vs Estimated Coefficients')
    plt.xticks(x, features, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_survival_curves(
    model: CoxModel,
    data: pd.DataFrame,
    true_coefficients: Dict[str, float],
    title: Optional[str] = None
) -> None:
    """Plot comparison of survival curves between true and estimated model.
    
    Args:
        model: Fitted CoxModel instance
        data: DataFrame containing covariates and survival data
        true_coefficients: Dictionary of true coefficient values
        title: Optional title for the plot
    """
    # Calculate true and estimated risk scores
    true_risk_scores = np.zeros(len(data))
    estimated_risk_scores = np.zeros(len(data))
    
    # Convert data to numpy for calculations
    for feature, coef in true_coefficients.items():
        true_risk_scores += coef * data[feature].to_numpy()
    
    for feature, coef in model.coef_.items():
        estimated_risk_scores += coef * data[feature].to_numpy()
    
    true_risk_scores = np.exp(true_risk_scores)
    estimated_risk_scores = np.exp(estimated_risk_scores)
    
    # Calculate survival curves
    times = np.sort(data['duration'].unique())
    times_2d = times.reshape(-1, 1)  # Make 2D for broadcasting
    
    # Calculate survival probabilities using broadcasting
    true_survival = np.exp(-model.baseline_hazard_ * np.outer(true_risk_scores, times))
    estimated_survival = np.exp(-model.baseline_hazard_ * np.outer(estimated_risk_scores, times))
    
    # Plot average survival curves
    plt.figure(figsize=(10, 6))
    plt.plot(times, true_survival.mean(axis=0), label='True')
    plt.plot(times, estimated_survival.mean(axis=0), '--', label='Estimated')
    
    plt.xlabel('Time')
    plt.ylabel('Survival Probability')
    plt.title(title or 'Comparison of True vs Estimated Survival Curves')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def test_distributions_visual() -> None:
    """Visual test comparing Cox model performance across different distributions.
    
    This test generates data from different distributions and compares:
    1. Coefficient estimation accuracy
    2. Survival curve estimation accuracy
    3. Model performance metrics
    
    For each distribution, we use the same covariate effects to ensure
    comparability across distributions.
    """
    # Common parameters
    sample_size = 2000
    censoring_rate = 0.2
    effect_size = 1.0
    
    # Test configurations
    configs = {
        "Weibull": CoxCohortConfig(
            name="Weibull Test",
            survival_distribution=WeibullParams(
                distribution=DistributionType.WEIBULL,
                sample_size=sample_size,
                censoring_rate=censoring_rate,
                shape=2.0,
                scale=15.0
            ),
            covariates=[
                CovariateConfig(
                    name="feature1",
                    distribution="normal",
                    params={"mean": 0, "std": 1},
                    effect_size=effect_size
                )
            ]
        ),
        "Exponential": CoxCohortConfig(
            name="Exponential Test",
            survival_distribution=ExponentialParams(
                distribution=DistributionType.EXPONENTIAL,
                sample_size=sample_size,
                censoring_rate=censoring_rate,
                scale=15.0
            ),
            covariates=[
                CovariateConfig(
                    name="feature1",
                    distribution="normal",
                    params={"mean": 0, "std": 1},
                    effect_size=effect_size
                )
            ]
        ),
        "Gamma": CoxCohortConfig(
            name="Gamma Test",
            survival_distribution=GammaParams(
                distribution=DistributionType.GAMMA,
                sample_size=sample_size,
                censoring_rate=censoring_rate,
                shape=2.0,
                scale=7.5
            ),
            covariates=[
                CovariateConfig(
                    name="feature1",
                    distribution="normal",
                    params={"mean": 0, "std": 1},
                    effect_size=effect_size
                )
            ]
        ),
        "LogNormal": CoxCohortConfig(
            name="LogNormal Test",
            survival_distribution=LogNormalParams(
                distribution=DistributionType.LOG_NORMAL,
                sample_size=sample_size,
                censoring_rate=censoring_rate,
                mu=2.0,
                sigma=0.5
            ),
            covariates=[
                CovariateConfig(
                    name="feature1",
                    distribution="normal",
                    params={"mean": 0, "std": 1},
                    effect_size=effect_size
                )
            ]
        )
    }
    
    # Create subplots for all distributions
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Test each distribution
    for i, (name, config) in enumerate(configs.items()):
        # Generate data and fit model
        data = generate_cox_data(config)
        model = CoxModel()
        result = model.fit(data)
        
        # Create DataFrame for plotting
        df = pd.DataFrame([
            {
                'duration': d.event_time,
                'event': d.event_status,
                **d.covariates
            }
            for d in data.data
        ])
        
        # Get true coefficients
        true_coefficients = {
            cov.name: 0.5 * cov.effect_size  # Account for 0.5 scale factor in generation
            for cov in config.covariates
        }
        
        # Plot survival curves for this distribution
        plt.sca(axes[i])
        
        # Calculate and plot survival curves
        true_risk_scores = np.zeros(len(df))
        estimated_risk_scores = np.zeros(len(df))
        
        for feature, coef in true_coefficients.items():
            true_risk_scores += coef * df[feature].to_numpy()
        
        for feature, coef in result.model.covariate_coefficients.items():
            estimated_risk_scores += coef * df[feature].to_numpy()
        
        true_risk_scores = np.exp(true_risk_scores)
        estimated_risk_scores = np.exp(estimated_risk_scores)
        
        times = np.sort(df['duration'].unique())
        true_survival = np.exp(-result.model.baseline_hazard * np.outer(true_risk_scores, times))
        estimated_survival = np.exp(-result.model.baseline_hazard * np.outer(estimated_risk_scores, times))
        
        plt.plot(times, true_survival.mean(axis=0), label='True')
        plt.plot(times, estimated_survival.mean(axis=0), '--', label='Estimated')
        
        plt.xlabel('Time')
        plt.ylabel('Survival Probability')
        plt.title(f'{name} Distribution\nC-index: {result.model.concordance_index:.3f}')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print numerical results
    print("\nNumerical Results:")
    for name, config in configs.items():
        data = generate_cox_data(config)
        model = CoxModel()
        result = model.fit(data)
        
        true_coef = 0.5 * config.covariates[0].effect_size
        est_coef = result.model.covariate_coefficients['feature1']
        
        print(f"\n{name} Distribution:")
        print(f"True Coefficient: {true_coef:.3f}")
        print(f"Estimated Coefficient: {est_coef:.3f}")
        print(f"Concordance Index: {result.model.concordance_index:.3f}")
        print(f"Log Likelihood: {result.model.log_likelihood:.3f}")


if __name__ == "__main__":
    test_distributions_visual() 