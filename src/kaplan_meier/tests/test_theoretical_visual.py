"""Visual test comparing Kaplan-Meier estimates with theoretical curves."""

from typing import Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from ..generate import (
    CohortConfig,
    WeibullParams,
    ExponentialParams,
    GammaParams,
    LogNormalParams,
    DistributionType,
    generate_survival_times
)
from ..models import KaplanMeierData


def plot_survival_comparison(
    km_data: Dict[str, KaplanMeierData],
    theoretical_data: Dict[str, Tuple[np.ndarray, np.ndarray]]
) -> None:
    """Plot KM estimates alongside theoretical survival curves.
    
    Args:
        km_data: Dictionary of KM survival data
        theoretical_data: Dictionary of theoretical survival curves
    """
    plt.figure(figsize=(12, 8))
    
    # Plot KM estimates
    colors = {
        "Weibull": "#2ecc71",      # Green
        "Exponential": "#3498db",   # Blue
        "Gamma": "#e74c3c",        # Red
        "LogNormal": "#9b59b6"     # Purple
    }
    
    for name, data in km_data.items():
        # Extract times and status
        times = np.array([d.event_time for d in data.data])
        events = np.array([d.event_status for d in data.data])
        
        # Sort data by time
        sort_idx = np.argsort(times)
        times = times[sort_idx]
        events = events[sort_idx]
        
        # Get unique time points and calculate survival probability
        unique_times = np.unique(times)
        survival_prob = np.ones(len(unique_times))
        
        # Calculate survival probability
        at_risk = len(times)
        prob_so_far = 1.0
        
        for i, t in enumerate(unique_times):
            # Count events at this time point
            mask = times == t
            events_at_t = events[mask].sum()
            
            # Calculate survival probability
            if at_risk > 0:
                prob_so_far *= (1 - events_at_t / at_risk)
            survival_prob[i] = prob_so_far
            
            # Update number at risk
            at_risk -= mask.sum()
        
        # Get distribution type from name
        dist_type = next(k for k in colors.keys() if k in name)
        color = colors[dist_type]
        plt.step(
            unique_times,
            survival_prob,
            where='post',
            label=name,
            color=color,
            alpha=0.8
        )
    
    # Add theoretical curves
    for name, (times, probs) in theoretical_data.items():
        dist_type = next(k for k in colors.keys() if k in name)
        color = colors[dist_type]
        plt.plot(
            times, 
            probs,
            '--',
            color=color,
            alpha=0.7,
            label=name,
            linewidth=2
        )
    
    plt.grid(True, alpha=0.3)
    plt.xlabel('Time')
    plt.ylabel('Survival Probability')
    plt.title("Kaplan-Meier vs Theoretical Survival Curves")
    plt.legend()
    plt.tight_layout()


def test_km_vs_theoretical() -> None:
    """Visual test comparing Kaplan-Meier estimates with theoretical survival curves.
    
    This test generates data from known distributions and compares the
    Kaplan-Meier estimates with the true survival functions. It helps validate
    that our KM implementation correctly estimates the underlying survival curves.
    
    For each distribution:
    1. Generate survival times from the distribution
    2. Calculate KM estimate from the generated data
    3. Plot both KM estimate and theoretical survival curve
    """
    # Time points for theoretical curves
    t = np.linspace(0, 30, 1000)
    
    # Dictionary to store both KM and theoretical data
    survival_data: Dict[str, KaplanMeierData] = {}
    theoretical_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    
    # 1. Weibull Distribution
    shape, scale = 2.5, 20
    weibull_config = CohortConfig(
        name="Weibull",
        distribution_params=WeibullParams(
            distribution=DistributionType.WEIBULL,
            sample_size=5000,  # Large sample for better comparison
            censoring_rate=0.0,  # No censoring for clearer comparison
            shape=shape,
            scale=scale
        )
    )
    survival_data["Weibull (KM)"] = generate_survival_times(weibull_config)
    theoretical_data["Weibull (True)"] = (
        t,
        np.exp(-(t/scale)**shape)  # Weibull survival function
    )
    
    # 2. Exponential Distribution
    scale_exp = 10
    exp_config = CohortConfig(
        name="Exponential",
        distribution_params=ExponentialParams(
            distribution=DistributionType.EXPONENTIAL,
            sample_size=5000,
            censoring_rate=0.0,
            scale=scale_exp
        )
    )
    survival_data["Exponential (KM)"] = generate_survival_times(exp_config)
    theoretical_data["Exponential (True)"] = (
        t,
        np.exp(-t/scale_exp)  # Exponential survival function
    )
    
    # 3. Gamma Distribution
    shape_gamma, scale_gamma = 2.0, 10.0
    gamma_config = CohortConfig(
        name="Gamma",
        distribution_params=GammaParams(
            distribution=DistributionType.GAMMA,
            sample_size=5000,
            censoring_rate=0.0,
            shape=shape_gamma,
            scale=scale_gamma
        )
    )
    survival_data["Gamma (KM)"] = generate_survival_times(gamma_config)
    theoretical_data["Gamma (True)"] = (
        t,
        1 - special.gammainc(shape_gamma, t/scale_gamma)  # Gamma survival function
    )
    
    # 4. Log-Normal Distribution
    mu, sigma = 2.0, 0.5
    lognorm_config = CohortConfig(
        name="LogNormal",
        distribution_params=LogNormalParams(
            distribution=DistributionType.LOG_NORMAL,
            sample_size=5000,
            censoring_rate=0.0,
            mu=mu,
            sigma=sigma
        )
    )
    survival_data["LogNormal (KM)"] = generate_survival_times(lognorm_config)
    theoretical_data["LogNormal (True)"] = (
        t,
        1 - special.ndtr((np.log(t) - mu) / sigma)  # Log-normal survival function
    )
    
    # Create the comparison plot
    plt.clf()  # Clear any existing plots
    plot_survival_comparison(survival_data, theoretical_data)
    plt.show()


if __name__ == "__main__":
    test_km_vs_theoretical() 