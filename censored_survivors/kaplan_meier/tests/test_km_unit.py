"""Unit tests for Kaplan-Meier model implementation."""

from typing import Dict, Tuple, List
import numpy as np
from scipy import stats
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
from ..run import calculate_survival_curve
from ..models import KaplanMeierData


def calculate_discrete_survival_prob(t: float, distribution: str, params: Dict) -> float:
    """Calculate survival probability for discrete time point.
    
    For discrete times, we need to account for the rounding in our data generation.
    The survival probability at time t should be the probability of surviving past t+0.5,
    since values are rounded to the nearest integer.
    
    Args:
        t: Time point
        distribution: Distribution type
        params: Distribution parameters
    
    Returns:
        Survival probability at time t
    """
    # Special case for time 0
    if t == 0:
        return 1.0
        
    # Add 0.5 to account for rounding
    t_adjusted = t + 0.5
    
    if distribution == "exponential":
        return np.exp(-t_adjusted/params["scale"])
    elif distribution == "weibull":
        return np.exp(-(t_adjusted/params["scale"])**params["shape"])
    elif distribution == "gamma":
        return 1 - special.gammainc(params["shape"], t_adjusted/params["scale"])
    elif distribution == "lognormal":
        return 1 - special.ndtr((np.log(t_adjusted) - params["mu"]) / params["sigma"])
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")


def check_survival_probability(
    km_prob: float,
    theo_prob: float,
    relative_tolerance: float = 0.05,
    absolute_tolerance: float = 0.01
) -> bool:
    """Check if KM estimate is close enough to theoretical probability.
    
    For small probabilities, we use absolute tolerance to avoid issues with
    relative errors becoming too large.
    
    Args:
        km_prob: Kaplan-Meier estimate
        theo_prob: Theoretical probability
        relative_tolerance: Maximum allowed relative error
        absolute_tolerance: Maximum allowed absolute error
    
    Returns:
        True if the estimate is within tolerance
    """
    abs_diff = abs(km_prob - theo_prob)
    
    # For small probabilities, use absolute tolerance
    if theo_prob < 0.1:
        return abs_diff < absolute_tolerance
    
    # Otherwise use relative tolerance
    return abs_diff / theo_prob < relative_tolerance


def test_exponential_distribution():
    """Test Kaplan-Meier estimates against exponential distribution.
    
    The exponential distribution is the simplest case for survival analysis
    with a constant hazard rate. This makes it a good baseline test.
    Even with constant hazard, we use time-dependent tolerances because
    later times have fewer subjects at risk.
    """
    # Parameters
    scale = 10.0  # Mean survival time
    sample_size = 10000
    max_time = 20  # Only test up to this time point
    
    # Generate data
    config = CohortConfig(
        name="Exponential Test",
        distribution_params=ExponentialParams(
            distribution=DistributionType.EXPONENTIAL,
            sample_size=sample_size,
            censoring_rate=0.0,  # No censoring for this test
            scale=scale
        )
    )
    km_data = generate_survival_times(config)
    
    # Calculate KM estimates
    times, survival_probs = calculate_survival_curve(km_data)
    
    # Calculate theoretical survival probabilities (accounting for discretization)
    theoretical_probs = np.array([
        calculate_discrete_survival_prob(t, "exponential", {"scale": scale})
        for t in times
    ])
    
    # Test at multiple time points with time-dependent tolerances
    for t, km_prob, theo_prob in zip(times, survival_probs, theoretical_probs):
        if t > max_time:
            break
        
        # For early times (t ≤ 5), use stricter tolerances
        if t <= 5:
            assert check_survival_probability(km_prob, theo_prob, 0.05, 0.01), \
                f"At time {t}, KM estimate ({km_prob:.3f}) differs significantly from theoretical ({theo_prob:.3f})"
        # For mid times (5 < t ≤ 10), use moderate tolerances
        elif t <= 10:
            assert check_survival_probability(km_prob, theo_prob, 0.08, 0.02), \
                f"At time {t}, KM estimate ({km_prob:.3f}) differs significantly from theoretical ({theo_prob:.3f})"
        # For later times (10 < t ≤ 15), use more relaxed tolerances
        elif t <= 15:
            assert check_survival_probability(km_prob, theo_prob, 0.10, 0.02), \
                f"At time {t}, KM estimate ({km_prob:.3f}) differs significantly from theoretical ({theo_prob:.3f})"
        # For very late times (t > 15), use even more relaxed tolerances
        else:
            assert check_survival_probability(km_prob, theo_prob, 0.15, 0.03), \
                f"At time {t}, KM estimate ({km_prob:.3f}) differs significantly from theoretical ({theo_prob:.3f})"


def test_weibull_distribution():
    """Test Kaplan-Meier estimates against Weibull distribution.
    
    The Weibull distribution is more general than exponential and allows
    for increasing or decreasing hazard rates. Due to the changing hazard
    rate, we use time-dependent tolerances and focus more on earlier times
    where estimates are more reliable.
    """
    # Parameters
    shape, scale = 2.5, 20.0
    sample_size = 10000
    max_time = 20  # Only test up to this time point
    
    # Generate data
    config = CohortConfig(
        name="Weibull Test",
        distribution_params=WeibullParams(
            distribution=DistributionType.WEIBULL,
            sample_size=sample_size,
            censoring_rate=0.0,
            shape=shape,
            scale=scale
        )
    )
    km_data = generate_survival_times(config)
    
    # Calculate KM estimates
    times, survival_probs = calculate_survival_curve(km_data)
    
    # Calculate theoretical survival probabilities (accounting for discretization)
    theoretical_probs = np.array([
        calculate_discrete_survival_prob(t, "weibull", {"shape": shape, "scale": scale})
        for t in times
    ])
    
    # Test at multiple time points with time-dependent tolerances
    for t, km_prob, theo_prob in zip(times, survival_probs, theoretical_probs):
        if t > max_time:
            break
            
        # For early times, use stricter tolerances
        if t <= 5:
            assert check_survival_probability(km_prob, theo_prob, 0.05, 0.01), \
                f"At time {t}, KM estimate ({km_prob:.3f}) differs significantly from theoretical ({theo_prob:.3f})"
        # For mid times, use moderate tolerances
        elif t <= 10:
            assert check_survival_probability(km_prob, theo_prob, 0.08, 0.02), \
                f"At time {t}, KM estimate ({km_prob:.3f}) differs significantly from theoretical ({theo_prob:.3f})"
        # For later times, use more relaxed tolerances
        else:
            assert check_survival_probability(km_prob, theo_prob, 0.10, 0.02), \
                f"At time {t}, KM estimate ({km_prob:.3f}) differs significantly from theoretical ({theo_prob:.3f})"


def test_censored_data():
    """Test Kaplan-Meier estimates with censored data.
    
    The KM estimator should handle censored data correctly.
    We use exponential distribution for simplicity.
    """
    # Parameters
    scale = 10.0
    sample_size = 10000
    censoring_rate = 0.3  # 30% censoring
    
    # Generate data
    config = CohortConfig(
        name="Censored Test",
        distribution_params=ExponentialParams(
            distribution=DistributionType.EXPONENTIAL,
            sample_size=sample_size,
            censoring_rate=censoring_rate,
            scale=scale
        )
    )
    km_data = generate_survival_times(config)
    
    # Calculate KM estimates
    times, survival_probs = calculate_survival_curve(km_data)
    
    # Calculate theoretical survival probabilities (accounting for discretization)
    theoretical_probs = np.array([
        calculate_discrete_survival_prob(t, "exponential", {"scale": scale})
        for t in times
    ])
    
    # With censoring, we expect more variance in our estimates
    # Use larger tolerances due to random censoring
    for t, km_prob, theo_prob in zip(times, survival_probs, theoretical_probs):
        assert check_survival_probability(km_prob, theo_prob, 0.20, 0.03), \
            f"At time {t}, KM estimate ({km_prob:.3f}) differs significantly from theoretical ({theo_prob:.3f})"


def test_small_sample():
    """Test Kaplan-Meier estimates with small sample size.
    
    The KM estimator should still provide reasonable estimates
    with smaller samples, though with larger variance.
    For small samples, we focus on earlier time points where
    more subjects are at risk and estimates are more reliable.
    """
    # Parameters
    scale = 10.0
    sample_size = 200  # Increased from 100 to 200
    max_time = 10  # Only test up to this time point
    
    # Generate data
    config = CohortConfig(
        name="Small Sample Test",
        distribution_params=ExponentialParams(
            distribution=DistributionType.EXPONENTIAL,
            sample_size=sample_size,
            censoring_rate=0.0,
            scale=scale
        )
    )
    km_data = generate_survival_times(config)
    
    # Calculate KM estimates
    times, survival_probs = calculate_survival_curve(km_data)
    
    # Calculate theoretical survival probabilities (accounting for discretization)
    theoretical_probs = np.array([
        calculate_discrete_survival_prob(t, "exponential", {"scale": scale})
        for t in times
    ])
    
    # With small samples, we expect more variance
    # Use larger tolerances and only test early time points
    for t, km_prob, theo_prob in zip(times, survival_probs, theoretical_probs):
        if t > max_time:
            break
        
        # For very early times, use stricter tolerances
        if t <= 3:
            assert check_survival_probability(km_prob, theo_prob, 0.20, 0.05), \
                f"At time {t}, KM estimate ({km_prob:.3f}) differs significantly from theoretical ({theo_prob:.3f})"
        else:
            assert check_survival_probability(km_prob, theo_prob, 0.25, 0.08), \
                f"At time {t}, KM estimate ({km_prob:.3f}) differs significantly from theoretical ({theo_prob:.3f})"


def test_monotonicity():
    """Test that Kaplan-Meier estimates are monotonically decreasing.
    
    The survival function should never increase over time.
    """
    # Generate data using any distribution
    config = CohortConfig(
        name="Monotonicity Test",
        distribution_params=WeibullParams(
            distribution=DistributionType.WEIBULL,
            sample_size=1000,
            censoring_rate=0.2,
            shape=2.0,
            scale=15.0
        )
    )
    km_data = generate_survival_times(config)
    
    # Calculate KM estimates
    times, survival_probs = calculate_survival_curve(km_data)
    
    # Check monotonicity
    for i in range(1, len(survival_probs)):
        assert survival_probs[i] <= survival_probs[i-1], \
            f"Non-monotonic behavior detected at time {times[i]}"


def test_initial_probability():
    """Test that initial survival probability is 1.0.
    
    The survival probability at time 0 should always be 1.0.
    """
    # Generate data using any distribution
    config = CohortConfig(
        name="Initial Probability Test",
        distribution_params=GammaParams(
            distribution=DistributionType.GAMMA,
            sample_size=1000,
            censoring_rate=0.0,
            shape=2.0,
            scale=10.0
        )
    )
    km_data = generate_survival_times(config)
    
    # Calculate KM estimates
    times, survival_probs = calculate_survival_curve(km_data)
    
    # Check initial probability
    assert np.isclose(survival_probs[0], 1.0), \
        f"Initial survival probability is {survival_probs[0]}, expected 1.0"


def test_gamma_distribution():
    """Test Kaplan-Meier estimates against gamma distribution.
    
    The gamma distribution is useful for modeling time-to-event data
    with increasing hazard rates. Like Weibull, we use time-dependent
    tolerances due to the changing hazard rate.
    """
    # Parameters
    shape, scale = 2.0, 10.0  # Mean = shape * scale = 20
    sample_size = 10000
    max_time = 20  # Only test up to this time point
    
    # Generate data
    config = CohortConfig(
        name="Gamma Test",
        distribution_params=GammaParams(
            distribution=DistributionType.GAMMA,
            sample_size=sample_size,
            censoring_rate=0.0,
            shape=shape,
            scale=scale
        )
    )
    km_data = generate_survival_times(config)
    
    # Calculate KM estimates
    times, survival_probs = calculate_survival_curve(km_data)
    
    # Calculate theoretical survival probabilities (accounting for discretization)
    theoretical_probs = np.array([
        calculate_discrete_survival_prob(t, "gamma", {"shape": shape, "scale": scale})
        for t in times
    ])
    
    # Test at multiple time points with time-dependent tolerances
    for t, km_prob, theo_prob in zip(times, survival_probs, theoretical_probs):
        if t > max_time:
            break
            
        # For early times, use stricter tolerances
        if t <= 5:
            assert check_survival_probability(km_prob, theo_prob, 0.05, 0.01), \
                f"At time {t}, KM estimate ({km_prob:.3f}) differs significantly from theoretical ({theo_prob:.3f})"
        # For mid times, use moderate tolerances
        elif t <= 10:
            assert check_survival_probability(km_prob, theo_prob, 0.08, 0.02), \
                f"At time {t}, KM estimate ({km_prob:.3f}) differs significantly from theoretical ({theo_prob:.3f})"
        # For later times, use more relaxed tolerances
        else:
            assert check_survival_probability(km_prob, theo_prob, 0.10, 0.02), \
                f"At time {t}, KM estimate ({km_prob:.3f}) differs significantly from theoretical ({theo_prob:.3f})"


def test_lognormal_distribution():
    """Test Kaplan-Meier estimates against log-normal distribution.
    
    The log-normal distribution has a non-monotonic hazard rate,
    making it a good test case for the KM estimator's ability to
    handle complex hazard patterns.
    """
    # Parameters
    mu, sigma = 2.0, 0.5  # Mean ≈ 8.4
    sample_size = 10000
    max_time = 20  # Only test up to this time point
    
    # Generate data
    config = CohortConfig(
        name="LogNormal Test",
        distribution_params=LogNormalParams(
            distribution=DistributionType.LOG_NORMAL,
            sample_size=sample_size,
            censoring_rate=0.0,
            mu=mu,
            sigma=sigma
        )
    )
    km_data = generate_survival_times(config)
    
    # Calculate KM estimates
    times, survival_probs = calculate_survival_curve(km_data)
    
    # Calculate theoretical survival probabilities (accounting for discretization)
    theoretical_probs = np.array([
        calculate_discrete_survival_prob(t, "lognormal", {"mu": mu, "sigma": sigma})
        for t in times
    ])
    
    # Test at multiple time points with time-dependent tolerances
    for t, km_prob, theo_prob in zip(times, survival_probs, theoretical_probs):
        if t > max_time:
            break
            
        # For early times, use stricter tolerances
        if t <= 5:
            assert check_survival_probability(km_prob, theo_prob, 0.05, 0.01), \
                f"At time {t}, KM estimate ({km_prob:.3f}) differs significantly from theoretical ({theo_prob:.3f})"
        # For mid times, use moderate tolerances
        elif t <= 10:
            assert check_survival_probability(km_prob, theo_prob, 0.08, 0.02), \
                f"At time {t}, KM estimate ({km_prob:.3f}) differs significantly from theoretical ({theo_prob:.3f})"
        # For later times, use more relaxed tolerances
        else:
            assert check_survival_probability(km_prob, theo_prob, 0.10, 0.02), \
                f"At time {t}, KM estimate ({km_prob:.3f}) differs significantly from theoretical ({theo_prob:.3f})"


def test_all_distributions_with_censoring():
    """Test all distributions with censoring enabled.
    
    This test ensures that our censoring mechanism works correctly
    across all supported distributions.
    """
    sample_size = 10000
    censoring_rate = 0.3
    max_time = 15
    
    # Test configurations for each distribution
    configs = [
        CohortConfig(
            name="Exponential",
            distribution_params=ExponentialParams(
                distribution=DistributionType.EXPONENTIAL,
                sample_size=sample_size,
                censoring_rate=censoring_rate,
                scale=10.0
            )
        ),
        CohortConfig(
            name="Weibull",
            distribution_params=WeibullParams(
                distribution=DistributionType.WEIBULL,
                sample_size=sample_size,
                censoring_rate=censoring_rate,
                shape=2.5,
                scale=20.0
            )
        ),
        CohortConfig(
            name="Gamma",
            distribution_params=GammaParams(
                distribution=DistributionType.GAMMA,
                sample_size=sample_size,
                censoring_rate=censoring_rate,
                shape=2.0,
                scale=10.0
            )
        ),
        CohortConfig(
            name="LogNormal",
            distribution_params=LogNormalParams(
                distribution=DistributionType.LOG_NORMAL,
                sample_size=sample_size,
                censoring_rate=censoring_rate,
                mu=2.0,
                sigma=0.5
            )
        )
    ]
    
    # Test each distribution
    for config in configs:
        # Generate data
        km_data = generate_survival_times(config)
        
        # Calculate KM estimates
        times, survival_probs = calculate_survival_curve(km_data)
        
        # Get distribution parameters for theoretical calculation
        params = config.distribution_params
        if isinstance(params, ExponentialParams):
            dist_type = "exponential"
            theo_params = {"scale": params.scale}
        elif isinstance(params, WeibullParams):
            dist_type = "weibull"
            theo_params = {"shape": params.shape, "scale": params.scale}
        elif isinstance(params, GammaParams):
            dist_type = "gamma"
            theo_params = {"shape": params.shape, "scale": params.scale}
        else:  # LogNormalParams
            dist_type = "lognormal"
            theo_params = {"mu": params.mu, "sigma": params.sigma}
        
        # Calculate theoretical probabilities
        theoretical_probs = np.array([
            calculate_discrete_survival_prob(t, dist_type, theo_params)
            for t in times
        ])
        
        # Test at multiple time points with relaxed tolerances due to censoring
        for t, km_prob, theo_prob in zip(times, survival_probs, theoretical_probs):
            if t > max_time:
                break
                
            assert check_survival_probability(km_prob, theo_prob, 0.20, 0.03), \
                f"For {config.name} at time {t}, KM estimate ({km_prob:.3f}) differs significantly from theoretical ({theo_prob:.3f})" 