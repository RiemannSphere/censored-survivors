from typing import Literal, Union, Optional, Dict, List
from enum import Enum
import numpy as np
from pydantic import BaseModel, Field


class DistributionType(str, Enum):
    """Supported probability distributions for survival time generation."""
    WEIBULL = "weibull"
    EXPONENTIAL = "exponential"
    GAMMA = "gamma"
    LOG_NORMAL = "log_normal"


class BaseDistributionParams(BaseModel):
    """Base class for distribution parameters."""
    distribution: DistributionType
    sample_size: int = Field(gt=0, description="Number of samples to generate")
    censoring_rate: float = Field(ge=0.0, le=1.0, description="Proportion of censored observations")


class WeibullParams(BaseDistributionParams):
    """Parameters for Weibull distribution."""
    distribution: Literal[DistributionType.WEIBULL]
    shape: float = Field(gt=0, description="Shape parameter (k)")
    scale: float = Field(gt=0, description="Scale parameter (lambda)")


class ExponentialParams(BaseDistributionParams):
    """Parameters for Exponential distribution."""
    distribution: Literal[DistributionType.EXPONENTIAL]
    scale: float = Field(gt=0, description="Scale parameter (lambda)")


class GammaParams(BaseDistributionParams):
    """Parameters for Gamma distribution."""
    distribution: Literal[DistributionType.GAMMA]
    shape: float = Field(gt=0, description="Shape parameter (k)")
    scale: float = Field(gt=0, description="Scale parameter (theta)")


class LogNormalParams(BaseDistributionParams):
    """Parameters for Log-Normal distribution."""
    distribution: Literal[DistributionType.LOG_NORMAL]
    mu: float = Field(description="Location parameter")
    sigma: float = Field(gt=0, description="Scale parameter")


# Union type for all possible distribution configurations
DistributionParams = Union[
    WeibullParams,
    ExponentialParams,
    GammaParams,
    LogNormalParams
]


class CovariateConfig(BaseModel):
    """Configuration for generating covariate data.
    
    Attributes:
        name: Name of the covariate
        distribution: Distribution type for generating values
        params: Parameters for the distribution
        effect_size: Effect size (coefficient) in the hazard function
    """
    name: str = Field(description="Name of the covariate")
    distribution: Literal["normal", "uniform", "binary"] = Field(
        description="Distribution type for the covariate"
    )
    params: Dict[str, float] = Field(
        description="Distribution parameters (e.g., mean and std for normal)"
    )
    effect_size: float = Field(
        description="Effect size in the hazard function"
    )


def generate_covariate_data(config: CovariateConfig, size: int) -> np.ndarray:
    """Generate covariate data based on configuration.
    
    Args:
        config: Configuration for the covariate
        size: Number of samples to generate
    
    Returns:
        Array of generated covariate values
    """
    if config.distribution == "normal":
        mean = config.params.get("mean", 0)
        std = config.params.get("std", 1)
        return np.random.normal(mean, std, size)
    elif config.distribution == "uniform":
        low = config.params.get("low", 0)
        high = config.params.get("high", 1)
        return np.random.uniform(low, high, size)
    elif config.distribution == "binary":
        p = config.params.get("p", 0.5)
        return np.random.binomial(1, p, size)
    else:
        raise ValueError(f"Unsupported covariate distribution: {config.distribution}")


def generate_survival_times(params: DistributionParams) -> tuple[np.ndarray, np.ndarray]:
    """Generate survival time data based on distribution parameters.
    
    Args:
        params: Parameters for the survival time distribution
    
    Returns:
        Tuple of (observed_times, event_status)
    """
    size = params.sample_size
    
    # Generate survival times based on distribution type
    if isinstance(params, WeibullParams):
        times = np.random.weibull(params.shape, size) * params.scale
    elif isinstance(params, ExponentialParams):
        times = np.random.exponential(params.scale, size)
    elif isinstance(params, GammaParams):
        times = np.random.gamma(params.shape, params.scale, size)
    elif isinstance(params, LogNormalParams):
        times = np.random.lognormal(params.mu, params.sigma, size)
    else:
        raise ValueError(f"Unsupported distribution: {params.distribution}")
    
    # Generate censoring times from the same distribution family
    # but with a larger scale to ensure reasonable censoring
    if isinstance(params, WeibullParams):
        censoring_times = np.random.weibull(params.shape, size) * (params.scale * 1.5)
    elif isinstance(params, ExponentialParams):
        censoring_times = np.random.exponential(params.scale * 1.5, size)
    elif isinstance(params, GammaParams):
        censoring_times = np.random.gamma(params.shape, params.scale * 1.5, size)
    elif isinstance(params, LogNormalParams):
        censoring_times = np.random.lognormal(params.mu, params.sigma * 1.2, size)
    
    # Round times to integers
    times = np.round(times).astype(int)
    censoring_times = np.round(censoring_times).astype(int)
    
    # Determine event status based on whether event occurs before censoring
    event_status = (times <= censoring_times).astype(int)
    
    # Adjust observed times to be the minimum of event and censoring times
    observed_times = np.minimum(times, censoring_times)
    
    # Ensure minimum time is 1
    observed_times = np.maximum(observed_times, 1)
    
    return observed_times, event_status 