from typing import Literal, Union, Optional
from enum import Enum
import numpy as np
from pydantic import BaseModel, Field
try:
    from .models import KaplanMeierData, KaplanMeierInput
except ImportError:
    from models import KaplanMeierData, KaplanMeierInput
import matplotlib.pyplot as plt
import seaborn as sns


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


class CohortConfig(BaseModel):
    """Configuration for a single cohort."""
    name: str = Field(description="Unique identifier for the cohort")
    distribution_params: DistributionParams


def generate_survival_times(config: CohortConfig) -> KaplanMeierData:
    """Generate survival time data based on cohort configuration.
    
    Args:
        config: Configuration for the cohort including distribution parameters
    
    Returns:
        KaplanMeierData containing generated survival times and event indicators
    """
    params = config.distribution_params
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
    
    # Create KaplanMeierInput objects
    km_inputs = [
        KaplanMeierInput(
            entity_id=i + 1,
            event_time=t,
            event_status=s
        )
        for i, (t, s) in enumerate(zip(observed_times, event_status))
    ]
    
    return KaplanMeierData(data=km_inputs)


def generate_multi_cohort_data(*configs: CohortConfig) -> KaplanMeierData:
    """Generate survival time data for multiple cohorts.
    
    Args:
        *configs: Variable number of CohortConfig objects
    
    Returns:
        Combined KaplanMeierData for all cohorts
    """
    all_data = []
    base_id = 0
    
    for config in configs:
        cohort_data = generate_survival_times(config)
        # Adjust entity_ids to ensure uniqueness across cohorts
        for item in cohort_data.data:
            item.entity_id += base_id
        
        all_data.extend(cohort_data.data)
        base_id = max(item.entity_id for item in all_data)
    
    return KaplanMeierData(data=all_data)


def plot_survival_data(data: KaplanMeierData, title: Optional[str] = None) -> None:
    """Plot the survival data distribution with censoring status.
    
    Args:
        data: KaplanMeierData object containing survival times and event status
        title: Optional title for the plot
    """
    # Extract times and status
    times = [d.event_time for d in data.data]
    status = [d.event_status for d in data.data]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Distribution of event times
    sns.histplot(data=times, bins=30, ax=ax1)
    ax1.set_title("Distribution of Event Times")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Count")
    
    # Plot 2: Event times by status (censored vs event)
    colors = ['red' if s == 1 else 'blue' for s in status]
    
    ax2.scatter(times, range(len(times)), c=colors, alpha=0.5)
    ax2.set_title("Event Times by Status")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Index")
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
               label='Event', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
               label='Censored', markersize=10)
    ]
    ax2.legend(handles=legend_elements)
    
    if title:
        fig.suptitle(title, fontsize=14)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    config1 = CohortConfig(
        name="Power Users",
        distribution_params=WeibullParams(
            distribution=DistributionType.WEIBULL,
            sample_size=1000,
            censoring_rate=0.3,
            shape=1.5,
            scale=10
        )
    )

    config2 = CohortConfig(
        name="Churning Users",
        distribution_params=ExponentialParams(
            distribution=DistributionType.EXPONENTIAL,
            sample_size=1000,
            censoring_rate=0.3,
            scale=10
        )
    )

    # Generate and plot data for each cohort separately
    for config in [config1, config2]:
        data = generate_survival_times(config)
        plot_survival_data(data, f"Survival Data Distribution - {config.name}")

    # Plot combined data
    combined_data = generate_multi_cohort_data(config1, config2)
    plot_survival_data(combined_data, "Combined Cohorts Survival Data")
