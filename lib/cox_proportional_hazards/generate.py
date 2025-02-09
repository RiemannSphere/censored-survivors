from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pydantic import BaseModel, Field

try:
    from ..shared.distributions import (
        DistributionParams,
        CovariateConfig,
        generate_survival_times,
        generate_covariate_data
    )
    from .models import CoxData, CoxInput
except ImportError:
    from shared.distributions import (
        DistributionParams,
        CovariateConfig,
        generate_survival_times,
        generate_covariate_data
    )
    from models import CoxData, CoxInput


class CoxCohortConfig(BaseModel):
    """Configuration for a single Cox model cohort.
    
    Attributes:
        name: Unique identifier for the cohort
        survival_distribution: Distribution parameters for survival times
        covariates: List of covariate configurations
    """
    name: str = Field(description="Unique identifier for the cohort")
    survival_distribution: "DistributionParams"
    covariates: List["CovariateConfig"]


def generate_cox_data(config: CoxCohortConfig) -> CoxData:
    """Generate survival and covariate data for Cox Proportional Hazards model.
    
    Args:
        config: Configuration for the cohort including survival and covariate parameters
    
    Returns:
        CoxData containing generated survival times, event indicators, and covariates
    """
    size = config.survival_distribution.sample_size
    
    # Generate covariates first
    covariates_dict = {}
    linear_predictor = np.zeros(size)
    
    # Scale factor for effect sizes
    # Using 0.5 to get coefficient estimates closer to true values
    scale_factor = 0.5
    
    for cov_config in config.covariates:
        # Generate covariate values
        values = generate_covariate_data(cov_config, size)
        
        # Normalize values to [-0.5, 0.5] range for better coefficient estimation
        values = values - np.mean(values)
        values = values / (4 * np.std(values))  # Divide by 4*std to keep values in [-0.5, 0.5]
        
        covariates_dict[cov_config.name] = values
        
        # Add contribution to linear predictor
        linear_predictor += scale_factor * cov_config.effect_size * values
    
    # Generate base survival times with adjusted scale
    base_config = config.survival_distribution.model_copy(deep=True)
    if hasattr(base_config, 'scale'):
        base_config.scale *= 2  # Double the scale for more spread
    base_times, _ = generate_survival_times(base_config)
    
    # Apply covariate effects to survival times
    hazard_ratios = np.exp(linear_predictor)
    observed_times = np.round(base_times / hazard_ratios).astype(int)
    observed_times = np.maximum(observed_times, 1)  # Ensure minimum time is 1
    
    # Generate censoring times with higher scale
    censoring_config = config.survival_distribution.model_copy(deep=True)
    if hasattr(censoring_config, 'scale'):
        censoring_config.scale *= 3  # Triple the scale for censoring times
    censoring_times, _ = generate_survival_times(censoring_config)
    censoring_times = np.round(censoring_times).astype(int)
    
    # Determine event status
    event_status = (observed_times <= censoring_times).astype(int)
    
    # Use censoring time if event is censored
    final_times = np.where(event_status == 1, observed_times, censoring_times)
    
    # Create CoxInput objects
    cox_inputs = []
    for i in range(size):
        covariates = {name: values[i] for name, values in covariates_dict.items()}
        cox_inputs.append(
            CoxInput(
                entity_id=i + 1,
                event_time=final_times[i],
                event_status=event_status[i],
                covariates=covariates
            )
        )
    
    return CoxData(data=cox_inputs)


def generate_multi_cohort_data(*configs: CoxCohortConfig) -> CoxData:
    """Generate survival and covariate data for multiple cohorts.
    
    Args:
        *configs: Variable number of CoxCohortConfig objects
    
    Returns:
        Combined CoxData for all cohorts
    """
    all_data = []
    base_id = 0
    
    for config in configs:
        cohort_data = generate_cox_data(config)
        # Adjust entity_ids to ensure uniqueness across cohorts
        for item in cohort_data.data:
            item.entity_id += base_id
        
        all_data.extend(cohort_data.data)
        base_id = max(item.entity_id for item in all_data)
    
    return CoxData(data=all_data)


def plot_cox_data(data: CoxData, title: Optional[str] = None) -> None:
    """Plot the Cox model data distribution with covariates.
    
    Args:
        data: CoxData object containing survival times, event status, and covariates
        title: Optional title for the plot
    """
    # Extract data
    times = [d.event_time for d in data.data]
    status = [d.event_status for d in data.data]
    covariates = {
        name: [d.covariates[name] for d in data.data]
        for name in data.data[0].covariates.keys()
    }
    
    # Calculate number of subplots needed
    n_covariates = len(covariates)
    n_cols = 3
    n_rows = (n_covariates + 3) // n_cols  # +3 for survival time plots
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()
    
    # Plot 1: Distribution of event times
    sns.histplot(data=times, bins=30, ax=axes[0])
    axes[0].set_title("Distribution of Event Times")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Count")
    
    # Plot 2: Event times by status
    colors = ['red' if s == 1 else 'blue' for s in status]
    axes[1].scatter(times, range(len(times)), c=colors, alpha=0.5)
    axes[1].set_title("Event Times by Status")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Index")
    
    # Add legend for status
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
               label='Event', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
               label='Censored', markersize=10)
    ]
    axes[1].legend(handles=legend_elements)
    
    # Plot 3: Covariate distributions
    for i, (name, values) in enumerate(covariates.items()):
        sns.histplot(data=values, bins=30, ax=axes[i + 2])
        axes[i + 2].set_title(f"Distribution of {name}")
        axes[i + 2].set_xlabel("Value")
        axes[i + 2].set_ylabel("Count")
    
    # Remove empty subplots
    for i in range(n_covariates + 2, len(axes)):
        fig.delaxes(axes[i])
    
    if title:
        fig.suptitle(title, fontsize=14)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from shared.distributions import WeibullParams, DistributionType
    
    # Example configuration
    config = CoxCohortConfig(
        name="Test Cohort",
        survival_distribution=WeibullParams(
            distribution=DistributionType.WEIBULL,
            sample_size=1000,
            censoring_rate=0.3,
            shape=1.5,
            scale=10
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
    
    # Generate and plot data
    data = generate_cox_data(config)
    plot_cox_data(data, "Cox Model Data Distribution") 