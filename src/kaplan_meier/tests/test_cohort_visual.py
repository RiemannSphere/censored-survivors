"""Visual test for comparing survival curves between different user cohorts."""

from typing import Dict
from ..generate import (
    CohortConfig,
    WeibullParams,
    ExponentialParams,
    DistributionType,
    generate_survival_times
)
from ..run import plot_multiple_survival_curves
from ..models import KaplanMeierData


def test_cohort_comparison() -> None:
    """Visual test comparing survival curves of different user cohorts.
    
    This test generates and plots survival curves for three distinct user cohorts:
    1. Power Users: High retention, gradual decline (Weibull with high shape)
    2. Average Users: Medium retention, steady decline (Exponential)
    3. Churning Users: Low retention, steep initial drop (Weibull with low shape)
    
    The test helps validate that the Kaplan-Meier estimator correctly
    distinguishes between different user behaviors and produces visually
    distinct and interpretable survival curves.
    """
    # Define different cohort configurations
    cohorts: Dict[str, CohortConfig] = {
        "Power Users": CohortConfig(
            name="Power Users",
            distribution_params=WeibullParams(
                distribution=DistributionType.WEIBULL,
                sample_size=1000,
                censoring_rate=0.3,
                shape=2.5,  # Higher shape -> more gradual decline
                scale=20    # Higher scale -> longer survival times
            )
        ),
        "Average Users": CohortConfig(
            name="Average Users",
            distribution_params=ExponentialParams(
                distribution=DistributionType.EXPONENTIAL,
                sample_size=1000,
                censoring_rate=0.3,
                scale=10    # Moderate survival times
            )
        ),
        "Churning Users": CohortConfig(
            name="Churning Users",
            distribution_params=WeibullParams(
                distribution=DistributionType.WEIBULL,
                sample_size=1000,
                censoring_rate=0.3,
                shape=0.8,  # Lower shape -> steeper initial decline
                scale=5     # Lower scale -> shorter survival times
            )
        )
    }
    
    # Generate data for each cohort
    cohort_data: Dict[str, KaplanMeierData] = {
        name: generate_survival_times(config)
        for name, config in cohorts.items()
    }
    
    # Custom colors for each cohort
    cohort_colors = {
        "Power Users": "#2ecc71",     # Green
        "Average Users": "#3498db",    # Blue
        "Churning Users": "#e74c3c"    # Red
    }
    
    # Plot comparison of survival curves
    plot_multiple_survival_curves(
        cohort_data,
        title="Comparison of User Cohort Survival Curves",
        colors=cohort_colors
    )


if __name__ == "__main__":
    test_cohort_comparison() 