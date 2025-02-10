from typing import List, Optional, Dict
import numpy as np
from pydantic import BaseModel, Field

try:
    from ..shared.distributions import (
        DistributionParams,
        CovariateConfig,
        generate_survival_times
    )
    from .models import WeibullRNNData, WeibullRNNInput
except ImportError:
    from shared.distributions import (
        DistributionParams,
        CovariateConfig,
        generate_survival_times
    )
    from models import WeibullRNNData, WeibullRNNInput


class SequenceConfig(BaseModel):
    """Configuration for a single time series feature.
    
    Attributes:
        name: Name of the feature
        base_mean: Base mean value for the sequence
        base_std: Base standard deviation for the sequence
        trend_coefficient: Coefficient for linear trend
        seasonality_amplitude: Amplitude of seasonal component
        seasonality_period: Period of seasonal component
        noise_std: Standard deviation of random noise
    """
    name: str = Field(description="Name of the feature")
    base_mean: float = Field(description="Base mean value")
    base_std: float = Field(gt=0, description="Base standard deviation")
    trend_coefficient: float = Field(description="Trend coefficient")
    seasonality_amplitude: float = Field(ge=0, description="Seasonality amplitude")
    seasonality_period: int = Field(gt=0, description="Seasonality period")
    noise_std: float = Field(ge=0, description="Noise standard deviation")
    effect_on_survival: float = Field(description="Effect size on survival time")


class WeibullRNNConfig(BaseModel):
    """Configuration for generating Weibull RNN data.
    
    Attributes:
        name: Unique identifier for the configuration
        survival_distribution: Distribution parameters for survival times
        sequence_length: Length of time series sequences
        sequences: List of sequence configurations
    """
    name: str = Field(description="Unique identifier for the configuration")
    survival_distribution: DistributionParams
    sequence_length: int = Field(gt=0, description="Length of time series")
    sequences: List[SequenceConfig]


def generate_sequence(
    config: SequenceConfig,
    sequence_length: int,
    random_state: Optional[int] = None
) -> List[float]:
    """Generate a single time series sequence.
    
    Args:
        config: Configuration for the sequence
        sequence_length: Length of the sequence
        random_state: Random seed for reproducibility
    
    Returns:
        List of time series values
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Time points
    t = np.arange(sequence_length)
    
    # Base level with random starting point
    base = config.base_mean + np.random.normal(0, config.base_std)
    
    # Trend component
    trend = config.trend_coefficient * t
    
    # Seasonal component
    seasonal = config.seasonality_amplitude * np.sin(
        2 * np.pi * t / config.seasonality_period
    )
    
    # Random noise
    noise = np.random.normal(0, config.noise_std, sequence_length)
    
    # Combine components and ensure mean is close to base_mean
    sequence = base + trend + seasonal + noise
    
    # Adjust to maintain base_mean
    sequence = sequence - np.mean(sequence) + config.base_mean
    
    return sequence.tolist()


def generate_weibull_rnn_data(config: WeibullRNNConfig) -> WeibullRNNData:
    """Generate data for Weibull Time-To-Event RNN model.
    
    Args:
        config: Configuration for data generation
    
    Returns:
        WeibullRNNData containing generated sequences and survival data
    """
    size = config.survival_distribution.sample_size
    
    # Generate sequences first
    all_sequences = {}
    sequence_effects = np.zeros(size)
    
    for seq_config in config.sequences:
        sequences = []
        effects = np.zeros(size)
        
        for i in range(size):
            seq = generate_sequence(
                seq_config,
                config.sequence_length,
                random_state=i
            )
            sequences.append(seq)
            
            # Calculate effect based on sequence characteristics
            # Using mean value and trend as predictors
            mean_value = np.mean(seq)
            trend = (seq[-1] - seq[0]) / len(seq)
            effects[i] = seq_config.effect_on_survival * (mean_value + trend)
        
        all_sequences[seq_config.name] = sequences
        sequence_effects += effects
    
    # Generate base survival times
    base_config = config.survival_distribution.model_copy(deep=True)
    base_times, event_status = generate_survival_times(base_config)
    
    # Adjust times based on sequence effects
    # Using exponential effect to ensure positive times
    adjusted_times = np.round(base_times * np.exp(sequence_effects)).astype(int)
    adjusted_times = np.maximum(adjusted_times, 1)  # Ensure minimum time is 1
    
    # Create WeibullRNNInput objects
    rnn_inputs = []
    for i in range(size):
        sequence_data = {
            name: sequences[i]
            for name, sequences in all_sequences.items()
        }
        
        rnn_inputs.append(
            WeibullRNNInput(
                entity_id=i + 1,
                event_time=adjusted_times[i],
                event_status=event_status[i],
                sequence_data=sequence_data
            )
        )
    
    return WeibullRNNData(data=rnn_inputs)


def plot_weibull_rnn_data(data: WeibullRNNData, title: Optional[str] = None) -> None:
    """Plot the generated sequences and survival data.
    
    Args:
        data: Generated WeibullRNNData
        title: Optional title for the plot
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Extract data
    times = [d.event_time for d in data.data]
    status = [d.event_status for d in data.data]
    sequences = {
        name: [d.sequence_data[name] for d in data.data]
        for name in data.data[0].sequence_data.keys()
    }
    
    # Create figure
    n_sequences = len(sequences)
    fig, axes = plt.subplots(n_sequences + 2, 1, figsize=(12, 4 * (n_sequences + 2)))
    
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
    
    # Plot sequences
    for i, (name, sequence_list) in enumerate(sequences.items()):
        ax = axes[i + 2]
        
        # Plot first few sequences
        for seq in sequence_list[:5]:
            ax.plot(seq, alpha=0.5)
        
        ax.set_title(f"Example Sequences: {name}")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Value")
    
    if title:
        fig.suptitle(title, fontsize=14)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from shared.distributions import WeibullParams, DistributionType
    
    # Example configuration
    config = WeibullRNNConfig(
        name="Test Config",
        survival_distribution=WeibullParams(
            distribution=DistributionType.WEIBULL,
            sample_size=1000,
            censoring_rate=0.3,
            shape=1.5,
            scale=100
        ),
        sequence_length=52,  # Weekly data for a year
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
            ),
            SequenceConfig(
                name="native_posts_ratio",
                base_mean=0.3,
                base_std=0.1,
                trend_coefficient=0.01,
                seasonality_amplitude=0.1,
                seasonality_period=13,
                noise_std=0.05,
                effect_on_survival=0.7
            )
        ]
    )
    
    # Generate and plot data
    data = generate_weibull_rnn_data(config)
    plot_weibull_rnn_data(data, "Weibull RNN Data Example") 