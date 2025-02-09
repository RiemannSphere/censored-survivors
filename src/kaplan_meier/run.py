"""Kaplan-Meier survival analysis model implementation."""

from typing import List, Tuple, Optional, Dict
import numpy as np
import matplotlib.pyplot as plt
from .models import KaplanMeierData


def calculate_survival_curve(data: KaplanMeierData) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the Kaplan-Meier survival curve.
    
    Args:
        data: KaplanMeierData object containing survival times and event status
        
    Returns:
        Tuple containing:
            - Unique time points
            - Survival probabilities at each time point
    """
    # Extract times and status
    times = np.array([d.event_time for d in data.data])
    events = np.array([d.event_status for d in data.data])
    
    # Sort data by time
    sort_idx = np.argsort(times)
    times = times[sort_idx]
    events = events[sort_idx]
    
    # Get unique time points
    unique_times = np.unique(times)
    
    # Add time 0 if not present
    if unique_times[0] > 0:
        unique_times = np.concatenate(([0], unique_times))
    
    survival_prob = np.ones(len(unique_times))
    
    # Calculate survival probability
    n_subjects = len(times)
    
    for i, t in enumerate(unique_times):
        if t == 0:
            survival_prob[i] = 1.0
            continue
            
        # Count events and at-risk subjects at this time
        at_risk = np.sum(times >= t)  # Number still at risk
        events_at_t = np.sum(events[times == t])  # Number of events at this time
        
        # Calculate survival probability
        if at_risk > 0:
            survival_prob[i] = survival_prob[i-1] * (1 - events_at_t / at_risk)
        else:
            survival_prob[i] = survival_prob[i-1]  # If no subjects at risk, maintain previous probability
    
    return unique_times, survival_prob


def plot_survival_curve(
    data: KaplanMeierData,
    title: Optional[str] = None,
    show_censored: bool = True
) -> None:
    """Plot the Kaplan-Meier survival curve.
    
    Args:
        data: KaplanMeierData object containing survival times and event status
        title: Optional title for the plot
        show_censored: Whether to show censored points on the curve
    """
    times, survival_prob = calculate_survival_curve(data)
    
    plt.figure(figsize=(10, 6))
    
    # Plot survival curve
    plt.step(times, survival_prob, where='post', label='Survival Probability')
    
    if show_censored:
        # Add censored points
        censored_times = [d.event_time for d in data.data if d.event_status == 0]
        if censored_times:
            # Find survival probabilities at censored times
            censored_probs = []
            for t in censored_times:
                idx = np.searchsorted(times, t, side='right') - 1
                censored_probs.append(survival_prob[idx])
            
            plt.scatter(
                censored_times,
                censored_probs,
                marker='|',
                color='red',
                label='Censored',
                zorder=3
            )
    
    plt.grid(True, alpha=0.3)
    plt.xlabel('Time')
    plt.ylabel('Survival Probability')
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_multiple_survival_curves(
    cohort_data: Dict[str, KaplanMeierData],
    title: str = "Comparison of Survival Curves",
    show_censored: bool = True,
    colors: Optional[Dict[str, str]] = None
) -> None:
    """Plot multiple survival curves on the same graph for comparison.
    
    Args:
        cohort_data: Dictionary mapping cohort names to their KaplanMeierData
        title: Title for the plot
        show_censored: Whether to show censored points
        colors: Optional dictionary mapping cohort names to colors
    """
    plt.figure(figsize=(12, 8))
    
    if colors is None:
        # Default color cycle
        colors = {
            name: plt.cm.Set2(i) 
            for i, name in enumerate(cohort_data.keys())
        }
    
    for name, data in cohort_data.items():
        times, survival_prob = calculate_survival_curve(data)
        
        # Plot survival curve
        plt.step(
            times,
            survival_prob,
            where='post',
            label=name,
            color=colors.get(name),
            linewidth=2
        )
        
        if show_censored:
            # Add censored points
            censored_times = [d.event_time for d in data.data if d.event_status == 0]
            if censored_times:
                censored_probs = []
                for t in censored_times:
                    idx = np.searchsorted(times, t, side='right') - 1
                    censored_probs.append(survival_prob[idx])
                
                plt.scatter(
                    censored_times,
                    censored_probs,
                    marker='|',
                    color=colors.get(name),
                    alpha=0.5,
                    zorder=3
                )
    
    plt.grid(True, alpha=0.3)
    plt.xlabel('Time')
    plt.ylabel('Survival Probability')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
