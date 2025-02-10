from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2
from lifelines import CoxPHFitter
import pandas as pd
from pydantic import BaseModel, Field

try:
    from .models import CoxData, CoxInput, CoxOutput, CoxResult, CoxPrediction
except ImportError:
    from models import CoxData, CoxInput, CoxOutput, CoxResult, CoxPrediction


class CoxModel:
    """Implementation of Cox Proportional Hazards model.
    
    This class provides methods for fitting the Cox model, calculating hazard ratios,
    and making predictions. It uses both a custom implementation and lifelines library
    for validation.
    """
    
    def __init__(self):
        """Initialize the Cox model."""
        self.coef_: Optional[Dict[str, float]] = None
        self.baseline_hazard_: Optional[float] = None
        self.log_likelihood_: Optional[float] = None
        self.concordance_index_: Optional[float] = None
        self._lifelines_model = CoxPHFitter()
    
    def _prepare_data(self, data: CoxData) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Prepare data for model fitting.
        
        Args:
            data: Input data in CoxData format
        
        Returns:
            Tuple of (X, T, E, feature_names) where:
                X: Covariate matrix
                T: Event/censoring times
                E: Event indicators
                feature_names: List of covariate names
        """
        # Extract feature names from first record
        feature_names = list(data.data[0].covariates.keys())
        
        # Create arrays
        n_samples = len(data.data)
        n_features = len(feature_names)
        X = np.zeros((n_samples, n_features))
        T = np.zeros(n_samples)
        E = np.zeros(n_samples)
        
        # Fill arrays
        for i, record in enumerate(data.data):
            T[i] = record.event_time
            E[i] = record.event_status
            for j, feature in enumerate(feature_names):
                X[i, j] = record.covariates[feature]
        
        return X, T, E, feature_names
    
    def _negative_log_likelihood(
        self,
        beta: np.ndarray,
        X: np.ndarray,
        T: np.ndarray,
        E: np.ndarray,
        alpha: float = 0.1  # L2 regularization strength
    ) -> float:
        """Calculate negative log likelihood for optimization.
        
        Args:
            beta: Model coefficients
            X: Covariate matrix
            T: Event/censoring times
            E: Event indicators
            alpha: L2 regularization strength
        
        Returns:
            Negative log likelihood value
        """
        # Sort by time for proper risk set calculation
        order = np.argsort(T)
        X = X[order]
        T = T[order]
        E = E[order]
        
        # Calculate linear predictors for all samples
        eta = np.dot(X, beta)  # Linear predictor
        risk_scores = np.exp(eta)
        
        # Initialize log likelihood
        log_lik = 0
        
        # Get unique event times and their indices
        unique_times, counts = np.unique(T[E == 1], return_counts=True)
        
        # For each unique event time
        for t, d in zip(unique_times, counts):
            # Find indices of samples in risk set (those with time >= t)
            risk_set = T >= t
            
            # Calculate sum of risk scores in risk set
            risk_sum = np.sum(risk_scores[risk_set])
            
            # Find samples that had an event at time t
            events_at_t = (T == t) & (E == 1)
            
            # Add to log likelihood:
            # For each event: add linear predictor - log(sum of risk scores)
            log_lik += np.sum(eta[events_at_t]) - d * np.log(risk_sum)
        
        # Add L2 regularization term
        log_lik -= (alpha / 2) * np.sum(beta ** 2)
        
        return -log_lik
    
    def _calculate_baseline_hazard(
        self,
        X: np.ndarray,
        T: np.ndarray,
        E: np.ndarray
    ) -> float:
        """Calculate baseline hazard rate.
        
        Args:
            X: Covariate matrix
            T: Event/censoring times
            E: Event indicators
        
        Returns:
            Baseline hazard rate
        """
        if self.coef_ is None:
            raise ValueError("Model must be fitted before calculating baseline hazard")
        
        # Sort by time
        order = np.argsort(T)
        X = X[order]
        T = T[order]
        E = E[order]
        
        beta = np.array(list(self.coef_.values()))
        risk_scores = np.exp(np.dot(X, beta))
        
        # Calculate baseline hazard at each event time
        unique_times = np.unique(T[E == 1])
        hazards = []
        
        for t in unique_times:
            # Find risk set
            risk_set = T >= t
            events_at_t = (T == t) & (E == 1)
            
            # Calculate hazard
            n_events = np.sum(events_at_t)
            risk_sum = np.sum(risk_scores[risk_set])
            
            if risk_sum > 0:
                hazards.append(n_events / risk_sum)
        
        # Return average baseline hazard
        return np.mean(hazards) if hazards else 0.0
    
    def _calculate_concordance_index(
        self,
        X: np.ndarray,
        T: np.ndarray,
        E: np.ndarray
    ) -> float:
        """Calculate concordance index (C-index).
        
        Args:
            X: Covariate matrix
            T: Event/censoring times
            E: Event indicators
        
        Returns:
            Concordance index value
        """
        if self.coef_ is None:
            raise ValueError("Model must be fitted before calculating concordance index")
        
        beta = np.array(list(self.coef_.values()))
        risk_scores = np.dot(X, beta)
        
        concordant = 0
        total = 0
        
        # Compare all pairs
        for i in range(len(T)):
            if E[i]:  # Only consider pairs where first subject had an event
                for j in range(len(T)):
                    if T[j] > T[i]:  # Second subject survived longer
                        total += 1
                        if risk_scores[i] > risk_scores[j]:  # Concordant pair
                            concordant += 1
        
        return concordant / total if total > 0 else 0.0
    
    def fit(self, data: CoxData) -> CoxResult:
        """Fit the Cox Proportional Hazards model.
        
        Args:
            data: Training data in CoxData format
        
        Returns:
            CoxResult containing fitted model parameters and metrics
        """
        # Prepare data
        X, T, E, feature_names = self._prepare_data(data)
        
        # Initial coefficients
        initial_beta = np.zeros(len(feature_names))
        
        # Optimize negative log likelihood with bounds to prevent extreme values
        bounds = [(-10, 10) for _ in range(len(feature_names))]  # Reasonable coefficient range
        result = minimize(
            self._negative_log_likelihood,
            initial_beta,
            args=(X, T, E),
            method='L-BFGS-B',
            bounds=bounds
        )
        
        # Store coefficients
        self.coef_ = dict(zip(feature_names, result.x))
        
        # Calculate baseline hazard
        self.baseline_hazard_ = self._calculate_baseline_hazard(X, T, E)
        
        # Calculate log likelihood and concordance index
        self.log_likelihood_ = -result.fun
        self.concordance_index_ = self._calculate_concordance_index(X, T, E)
        
        # Create output
        model_output = CoxOutput(
            covariate_coefficients=self.coef_,
            baseline_hazard=self.baseline_hazard_,
            log_likelihood=self.log_likelihood_,
            concordance_index=self.concordance_index_
        )
        
        # Fit lifelines model for validation
        df = pd.DataFrame(
            X,
            columns=feature_names
        )
        df['duration'] = T
        df['event'] = E
        self._lifelines_model.fit(df, 'duration', 'event')
        
        return CoxResult(model=model_output)
    
    def predict(self, data: CoxData) -> CoxResult:
        """Make predictions for new data.
        
        Args:
            data: Test data in CoxData format
        
        Returns:
            CoxResult containing model parameters and predictions
        """
        if self.coef_ is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # Prepare data
        X, T, E, feature_names = self._prepare_data(data)
        
        # Calculate risk scores
        beta = np.array([self.coef_[f] for f in feature_names])
        risk_scores = np.exp(np.dot(X, beta))
        
        # Calculate survival probabilities
        baseline_survival = np.exp(-self.baseline_hazard_)
        survival_probs = baseline_survival ** risk_scores
        
        # Create predictions
        predictions = []
        for i, record in enumerate(data.data):
            predictions.append(
                CoxPrediction(
                    entity_id=record.entity_id,
                    survival_probability=float(survival_probs[i]),
                    hazard_ratio=float(risk_scores[i])
                )
            )
        
        # Create output
        model_output = CoxOutput(
            covariate_coefficients=self.coef_,
            baseline_hazard=self.baseline_hazard_,
            log_likelihood=self.log_likelihood_,
            concordance_index=self.concordance_index_
        )
        
        return CoxResult(
            model=model_output,
            predictions=predictions
        )


if __name__ == "__main__":
    from generate import CoxCohortConfig, generate_cox_data
    from shared.distributions import WeibullParams, DistributionType, CovariateConfig
    
    # Generate example data
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
    
    # Generate data
    data = generate_cox_data(config)
    
    # Create and fit model
    model = CoxModel()
    result = model.fit(data)
    
    # Print results
    print("Model Results:")
    print(f"Coefficients: {result.model.covariate_coefficients}")
    print(f"Baseline Hazard: {result.model.baseline_hazard:.4f}")
    print(f"Log Likelihood: {result.model.log_likelihood:.4f}")
    print(f"Concordance Index: {result.model.concordance_index:.4f}")
    
    # Make predictions
    predictions = model.predict(data)
    print("\nExample Predictions:")
    for pred in predictions.predictions[:5]:
        print(f"Entity {pred.entity_id}:")
        print(f"  Survival Probability: {pred.survival_probability:.4f}")
        print(f"  Hazard Ratio: {pred.hazard_ratio:.4f}") 