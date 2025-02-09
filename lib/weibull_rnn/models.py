from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class WeibullRNNInput(BaseModel):
    """Single data point for Weibull Time-To-Event RNN analysis input.
    
    Attributes:
        entity_id: Unique identifier for the entity (e.g., customer ID)
        event_time: Time until event or censoring occurred
        event_status: Binary indicator (1 = event occurred, 0 = censored)
        sequence_data: Dictionary of feature names and their time series values
    """
    entity_id: int = Field(gt=0, description="Unique identifier for the entity")
    event_time: int = Field(ge=0, description="Time until event or censoring")
    event_status: int = Field(ge=0, le=1, description="Event status (1=event occurred, 0=censored)")
    sequence_data: Dict[str, List[float]] = Field(
        description="Dictionary of feature names and their time series values"
    )


class WeibullParameters(BaseModel):
    """Weibull distribution parameters for each time step.
    
    Attributes:
        shape: Shape parameter (k) of the Weibull distribution
        scale: Scale parameter (λ) of the Weibull distribution
    """
    shape: float = Field(gt=0.0, description="Shape parameter (k)")
    scale: float = Field(gt=0.0, description="Scale parameter (λ)")


class WeibullRNNOutput(BaseModel):
    """Output for Weibull Time-To-Event RNN analysis.
    
    Attributes:
        weibull_params: Predicted Weibull parameters
        predicted_time: Expected time-to-event
        survival_probability: Survival probability at predicted time
        log_likelihood: Log likelihood of the prediction
    """
    weibull_params: WeibullParameters
    predicted_time: float = Field(ge=0.0, description="Expected time-to-event")
    survival_probability: float = Field(
        ge=0.0,
        le=1.0,
        description="Survival probability at predicted time"
    )
    log_likelihood: Optional[float] = Field(
        default=None,
        description="Log likelihood of the prediction"
    )


class WeibullRNNData(BaseModel):
    """Container for multiple Weibull Time-To-Event RNN input records.
    
    Attributes:
        data: List of WeibullRNNInput records
    """
    data: List[WeibullRNNInput]


class WeibullRNNPrediction(BaseModel):
    """Prediction output for a single entity.
    
    Attributes:
        entity_id: Identifier of the entity
        predicted_time: Predicted time-to-event
        survival_probability: Survival probability at predicted time
        weibull_params: Predicted Weibull parameters
    """
    entity_id: int = Field(gt=0, description="Entity identifier")
    predicted_time: float = Field(ge=0.0, description="Predicted time-to-event")
    survival_probability: float = Field(
        ge=0.0,
        le=1.0,
        description="Survival probability at predicted time"
    )
    weibull_params: WeibullParameters


class WeibullRNNResult(BaseModel):
    """Container for Weibull Time-To-Event RNN analysis results.
    
    Attributes:
        model_metrics: Overall model metrics
        predictions: Optional list of predictions for entities
    """
    model_metrics: WeibullRNNOutput
    predictions: Optional[List[WeibullRNNPrediction]] = None 