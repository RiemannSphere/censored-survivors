from typing import Dict, List, Union, Optional
from pydantic import BaseModel, Field, field_validator


class CoxInput(BaseModel):
    """Single data point for Cox Proportional Hazards analysis input.
    
    Attributes:
        entity_id: Unique identifier for the entity (e.g., customer ID)
        event_time: Time until event or censoring occurred
        event_status: Binary indicator (1 = event occurred, 0 = censored)
        covariates: Dictionary of covariate names and their values
    """
    entity_id: int = Field(gt=0, description="Unique identifier for the entity")
    event_time: int = Field(ge=0, description="Time until event or censoring")
    event_status: int = Field(ge=0, le=1, description="Event status (1=event occurred, 0=censored)")
    covariates: Dict[str, float] = Field(
        description="Dictionary of covariates and their values"
    )
    
    @field_validator('event_time')
    def validate_event_time(cls, v: int) -> int:
        """Validate that event_time is non-negative."""
        if v < 0:
            raise ValueError("event_time must be non-negative")
        return v


class CoxOutput(BaseModel):
    """Output for Cox Proportional Hazards analysis.
    
    Attributes:
        covariate_coefficients: Dictionary of covariate names and their coefficients
        baseline_hazard: Baseline hazard rate at time t
        log_likelihood: Log likelihood of the fitted model
        concordance_index: C-index measuring model discrimination
    """
    covariate_coefficients: Dict[str, float] = Field(
        description="Coefficients for each covariate"
    )
    baseline_hazard: float = Field(
        ge=0.0,
        description="Baseline hazard rate"
    )
    log_likelihood: Optional[float] = Field(
        default=None,
        description="Log likelihood of the fitted model"
    )
    concordance_index: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="C-index (concordance index) of the model"
    )


class CoxData(BaseModel):
    """Container for multiple Cox Proportional Hazards input records.
    
    Attributes:
        data: List of CoxInput records
    """
    data: List[CoxInput]


class CoxPrediction(BaseModel):
    """Prediction output for a single entity.
    
    Attributes:
        entity_id: Identifier of the entity
        survival_probability: Predicted survival probability at specified time
        hazard_ratio: Hazard ratio for this entity relative to baseline
    """
    entity_id: int = Field(gt=0, description="Entity identifier")
    survival_probability: float = Field(
        ge=0.0,
        le=1.0,
        description="Predicted survival probability"
    )
    hazard_ratio: float = Field(
        ge=0.0,
        description="Hazard ratio relative to baseline"
    )


class CoxResult(BaseModel):
    """Container for Cox Proportional Hazards analysis results.
    
    Attributes:
        model: Fitted model parameters and metrics
        predictions: Optional list of predictions for entities
    """
    model: CoxOutput
    predictions: Optional[List[CoxPrediction]] = None 