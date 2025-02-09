from typing import List
from pydantic import BaseModel, Field, field_validator


class KaplanMeierInput(BaseModel):
    """Single data point for Kaplan-Meier analysis input.
    
    Attributes:
        entity_id: Unique identifier for the entity (e.g., customer ID)
        event_time: Time until event or censoring occurred
        event_status: Binary indicator (1 = event occurred, 0 = censored)
    """
    entity_id: int = Field(gt=0, description="Unique identifier for the entity")
    event_time: int = Field(ge=0, description="Time until event or censoring")
    event_status: int = Field(ge=0, le=1, description="Event status (1=event occurred, 0=censored)")
    
    @field_validator('event_time')
    def validate_event_time(cls, v: int) -> int:
        """Validate that event_time is non-negative."""
        if v < 0:
            raise ValueError("event_time must be non-negative")
        return v


class KaplanMeierOutput(BaseModel):
    """Single data point for Kaplan-Meier analysis output.
    
    Attributes:
        time: Time point at which survival probability is calculated
        probability: Estimated survival probability at the given time point
    """
    time: int = Field(ge=0, description="Time point")
    probability: float = Field(
        ge=0.0, 
        le=1.0, 
        description="Survival probability at this time point"
    )


class KaplanMeierData(BaseModel):
    """Container for multiple Kaplan-Meier input records.
    
    Attributes:
        data: List of KaplanMeierInput records
    """
    data: List[KaplanMeierInput]


class KaplanMeierResult(BaseModel):
    """Container for Kaplan-Meier analysis results.
    
    Attributes:
        estimates: List of survival probability estimates at different time points
    """
    estimates: List[KaplanMeierOutput] 