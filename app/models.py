from pydantic import BaseModel

from pydantic import BaseModel, Field

class CustomerFeatures(BaseModel):
    CreditScore: int = Field(..., ge=300, le=900)
    Age: int = Field(..., ge=18, le=100)
    Tenure: int = Field(..., ge=0, le=10)
    Balance: float = Field(..., ge=0)
    NumOfProducts: int = Field(..., ge=1, le=4)
    HasCrCard: int = Field(..., ge=0, le=1)
    IsActiveMember: int = Field(..., ge=0, le=1)
    EstimatedSalary: float = Field(..., ge=0)
    Geography_Germany: int = Field(..., ge=0, le=1)
    Geography_Spain: int = Field(..., ge=0, le=1)


class PredictionResponse(BaseModel):
    churn_probability: float
    prediction: int
    risk_level: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
