from pydantic import BaseModel
from typing import List, Optional


class PredictRequest(BaseModel):
    source: str = "mock"
    snr: Optional[float] = None


class PredictResponse(BaseModel):
    estimated_k: int
    doas_deg: List[float]
    confidence: float
    spectrum: List[float]
    message: str