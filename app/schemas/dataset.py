from datetime import datetime
from pydantic import BaseModel

class DatasetCreate(BaseModel):
    input_query: str
    expected_output: str

class DatasetResponse(BaseModel):
    id: int
    input_query: str
    expected_output: str
    created_at: datetime
    model_config = {"from_attributes": True}