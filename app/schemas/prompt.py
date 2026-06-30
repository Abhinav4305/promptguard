from datetime import datetime
from pydantic import BaseModel

class PromptCreate(BaseModel):
    version_tag: str
    prompt_text: str
    model_name: str
    model_config = {"protected_namespaces": ()}

class PromptResponse(BaseModel):
    id: int
    version_tag: str
    prompt_text: str
    model_name: str
    created_at: datetime
    model_config = {"from_attributes": True, "protected_namespaces": ()}