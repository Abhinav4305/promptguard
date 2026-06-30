from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.models.prompt import Prompt
from app.db.session import get_db
from app.schemas.prompt import PromptCreate, PromptResponse

router = APIRouter(prefix="/prompts", tags=["prompts"])

@router.post("/", response_model=PromptResponse, status_code=201)
def create_prompt(payload: PromptCreate, db: Session = Depends(get_db)):
    prompt = Prompt(**payload.model_dump())
    db.add(prompt)
    db.commit()
    db.refresh(prompt)
    return prompt

@router.get("/", response_model=list[PromptResponse])
def list_prompts(db: Session = Depends(get_db)):
    return db.query(Prompt).all()

@router.get("/{prompt_id}", response_model=PromptResponse)
def get_prompt(prompt_id: int, db: Session = Depends(get_db)):
    prompt = db.get(Prompt, prompt_id)
    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return prompt