from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql://promptguard:promptguard@db:5432/promptguard"
    redis_url: str = "redis://redis:6379/0"
    celery_broker_url: str = "redis://redis:6379/0"
    celery_result_backend: str = "redis://redis:6379/0"

    # OpenAI API key
    openai_api_key: str = ""

    # Default model used when creating prompts without an explicit model_name.
    default_model: str = "gpt-4o-mini"

    # Phase 4 regression thresholds (relative, 0.0–1.0)
    regression_similarity_drop_threshold: float = 0.10   # fail if similarity drops > 10%
    regression_latency_spike_threshold: float = 0.50     # fail if avg latency rises > 50%
    regression_cost_spike_threshold: float = 0.20        # fail if total cost rises > 20%

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
