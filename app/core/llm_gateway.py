"""
LLM Gateway — calls a local Ollama instance running on the host machine.
No API key required. Ollama must be running on the host (ollama serve)
and reachable from inside Docker via host.docker.internal.
"""

import logging
import time
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

# host.docker.internal lets containers reach services running on the Mac host
OLLAMA_URL = "http://host.docker.internal:11434/api/chat"

# Ollama is local and free — cost is always zero
_COST_PER_TOKEN = 0.0


@dataclass
class LLMCallResult:
    output: str
    latency_seconds: float
    token_cost: float
    prompt_tokens: int
    completion_tokens: int


def call_llm(
    model_name: str,
    system_prompt: str,
    user_query: str,
    temperature: float = 0.0,
) -> LLMCallResult:
    # Strip any provider prefix if present, e.g. "ollama/llama3.2" -> "llama3.2"
    bare_model = model_name.split("/")[-1]

    payload = {
        "model": bare_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
        },
    }

    start = time.perf_counter()
    with httpx.Client(timeout=120.0) as client:
        response = client.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
    latency = time.perf_counter() - start

    data = response.json()
    output_text: str = data.get("message", {}).get("content", "")

    prompt_tokens: int = data.get("prompt_eval_count", 0)
    completion_tokens: int = data.get("eval_count", 0)

    logger.debug(
        "Ollama call model=%s latency=%.3fs tokens=%d",
        bare_model, latency, prompt_tokens + completion_tokens,
    )

    return LLMCallResult(
        output=output_text,
        latency_seconds=latency,
        token_cost=_COST_PER_TOKEN,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )