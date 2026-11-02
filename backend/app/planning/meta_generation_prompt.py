"""Optional Meta Muse Spark prompt enhancement for text-to-image generation."""
from __future__ import annotations

import json
import time
from threading import Lock
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.planning.meta_api import MetaModelAPI


class GenerationQuestion(BaseModel):
    id: str
    question: str
    reason: str
    options: list[str] = Field(min_length=2, max_length=4)

    @field_validator("options")
    @classmethod
    def unique_options(cls, value):
        if len(set(value)) != len(value):
            raise ValueError("Question options must be unique")
        return value


class GenerationPromptPlan(BaseModel):
    status: Literal["needs_answers", "ready"]
    improved_prompt: str
    questions: list[GenerationQuestion] = Field(default_factory=list, max_length=3)
    detected_details: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    message: str


class GenerationPromptEnhancer:
    """Text-only Meta prompt designer; asks only high-impact missing questions."""
    def __init__(self, api_key: str | None = None, model: str | None = None, *,
                 client=None, min_interval_seconds: float = 1.0,
                 read_timeout_seconds: float = 120.0, max_attempts: int = 2):
        self.api = MetaModelAPI(api_key, model, client=client, timeout_seconds=read_timeout_seconds)
        self.model = self.api.model
        self.max_attempts = max(1, max_attempts)
        self.min_interval_seconds = min_interval_seconds
        self._last_call = 0.0
        self._lock = Lock()

    def _throttle(self):
        with self._lock:
            now = time.monotonic()
            wait = self.min_interval_seconds - (now - self._last_call)
            if self._last_call and wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()

    def plan(self, user_prompt: str, answers: list[dict[str, str]] | None = None) -> GenerationPromptPlan:
        user_prompt = user_prompt.strip()
        if not self.api.api_key:
            raise RuntimeError("MODEL_API_KEY is not configured. Prompt improvement is optional; you can generate directly with your original prompt.")
        schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["needs_answers", "ready"]},
                "improved_prompt": {"type": "string"},
                "questions": {"type": "array", "maxItems": 3, "items": {"type": "object", "properties": {
                    "id": {"type": "string"}, "question": {"type": "string"}, "reason": {"type": "string"},
                    "options": {"type": "array", "minItems": 2, "maxItems": 4, "items": {"type": "string"}}
                }, "required": ["id", "question", "reason", "options"], "additionalProperties": False}},
                "detected_details": {"type": "array", "items": {"type": "string"}},
                "assumptions": {"type": "array", "items": {"type": "string"}},
                "message": {"type": "string"}
            },
            "required": ["status", "improved_prompt", "questions", "detected_details", "assumptions", "message"],
            "additionalProperties": False,
        }
        context = json.dumps(answers or [], ensure_ascii=False)
        instruction = f"""You are Pixl's optional text-to-image prompt designer. Turn the user's idea into a detailed, coherent prompt for a text-to-image model. Do not change the user's core intent or invent named entities/brand identities.

Evaluate only useful dimensions: subject and defining attributes; action/pose; environment; composition/framing; medium/style; lighting; palette/mood/materials; focus/depth of field; aspect ratio/orientation; quality and avoidances.
Infer sensible low-risk details and record them as assumptions. Ask at most 3 concise multiple-choice questions, only for high-impact ambiguities that materially change the image and are not answered by the prompt or prior answers. If sufficiently specified, return status='ready' and no questions. If questions remain, return status='needs_answers'. Always provide a useful improved_prompt even while questions remain. Incorporate answers and do not repeat answered questions. Keep the prompt concrete, visually specific, internally consistent and usually 80-180 words. Avoid unsupported model parameters.

Original user prompt:\n{user_prompt}

User answers (JSON):\n{context}

Return JSON matching the schema."""
        last_error = None
        for attempt in range(self.max_attempts):
            self._throttle()
            try:
                content = self.api.complete_json(
                    messages=[{"role": "user", "content": instruction}],
                    schema_name="pixl_generation_prompt_plan", schema=schema,
                )
                result = GenerationPromptPlan.model_validate_json(content)
                if result.status == "ready":
                    result.questions = []
                elif not result.questions:
                    result.status = "ready"
                return result
            except TimeoutError as exc:
                last_error = exc
                if attempt + 1 >= self.max_attempts:
                    raise TimeoutError(f"Meta prompt planning timed out after {self.max_attempts} attempt(s). Your original prompt is unchanged; retry or generate directly.") from exc
                time.sleep(min(2 ** attempt, 3))
            except Exception as exc:
                # Retry transport-level errors only; schema and HTTP errors are actionable as-is.
                import httpx
                if isinstance(exc, httpx.TransportError) and attempt + 1 < self.max_attempts:
                    last_error = exc
                    time.sleep(min(2 ** attempt, 3))
                    continue
                raise
        raise RuntimeError("Meta prompt planning did not return a response.") from last_error
