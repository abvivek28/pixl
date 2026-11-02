"""Small HTTP client for Meta Model API structured chat completions."""
from __future__ import annotations

import os
import httpx

META_API_BASE_URL = "https://api.meta.ai/v1"
DEFAULT_META_MODEL = "muse-spark-1.3-contributor"


class MetaModelAPI:
    def __init__(self, api_key: str | None = None, model: str | None = None, *,
                 client: httpx.Client | None = None, timeout_seconds: float = 120.0):
        self.api_key = api_key if api_key is not None else os.getenv("MODEL_API_KEY", "")
        self.model = model or os.getenv("META_MODEL", DEFAULT_META_MODEL)
        self.client = client or httpx.Client(timeout=httpx.Timeout(
            connect=10.0, read=timeout_seconds, write=30.0, pool=10.0
        ))
        self.timeout_seconds = timeout_seconds

    def complete_json(self, *, messages: list[dict], schema_name: str, schema: dict) -> str:
        if not self.api_key:
            raise RuntimeError(
                "MODEL_API_KEY is not configured. Add your Meta Model API key to backend/.env."
            )
        body = {
            "model": self.model,
            "messages": messages,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": schema},
            },
        }
        try:
            response = self.client.post(
                f"{META_API_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json=body,
                timeout=self.timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            raise TimeoutError("Meta planning timed out. Retry planning or continue without the optional plan.") from exc
        except httpx.TransportError as exc:
            raise RuntimeError("Could not connect to Meta Model API. Check the network and try again.") from exc

        if response.is_error:
            try:
                payload = response.json()
                message = str((payload.get("error") or {}).get("message", "")) if isinstance(payload, dict) else ""
            except ValueError:
                message = ""
            detail = f"Meta Model API returned HTTP {response.status_code}"
            if message:
                detail += f": {message[:500]}"
            raise RuntimeError(detail)
        try:
            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise TypeError("Expected message content to be a JSON string")
            return content
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise RuntimeError("Meta returned an unexpected chat-completion response shape.") from exc
