from __future__ import annotations

from app import api_v3
from app.main import app
from app.planning.meta_generation_prompt import GenerationPromptPlan, GenerationQuestion
from fastapi.testclient import TestClient


def test_generation_prompt_plan_and_answer_flow(monkeypatch, tmp_path):
    class StubEnhancer:
        def __init__(self):
            self.calls = []

        def plan(self, prompt, answers=None):
            self.calls.append((prompt, answers or []))
            if not answers:
                return GenerationPromptPlan(
                    status="needs_answers",
                    improved_prompt="A cinematic photograph of a ceramic vase on a stone plinth.",
                    questions=[GenerationQuestion(id="style", question="Which visual style?", reason="Style changes the entire image.", options=["Photorealistic product photography", "Editorial illustration"])],
                    detected_details=["Main subject: ceramic vase"],
                    assumptions=["Use a restrained composition"],
                    message="A couple of details could materially improve the result.",
                )
            return GenerationPromptPlan(
                status="ready",
                improved_prompt="Photorealistic product photograph of a ceramic vase on a limestone plinth, soft side light, neutral palette, clean editorial composition.",
                questions=[], detected_details=["Ceramic vase", "Product photography"],
                assumptions=["Neutral palette"], message="Prompt is ready to review.",
            )

    stub = StubEnhancer()
    monkeypatch.setattr(api_v3, "_generation_enhancer", stub)
    monkeypatch.setattr(api_v3, "_generation_plans", {})
    client = TestClient(app)

    planned = client.post("/api/generation-plan", json={"prompt": "A ceramic vase"})
    assert planned.status_code == 200, planned.text
    body = planned.json()
    assert body["status"] == "needs_answers"
    assert body["questions"][0]["id"] == "style"

    answered = client.post("/api/generation-answer", json={
        "plan_id": body["plan_id"],
        "answers": [{"id": "style", "question": "Which visual style?", "answer": "Photorealistic product photography"}],
    })
    assert answered.status_code == 200, answered.text
    assert answered.json()["status"] == "ready"
    assert "Photorealistic product photograph" in answered.json()["improved_prompt"]
    assert stub.calls[1][0] == "A ceramic vase"
    assert stub.calls[1][1][0]["answer"] == "Photorealistic product photography"


def test_generation_answer_unknown_plan_returns_404(monkeypatch):
    monkeypatch.setattr(api_v3, "_generation_plans", {})
    response = TestClient(app).post("/api/generation-answer", json={"plan_id": "missing", "answers": [{"id": "x", "answer": "y"}]})
    assert response.status_code == 404


def test_generation_prompt_retries_transient_timeout_then_succeeds():
    import httpx
    from app.planning.meta_generation_prompt import GenerationPromptEnhancer

    class FakeResponse:
        is_error = False
        def json(self):
            return {"choices": [{"message": {"content": '{"status":"ready","improved_prompt":"A detailed image prompt.","questions":[],"detected_details":[],"assumptions":[],"message":"Ready."}'}}]}

    class FakeClient:
        def __init__(self):
            self.calls = 0
        def post(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise httpx.ReadTimeout("simulated timeout")
            return FakeResponse()

    client = FakeClient()
    enhancer = GenerationPromptEnhancer(api_key="test-key", model="test-model", client=client,
                                         min_interval_seconds=0, max_attempts=2)
    result = enhancer.plan("A vase")
    assert result.status == "ready"
    assert client.calls == 2


def test_generation_prompt_timeout_returns_actionable_http_504(monkeypatch):
    class TimeoutEnhancer:
        def plan(self, prompt, answers=None):
            raise TimeoutError("Meta prompt planning timed out; retry or generate directly.")

    monkeypatch.setattr(api_v3, "_generation_enhancer", TimeoutEnhancer())
    response = TestClient(app).post("/api/generation-plan", json={"prompt": "A vase"})
    assert response.status_code == 504
    assert "generate directly" in response.json()["detail"]
