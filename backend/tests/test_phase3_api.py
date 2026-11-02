from __future__ import annotations

from io import BytesIO
import time

from fastapi.testclient import TestClient
from PIL import Image

import app.api_v3 as api_v3
from app.main import app
from app.planning.meta_planner import PlanningResult


def png_bytes(size=(80, 60), color=(120, 80, 40)):
    buf = BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def client_with_temp_dirs(monkeypatch, tmp_path):
    uploads = tmp_path / "uploads"
    jobs = tmp_path / "jobs"
    uploads.mkdir()
    jobs.mkdir()
    monkeypatch.setattr(api_v3, "UPLOADS_DIR", uploads)
    monkeypatch.setattr(api_v3, "JOBS_DIR", jobs)
    return TestClient(app)


def wait_for_job(client, job_id, timeout=5):
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200
        data = response.json()
        if data["status"] in {"completed", "failed"}:
            return data
        time.sleep(0.03)
    raise AssertionError("Job did not finish before timeout")


def test_upload_saves_normalized_master_and_serves_it(monkeypatch, tmp_path):
    client = client_with_temp_dirs(monkeypatch, tmp_path)
    response = client.post("/api/upload", files={"file": ("test.png", png_bytes(), "image/png")})
    assert response.status_code == 201, response.text
    record = response.json()
    assert record["image_id"]
    assert record["width"] == 80 and record["height"] == 60
    assert len(record["sha256"]) == 64
    fetched = client.get(record["url"])
    assert fetched.status_code == 200
    assert fetched.headers["content-type"].startswith("image/png")


def test_edit_job_returns_result_heatmap_receipt_and_original(monkeypatch, tmp_path):
    client = client_with_temp_dirs(monkeypatch, tmp_path)
    uploaded = client.post("/api/upload", files={"file": ("test.png", png_bytes(), "image/png")})
    image_id = uploaded.json()["image_id"]
    response = client.post("/api/edit", json={
        "image_id": image_id,
        "instruction": "Change the selected region",
        "box": [15, 10, 55, 50],
        "provider": "fake",
        "seed": 0,
    })
    assert response.status_code == 202, response.text
    job = wait_for_job(client, response.json()["job_id"])
    assert job["status"] == "completed", job
    assert {"original", "result", "heatmap", "receipt"}.issubset(job["artifacts"])
    receipt = client.get(job["artifacts"]["receipt"])
    assert receipt.status_code == 200
    assert receipt.json()["metrics"]["outside_changed_pixels"] == 0
    assert receipt.json()["boxes"]["selected"] == {"x1": 15, "y1": 10, "x2": 55, "y2": 50}
    assert receipt.json()["boxes"]["expanded"] == {"x1": 0, "y1": -5, "x2": 70, "y2": 65}
    assert client.get(job["artifacts"]["result"]).status_code == 200
    assert client.get(job["artifacts"]["heatmap"]).status_code == 200


def test_edit_requires_manual_box_until_coordinate_verification(monkeypatch, tmp_path):
    client = client_with_temp_dirs(monkeypatch, tmp_path)
    uploaded = client.post("/api/upload", files={"file": ("test.png", png_bytes(), "image/png")})
    response = client.post("/api/edit", json={
        "image_id": uploaded.json()["image_id"],
        "instruction": "Change the object",
    })
    assert response.status_code == 422
    assert "confirmed box" in response.json()["detail"]


def test_edit_rejects_out_of_bounds_box(monkeypatch, tmp_path):
    client = client_with_temp_dirs(monkeypatch, tmp_path)
    uploaded = client.post("/api/upload", files={"file": ("test.png", png_bytes(), "image/png")})
    response = client.post("/api/edit", json={
        "image_id": uploaded.json()["image_id"],
        "instruction": "Change the object",
        "box": [10, 10, 100, 50],
    })
    assert response.status_code == 422


def test_plan_and_answer_round_trip(monkeypatch, tmp_path):
    class StubPlanner:
        def plan(self, image, instruction):
            return PlanningResult(status="needs_choice", message="Choose a target")

    monkeypatch.setattr(api_v3, "MetaPlanner", StubPlanner)
    monkeypatch.setattr(api_v3, "_planner_instance", None)
    client = client_with_temp_dirs(monkeypatch, tmp_path)
    uploaded = client.post("/api/upload", files={"file": ("test.png", png_bytes(), "image/png")})
    planned = client.post("/api/plan", json={
        "image_id": uploaded.json()["image_id"], "instruction": "Change the cup color"
    })
    assert planned.status_code == 200, planned.text
    plan_id = planned.json()["plan_id"]
    answered = client.post("/api/answer", json={"plan_id": plan_id, "answer": "The cup on the left"})
    assert answered.status_code == 200, answered.text
    assert answered.json()["answers"] == ["The cup on the left"]


def test_unknown_job_and_artifact_return_404(monkeypatch, tmp_path):
    client = client_with_temp_dirs(monkeypatch, tmp_path)
    assert client.get("/api/jobs/no-such-job").status_code == 404
    assert client.get("/api/jobs/no-such-job/artifacts/result").status_code == 404


def test_generation_sends_only_supported_flux_schnell_fields(monkeypatch, tmp_path):
    import httpx

    client = client_with_temp_dirs(monkeypatch, tmp_path)
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "test-account")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "test-token")
    captured = {}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, headers=None, json=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            request = httpx.Request("POST", url)
            return httpx.Response(200, content=png_bytes(), headers={"content-type": "image/png"}, request=request)

    monkeypatch.setattr(api_v3.httpx, "Client", FakeClient)
    started = client.post("/api/generate", json={
        "prompt": "a small orange cube on a white background",
        "width": 1280,
        "height": 720,
        "steps": 4,
        "seed": 17,
    })
    assert started.status_code == 202, started.text
    job = wait_for_job(client, started.json()["job_id"])
    assert job["status"] == "completed", job
    assert captured["json"] == {"prompt": "a small orange cube on a white background", "steps": 4}
    assert captured["url"].endswith("/@cf/black-forest-labs/flux-1-schnell")


def test_generation_includes_cloudflare_error_detail(monkeypatch, tmp_path):
    import httpx

    client = client_with_temp_dirs(monkeypatch, tmp_path)
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "test-account")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "test-token")

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, headers=None, json=None):
            request = httpx.Request("POST", url)
            return httpx.Response(400, json={"success": False, "errors": [{"code": 5004, "message": "Invalid input for test"}]}, request=request)

    monkeypatch.setattr(api_v3.httpx, "Client", FakeClient)
    started = client.post("/api/generate", json={"prompt": "test image"})
    assert started.status_code == 202, started.text
    job = wait_for_job(client, started.json()["job_id"])
    assert job["status"] == "failed"
    assert "HTTP 400" in job["error"]["message"]
    assert "Invalid input for test" in job["error"]["message"]


def test_edit_uses_the_selected_meta_plan_instruction_and_preserve_list(monkeypatch, tmp_path):
    client = client_with_temp_dirs(monkeypatch, tmp_path)
    uploaded = client.post("/api/upload", files={"file": ("test.png", png_bytes(), "image/png")})
    image_id = uploaded.json()["image_id"]
    plan_id = "test-plan-for-edit"
    api_v3._plans[plan_id] = {
        "plan_id": plan_id,
        "image_id": image_id,
        "instruction": "Change the cup color",
        "answers": [],
        "result": {
            "status": "planned",
            "plan": {
                "target_phrase": "cup", "edit_type": "recolor",
                "instruction": "make the selected cup blue", "preserve": ["table", "background"],
                "box_2d": [100, 100, 800, 800], "confidence": 0.9,
                "questions": [], "match_count": 1,
            },
        },
    }
    response = client.post("/api/edit", json={
        "image_id": image_id, "instruction": "Change the cup color",
        "box": [15, 10, 55, 50], "provider": "fake", "plan_id": plan_id,
    })
    assert response.status_code == 202, response.text
    job = wait_for_job(client, response.json()["job_id"])
    assert job["status"] == "completed", job
    receipt = client.get(job["artifacts"]["receipt"]).json()
    assert receipt["prompt"] == "make the selected cup blue"
    assert receipt["plan"]["preserve"] == ["table", "background"]
    assert receipt["boxes"]["selected"] == {"x1": 15, "y1": 10, "x2": 55, "y2": 50}


def test_pdf_receipt_includes_visual_artifacts_and_downloads(monkeypatch, tmp_path):
    client = client_with_temp_dirs(monkeypatch, tmp_path)
    uploaded = client.post("/api/upload", files={"file": ("test.png", png_bytes(), "image/png")})
    response = client.post("/api/edit", json={
        "image_id": uploaded.json()["image_id"],
        "instruction": "Change the selected region",
        "box": [15, 10, 55, 50],
        "provider": "fake",
        "seed": 0,
    })
    job = wait_for_job(client, response.json()["job_id"])
    assert job["status"] == "completed", job

    pdf = client.get(f"/api/jobs/{job['job_id']}/receipt.pdf")
    assert pdf.status_code == 200, pdf.text
    assert pdf.headers["content-type"] == "application/pdf"
    assert pdf.content.startswith(b"%PDF-")

    refreshed = client.get(f"/api/jobs/{job['job_id']}").json()
    assert "receipt_pdf" in refreshed["artifacts"]
    artifact = client.get(refreshed["artifacts"]["receipt_pdf"])
    assert artifact.status_code == 200
    assert artifact.content.startswith(b"%PDF-")


def test_freehand_edit_uses_polygon_and_preserves_outside_pixels(monkeypatch, tmp_path):
    client = client_with_temp_dirs(monkeypatch, tmp_path)
    uploaded = client.post("/api/upload", files={"file": ("test.png", png_bytes((100, 100)), "image/png")})
    image_id = uploaded.json()["image_id"]
    pts = [[30, 30], [60, 30], [60, 60], [30, 60]]
    response = client.post("/api/edit", json={
        "image_id": image_id,
        "instruction": "Recolor the polygon area",
        "points": pts,
        "provider": "fake",
        "seed": 0,
    })
    assert response.status_code == 202, response.text
    job = wait_for_job(client, response.json()["job_id"])
    assert job["status"] == "completed", job
    receipt = client.get(job["artifacts"]["receipt"]).json()
    assert receipt["selection_type"] == "freehand"
    assert receipt["points"] == pts
    assert receipt["metrics"]["outside_changed_pixels"] == 0


def test_plan_with_target_region_gives_meta_only_cropped_context(monkeypatch, tmp_path):
    received_sizes = []

    class CapturingPlanner:
        def plan(self, image, instruction):
            if isinstance(image, Image.Image):
                received_sizes.append(image.size)
            else:
                with Image.open(image) as im:
                    received_sizes.append(im.size)
            return PlanningResult(status="planned", message="Plan ready", plan=None)

    monkeypatch.setattr(api_v3, "MetaPlanner", CapturingPlanner)
    monkeypatch.setattr(api_v3, "_planner_instance", None)
    client = client_with_temp_dirs(monkeypatch, tmp_path)
    uploaded = client.post("/api/upload", files={"file": ("test.png", png_bytes((200, 200)), "image/png")})
    image_id = uploaded.json()["image_id"]

    planned = client.post("/api/plan", json={
        "image_id": image_id,
        "instruction": "Change the cup color",
        "box": [50, 50, 90, 90],
    })
    assert planned.status_code == 200, planned.text
    assert len(received_sizes) == 1
    assert received_sizes[0] == (70, 70)
    assert received_sizes[0] != (200, 200)

