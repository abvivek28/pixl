"""Phase 3 HTTP API for Pixl.

The legacy routes in app.main remain available for the existing frontend. This module
adds the job-oriented API and is mounted from app.main below.
"""
from __future__ import annotations

import base64
import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, ImageOps, UnidentifiedImageError
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass
from pydantic import BaseModel, Field

from app.engine.deterministic import Box, delta_e
from app.engine.orchestrator import FakeHueShiftProvider, run_edit
from app.planning.meta_planner import MetaPlanner
from app.planning.meta_generation_prompt import GenerationPromptEnhancer
from app.providers.cloudflare_klein4b import CloudflareKlein4B, ProviderError

router = APIRouter(prefix="/api", tags=["Phase 3 API"])
BACKEND_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("PIXL_DATA_DIR", str(BACKEND_DIR / "data"))).resolve()
UPLOADS_DIR = DATA_DIR / "uploads"
JOBS_DIR = DATA_DIR / "jobs"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DIR.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_FORMATS = {"PNG", "JPEG", "WEBP", "GIF"}
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="pixl-job")
_lock = threading.RLock()
_jobs: dict[str, dict[str, Any]] = {}
_plans: dict[str, dict[str, Any]] = {}
_planner_instance: MetaPlanner | None = None
_generation_enhancer: GenerationPromptEnhancer | None = None
_generation_plans: dict[str, dict[str, Any]] = {}


def _get_generation_enhancer() -> GenerationPromptEnhancer:
    """Share the text-only enhancer so its request throttle applies across calls."""
    global _generation_enhancer
    with _lock:
        if _generation_enhancer is None:
            _generation_enhancer = GenerationPromptEnhancer()
        return _generation_enhancer


def _get_planner() -> MetaPlanner:
    """Share one planner instance so its request throttle applies across API calls."""
    global _planner_instance
    with _lock:
        if _planner_instance is None:
            _planner_instance = MetaPlanner()
        return _planner_instance


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(data: bytes) -> str:
    import hashlib
    return hashlib.sha256(data).hexdigest()


def _safe_image(content: bytes) -> tuple[Image.Image, str]:
    try:
        with Image.open(BytesIO(content)) as opened:
            if opened.format not in ALLOWED_FORMATS:
                raise HTTPException(status_code=415, detail="Unsupported image format. Use PNG, JPEG, WebP, or GIF.")
            image = ImageOps.exif_transpose(opened).convert("RGB").copy()
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.") from exc
    return image, "PNG"


async def _read_upload(file: UploadFile) -> tuple[bytes, str]:
    filename = Path(file.filename or "image.png").name
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Maximum image size is 10 MB.")
    _safe_image(content)
    return content, filename


def _save_master(content: bytes, filename: str) -> dict[str, Any]:
    image, _ = _safe_image(content)
    image_id = str(uuid.uuid4())
    folder = UPLOADS_DIR / image_id
    folder.mkdir(parents=True, exist_ok=False)
    master_path = folder / "master.png"
    image.save(master_path, format="PNG", optimize=False)
    saved = master_path.read_bytes()
    record = {
        "image_id": image_id,
        "filename": filename,
        "width": image.width,
        "height": image.height,
        "size_bytes": len(saved),
        "sha256": _sha256(saved),
        "created_at": _now(),
        "url": f"/api/uploads/{image_id}",
        "path": str(master_path),
    }
    (folder / "metadata.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def _get_image(image_id: str) -> tuple[dict[str, Any], Path]:
    folder = UPLOADS_DIR / image_id
    meta_path = folder / "metadata.json"
    image_path = folder / "master.png"
    if not meta_path.is_file() or not image_path.is_file():
        raise HTTPException(status_code=404, detail="Uploaded image not found.")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail="Image metadata is unreadable.") from exc
    return meta, image_path


def _public_job(job: dict[str, Any]) -> dict[str, Any]:
    result = {key: value for key, value in job.items() if key not in {"_future", "_lock", "output_dir"}}
    if result.get("artifacts"):
        result["artifacts"] = {
            name: f"/api/jobs/{job['job_id']}/artifacts/{name}"
            for name, path in job["artifacts"].items() if Path(path).is_file()
        }
    return result


def _new_job(kind: str, input_data: dict[str, Any]) -> dict[str, Any]:
    job_id = str(uuid.uuid4())
    output_dir = JOBS_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=False)
    job = {
        "job_id": job_id,
        "kind": kind,
        "status": "queued",
        "progress": 0,
        "stage": "queued",
        "created_at": _now(),
        "updated_at": _now(),
        "input": input_data,
        "output_dir": str(output_dir),
        "artifacts": {},
        "error": None,
    }
    with _lock:
        _jobs[job_id] = job
    return job


def _update_job(job_id: str, **changes: Any) -> None:
    with _lock:
        job = _jobs[job_id]
        job.update(changes)
        job["updated_at"] = _now()


def _run_edit_job(job_id: str, request_data: dict[str, Any]) -> None:
    _update_job(job_id, status="running", progress=10, stage="preparing_image")
    try:
        _, image_path = _get_image(request_data["image_id"])
        output_dir = Path(_jobs[job_id]["output_dir"])
        with Image.open(image_path) as image:
            original = image.convert("RGB").copy()
        original.save(output_dir / "original.png", format="PNG")
        _update_job(job_id, progress=25, stage="running_edit")
        box_values = request_data.get("box")
        selected_box = Box(*box_values) if box_values else None
        points_values = request_data.get("points")
        provider_name = request_data.get("provider", "fake")
        provider = CloudflareKlein4B() if provider_name == "cloudflare" else FakeHueShiftProvider()
        result = run_edit(
            image_path=image_path,
            instruction=request_data["instruction"],
            output_dir=output_dir,
            selected_box=selected_box,
            points=points_values,
            box_source="freehand" if points_values else ("manual" if selected_box else "manual_required"),
            provider=provider,
            seed=request_data.get("seed", 0),
            plan_data=request_data.get("plan_data"),
        )
        if result.get("status") in {"needs_choice", "needs_manual_box", "manual_fallback", "global_warning"}:
            raise ValueError(result.get("message", "The edit needs additional input."))
        artifacts = {
            "original": str(output_dir / "original.png"),
            "result": str(output_dir / "result.png"),
            "heatmap": str(output_dir / "heatmap.png"),
            "receipt": str(output_dir / "receipt.json"),
        }
        (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        artifacts["result_json"] = str(output_dir / "result.json")
        _update_job(job_id, status="completed", progress=100, stage="completed", artifacts=artifacts,
                    summary=result.get("summary"), metrics=result.get("metrics"), checks=result.get("checks"))
    except Exception as exc:
        _update_job(job_id, status="failed", progress=100, stage="failed",
                    error={"type": type(exc).__name__, "message": str(exc)})


def _decode_image_response(response: httpx.Response) -> bytes:
    if response.headers.get("content-type", "").lower().startswith("image/"):
        return response.content
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError("Image generation provider returned an unknown response format.") from exc
    result = payload.get("result", payload) if isinstance(payload, dict) else {}
    for key in ("image", "data", "output"):
        value = result.get(key) if isinstance(result, dict) else None
        if isinstance(value, str):
            try:
                return base64.b64decode(value, validate=True)
            except Exception:
                continue
        if isinstance(value, list) and all(isinstance(n, int) and 0 <= n <= 255 for n in value):
            return bytes(value)
    raise RuntimeError(f"Generation response did not contain image bytes: {str(payload)[:400]}")


def _run_generation_job(job_id: str, request_data: dict[str, Any]) -> None:
    _update_job(job_id, status="running", progress=10, stage="connecting_to_provider")
    try:
        account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
        token = os.getenv("CLOUDFLARE_API_TOKEN", "")
        if not account_id or not token:
            raise ProviderError("Set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN to use image generation.")
        model = os.getenv("CLOUDFLARE_FLUX_SCHNELL_MODEL", "@cf/black-forest-labs/flux-1-schnell")
        url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"
        # FLUX.1 schnell's documented Workers AI schema accepts prompt and steps.
        # Do not forward width/height (or seed) from the UI: unsupported fields can
        # cause Cloudflare to reject the request with HTTP 400. The model determines
        # the output dimensions; we record the actual dimensions in the receipt.
        payload: dict[str, Any] = {
            "prompt": request_data["prompt"],
            "steps": request_data.get("steps", 4),
        }
        _update_job(job_id, progress=35, stage="generating_image")
        with httpx.Client(timeout=float(os.getenv("PIXL_GENERATION_TIMEOUT_SECONDS", "120"))) as client:
            response = client.post(url, headers={"Authorization": f"Bearer {token}"}, json=payload)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                detail = ""
                try:
                    provider_body = response.json()
                    errors = provider_body.get("errors", []) if isinstance(provider_body, dict) else []
                    if errors:
                        detail = "; ".join(
                            str(item.get("message", item)) if isinstance(item, dict) else str(item)
                            for item in errors
                        )
                    else:
                        detail = str(provider_body)[:600]
                except (ValueError, TypeError):
                    detail = response.text[:600]
                message = f"Cloudflare image generation failed (HTTP {response.status_code})"
                if detail:
                    message += f": {detail}"
                raise ProviderError(message) from exc
        generated_bytes = _decode_image_response(response)
        image, _ = _safe_image(generated_bytes)
        output_dir = Path(_jobs[job_id]["output_dir"])
        image.save(output_dir / "result.png", format="PNG")
        (output_dir / "receipt.json").write_text(json.dumps({
            "operation": "text_to_image_generation", "provider": "cloudflare", "model": model,
            "prompt": request_data["prompt"], "seed": request_data.get("seed"),
            "size": {"width": image.width, "height": image.height},
            "result_sha256": _sha256((output_dir / "result.png").read_bytes()), "created_at": _now(),
        }, indent=2), encoding="utf-8")
        _update_job(job_id, status="completed", progress=100, stage="completed", artifacts={
            "result": str(output_dir / "result.png"), "receipt": str(output_dir / "receipt.json")},
            summary="Image generated successfully.")
    except Exception as exc:
        _update_job(job_id, status="failed", progress=100, stage="failed",
                    error={"type": type(exc).__name__, "message": str(exc)})


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    width: int = Field(default=1024, ge=256, le=1440)
    height: int = Field(default=1024, ge=256, le=1440)
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)
    steps: int = Field(default=4, ge=1, le=4)


class GenerationPromptRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)


class GenerationAnswerRequest(BaseModel):
    plan_id: str
    answers: list[dict[str, str]] = Field(min_length=1, max_length=3)


class PlanRequest(BaseModel):
    image_id: str
    instruction: str = Field(min_length=1, max_length=4000)
    box: list[int] | None = Field(default=None, min_length=4, max_length=4, description="[x1,y1,x2,y2] pixel coordinates of selected region")
    points: list[list[int]] | None = Field(default=None, description="Freehand polygon vertices [[x, y], ...]")


class AnswerRequest(BaseModel):
    plan_id: str
    answer: str = Field(min_length=1, max_length=1000)


class EditRequest(BaseModel):
    image_id: str
    instruction: str = Field(min_length=1, max_length=4000)
    box: list[int] | None = Field(default=None, min_length=4, max_length=4, description="[x1,y1,x2,y2] pixel coordinates")
    points: list[list[int]] | None = Field(default=None, description="Freehand polygon vertices [[x, y], ...]")
    provider: Literal["fake", "cloudflare"] = "fake"
    seed: int | None = Field(default=0, ge=0, le=2**31 - 1)
    plan_id: str | None = None

    @classmethod
    def model_validate_box(cls, values):
        return values


@router.post("/upload", status_code=201)
async def upload(file: UploadFile = File(...)):
    """Save a validated, EXIF-corrected RGB PNG master and return its image_id."""
    content, filename = await _read_upload(file)
    record = _save_master(content, filename)
    return {key: value for key, value in record.items() if key != "path"}


@router.get("/uploads/{image_id}")
async def uploaded_image(image_id: str):
    record, path = _get_image(image_id)
    return FileResponse(path, media_type="image/png", filename=f"{image_id}.png", content_disposition_type="inline")


@router.post("/generate", status_code=202)
def generate(request: GenerateRequest):
    """Queue a Cloudflare FLUX.1 schnell text-to-image generation job."""
    job = _new_job("generate", request.model_dump())
    _executor.submit(_run_generation_job, job["job_id"], request.model_dump())
    return _public_job(job)


@router.post("/generation-plan")
def plan_generation_prompt(request: GenerationPromptRequest):
    """Optionally enrich a text-to-image prompt and ask only high-impact missing details."""
    enhancer = _get_generation_enhancer()
    try:
        result = enhancer.plan(request.prompt)
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    plan_id = str(uuid.uuid4())
    with _lock:
        _generation_plans[plan_id] = {"plan_id": plan_id, "original_prompt": request.prompt.strip(),
                                     "answers": [], "created_at": _now()}
    return {"plan_id": plan_id, **result.model_dump(mode="json")}


@router.post("/generation-answer")
def answer_generation_prompt(request: GenerationAnswerRequest):
    """Apply multiple-choice answers and return the next questions or final improved prompt."""
    with _lock:
        saved = _generation_plans.get(request.plan_id)
        if not saved:
            raise HTTPException(status_code=404, detail="Generation prompt plan not found or expired for this server process.")
        saved["answers"].extend(request.answers)
        answers = list(saved["answers"])
        original_prompt = saved["original_prompt"]
    try:
        result = _get_generation_enhancer().plan(original_prompt, answers)
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"plan_id": request.plan_id, **result.model_dump(mode="json"), "answers": answers}


@router.post("/plan")
def plan_edit(request: PlanRequest):
    """Use Meta Muse Spark to analyze the selected region of an image and produce an edit plan."""
    _, image_path = _get_image(request.image_id)
    with Image.open(image_path) as opened:
        master = opened.convert("RGB").copy()

    box_coords = request.box
    if not box_coords and request.points and len(request.points) >= 3:
        xs = [p[0] for p in request.points]
        ys = [p[1] for p in request.points]
        box_coords = [max(0, min(xs)), max(0, min(ys)), min(master.width, max(xs) + 1), min(master.height, max(ys) + 1)]

    # Provide only the context of the selected region (with 15px margin) rather than the complete image
    if box_coords:
        x1, y1, x2, y2 = box_coords
        w, h = master.size
        context_margin = 15
        cx1, cy1 = max(0, x1 - context_margin), max(0, y1 - context_margin)
        cx2, cy2 = min(w, x2 + context_margin), min(h, y2 + context_margin)
        image = master.crop((cx1, cy1, cx2, cy2))
    else:
        image = master

    planner = _get_planner()
    result = planner.plan(image, request.instruction)
    plan_id = str(uuid.uuid4())
    with _lock:
        _plans[plan_id] = {
            "plan_id": plan_id,
            "image_id": request.image_id,
            "instruction": request.instruction,
            "box": box_coords,
            "points": request.points,
            "answers": [],
            "result": result.model_dump(mode="json"),
            "created_at": _now()
        }
    return {"plan_id": plan_id, **result.model_dump(mode="json")}


@router.post("/answer")
def answer_question(request: AnswerRequest):
    """Record a clarification answer and rerun planning on the selected region."""
    with _lock:
        saved = _plans.get(request.plan_id)
    if not saved:
        raise HTTPException(status_code=404, detail="Plan not found or expired for this server process.")
    saved["answers"].append(request.answer)
    _, image_path = _get_image(saved["image_id"])
    with Image.open(image_path) as opened:
        master = opened.convert("RGB").copy()

    box_coords = saved.get("box")
    if box_coords:
        x1, y1, x2, y2 = box_coords
        w, h = master.size
        context_margin = 15
        cx1, cy1 = max(0, x1 - context_margin), max(0, y1 - context_margin)
        cx2, cy2 = min(w, x2 + context_margin), min(h, y2 + context_margin)
        image = master.crop((cx1, cy1, cx2, cy2))
    else:
        image = master

    answer_context = "\n".join(f"Clarification answer {i + 1}: {answer}" for i, answer in enumerate(saved["answers"]))
    revised_instruction = f"{saved['instruction']}\n\nUser clarifications:\n{answer_context}"
    result = _get_planner().plan(image, revised_instruction)
    with _lock:
        saved["result"] = result.model_dump(mode="json")
        saved["updated_at"] = _now()
    return {"plan_id": request.plan_id, "answers": saved["answers"], **result.model_dump(mode="json")}


@router.post("/edit", status_code=202)
def start_edit(request: EditRequest):
    """Queue an isolated region edit; a confirmed manual pixel box or freehand points are required."""
    _, image_path = _get_image(request.image_id)
    with Image.open(image_path) as opened:
        width, height = opened.size
    if request.box is None and request.points and len(request.points) >= 3:
        xs = [p[0] for p in request.points]
        ys = [p[1] for p in request.points]
        request.box = [max(0, min(xs)), max(0, min(ys)), min(width, max(xs) + 1), min(height, max(ys) + 1)]
    if request.box is None:
        raise HTTPException(status_code=422, detail="A confirmed box or freehand region is required.")
    x1, y1, x2, y2 = request.box
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise HTTPException(status_code=422, detail=f"Box must satisfy 0 <= x1 < x2 <= {width} and 0 <= y1 < y2 <= {height}.")
    data = request.model_dump()
    # A Meta plan is advisory for target identification, but its instruction and
    # preserve list must reach the edit provider. The confirmed manual box remains
    # authoritative until Meta's coordinate convention is verified.
    if request.plan_id:
        with _lock:
            saved_plan = _plans.get(request.plan_id)
        if not saved_plan:
            raise HTTPException(status_code=404, detail="Meta plan not found or expired. Run Plan with Meta again.")
        if saved_plan.get("image_id") != request.image_id:
            raise HTTPException(status_code=422, detail="The Meta plan belongs to a different source image. Plan again for this image.")
        plan_result = saved_plan.get("result", {})
        if plan_result.get("plan"):
            data["plan_data"] = plan_result["plan"]
    job = _new_job("edit", data)
    _executor.submit(_run_edit_job, job["job_id"], data)
    return _public_job(job)


@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found. Jobs are kept in memory for this server process.")
    return _public_job(job)


def _build_receipt_pdf(job: dict[str, Any]) -> Path:
    """Build a Pixl-branded, bill-like receipt with large visual evidence and four requested stats."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as PDFImage, HRFlowable, KeepTogether
        from xml.sax.saxutils import escape
    except ImportError as exc:
        raise HTTPException(status_code=503, detail="PDF receipts require reportlab. Install backend requirements and restart Pixl.") from exc

    output_dir = Path(job["output_dir"])
    receipt_path = Path(job.get("artifacts", {}).get("receipt", ""))
    if not receipt_path.is_file():
        raise HTTPException(status_code=409, detail="The edit receipt is not available yet.")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail="The edit receipt could not be read.") from exc

    original_path = Path(job.get("artifacts", {}).get("original", ""))
    result_path = Path(job.get("artifacts", {}).get("result", ""))
    heatmap_path = Path(job.get("artifacts", {}).get("heatmap", ""))
    for path, label in ((original_path, "original image"), (result_path, "edited image"), (heatmap_path, "heatmap")):
        if not path.is_file():
            raise HTTPException(status_code=409, detail=f"The {label} artifact is not available.")

    with Image.open(original_path) as im:
        original = im.convert("RGB").copy()
    with Image.open(result_path) as im:
        edited = im.convert("RGB").copy()
    if original.size != edited.size:
        edited = edited.resize(original.size, Image.Resampling.LANCZOS)
    import numpy as np
    delta = delta_e(np.asarray(original, dtype=np.uint8), np.asarray(edited, dtype=np.uint8))
    changed_pct = float(np.mean(delta > 0.05) * 100.0) if delta.size else 0.0
    retained_pct = max(0.0, 100.0 - changed_pct)
    metrics = dict(receipt.get("metrics") or {})
    selected_changed = metrics.get("percent_box_changed", metrics.get("percent_selected_area_changed", "N/A"))
    outside_changed = metrics.get("outside_changed_pixels", "N/A")

    pdf_path = output_dir / "receipt.pdf"
    page_size = landscape(A4)
    page_w, page_h = page_size
    ink = colors.HexColor("#111111")
    orange = colors.HexColor("#FF5A1F")
    paper = colors.HexColor("#EFEFE9")
    muted = colors.HexColor("#686862")
    line = colors.HexColor("#D0CEC6")
    white = colors.white

    def page_decor(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(paper)
        canvas.rect(0, 0, page_w, page_h, stroke=0, fill=1)
        canvas.setStrokeColor(colors.HexColor("#E5E3DC")); canvas.setLineWidth(0.35)
        step = 22 * mm
        for x in range(0, int(page_w) + 1, int(step)):
            canvas.line(x, 0, x, page_h)
        for y in range(0, int(page_h) + 1, int(step)):
            canvas.line(0, y, page_w, y)
        canvas.setFillColor(white); canvas.setStrokeColor(ink); canvas.setLineWidth(1.5)
        canvas.rect(doc.leftMargin - 4*mm, 10*mm, page_w - doc.leftMargin - doc.rightMargin + 8*mm, page_h - 20*mm, stroke=1, fill=1)
        canvas.setFillColor(orange); canvas.rect(doc.leftMargin - 4*mm, page_h - 20*mm, 7*mm, 7*mm, stroke=0, fill=1)
        canvas.setFillColor(ink); canvas.setFont("Helvetica-Bold", 13)
        canvas.drawString(doc.leftMargin + 6*mm, page_h - 16*mm, "pixl")
        canvas.setFont("Courier", 7); canvas.setFillColor(muted)
        canvas.drawString(doc.leftMargin + 20*mm, page_h - 15.5*mm, "IMAGE INTELLIGENCE  /  CHANGE RECEIPT")
        canvas.setFont("Courier-Bold", 7); canvas.setFillColor(orange)
        canvas.drawRightString(page_w - doc.rightMargin, page_h - 15.5*mm, "PAID IN PIXELS")
        canvas.setStrokeColor(line); canvas.setDash(2, 2)
        canvas.line(doc.leftMargin - 2*mm, 15*mm, page_w - doc.rightMargin + 2*mm, 15*mm)
        canvas.setDash()
        canvas.setFont("Courier", 5.8); canvas.setFillColor(muted)
        canvas.drawString(doc.leftMargin - 2*mm, 11.5*mm, f"RECEIPT {job.get('job_id', 'UNKNOWN')}  /  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
        canvas.drawRightString(page_w - doc.rightMargin + 2*mm, 11.5*mm, f"SOURCE SHA256 {(receipt.get('image') or {}).get('master_sha256', 'not-recorded')[:16]}…")
        canvas.restoreState()

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReceiptTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=22, leading=25, textColor=ink, alignment=TA_LEFT, spaceAfter=2))
    styles.add(ParagraphStyle(name="ReceiptSub", parent=styles["Normal"], fontName="Helvetica", fontSize=8, leading=10, textColor=muted))
    styles.add(ParagraphStyle(name="ReceiptLabel", parent=styles["Normal"], fontName="Courier-Bold", fontSize=7, leading=9, textColor=muted))
    styles.add(ParagraphStyle(name="ReceiptValue", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=17, leading=19, textColor=ink))
    styles.add(ParagraphStyle(name="ReceiptSection", parent=styles["Normal"], fontName="Courier-Bold", fontSize=8, leading=10, textColor=ink))
    styles.add(ParagraphStyle(name="ReceiptTiny", parent=styles["Normal"], fontName="Courier", fontSize=6, leading=8, textColor=muted))
    doc = SimpleDocTemplate(str(pdf_path), pagesize=page_size, rightMargin=17*mm, leftMargin=17*mm, topMargin=25*mm, bottomMargin=20*mm, title="Pixl Change Receipt", author="Pixl")
    available = page_w - doc.leftMargin - doc.rightMargin
    story: list[Any] = []

    receipt_title = Table([
        [Paragraph("CHANGE RECEIPT", styles["ReceiptTitle"]), Paragraph("ORIGINAL → EDITED", ParagraphStyle(name="ReceiptRight", parent=styles["ReceiptLabel"], alignment=TA_RIGHT))],
        [Paragraph("A visual record of what changed — and what stayed.", styles["ReceiptSub"]), Paragraph(escape(datetime.now(timezone.utc).strftime("%d %b %Y · %H:%M UTC")), ParagraphStyle(name="ReceiptDate", parent=styles["ReceiptTiny"], alignment=TA_RIGHT))],
    ], colWidths=[available * .66, available * .34])
    receipt_title.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "BOTTOM"), ("LEFTPADDING", (0,0), (-1,-1), 0), ("RIGHTPADDING", (0,0), (-1,-1), 0), ("TOPPADDING", (0,0), (-1,-1), 0), ("BOTTOMPADDING", (0,0), (-1,-1), 2)]))
    story.extend([receipt_title, Spacer(1, 4*mm)])

    # Bill-like line items: just the four statistics requested, with large, easy-to-scan values.
    def fmt_pct(value):
        try: return f"{float(value):.2f}%"
        except (TypeError, ValueError): return "N/A"
    def fmt_count(value):
        try: return f"{int(round(float(value))):,}"
        except (TypeError, ValueError): return "N/A"
    stat_items = [
        ("WHOLE IMAGE CHANGED", fmt_pct(changed_pct), True),
        ("WHOLE IMAGE RETAINED", fmt_pct(retained_pct), False),
        ("SELECTED AREA CHANGED", fmt_pct(selected_changed), False),
        ("OUTSIDE CHANGES (PIXELS)", fmt_count(outside_changed), False),
    ]
    stat_cells = []
    stat_w = available / 4
    for label, value, accent in stat_items:
        cell = Table([[Paragraph(label, styles["ReceiptLabel"])], [Paragraph(escape(value), styles["ReceiptValue"])]], colWidths=[stat_w - 3*mm])
        cell.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#FFF0E9") if accent else colors.HexColor("#F7F6F1")),
            ("BOX", (0,0), (-1,-1), 1, ink), ("LINEBELOW", (0,0), (-1,0), .5, line),
            ("LEFTPADDING", (0,0), (-1,-1), 7), ("RIGHTPADDING", (0,0), (-1,-1), 4),
            ("TOPPADDING", (0,0), (-1,0), 8), ("BOTTOMPADDING", (0,0), (-1,0), 5),
            ("TOPPADDING", (0,1), (-1,1), 7), ("BOTTOMPADDING", (0,1), (-1,1), 9),
        ]))
        stat_cells.append(cell)
    stat_table = Table([stat_cells], colWidths=[stat_w]*4, hAlign="LEFT")
    stat_table.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "TOP"), ("LEFTPADDING", (0,0), (-1,-1), 0), ("RIGHTPADDING", (0,0), (-1,-1), 3), ("TOPPADDING", (0,0), (-1,-1), 0), ("BOTTOMPADDING", (0,0), (-1,-1), 0)]))
    story.extend([stat_table, Spacer(1, 5*mm), HRFlowable(width="100%", thickness=.7, color=line, dash=[2,2]), Spacer(1, 3*mm), Paragraph("VISUAL EVIDENCE", styles["ReceiptSection"]), Spacer(1, 2*mm)])

    def fitted_image(path: Path, max_w: float, max_h: float):
        with Image.open(path) as im:
            iw, ih = im.size
        scale = min(max_w / max(iw, 1), max_h / max(ih, 1))
        return PDFImage(str(path), width=iw*scale, height=ih*scale)

    gap = 4*mm
    col_w = (available - 2*gap) / 3
    image_cells = []
    for path, title in ((original_path, "01  /  ORIGINAL"), (result_path, "02  /  EDITED"), (heatmap_path, "03  /  HEATMAP")):
        img = fitted_image(path, col_w - 5*mm, 61*mm)
        cell = Table([[Paragraph(title, styles["ReceiptLabel"])], [img]], colWidths=[col_w])
        cell.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#F7F6F1")),
            ("BOX", (0,0), (-1,-1), 1, ink), ("ALIGN", (0,0), (-1,-1), "CENTER"),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("LEFTPADDING", (0,0), (-1,-1), 4), ("RIGHTPADDING", (0,0), (-1,-1), 4),
            ("TOPPADDING", (0,0), (0,0), 7), ("BOTTOMPADDING", (0,0), (0,0), 5),
            ("TOPPADDING", (0,1), (0,1), 4), ("BOTTOMPADDING", (0,1), (0,1), 7),
        ]))
        image_cells.append(cell)
    visual = Table([image_cells], colWidths=[col_w]*3, hAlign="LEFT")
    visual.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "TOP"), ("LEFTPADDING", (0,0), (-1,-1), 0), ("RIGHTPADDING", (0,0), (-1,-1), gap), ("TOPPADDING", (0,0), (-1,-1), 0), ("BOTTOMPADDING", (0,0), (-1,-1), 0)]))
    story.extend([visual, Spacer(1, 3*mm)])

    prompt = str(receipt.get("prompt") or "")
    if prompt:
        story.extend([HRFlowable(width="100%", thickness=.7, color=line, dash=[2,2]), Spacer(1, 2*mm), Paragraph("EDIT REQUEST", styles["ReceiptLabel"]), Spacer(1, 1*mm), Paragraph(escape(prompt[:420]), styles["ReceiptSub"])])
    doc.build(story, onFirstPage=page_decor, onLaterPages=page_decor)
    with _lock:
        job.setdefault("artifacts", {})["receipt_pdf"] = str(pdf_path)
    return pdf_path

@router.get("/jobs/{job_id}/receipt.pdf")
def download_receipt_pdf(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.get("status") != "completed":
        raise HTTPException(status_code=409, detail="The edit must complete before a PDF receipt can be generated.")
    path = _build_receipt_pdf(job)
    return FileResponse(path, media_type="application/pdf", filename="pixl-edit-receipt.pdf")


@router.get("/jobs/{job_id}/artifacts/{artifact_name}")
def get_artifact(job_id: str, artifact_name: str):
    allowed = {"original", "result", "heatmap", "receipt", "receipt_pdf", "result_json"}
    if artifact_name not in allowed:
        raise HTTPException(status_code=404, detail="Unknown artifact.")
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    path = job.get("artifacts", {}).get(artifact_name)
    if not path or not Path(path).is_file():
        raise HTTPException(status_code=404, detail="Artifact is not available yet.")
    suffix = Path(path).suffix.lower()
    media_type = {".png": "image/png", ".json": "application/json", ".pdf": "application/pdf"}.get(suffix, "application/octet-stream")
    return FileResponse(path, media_type=media_type, filename=Path(path).name)
