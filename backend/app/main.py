from pathlib import Path
from uuid import uuid4
import json
import mimetypes
import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Load backend/.env before reading deployment configuration such as CORS origins.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
METADATA_FILE = UPLOAD_DIR / "metadata.json"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_TYPES = {"image/png", "image/jpeg", "image/gif", "image/webp"}
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}

app = FastAPI(
    title="Pixl API",
    description="Pixl image upload, generation, planning, and deterministic regional editing API.",
    version="3.0.0",
)
def get_cors_origins() -> list[str]:
    """Read allowed browser origins from BACKEND_CORS_ORIGINS (comma-separated)."""
    configured = os.getenv("BACKEND_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
    return [origin.strip().rstrip("/") for origin in configured.split(",") if origin.strip()]


app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ImageRecord(BaseModel):
    id: str
    filename: str
    content_type: str
    size: int
    url: str
    operation: str = "upload"
    created_at: str | None = None


def read_metadata() -> list[dict]:
    if not METADATA_FILE.exists():
        return []
    try:
        data = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def write_metadata(records: list[dict]) -> None:
    temp = METADATA_FILE.with_suffix(".tmp")
    temp.write_text(json.dumps(records, indent=2), encoding="utf-8")
    temp.replace(METADATA_FILE)


def public_record(record: dict) -> dict:
    return {key: record.get(key) for key in ("id", "filename", "content_type", "size", "url", "operation", "created_at")}


def get_record(image_id: str) -> dict:
    record = next((item for item in read_metadata() if item.get("id") == image_id), None)
    if not record:
        raise HTTPException(status_code=404, detail="Image not found.")
    path = UPLOAD_DIR / record.get("stored_name", "")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Image file is missing.")
    return record


def save_image_bytes(content: bytes, filename: str, content_type: str, operation: str) -> dict:
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS or content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported image. Use PNG, JPEG, GIF, or WebP.")
    if not content:
        raise HTTPException(status_code=400, detail="The image is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Maximum image size is 10 MB.")

    image_id = str(uuid4())
    stored_name = f"{image_id}{extension}"
    (UPLOAD_DIR / stored_name).write_bytes(content)
    from datetime import datetime, timezone
    record = {
        "id": image_id,
        "filename": Path(filename).name,
        "stored_name": stored_name,
        "content_type": content_type,
        "size": len(content),
        "url": f"/api/images/{image_id}",
        "operation": operation,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    records = read_metadata()
    records.insert(0, record)
    write_metadata(records)
    return public_record(record)


async def read_upload(file: UploadFile) -> tuple[bytes, str, str]:
    filename = Path(file.filename or "image.png").name
    extension = Path(filename).suffix.lower()
    content_type = (file.content_type or mimetypes.guess_type(filename)[0] or "").lower()
    if content_type == "image/jpg":
        content_type = "image/jpeg"
    if extension not in ALLOWED_EXTENSIONS or content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported image. Use PNG, JPEG, GIF, or WebP.")
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Maximum image size is 10 MB.")
    if not content:
        raise HTTPException(status_code=400, detail="The image is empty.")
    return content, filename, content_type


@app.get("/api/", tags=["Homepage"])
async def homepage():
    return {
        "name": "Pixl API",
        "message": "Fewer iterations. More intentional images.",
        "status": "ok",
        "services": {
            "upload": "POST /api/images",
            "list": "GET /api/images",
            "show": "GET /api/images/{image_id}",
            "generate": "POST /api/generate",
            "edit": "POST /api/edit",
            "compare": "GET /api/compare?original_id=...&edited_id=...",
            "docs": "/docs",
        },
    }


@app.get("/api/health", tags=["Homepage"])
async def health():
    return {"status": "ok", "service": "pixl-api"}


@app.post("/api/images", response_model=ImageRecord, status_code=201, tags=["Images"])
async def upload_image(file: UploadFile = File(...)):
    content, filename, content_type = await read_upload(file)
    return save_image_bytes(content, filename, content_type, "upload")


@app.get("/api/images", response_model=list[ImageRecord], tags=["Images"])
async def list_images():
    records = read_metadata()
    return [public_record(item) for item in records if (UPLOAD_DIR / item.get("stored_name", "")).is_file()]


@app.get("/api/images/{image_id}", tags=["Images"])
async def show_image(image_id: str):
    record = get_record(image_id)
    path = UPLOAD_DIR / record["stored_name"]
    # Return inline image bytes so browsers can display them in img elements.
    return FileResponse(path, media_type=record["content_type"], content_disposition_type="inline")


@app.get("/api/compare", tags=["Image workflow"])
async def compare_images(original_id: str, edited_id: str):
    """Comparison placeholder: returns metadata and display URLs for two saved images."""
    original = get_record(original_id)
    edited = get_record(edited_id)
    return {
        "original": public_record(original),
        "edited": public_record(edited),
        "same_bytes_placeholder": original["size"] == edited["size"] and (UPLOAD_DIR / original["stored_name"]).read_bytes() == (UPLOAD_DIR / edited["stored_name"]).read_bytes(),
        "message": "Comparison UI placeholder. The editor can later provide real before/after metrics.",
    }


# Phase 3 job-oriented API routes.
from app.api_v3 import router as phase3_router
app.include_router(phase3_router)
