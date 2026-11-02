from __future__ import annotations
from enum import Enum
from pathlib import Path
from io import BytesIO
import base64
from typing import Literal

from pydantic import BaseModel, Field, field_validator
from PIL import Image

from app.planning.meta_api import MetaModelAPI


class EditType(str, Enum):
    recolor = "recolor"
    replace = "replace"
    remove = "remove"
    add = "add"
    restyle = "restyle"
    global_edit = "global"


class PlanQuestion(BaseModel):
    question: str
    options: list[str] = Field(min_length=2, max_length=4)

    @field_validator("options")
    @classmethod
    def options_unique(cls, value):
        if len(set(value)) != len(value):
            raise ValueError("Question options must be unique")
        return value


class MetaEditPlan(BaseModel):
    target_phrase: str
    edit_type: EditType
    instruction: str
    preserve: list[str] = Field(default_factory=list)
    box_2d: list[int] | None = Field(default=None, description="Raw normalized model coordinates; never auto-applied")
    confidence: float = Field(ge=0.0, le=1.0)
    questions: list[PlanQuestion] = Field(default_factory=list)
    match_count: int = Field(default=1, ge=0)


class PlanningResult(BaseModel):
    status: Literal["planned", "needs_choice", "needs_manual_box", "global_warning", "manual_fallback"]
    plan: MetaEditPlan | None = None
    message: str
    warnings: list[str] = Field(default_factory=list)
    coordinate_status: str = "unverified_raw_coordinates"


class MetaPlanner:
    """Meta Muse Spark vision planner with schema validation and safe manual fallback."""
    def __init__(self, api_key: str | None = None, model: str | None = None, *,
                 client=None, min_interval_seconds: float = 0.0, sleep=None, clock=None):
        import time
        from threading import Lock
        self.api = MetaModelAPI(api_key, model, client=client)
        self.model = self.api.model
        self.min_interval_seconds = min_interval_seconds
        self._sleep = sleep or time.sleep
        self._clock = clock or time.monotonic
        self._last_call = 0.0
        self._lock = Lock()

    def _throttle(self):
        with self._lock:
            now = self._clock()
            wait = self.min_interval_seconds - (now - self._last_call)
            if self._last_call and wait > 0:
                self._sleep(wait)
            self._last_call = self._clock()

    @staticmethod
    def _image_data_url(image: Image.Image) -> str:
        preview = image.convert("RGB").copy()
        preview.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        buf = BytesIO()
        preview.save(buf, format="JPEG", quality=88)
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def plan(self, image: Image.Image | str | Path, user_text: str) -> PlanningResult:
        if isinstance(image, (str, Path)):
            with Image.open(image) as opened:
                img = opened.convert("RGB").copy()
        else:
            img = image.convert("RGB").copy()
        if not self.api.api_key:
            return PlanningResult(status="manual_fallback", message="Meta planning is not configured; draw or supply a box manually.", warnings=["MODEL_API_KEY is missing."])

        schema = {
            "type": "object",
            "properties": {
                "target_phrase": {"type": "string"},
                "edit_type": {"type": "string", "enum": ["recolor", "replace", "remove", "add", "restyle", "global"]},
                "instruction": {"type": "string"},
                "preserve": {"type": "array", "items": {"type": "string"}},
                "box_2d": {"anyOf": [{"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4}, {"type": "null"}]},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "questions": {"type": "array", "items": {"type": "object", "properties": {
                    "question": {"type": "string"},
                    "options": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 4}
                }, "required": ["question", "options"], "additionalProperties": False}},
                "match_count": {"type": "integer", "minimum": 0}
            },
            "required": ["target_phrase", "edit_type", "instruction", "preserve", "box_2d", "confidence", "questions", "match_count"],
            "additionalProperties": False,
        }
        prompt = f"""You are Pixl's image-edit planning component. Analyze the supplied image and user request.
Return one JSON object matching the schema. User request: {user_text}
Choose edit_type from recolor, replace, remove, add, restyle, global. Write one precise instruction for the image editor.
List visual elements that must be preserved. If multiple plausible target objects exist, set match_count > 1 and add one multiple-choice question with 2-4 concise options. If no target object can be identified, set match_count=0 and box_2d=null. For a single clear target, provide box_2d as raw normalized coordinates in [ymin,xmin,ymax,xmax] order, scaled 0..1000, and confidence. Never return pixel coordinates. For add, identify an anchor object and box or leave box null if no anchor exists. Global changes must use edit_type='global'.
Image dimensions: {img.width} x {img.height}. Model coordinates are advisory only; do not assume they will be applied automatically."""
        try:
            self._throttle()
            content = self.api.complete_json(
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": self._image_data_url(img)}},
                ]}],
                schema_name="pixl_edit_plan", schema=schema,
            )
            plan = MetaEditPlan.model_validate_json(content)
        except Exception as exc:
            return PlanningResult(status="manual_fallback", message="Meta planning failed; continue with a manually drawn box.", warnings=[f"Planner failure: {type(exc).__name__}: {str(exc)[:700]}"])

        if plan.edit_type == EditType.global_edit:
            return PlanningResult(status="global_warning", plan=plan, message="This is a global edit; region isolation does not apply.", warnings=["Global edits can change pixels outside a selected region."])
        if plan.match_count > 1 or plan.questions:
            return PlanningResult(status="needs_choice", plan=plan, message="Several possible targets were found. Choose the intended object.")
        if plan.match_count == 0 or plan.box_2d is None:
            msg = "Choose an anchor object or draw a manual box before adding the new element." if plan.edit_type == EditType.add else "No reliable target box was found. Draw a box around the target to continue."
            return PlanningResult(status="needs_manual_box", plan=plan, message=msg)
        return PlanningResult(status="planned", plan=plan, message="Plan ready. Confirm the target box before editing.")


def convert_box_2d_to_pixels(raw_box: list[int], image_width: int, image_height: int, *, verified_format: str | None = None) -> tuple[int, int, int, int]:
    """Convert coordinates only after an explicit five-image convention verification."""
    if verified_format != "ymin_xmin_ymax_xmax_0_1000":
        raise ValueError("Meta box coordinate mapping is unverified. Validate 5 images, then pass the confirmed format explicitly.")
    if len(raw_box) != 4:
        raise ValueError("Expected four raw box coordinates")
    ymin, xmin, ymax, xmax = [max(0, min(1000, int(v))) for v in raw_box]
    if ymax <= ymin or xmax <= xmin:
        raise ValueError("Invalid normalized box coordinates")
    x1, y1 = round(xmin / 1000 * image_width), round(ymin / 1000 * image_height)
    x2, y2 = round(xmax / 1000 * image_width), round(ymax / 1000 * image_height)
    return x1, y1, x2, y2
