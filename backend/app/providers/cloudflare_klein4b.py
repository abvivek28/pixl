from __future__ import annotations
from abc import ABC, abstractmethod
from hashlib import sha256
from pathlib import Path
from io import BytesIO
import base64, os, time
import httpx
from PIL import Image

class EditProvider(ABC):
    """Provider contract: edit a crop and return a PIL image."""
    name = "abstract"
    model = "abstract"
    @abstractmethod
    def edit(self, image: Image.Image, instruction: str, out_size: tuple[int, int],
             seed: int | None, *, preserve: list[str] | None = None) -> Image.Image:
        raise NotImplementedError

def edit_prompt(instruction: str, preserve: list[str] | None = None) -> str:
    keep = preserve or ["all objects, composition, geometry, and details outside the requested edit"]
    preserve_text = "; ".join(p.strip() for p in keep if p.strip())
    return (
        f"Edit image 0: {instruction.strip()}.\n\n"
        "PIXL IMAGE-EDITING GUARDRAILS (apply together with the user's requested edit):\n"
        "1. Maintain the exact framing, composition, and aspect ratio of the supplied image; do not stretch, squeeze, or change its proportions.\n"
        "2. Preserve the existing composition and geometry: do not zoom, reframe, rotate, shift, stretch, squeeze, or crop the scene.\n"
        "3. Maintain the relative size, position, shape, and proportions of every object and feature. Do not resize objects relative to one another.\n"
        "4. Change only what the user's instruction explicitly asks to change. Keep all other objects, edges, textures, fine details, and spatial relationships unchanged.\n"
        "5. Preserve the original background and overall colour grade, white balance, exposure, contrast, saturation, shadows, and lighting. Do not apply a global style or colour-grade change unless explicitly requested.\n"
        "6. Match the surrounding lighting, colour temperature, grain, texture, and sharpness at the edited area so it blends naturally.\n"
        f"7. Additional details to preserve: {preserve_text}.\n"
        "8. Return only the edited image, at the exact same framing and dimensions as the input.\n"
        "IT IS VERY IMPORTANT TO NOT CHANGE COLOUR GRADING OF IMAGE OR BACKGROUND. THE COMPLETE IMAGE PROPORTIONS MUST BE SAME AS INPUT IMAGE. YOU SHOULD ONLY MAKE THE LOCALIZED CHANGE AS SPECIFIED BY USER."
    )

class ProviderError(RuntimeError):
    pass

class CloudflareKlein4B(EditProvider):
    """Cloudflare Workers AI FLUX.2 [klein] 4B multipart provider with retries and disk cache."""
    name = "cloudflare"
    model = "@cf/black-forest-labs/flux-2-klein-4b"

    def __init__(self, account_id: str | None = None, api_token: str | None = None, *,
                 model: str | None = None, cache_dir: str | Path | None = None,
                 timeout: float | None = None, retries: int | None = None,
                 client: httpx.Client | None = None, sleep=time.sleep):
        self.account_id = account_id or os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
        self.api_token = api_token or os.getenv("CLOUDFLARE_API_TOKEN", "")
        self.model = model or os.getenv("CLOUDFLARE_KLEIN_MODEL", self.model)
        self.cache_dir = Path(cache_dir or os.getenv("PIXL_PROVIDER_CACHE_DIR", str(Path(__file__).resolve().parents[2] / "data" / "provider_cache")))
        self.timeout = timeout or float(os.getenv("PIXL_PROVIDER_TIMEOUT_SECONDS", "90"))
        self.retries = retries if retries is not None else int(os.getenv("PIXL_PROVIDER_RETRIES", "3"))
        self.client = client or httpx.Client(timeout=self.timeout)
        self._sleep = sleep

    def _cache_key(self, image_bytes: bytes, prompt: str, out_size: tuple[int, int], seed: int | None) -> str:
        payload = b"\0".join([image_bytes, prompt.encode(), f"{out_size[0]}x{out_size[1]}".encode(), str(seed).encode(), self.name.encode(), self.model.encode()])
        return sha256(payload).hexdigest()

    @staticmethod
    def _decode_response(response: httpx.Response) -> bytes:
        if response.headers.get("content-type", "").lower().startswith("image/"):
            return response.content
        try:
            payload = response.json()
        except ValueError as exc:
            raise ProviderError("Cloudflare returned neither image bytes nor JSON.") from exc
        result = payload.get("result", payload) if isinstance(payload, dict) else {}
        candidates = [result.get("image"), result.get("data"), result.get("output")] if isinstance(result, dict) else []
        for candidate in candidates:
            if isinstance(candidate, str):
                try:
                    return base64.b64decode(candidate, validate=True)
                except Exception:
                    pass
            if isinstance(candidate, list) and all(isinstance(x, int) for x in candidate):
                return bytes(candidate)
        raise ProviderError(f"Cloudflare response did not contain an image field: {str(payload)[:500]}")

    def edit(self, image: Image.Image, instruction: str, out_size: tuple[int, int], seed: int | None,
             *, preserve: list[str] | None = None) -> Image.Image:
        if not self.account_id or not self.api_token:
            raise ProviderError("Set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN before using CloudflareKlein4B.")
        buf = BytesIO(); image.convert("RGB").save(buf, format="PNG"); image_bytes = buf.getvalue()
        prompt = edit_prompt(instruction, preserve)
        key = self._cache_key(image_bytes, prompt, out_size, seed)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.cache_dir / f"{key}.png"
        if cache_path.exists():
            with Image.open(cache_path) as cached: return cached.convert("RGB").copy()
        url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run/{self.model}"
        fields = {"prompt": prompt, "width": str(int(out_size[0])), "height": str(int(out_size[1]))}
        if seed is not None: fields["seed"] = str(int(seed))
        last_error = None
        for attempt in range(max(0, self.retries) + 1):
            try:
                response = self.client.post(url, headers={"Authorization": f"Bearer {self.api_token}"},
                    data=fields, files={"input_image_0": ("input.png", image_bytes, "image/png")}, timeout=self.timeout)
                status = response.status_code
                if (status == 429 or 500 <= status <= 599) and attempt < self.retries:
                    try: wait = min(30.0, max(0.0, float(response.headers.get("retry-after", ""))))
                    except ValueError: wait = min(8.0, .75 * (2 ** attempt))
                    self._sleep(wait); continue
                response.raise_for_status()
                output_bytes = self._decode_response(response)
                try:
                    with Image.open(BytesIO(output_bytes)) as generated: output = generated.convert("RGB").copy()
                except Exception as exc: raise ProviderError("Cloudflare response image could not be decoded.") from exc
                output.save(cache_path, format="PNG")
                return output
            except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError, ProviderError) as exc:
                last_error = exc
                status = getattr(getattr(exc, "response", None), "status_code", None)
                retryable = isinstance(exc, (httpx.TimeoutException, httpx.TransportError)) or status == 429 or (status is not None and 500 <= status <= 599)
                if attempt < self.retries and retryable:
                    self._sleep(min(8.0, .75 * (2 ** attempt))); continue
                if isinstance(exc, ProviderError): raise
                raise ProviderError(f"Cloudflare image edit failed: {exc}") from exc
        raise ProviderError(f"Cloudflare image edit failed after retries: {last_error}")
