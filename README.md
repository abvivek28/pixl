# Pixl — AI Image Workflow Boilerplate

Pixl is a FastAPI + React/Vite starter for an AI-powered image editor. Its product idea is to help people reach a high-quality result with fewer, more deliberate iterations by generating a starting image, refining specific parts, and comparing versions.

**Current status:** Phase 4 connects the React/Vite frontend to the Phase 3 job API. The UI supports image generation, uploads, manual target selection, optional Meta Muse Spark planning and multiple-choice clarification, background job progress, before/after comparison, heatmap display, receipt checks, and downloads. Generation and Quality edits require live Cloudflare credentials; Free edit mode uses the deterministic local fake provider.

## Phase 4 frontend workflow

The frontend is a single connected workspace with three steps:

- **Create:** prompt-to-image generation through the Phase 3 Cloudflare FLUX.1 schnell job endpoint, model-default output dimensions, job progress, generated PNG and send-to-editor action.
- **Edit:** upload an image, enter an instruction, draw/move/resize a manual region, optionally request a Meta Muse Spark plan and answer clarification questions, choose Free (local fake provider) or Quality (Cloudflare Klein 4B), then start and track a background edit.
- **Results:** interactive before/after reveal slider, heatmap toggle, quality-check list, metrics, warnings, result PNG download and JSON receipt download.

The UI never auto-applies model-predicted coordinates: a manually confirmed pixel region remains authoritative for safe localized edits.

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/` | API homepage/service index |
| GET | `/api/health` | Health check |
| POST | `/api/images` | Upload and store an image |
| GET | `/api/images` | List stored image metadata |
| GET | `/api/images/{image_id}` | Serve an image inline |
| POST | `/api/upload` | Validate, normalize, and save an image master |
| GET | `/api/uploads/{image_id}` | Serve the normalized PNG master |
| POST | `/api/generate` | Queue a Cloudflare FLUX.1 schnell generation job |
| POST | `/api/plan` | Create a Meta Muse Spark edit plan |
| POST | `/api/answer` | Submit clarification and re-plan |
| POST | `/api/edit` | Queue a deterministic localized edit job |
| GET | `/api/jobs/{job_id}` | Read job status and progress |
| GET | `/api/jobs/{job_id}/artifacts/{name}` | Fetch original/result/heatmap/receipt artifacts |
| GET | `/api/compare?original_id=...&edited_id=...` | Return metadata for a before/after comparison |

Phase 3 upload accepts valid PNG, JPEG, GIF, and WebP images up to 10 MB, applies EXIF orientation, and stores a normalized PNG master under `backend/data/uploads/`. Job artifacts are stored under `backend/data/jobs/`.

## Run on Windows PowerShell

Terminal 1 — backend:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn app.main:app --reload
```

API: http://127.0.0.1:8000  
Swagger docs: http://127.0.0.1:8000/docs

Terminal 2 — frontend, from the project root:

```powershell
cd frontend
npm install
npm run dev
```

Frontend: http://localhost:5173

Vite proxies `/api` requests to FastAPI at `http://127.0.0.1:8000`.

## Design notes

The global background retains one pattern at a time (a light-grey grid by default). Main content uses solid black, white, and orange, with restrained grey accents. Panels have hard black borders and solid offset projection shadows to give a 3D/extruded edge; there are no gradients on UI surfaces.

## Before production

Before production, add authentication and authorization, durable database-backed job metadata, cloud/object storage, malware scanning, rate limits, observability, and production deployment settings. Phase 3 job state is currently held in memory and is reset when the API process restarts.


## Phase 1 — deterministic edit engine

The backend now includes a deterministic, no-AI engine that ingests an image into a normalized RGB PNG master, expands a selected pixel box, prepares model-sized crop metadata, runs a local hue-shift fake edit, aligns colors in OKLab, composites only the selected box with an inside-only feather, and writes a full-size heatmap plus a JSON receipt.

Install backend dependencies, then run tests:

```powershell
cd backend
python -m pip install -r requirements.txt
python -m pytest -q
```

Run the command-line demo:

```powershell
python scripts/run_fake_edit.py C:\path\to\photo.jpg --box 100 100 400 350 --output data\demo
```

The output folder contains `master.png`, `result.png`, `heatmap.png`, and `receipt.json`. Box coordinates use pixel-space half-open rectangles: `x1 y1 x2 y2`.

Phase 1 intentionally uses a deterministic local fake edit. Cloud providers and Meta Muse Spark planning are implemented in Phase 2; the HTTP job API is implemented in Phase 3.

## Phase 2 — providers and Meta Muse Spark planning

Phase 2 adds a provider interface, Cloudflare FLUX.2 [klein] 4B multipart image editing, retries for rate limits/server errors, timeout handling, disk caching keyed by image/prompt/size/seed/provider/model, Meta Muse Spark 1.3 Contributor structured planning, throttling, manual-box fallback, and a CLI orchestrator.

### Configure credentials

Copy `backend/.env.example` to `backend/.env` and set `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN`, and optionally `MODEL_API_KEY`. The Phase 3 API loads `backend/.env` automatically when `python-dotenv` is installed; CLI scripts can also use environment variables exported in your shell. Never commit real tokens.

PowerShell example:

```powershell
$env:CLOUDFLARE_ACCOUNT_ID="your_account_id"
$env:CLOUDFLARE_API_TOKEN="your_token"
$env:MODEL_API_KEY="your_meta_key"
```

Cloudflare's configured default model is `@cf/black-forest-labs/flux-2-klein-4b`, which accepts multipart form data including `input_image_0`, prompt, dimensions, and seed. See the official [Cloudflare FLUX.2 Klein 4B documentation](https://developers.cloudflare.com/workers-ai/models/flux-2-klein-4b/). Meta Muse Spark uses the official Meta Model API Chat Completions endpoint with Bearer authentication and schema-constrained JSON output; image planning sends a base64 data URL in a user-message `image_url` content part. See the [Meta API overview](https://dev.meta.ai/docs/overview), [structured-output guide](https://dev.meta.ai/docs/structured-output), and [image-understanding guide](https://dev.meta.ai/docs/image-understanding).

### Run the CLI

Offline fake-provider path:

```powershell
cd backend
python scripts/run_edit.py "C:\path\to\photo.jpg" "make the jacket blue" --box 100 80 450 500 --output data\demo
```

Real Cloudflare provider (requires credentials):

```powershell
python scripts/run_edit.py "C:\path\to\photo.jpg" "make the jacket blue" --box 100 80 450 500 --provider cloudflare --output data\cloudflare-demo
```

Enable Meta Muse Spark planning while still supplying a manual box:

```powershell
python scripts/run_edit.py "C:\path\to\photo.jpg" "make the jacket blue" --box 100 80 450 500 --plan-with-meta --provider cloudflare --output data\planned-demo
```

For a 10-photo smoke run, create a JSON manifest as described in `backend/scripts/run_batch.py`'s module docstring, then run `python scripts/run_batch.py manifest.json --provider fake` (or `--provider cloudflare`).

### Important coordinate safety gate

Meta Muse Spark's raw `box_2d` is kept unconverted by default. The project intentionally does not enable automatic coordinate conversion for model-predicted boxes. Keep using a confirmed manual box; this prevents a plausible-looking but incorrectly converted box from silently editing the wrong area.

### Tests

```powershell
cd backend
python -m pytest -q
```

Provider/Meta Muse Spark HTTP tests are mocked and require no paid API keys. Live Cloudflare/Meta Muse Spark requests still need to be run with your credentials; they are not claimed as live-verified in this package.


## Phase 3 — HTTP API and background jobs

Phase 3 adds the job-oriented FastAPI endpoints while preserving the legacy `/api/images` routes used by the existing frontend. Uploads are decoded, EXIF orientation is applied, and the stored master is normalized to RGB PNG. Edit jobs use the deterministic Phase 1 compositor and the Phase 2 fake or Cloudflare edit provider. Generation jobs call Cloudflare FLUX.1 schnell. Jobs run in a background thread pool and expose status/progress and artifact URLs.

### Start the API (PowerShell)

From the project root (`pixl_image_editor`):

```powershell
python -m pip install -r backend\requirements.txt
python -m uvicorn app.main:app --app-dir backend --reload
```

Open `http://127.0.0.1:8000/docs` for interactive API documentation. For real Cloudflare generation/editing, set `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN` in `backend/.env`. The fake edit provider does not require API credentials. Meta Muse Spark planning requires `MODEL_API_KEY` and defaults to `muse-spark-1.3-contributor`. Important: Contributor mode allows Meta to use submitted prompts, responses, and image inputs to improve its models in exchange for lower pricing. Use `META_MODEL=muse-spark-1.3` for Standard mode if you do not want Contributor mode.

### Endpoints

- `POST /api/upload` — multipart form field `file`; returns `image_id`, dimensions, SHA-256, and a URL for the normalized master.
- `GET /api/uploads/{image_id}` — serves the normalized original PNG.
- `POST /api/generate` — JSON body such as `{"prompt":"a glass greenhouse in a field at sunrise","steps":4}`; returns HTTP 202 and a `job_id`. Uses Cloudflare `@cf/black-forest-labs/flux-1-schnell`. The request sends only the documented `prompt` and `steps` fields; output dimensions are selected by the model and recorded in the receipt. Cloudflare error messages are included in failed job details.
- `POST /api/plan` — JSON body `{"image_id":"...","instruction":"make the cup blue"}`; returns the Meta Muse Spark planning result and `plan_id`.
- `POST /api/answer` — JSON body `{"plan_id":"...","answer":"the cup on the left"}`; stores the answer and replans with clarification context.
- `POST /api/edit` — JSON body `{"image_id":"...","instruction":"make the cup blue","box":[100,80,450,400],"provider":"fake","seed":0}`; returns HTTP 202 and a `job_id`. `box` uses pixel coordinates `[x1,y1,x2,y2]`. A confirmed manual box is required; model-predicted coordinates are advisory only.
- `GET /api/jobs/{job_id}` — returns status, progress, stage, error (if any), metrics, checks, and artifact URLs.
- `GET /api/jobs/{job_id}/artifacts/{name}` — serves `original`, `result`, `heatmap`, `receipt`, or `result_json` when available.

### Example PowerShell smoke test

Upload a local image (replace `sample.png` with your file):

```powershell
$upload = curl.exe -s -X POST http://127.0.0.1:8000/api/upload -F "file=@sample.png" | ConvertFrom-Json
$upload | Format-List

$body = @{ image_id = $upload.image_id; instruction = "make the central area brighter"; box = @(20, 15, 75, 65); provider = "fake"; seed = 0 } | ConvertTo-Json
$job = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/edit -ContentType "application/json" -Body $body
$job
Invoke-RestMethod "http://127.0.0.1:8000/api/jobs/$($job.job_id)"
```

Poll `GET /api/jobs/{job_id}` until `status` is `completed` or `failed`. Artifact links are returned only once their files are available.

### Phase 3 tests

```powershell
$env:PYTHONPATH = "$PWD\backend"
python -m pytest backend\tests -q
```

The tests exercise upload validation/normalization, plan/answer flow, manual-box safety, background edit jobs, and artifact serving without making live paid provider calls. Live Cloudflare and Meta Muse Spark calls still require valid credentials.


## Phase 4 — connected frontend

The frontend uses Vite + React and the existing `/api` proxy. From the project root, start the backend in one terminal:

```powershell
python -m pip install -r backend\requirements.txt
$env:PYTHONPATH = "$PWD\backend"
python -m uvicorn app.main:app --app-dir backend --reload
```

In a second terminal, install frontend dependencies and start Vite:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The frontend polls background jobs until they complete or fail. Configure `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN` in `backend/.env` for generation and Quality edit mode; configure `MODEL_API_KEY` for the optional planning/clarification flow. Free edit mode does not require provider credentials.

**Verification note:** the 41 backend tests pass. JSX/JavaScript transpilation parsing was checked. A full Vite production build was not run in this environment because npm package registry access timed out; run `npm install` and `npm run build` locally to verify the installed dependency environment.
