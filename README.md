# Pixl — AI Image Editing Workspace

Pixl is an AI-powered image creation and localized image-editing application. It combines a React/Vite frontend with a FastAPI backend and Cloudflare Workers AI image models. Users can generate images, upload existing images, select a region with a freehand polygon, describe an edit, and review the resulting image alongside a verification receipt.

## Features

- **Image creation:** Generate images from text prompts using Cloudflare FLUX.1 schnell.
- **Localized editing:** Upload an image, draw a freehand selection, and request an edit using Cloudflare FLUX.2 Klein 4B.
- **Optional prompt planning:** Use Meta Muse Spark to help interpret an edit instruction and ask clarification questions. Manual user selections remain authoritative.
- **Context-aware generation:** The backend expands the selected region with surrounding image context before sending it to the editing model.
- **Geometric registration:** OpenCV-based phase correlation, constrained ECC initialization, and bounded scale/translation search estimate whether the generated crop needs alignment. A separate held-out context check decides whether a proposed transform is safe to apply.
- **Controlled compositing:** The generated crop is combined with the original through the user's selection mask with inward-only feathering.
- **Verification:** Inspect pixel-change, colour-drift, seam, halo, texture, context and alignment measurements.
- **Export:** Download the result image, heatmap, JSON receipt and PDF change receipt.

## Technology stack

| Layer | Technology |
|---|---|
| Frontend | React 18, Vite, JavaScript, CSS |
| Backend | Python, FastAPI, Uvicorn |
| Image processing | Pillow, NumPy, OpenCV |
| Image generation/editing | Cloudflare Workers AI — FLUX.1 schnell and FLUX.2 Klein 4B |
| Optional planning | Meta Muse Spark API |
| Receipt generation | JSON and ReportLab PDF |

## Project structure

```text
pixl/
├── backend/
│   ├── app/
│   │   ├── api_v3.py          # HTTP API, jobs and artifact delivery
│   │   ├── engine/            # Crop, normalization, alignment, compositing, verification
│   │   ├── planning/          # Optional Meta-assisted planning
│   │   └── providers/         # Image model integrations
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── src/                   # React application and styles
│   ├── public/                # Static assets and browser icon
│   ├── index.html
│   ├── package.json
│   └── .env.example
└── README.md
```

## Requirements

- Python 3.10 or newer
- Node.js 18 or newer and npm
- Cloudflare account ID and Workers AI API token for live image generation/editing
- Meta API key only if optional Meta planning is enabled

## Local development (Windows PowerShell)

### 1. Configure the backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set the credentials and configuration values in `backend/.env`. At minimum, configure:

```dotenv
CLOUDFLARE_ACCOUNT_ID=your_cloudflare_account_id
CLOUDFLARE_API_TOKEN=your_cloudflare_api_token
BACKEND_CORS_ORIGINS=http://localhost:5173
```

Optional Meta planning configuration:

```dotenv
MODEL_API_KEY=your_meta_api_key
META_MODEL=muse-spark-1.3-contributor
```

Start the API from the project root:

```powershell
$env:PYTHONPATH = "$PWDackend"
python -m uvicorn app.main:app --app-dir backend --reload
```

The API is available at `http://127.0.0.1:8000`; interactive API documentation is at `http://127.0.0.1:8000/docs`.

### 2. Configure and start the frontend

In a second terminal:

```powershell
cd frontend
npm install
Copy-Item .env.example .env.development
npm run dev
```

Open `http://localhost:5173`.

For local development, `VITE_API_BASE_URL` can remain empty and Vite proxies `/api` requests to the backend using `VITE_API_PROXY_TARGET`.

## Deployment configuration

The frontend and backend can be deployed separately.

- **Frontend (for example, Vercel):** Set `VITE_API_BASE_URL` to the backend origin, such as `https://your-api.onrender.com`. Do not include `/api`; the frontend API module adds the route prefix.
- **Backend (for example, Render):** Set `BACKEND_CORS_ORIGINS` to the deployed frontend origin. Configure `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN` in the backend environment.
- Set `MODEL_API_KEY` and `META_MODEL` only if Meta planning is required.
- Never place provider secrets in frontend/Vite environment variables or commit real `.env` files.

See `frontend/.env.example` and `backend/.env.example` for supported configuration names.

## API overview

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/health` | Backend health check |
| `POST` | `/api/upload` | Upload and normalize an image |
| `GET` | `/api/uploads/{image_id}` | Retrieve an uploaded image |
| `POST` | `/api/generate` | Queue text-to-image generation |
| `POST` | `/api/plan` | Request optional edit planning |
| `POST` | `/api/answer` | Submit a planning clarification |
| `POST` | `/api/edit` | Queue a localized image edit |
| `GET` | `/api/jobs/{job_id}` | Read job status, metrics and artifacts |
| `GET` | `/api/jobs/{job_id}/artifacts/{artifact_name}` | Retrieve an output artifact |
| `GET` | `/api/jobs/{job_id}/receipt.pdf` | Generate/download the PDF receipt |

## Editing and verification notes

The editing model receives an expanded rectangular crop and an instruction; it does not receive the frontend's freehand polygon as a dedicated inpainting mask. The polygon is used by Pixl's deterministic compositor after generation. Therefore, content changed by the model inside the selected polygon can be included in the result, even if it is surrounding background within that selection.

The registration stage estimates uniform scale and X/Y translation only. It does not correct perspective changes, object deformation or semantic reconstruction errors. If the candidate transform does not improve independent context validation sufficiently, Pixl retains the identity transform.

The receipt records calculated metrics, checks, thresholds, metric explanations, alignment diagnostics, seam diagnostics and artifact references. These checks are quantitative indicators rather than a guarantee of perceptual or semantic quality; results should still be visually reviewed.
