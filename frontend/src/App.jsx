import { useEffect, useMemo, useRef, useState } from "react";
import { API_BASE_URL, apiAssetUrl } from "./config/api";
import {
  ArrowRight, Check, CircleHelp, Download, FileImage, FileJson, FileText, ImagePlus,
  LoaderCircle, Move, Scan, Sparkles, UploadCloud, WandSparkles, X, Zap,
  ShieldCheck, RefreshCw, SlidersHorizontal, ChevronRight
} from "lucide-react";

const API = API_BASE_URL;
const assetUrl = apiAssetUrl;
const fmtBytes = (n = 0) => n < 1024 ? `${n} B` : n < 1048576 ? `${(n / 1024).toFixed(1)} KB` : `${(n / 1048576).toFixed(1)} MB`;
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  let data;
  try { data = await response.json(); } catch { data = {}; }
  if (!response.ok) {
    const detail = typeof data.detail === "string" ? data.detail : data.detail ? JSON.stringify(data.detail) : data.error?.message;
    throw new Error(detail || `Request failed (${response.status}).`);
  }
  return data;
}

async function waitForJob(jobId, onUpdate) {
  const started = Date.now();
  while (Date.now() - started < 240000) {
    const job = await requestJson(`${API}/jobs/${jobId}`);
    onUpdate?.(job);
    if (job.status === "completed") return job;
    if (job.status === "failed") throw new Error(job.error?.message || "The job failed.");
    await delay(1000);
  }
  throw new Error("This job is taking longer than expected. Check its status and try again.");
}

function Header({ view, setView, apiState }) {
  return <header className="p4-topbar">
    <button className="p4-brand" onClick={() => setView("home")} aria-label="Pixl home"><span className="p4-brand-mark">p<span>.</span></span><span className="p4-brand-copy"><strong>pixl</strong><small>IMAGE INTELLIGENCE</small></span></button>
    <nav className="p4-nav" aria-label="Main workflow">
      {[["home", "Home"], ["create", "Create"], ["edit", "Edit"], ["results", "Results"]].map(([id, label]) => <button key={id} className={view === id ? "active" : ""} onClick={() => setView(id)}>{label}</button>)}
    </nav>
    <div className="p4-api-state"><i className={apiState === "online" ? "online" : apiState === "offline" ? "offline" : ""} />{apiState === "online" ? "API CONNECTED" : apiState === "offline" ? "API OFFLINE" : "CHECKING API"}</div>
  </header>;
}

function StepHeading({ index, title, detail }) {
  return <div className="p4-step-heading"><span className="p4-step-index">{index}</span><div><h2>{title}</h2>{detail && <p>{detail}</p>}</div></div>;
}

function Notice({ type = "info", children, onClose }) {
  return <div className={`p4-notice ${type}`} role={type === "error" ? "alert" : "status"}><span className="p4-notice-mark">{type === "error" ? "!" : type === "success" ? "✓" : "i"}</span><div>{children}</div>{onClose && <button aria-label="Dismiss message" onClick={onClose}><X size={15} /></button>}</div>;
}

function UploadDrop({ onFile, fileName }) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);
  function accept(file) {
    if (!file) return;
    if (!file.type.startsWith("image/")) return;
    if (file.size > 10 * 1024 * 1024) return;
    onFile(file);
  }
  return <div className={`p4-dropzone ${dragging ? "dragging" : ""}`} onDragOver={(e) => { e.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={(e) => { e.preventDefault(); setDragging(false); accept(e.dataTransfer.files?.[0]); }}>
    <input ref={inputRef} className="p4-visually-hidden" type="file" accept="image/png,image/jpeg,image/webp,image/gif" onChange={(e) => { accept(e.target.files?.[0]); e.target.value = ""; }} />
    <div className="p4-drop-icon"><UploadCloud size={22} /></div>
    <strong>{fileName || (dragging ? "Drop it here" : "Drop an image here")}</strong>
    <span>PNG, JPG, WEBP or GIF · maximum 10 MB</span>
    <button type="button" className="p4-button p4-button-light" onClick={() => inputRef.current?.click()}>Browse files <ChevronRight size={15} /></button>
  </div>;
}

function ImageCanvas({ imageSrc, box, setBox, points, setPoints, dimensions, setDimensions, emptyLabel = "Upload an image to begin" }) {
  const wrapperRef = useRef(null);
  const currentPoints = useRef(points || []);
  currentPoints.current = points || [];
  const [drawing, setDrawing] = useState(false);
  const [hint, setHint] = useState("Click and drag on the image to draw a freehand target region");

  function imageMetrics() {
    const image = wrapperRef.current?.querySelector("img");
    const rect = image?.getBoundingClientRect();
    const width = image?.naturalWidth || dimensions.width;
    const height = image?.naturalHeight || dimensions.height;
    if (!rect || !rect.width || !rect.height || !width || !height) return null;
    return { rect, width, height };
  }
  function point(e) {
    const m = imageMetrics();
    if (!m) return { x: 0, y: 0 };
    return {
      x: Math.max(0, Math.min(m.width, Math.round(((e.clientX - m.rect.left) / m.rect.width) * m.width))),
      y: Math.max(0, Math.min(m.height, Math.round(((e.clientY - m.rect.top) / m.rect.height) * m.height))),
    };
  }
  function begin(e) {
    if (!imageSrc) return;
    e.preventDefault(); e.stopPropagation();
    wrapperRef.current?.setPointerCapture?.(e.pointerId);
    const p = point(e);
    currentPoints.current = [p];
    setPoints([p]);
    setBox(null);
    setDrawing(true);
  }
  function move(e) {
    if (!drawing) return;
    const p = point(e);
    const pts = currentPoints.current;
    const last = pts[pts.length - 1];
    if (last && Math.hypot(p.x - last.x, p.y - last.y) < 2) return;
    pts.push(p);
    setPoints([...pts]);
  }
  function end() {
    if (!drawing) return;
    setDrawing(false);
    const pts = currentPoints.current;
    if (pts && pts.length >= 3) {
      const xs = pts.map((p) => p.x);
      const ys = pts.map((p) => p.y);
      const minX = Math.max(0, Math.min(...xs));
      const minY = Math.max(0, Math.min(...ys));
      const maxX = Math.min(dimensions.width || 1, Math.max(...xs));
      const maxY = Math.min(dimensions.height || 1, Math.max(...ys));
      const w = dimensions.width || 1;
      const h = dimensions.height || 1;
      const b = {
        x: (minX / w) * 100,
        y: (minY / h) * 100,
        w: Math.max(0.5, ((maxX - minX) / w) * 100),
        h: Math.max(0.5, ((maxY - minY) / h) * 100),
      };
      setBox(b);
      setHint("Freehand region selected · click Clear selection to redraw");
    } else {
      setPoints(null);
      setBox(null);
      setHint("Click and drag on the image to draw a freehand target region");
    }
  }
  const selectedPixels = box && dimensions.width && dimensions.height
    ? `${Math.round(box.w * dimensions.width / 100)} × ${Math.round(box.h * dimensions.height / 100)} px · Freehand`
    : hint;
  return <div className="p4-canvas-shell">
    {imageSrc ? <div ref={wrapperRef} className={`p4-image-canvas ${drawing ? "drawing" : ""}`} onPointerDown={begin} onPointerMove={move} onPointerUp={end} onPointerCancel={end} style={{ touchAction: "none", userSelect: "none", cursor: "crosshair" }}>
      <img src={imageSrc} alt="Image being edited" draggable="false" onLoad={(e) => setDimensions({ width: e.currentTarget.naturalWidth, height: e.currentTarget.naturalHeight })} />
      {points && points.length > 1 && (
        <svg
          viewBox={`0 0 ${dimensions.width || 1} ${dimensions.height || 1}`}
          style={{ position: "absolute", left: 0, top: 0, width: "100%", height: "100%", pointerEvents: "none" }}
        >
          {drawing ? (
            <polyline
              points={points.map((p) => `${p.x},${p.y}`).join(" ")}
              fill="none"
              stroke="#ff5a1f"
              strokeWidth={Math.max(2.5, (dimensions.width || 500) / 280)}
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          ) : (
            <polygon
              points={points.map((p) => `${p.x},${p.y}`).join(" ")}
              fill="rgba(255, 90, 31, 0.22)"
              stroke="#ff5a1f"
              strokeWidth={Math.max(2, (dimensions.width || 500) / 320)}
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          )}
        </svg>
      )}
      {box && !drawing && (
        <div style={{ position: "absolute", left: `${box.x}%`, top: `${box.y}%`, pointerEvents: "none" }}>
          <span className="p4-selection-tag"><Move size={10} /> FREEHAND TARGET</span>
        </div>
      )}
    </div> : <div className="p4-canvas-empty"><ImagePlus size={30} /><strong>{emptyLabel}</strong><span>Your image stays unchanged outside the selected region.</span></div>}
    <div className="p4-canvas-footer"><span><i className="p4-orange-dot" />{selectedPixels}</span>{(box || points) && <button onClick={() => { setBox(null); setPoints(null); }}>Clear selection <X size={13} /></button>}</div>
  </div>;
}
function JobProgress({ job }) {
  if (!job) return null;
  const status = job.status || "queued";
  return <div className="p4-progress-card"><div className="p4-progress-top"><span><LoaderCircle className={status === "completed" ? "" : "p4-spin"} size={15} />{status === "completed" ? "Completed" : status === "failed" ? "Job failed" : (job.stage || status).replaceAll("_", " ")}</span><strong>{job.progress ?? 0}%</strong></div><div className="p4-progress-track"><span style={{ width: `${Math.max(0, Math.min(100, job.progress || 0))}%` }} /></div><small>{status === "completed" ? "Your artifacts are ready." : status === "failed" ? job.error?.message : "Pixl is processing this job. You can keep this tab open."}</small></div>;
}

function ReceiptPanel({ receipt, job, onDownloadPdf, onDownloadJson }) {
  const checks = receipt?.checks || job?.checks || [];
  const metrics = receipt?.metrics || job?.metrics || {};
  if (!receipt && !job) return <div className="p4-empty-receipt"><ShieldCheck size={21} /><span>The verification receipt will appear after an edit completes.</span></div>;
  return <div className="p4-receipt">
    <div className="p4-receipt-head"><div><span className="p4-overline">EDIT RECEIPT</span><h3>Verification summary</h3></div><span className="p4-receipt-stamp"><ShieldCheck size={16} /> {checks.length && checks.every((c) => c.pass) ? "PASS" : receipt ? "REVIEW" : "PENDING"}</span></div>
    {receipt?.summary && <p className="p4-receipt-summary">{receipt.summary}</p>}
    <div className="p4-check-list">{checks.length ? checks.map((check, i) => <div className="p4-check-row" key={`${check.name || check.check || "check"}-${i}`}><span className={check.pass ? "p4-check-icon pass" : "p4-check-icon fail"}>{check.pass ? <Check size={13} /> : <X size={13} />}</span><span>{(check.name || check.check || `Check ${i + 1}`).replaceAll("_", " ")}</span><strong>{check.pass ? "PASS" : "REVIEW"}</strong></div>) : <p className="p4-muted">Detailed checks are not available for this job type.</p>}</div>
    {Object.keys(metrics).length > 0 && <>
      <div className="p4-metric-section-title"><strong>Measured metrics</strong><span>{Object.keys(metrics).length} values recorded</span></div>
      <div className="p4-metric-grid p4-metric-grid-all">{Object.entries(metrics).map(([key, value]) => {
        const definition = receipt?.metric_legend?.[key];
        const label = definition?.label || key.replaceAll("_", " ");
        const display = typeof value === "number" ? Number(value.toFixed(4)).toString() : String(value);
        return <div key={key} title={definition?.purpose || label}><span>{label}</span><strong>{display}{definition?.unit === "percent" ? "%" : ""}</strong><small>{definition?.purpose || "Recorded processing metric."}</small></div>;
      })}</div>
      <details className="p4-metric-legend"><summary><CircleHelp size={14} /> Metric legend — what each value measures</summary>
        <div className="p4-metric-legend-list">{Object.entries(metrics).map(([key]) => {
          const definition = receipt?.metric_legend?.[key];
          return <div key={key}><strong>{definition?.label || key.replaceAll("_", " ")}</strong><span>{definition?.purpose || "Recorded processing metric."}</span><small>Unit: {definition?.unit || "as recorded"}</small></div>;
        })}</div>
      </details>
    </>}
    {receipt?.alignment && <div className="p4-alignment-summary">
      <div className="p4-alignment-heading"><strong>Geometric alignment</strong><span>{receipt.alignment.applied ? "APPLIED" : "IDENTITY RETAINED"}</span></div>
      <div className="p4-alignment-values">
        <div><span>Scale</span><strong>{Number(receipt.alignment.scale).toFixed(4)}x</strong></div>
        <div><span>Horizontal</span><strong>{Number(receipt.alignment.translation_x).toFixed(1)} px</strong></div>
        <div><span>Vertical</span><strong>{Number(receipt.alignment.translation_y).toFixed(1)} px</strong></div>
        <div><span>Applied-score improvement</span><strong>{Number(receipt.alignment.improvement_percent).toFixed(2)}%</strong></div>
        <div><span>Validated proposal zoom</span><strong>{Number(receipt.alignment.proposed_scale ?? 1).toFixed(4)}x</strong></div>
        <div><span>Fit-best zoom</span><strong>{Number(receipt.alignment.fit_best_scale ?? 1).toFixed(4)}x</strong></div>
        <div><span>Fit-best X / Y</span><strong>{Number(receipt.alignment.fit_best_translation_x ?? 0).toFixed(2)} / {Number(receipt.alignment.fit_best_translation_y ?? 0).toFixed(2)} px</strong></div>
        <div><span>Independent validation</span><strong>{Number(receipt.alignment.validation_improvement_percent || 0).toFixed(2)}%</strong></div>
        <div><span>Context SSIM</span><strong>{Number(receipt.alignment.context_ssim || 0).toFixed(3)}</strong></div>
        <div><span>Gradient match</span><strong>{Number(receipt.alignment.context_gradient_similarity || 0).toFixed(3)}</strong></div>
        <div><span>Context coverage</span><strong>{(Number(receipt.alignment.context_coverage || 0) * 100).toFixed(1)}%</strong></div>
        <div><span>Stable context pixels</span><strong>{Number(receipt.alignment.registration_context_pixels || 0).toLocaleString()}</strong></div>
        <div><span>Validation pixels</span><strong>{Number(receipt.alignment.validation_context_pixels || 0).toLocaleString()}</strong></div>
        <div><span>Candidates</span><strong>{Number(receipt.alignment.candidates_evaluated || 0).toLocaleString()}</strong></div>
      </div>
      <small>{receipt.alignment.reason}</small>
      {receipt.seam && <div className="p4-seam-summary"><strong>Polygon seam diagnostics</strong><span>Mean excess: {Number(receipt.seam.excess_boundary_step || 0).toFixed(2)} · P95 excess: {Number(receipt.seam.p95_excess_boundary_step || 0).toFixed(2)} · Halo: {Number(receipt.seam.halo_score || 0).toFixed(2)}</span></div>}
      {receipt.compositing && <div className="p4-seam-summary"><strong>Selection boundary integrity</strong><span>Changed outside selection: {Number(receipt.compositing.changed_outside_hard_selection_pixels || 0).toLocaleString()} px · Feather-band changed: {Number(receipt.metrics?.feather_band_changed_pixels || 0).toLocaleString()} px · Feather: inward only</span></div>}
    </div>}
    {receipt?.warnings?.length > 0 && <div className="p4-warnings"><strong>Warnings</strong>{receipt.warnings.map((warning, i) => <p key={i}>{warning}</p>)}</div>}
    {onDownloadPdf && <button className="p4-button p4-button-orange p4-full" onClick={onDownloadPdf}><FileText size={15} /> Get Receipt of Changes</button>}
    {onDownloadJson && <button className="p4-button p4-button-light p4-full p4-json-button" onClick={onDownloadJson}><FileJson size={15} /> Download raw JSON data</button>}
  </div>;
}

export default function App() {
  const [view, setView] = useState("home");
  const [apiState, setApiState] = useState("checking");
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [selectionError, setSelectionError] = useState("");
  const [busy, setBusy] = useState(false);
  const [receiptPrinting, setReceiptPrinting] = useState(false);
  const [sourceFile, setSourceFile] = useState(null);
  const [source, setSource] = useState(null);
  const [localPreview, setLocalPreview] = useState("");
  const [prompt, setPrompt] = useState("");
  const [generationJob, setGenerationJob] = useState(null);
  const [editJob, setEditJob] = useState(null);
  const [plan, setPlan] = useState(null);
  const [answerChoice, setAnswerChoice] = useState("");
  const [box, setBox] = useState(null);
  const [points, setPoints] = useState(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  const [resultUrl, setResultUrl] = useState("");
  const [heatmapUrl, setHeatmapUrl] = useState("");
  const [receipt, setReceipt] = useState(null);
  const [comparePosition, setComparePosition] = useState(50);
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [generatedPrompt, setGeneratedPrompt] = useState("");
  const [generationPlan, setGenerationPlan] = useState(null);
  const [generationAnswers, setGenerationAnswers] = useState({});
  const [generationImprovedPrompt, setGenerationImprovedPrompt] = useState("");

  const previousView = useRef(view);
  useEffect(() => { fetch(`${API}/health`).then((r) => { if (!r.ok) throw new Error(); setApiState("online"); }).catch(() => setApiState("offline")); }, []);
  useEffect(() => {
    const previous = previousView.current;
    if (previous !== view && (["create", "edit"].includes(previous) || ["create", "edit"].includes(view))) {
      setPrompt("");
      setPlan(null);
      setAnswerChoice("");
      setGenerationPlan(null);
      setGenerationAnswers({});
      setGenerationImprovedPrompt("");
      setError("");
      setSuccess("");
    }
    previousView.current = view;
  }, [view]);
  useEffect(() => () => { if (localPreview) URL.revokeObjectURL(localPreview); }, [localPreview]);
  const sourceUrl = useMemo(() => source ? assetUrl(source.url || `/uploads/${source.image_id}`) : localPreview, [source, localPreview]);

  function clearResult() { setEditJob(null); setResultUrl(""); setHeatmapUrl(""); setReceipt(null); setShowHeatmap(false); }
  function chooseSource(file) {
    if (file.size > 10 * 1024 * 1024) { setError("Image must be 10 MB or smaller."); return; }
    if (localPreview) URL.revokeObjectURL(localPreview);
    setSourceFile(file); setSource(null); setLocalPreview(URL.createObjectURL(file)); setBox(null); setPoints(null); setPlan(null); setPrompt(""); setDimensions({ width: 0, height: 0 }); clearResult(); setError(""); setSuccess("");
  }
  async function ensureUploaded() {
    if (source?.image_id) return source;
    if (!sourceFile) throw new Error("Upload an image first, or generate one in the Create tab.");
    const form = new FormData(); form.append("file", sourceFile);
    const uploaded = await requestJson(`${API}/upload`, { method: "POST", body: form });
    setSource(uploaded); setSourceFile(null);
    if (localPreview) URL.revokeObjectURL(localPreview);
    setLocalPreview("");
    return uploaded;
  }
  async function planGenerationPrompt() {
    if (!prompt.trim()) { setError("Describe the image you want to create before planning it."); return; }
    setBusy(true); setError(""); setSuccess(""); setGenerationPlan(null); setGenerationAnswers({}); setGenerationImprovedPrompt("");
    try {
      const data = await requestJson(`${API}/generation-plan`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ prompt: prompt.trim() }) });
      setGenerationPlan(data); setGenerationImprovedPrompt(data.improved_prompt || prompt.trim());
      if (data.status === "ready") setSuccess("Prompt improved. Review it, make any edits, then generate.");
    } catch (e) { setError(e.message); }
    finally { setBusy(false); }
  }
  async function submitGenerationAnswers() {
    if (!generationPlan?.plan_id) return;
    const questions = generationPlan.questions || [];
    if (!questions.length || questions.some((q) => !generationAnswers[q.id])) { setError("Answer each follow-up question before continuing, or generate directly with your original prompt."); return; }
    const answers = questions.map((q) => ({ id: q.id, question: q.question, answer: generationAnswers[q.id] }));
    setBusy(true); setError("");
    try {
      const data = await requestJson(`${API}/generation-answer`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ plan_id: generationPlan.plan_id, answers }) });
      setGenerationPlan(data); setGenerationAnswers({}); setGenerationImprovedPrompt(data.improved_prompt || prompt.trim());
      if (data.status === "ready") setSuccess("Prompt refined with your answers. Review the final prompt before generating.");
    } catch (e) { setError(e.message); }
    finally { setBusy(false); }
  }
  async function runGeneration() {
    if (!prompt.trim()) { setError("Describe the image you want to create."); return; }
    const finalPrompt = generationPlan?.status === "ready" && generationImprovedPrompt.trim() ? generationImprovedPrompt.trim() : prompt.trim();
    setBusy(true); setError(""); setSuccess(""); setGenerationJob(null);
    try {
      const started = await requestJson(`${API}/generate`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ prompt: finalPrompt, steps: 4 }) });
      setGenerationJob(started);
      const finished = await waitForJob(started.job_id, setGenerationJob);
      setGenerationJob(finished); setGeneratedPrompt(finalPrompt); setSuccess("Image generated. You can download it or send it to the editor.");
    } catch (e) { setError(e.message); }
    finally { setBusy(false); }
  }
  async function sendGeneratedToEditor() {
    const path = generationJob?.artifacts?.result;
    if (!path) return;
    setBusy(true); setError("");
    try {
      const response = await fetch(assetUrl(path)); if (!response.ok) throw new Error("Could not read the generated image artifact.");
      const blob = await response.blob(); const file = new File([blob], "pixl-generated.png", { type: "image/png" });
      const form = new FormData(); form.append("file", file);
      const uploaded = await requestJson(`${API}/upload`, { method: "POST", body: form });
      setSource(uploaded); setSourceFile(null); setLocalPreview(""); setBox(null); setPoints(null); setPlan(null); setPrompt(""); setDimensions({ width: uploaded.width, height: uploaded.height }); clearResult(); setView("edit"); setSuccess("Generated image is ready to edit.");
    } catch (e) { setError(e.message); }
    finally { setBusy(false); }
  }
  async function requestPlan() {
    if (!prompt.trim()) { setError("Describe what you want to change before asking Pixl to plan it."); return; }
    if (!boxPixels()) { setError("Draw a freehand target region on the image first so Meta has the exact region context."); return; }
    setBusy(true); setError(""); setPlan(null); setAnswerChoice("");
    try {
      const uploaded = await ensureUploaded();
      const rawPts = points && points.length > 2 ? points.map((p) => [Math.round(p.x), Math.round(p.y)]) : null;
      const data = await requestJson(`${API}/plan`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image_id: uploaded.image_id,
          instruction: prompt.trim(),
          box: boxPixels(),
          points: rawPts,
        }),
      });
      setPlan(data);
      if (data.status === "manual_fallback") setSuccess("Meta planning is unavailable; you can continue with your selected target region.");
    }
    catch (e) { setError(e.message); }
    finally { setBusy(false); }
  }
  async function submitAnswer() {
    if (!plan?.plan_id || !answerChoice) return;
    setBusy(true); setError("");
    try { const data = await requestJson(`${API}/answer`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ plan_id: plan.plan_id, answer: answerChoice }) }); setPlan(data); setAnswerChoice(""); }
    catch (e) { setError(e.message); }
    finally { setBusy(false); }
  }
  function boxPixels() {
    if (!box || !dimensions.width || !dimensions.height) return null;
    const x1 = Math.max(0, Math.min(dimensions.width - 1, Math.round(box.x * dimensions.width / 100)));
    const y1 = Math.max(0, Math.min(dimensions.height - 1, Math.round(box.y * dimensions.height / 100)));
    const x2 = Math.max(x1 + 1, Math.min(dimensions.width, Math.round((box.x + box.w) * dimensions.width / 100)));
    const y2 = Math.max(y1 + 1, Math.min(dimensions.height, Math.round((box.y + box.h) * dimensions.height / 100)));
    return [x1, y1, x2, y2];
  }
  async function runEdit() {
    if (!boxPixels()) { setSelectionError("Please select a region on the image before applying your edit."); return; }
    setSelectionError("");
    if (!prompt.trim()) { setError("Describe the change you want to make."); return; }
    setBusy(true); setError(""); setSuccess(""); clearResult();
    try {
      const uploaded = await ensureUploaded();
      const rawPts = points && points.length > 2 ? points.map((p) => [Math.round(p.x), Math.round(p.y)]) : null;
      const started = await requestJson(`${API}/edit`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image_id: uploaded.image_id,
          instruction: prompt.trim(),
          box: boxPixels(),
          points: rawPts,
          provider: "cloudflare",
          seed: 0,
          plan_id: plan?.plan ? plan.plan_id : undefined,
        }),
      });
      setEditJob(started);
      const finished = await waitForJob(started.job_id, setEditJob); setEditJob(finished);
      const resultPath = finished.artifacts?.result; const heatPath = finished.artifacts?.heatmap;
      if (!resultPath) throw new Error("The job completed but did not return a result image.");
      setResultUrl(assetUrl(resultPath)); setHeatmapUrl(heatPath ? assetUrl(heatPath) : "");
      const receiptPath = finished.artifacts?.receipt;
      if (receiptPath) { try { setReceipt(await requestJson(assetUrl(receiptPath))); } catch { setReceipt(null); } }
      setSuccess("Edit complete. Review the comparison and verification receipt below."); setView("results");
    } catch (e) { setError(e.message); }
    finally { setBusy(false); }
  }
  async function refineAgain() {
    if (!resultUrl) return;
    setBusy(true); setError(""); setSuccess("");
    try {
      const response = await fetch(resultUrl);
      if (!response.ok) throw new Error("Could not load the previous result as the next editing source.");
      const blob = await response.blob();
      const file = new File([blob], "pixl-refinement-source.png", { type: "image/png" });
      if (localPreview) URL.revokeObjectURL(localPreview);
      setSource(null);
      setSourceFile(file);
      setLocalPreview(URL.createObjectURL(blob));
      setBox(null);
      setPoints(null);
      setPlan(null);
      setAnswerChoice("");
      setPrompt("");
      setDimensions({ width: 0, height: 0 });
      clearResult();
      setView("edit");
      setSuccess("Previous result loaded as the new source. Draw a fresh target region for this refinement.");
    } catch (e) { setError(e.message); }
    finally { setBusy(false); }
  }
  async function downloadReceiptJson() {
    const path = editJob?.artifacts?.receipt;
    if (!path) return;
    try { const res = await fetch(assetUrl(path)); if (!res.ok) throw new Error(); const blob = await res.blob(); const url = URL.createObjectURL(blob); const a = document.createElement("a"); a.href = url; a.download = "pixl-receipt.json"; a.click(); URL.revokeObjectURL(url); }
    catch { setError("Could not download the receipt JSON."); }
  }
  async function downloadReceiptPdf() {
    if (!editJob?.job_id || receiptPrinting) return;
    setBusy(true); setReceiptPrinting(true); setError(""); setSuccess("");
    const animationStarted = Date.now();
    try {
      const res = await fetch(`${API}/jobs/${editJob.job_id}/receipt.pdf`);
      if (!res.ok) { let message = "Could not generate the PDF receipt."; try { const data = await res.json(); message = typeof data.detail === "string" ? data.detail : message; } catch { } throw new Error(message); }
      const blob = await res.blob();
      // Keep the paper-feed animation visible long enough to read even when PDF generation is fast.
      await delay(Math.max(0, 2300 - (Date.now() - animationStarted)));
      const url = URL.createObjectURL(blob); const a = document.createElement("a"); a.href = url; a.download = "pixl-edit-receipt.pdf"; document.body.appendChild(a); a.click(); a.remove(); window.setTimeout(() => URL.revokeObjectURL(url), 1500);
      setSuccess("Your Pixl receipt is ready and has been downloaded.");
    } catch (e) { setError(e.message || "Could not generate the PDF receipt."); }
    finally { setReceiptPrinting(false); setBusy(false); }
  }
  async function downloadImage(url, filename) {
    try { const res = await fetch(url); if (!res.ok) throw new Error(); const blob = await res.blob(); const obj = URL.createObjectURL(blob); const a = document.createElement("a"); a.href = obj; a.download = filename; a.click(); URL.revokeObjectURL(obj); }
    catch { setError("Could not download the image artifact."); }
  }
  function removeSourceImage() {
    if (localPreview) URL.revokeObjectURL(localPreview);
    setSource(null);
    setSourceFile(null);
    setLocalPreview("");
    setPrompt("");
    setBox(null);
    setPoints(null);
    setPlan(null);
    setAnswerChoice("");
    setDimensions({ width: 0, height: 0 });
    setSelectionError("");
    clearResult();
    setError("");
    setSuccess("Image removed. Upload another image to continue.");
  }
  function resetWorkspace() { removeSourceImage(); setSuccess(""); setView("create"); }

  const activeImage = !!sourceUrl;
  const planQuestions = plan?.plan?.questions || [];
  const currentQuestion = planQuestions[0];
  const currentReceipt = receipt || null;

  return <div className="p4-app">
    <Header view={view} setView={(next) => { setView(next); setError(""); }} apiState={apiState} />
    <main className="p4-main">
      {view !== "home" && <><section className="p4-intro"><div><div className="p4-eyebrow"><span /> AI IMAGE WORKFLOW / CONTROLLED EDITING</div><h1>Make the change.<br /><em>Keep the good.</em></h1><p>Generate a starting point, refine only what needs refining, then inspect exactly what changed.</p></div><div className="p4-intro-stamp"><strong>PIXL</strong><span>CONTROLLED<br />IMAGE EDITING</span><b>04</b></div></section>
        <div className="p4-workflow-rail"><span className={view === "create" ? "current" : "done"}><b>01</b> Create</span><i /><span className={view === "edit" ? "current" : view === "results" ? "done" : ""}><b>02</b> Refine</span><i /><span className={view === "results" ? "current" : ""}><b>03</b> Verify</span></div></>}
      {error && <Notice type="error" onClose={() => setError("")}>{error}</Notice>}
      {success && <Notice type="success" onClose={() => setSuccess("")}>{success}</Notice>}

      {view === "home" && <div className="p4-home-page">
        <section className="p4-home-hero">
          <div className="p4-home-copy"><div className="p4-eyebrow"><span /> PIXL / IMAGE INTELLIGENCE</div><h1>Change what matters.<br /><em>Protect what works.</em></h1><p className="p4-home-lead">AI image editing should not mean starting over every time one detail needs work.</p><p className="p4-home-body">Pixl is built for deliberate, localized edits: choose the area, give the model enough surrounding context, then composite only the region you approved. The rest of your image stays untouched by the compositing step.</p><div className="p4-home-actions"><button className="p4-button p4-button-orange" onClick={() => setView("create")}><Sparkles size={16} /> Create an image <ArrowRight size={16} /></button><button className="p4-button p4-button-light" onClick={() => setView("edit")}><WandSparkles size={16} /> Refine an image</button></div><div className="p4-home-footnote"><ShieldCheck size={14} /> Localized compositing · Visible differences · Exportable evidence</div></div>
          <div className="p4-home-art" aria-label="Illustration of a localized image edit"><div className="p4-home-art-top"><span>PIX / REGION CONTROL</span><span>01—03</span></div><div className="p4-home-art-image"><div className="p4-art-sun" /><div className="p4-art-hill hill-one" /><div className="p4-art-hill hill-two" /><div className="p4-art-object"><i /><i /><i /></div><div className="p4-art-selection"><span>APPROVED REGION</span></div></div><div className="p4-home-art-bottom"><span>CONTEXT IN</span><b>→</b><span>PIXELS OUTSIDE REGION: 0 CHANGED</span></div></div>
        </section>
        <section className="p4-home-why"><div className="p4-home-section-intro"><span className="p4-overline">WHY PIXL EXISTS</span><h2>Because one small change<br />shouldn't risk the whole image.</h2></div><p>Generative edits can alter nearby details, shift lighting, or introduce differences you never asked for. Repeating prompts alone doesn't provide a reliable boundary or a clear record of what changed. Pixl makes the edit region explicit, preserves the original outside it during compositing, and gives you evidence to review before you use the result.</p></section>
        <section className="p4-home-benefits"><article className="p4-home-benefit"><span className="p4-home-number">01 / CONTROL</span><div className="p4-home-benefit-icon orange"><Scan size={21} /></div><h3>Context without collateral changes</h3><p>The model sees an expanded crop for context. Pixl then aligns the output and composites only the original selection into the image.</p></article><article className="p4-home-benefit dark"><span className="p4-home-number">02 / EVIDENCE</span><div className="p4-home-benefit-icon white"><ShieldCheck size={21} /></div><h3>Inspect, don't just trust</h3><p>Compare the original and result, inspect the difference heatmap, and review metrics like changed pixels, color drift, seams and texture.</p></article><article className="p4-home-benefit"><span className="p4-home-number">03 / ITERATE</span><div className="p4-home-benefit-icon black"><RefreshCw size={21} /></div><h3>Refine without losing your baseline</h3><p>Continue from the last result with a new selection. Each edit creates its own artifacts and a receipt you can keep.</p></article></section>
        <section className="p4-home-method"><div><span className="p4-overline">THE PIXL METHOD</span><h2>Three steps.<br /><em>One clear boundary.</em></h2><p>Built to make image editing more intentional, inspectable and repeatable.</p></div><div className="p4-home-steps"><div><b>01</b><span><strong>Describe & select</strong><small>Explain the change. Use Meta planning when available, or draw the region yourself.</small></span></div><div><b>02</b><span><strong>Edit with context</strong><small>Pixl expands the crop for model context, then returns to your confirmed selection for compositing.</small></span></div><div><b>03</b><span><strong>Verify & export</strong><small>Review the before/after, heatmap and metrics. Generate a styled PDF receipt for your records.</small></span></div></div></section>
        <section className="p4-home-cta"><div><span className="p4-overline">READY WHEN YOU ARE</span><h2>Keep the image.<br /><em>Change the detail.</em></h2></div><button className="p4-button p4-button-orange" onClick={() => setView("edit")}>Open the editor <ArrowRight size={16} /></button></section>
      </div>}

      {view === "create" && <div className="p4-create-grid">
        <section className="p4-panel p4-panel-project"><StepHeading index="01" title="Create an image" detail="Describe the frame you have in mind." /><label className="p4-label" htmlFor="generation-prompt">YOUR IMAGE IDEA</label><textarea id="generation-prompt" className="p4-textarea" rows={5} maxLength={4000} value={prompt} onChange={(e) => { setPrompt(e.target.value); setGenerationPlan(null); setGenerationAnswers({}); setGenerationImprovedPrompt(""); }} placeholder="A minimal ceramic vase on a warm limestone plinth, soft afternoon light…" /><div className="p4-generation-plan-actions"><button className="p4-button p4-button-light p4-full" onClick={planGenerationPrompt} disabled={busy || apiState !== "online" || !prompt.trim()}>{busy ? <LoaderCircle className="p4-spin" size={15} /> : <Scan size={15} />} Plan with Meta <Sparkles size={14} /></button><span>Optional · fill important gaps and build a richer prompt. Meta Contributor mode may use submitted prompts to improve Meta models.</span></div>
          {generationPlan && <div className="p4-plan-card p4-generation-plan-card"><div className="p4-plan-head"><span className="p4-overline">PROMPT DESIGN · {generationPlan.status === "ready" ? "READY" : "NEEDS DETAILS"}</span><span>{generationPlan.questions?.length ? `${generationPlan.questions.length} question(s)` : "Draft ready"}</span></div><p>{generationPlan.message}</p>{generationPlan.detected_details?.length > 0 && <div className="p4-preserve-list"><span>DETAILS ALREADY COVERED</span>{generationPlan.detected_details.map((item, i) => <small key={i}><Check size={11} />{item}</small>)}</div>}{generationPlan.questions?.length > 0 && <div className="p4-generation-questions">{generationPlan.questions.map((q) => <div className="p4-question" key={q.id}><strong>{q.question}</strong><small className="p4-question-reason">{q.reason}</small>{q.options.map((option) => <label key={option} className={`p4-option ${generationAnswers[q.id] === option ? "chosen" : ""}`}><input type="radio" name={`generation-${q.id}`} checked={generationAnswers[q.id] === option} onChange={() => setGenerationAnswers((prev) => ({ ...prev, [q.id]: option }))} /><span>{option}</span></label>)}</div>)}<button className="p4-button p4-button-orange p4-full" onClick={submitGenerationAnswers} disabled={busy || generationPlan.questions.some((q) => !generationAnswers[q.id])}>Use answers & refine prompt <ArrowRight size={14} /></button></div>}{generationPlan.assumptions?.length > 0 && <div className="p4-generation-assumptions"><strong>Reasonable defaults</strong><span>{generationPlan.assumptions.join(" · ")}</span></div>}{generationPlan.status === "ready" && <><label className="p4-label" htmlFor="improved-generation-prompt">FINAL PROMPT SENT TO IMAGE MODEL · EDITABLE</label><textarea id="improved-generation-prompt" className="p4-textarea" rows={7} maxLength={4000} value={generationImprovedPrompt} onChange={(e) => setGenerationImprovedPrompt(e.target.value)} /></>}</div>}
          <div className="p4-field-row"><div className="p4-label">OUTPUT SIZE<span className="p4-static-value">Model default</span></div><div className="p4-label">ENGINE<span className="p4-static-value"><Zap size={14} /> FLUX.1 SCHNELL</span></div></div><button className="p4-button p4-button-orange p4-full" onClick={runGeneration} disabled={busy || apiState !== "online"}>{busy ? <LoaderCircle className="p4-spin" size={17} /> : <Sparkles size={17} />} Generate image <ArrowRight size={16} /></button><p className="p4-microcopy"><CircleHelp size={13} /> Prompt planning is optional. Without it, your original prompt is sent directly. Requires the backend to be configured with Cloudflare credentials.</p>
        </section>
        <section className="p4-panel p4-output-panel"><div className="p4-output-header"><span className="p4-overline">OUTPUT / 01</span><span className="p4-live-chip">{generationJob?.status || "WAITING"}</span></div>{generationJob?.artifacts?.result ? <div className="p4-generated-image"><img src={assetUrl(generationJob.artifacts.result)} alt="Generated image" /></div> : <div className="p4-output-empty"><div className="p4-output-shape"><span /><span /><span /></div><strong>Your next image starts here.</strong><span>Generated artwork and job progress will appear in this panel.</span></div>}
          <JobProgress job={generationJob} />
          {generationJob?.artifacts?.result && generationJob.status === "completed" && <div className="p4-output-actions"><button className="p4-button p4-button-light" onClick={() => downloadImage(assetUrl(generationJob.artifacts.result), "pixl-generated.png")}><Download size={15} /> Download PNG</button><button className="p4-button p4-button-orange" onClick={sendGeneratedToEditor} disabled={busy}><WandSparkles size={15} /> Edit this image <ArrowRight size={15} /></button></div>}
        </section>
      </div>}

      {view === "edit" && <div className="p4-edit-grid">
        <section className="p4-panel p4-editor-controls"><StepHeading index="02" title="Refine a region" detail="Draw a target region around the area you want to change." />
          {source && <div className="p4-source-pill"><FileImage size={15} /><span>{source.filename}</span><small>{source.width} × {source.height}</small><button onClick={resetWorkspace} title="Choose another image"><RefreshCw size={13} /></button></div>}
          <label className="p4-label" htmlFor="edit-prompt">WHAT SHOULD CHANGE?</label><textarea id="edit-prompt" className="p4-textarea" rows={4} maxLength={4000} value={prompt} onChange={(e) => { setPrompt(e.target.value); setPlan(null); }} placeholder="Change the wall to a muted sage green. Keep the furniture, framing and lighting unchanged." />
          <div className="p4-edit-engine-note"><Sparkles size={14} /><span><strong>Cloudflare AI editing</strong><small>Edits are processed using FLUX.2 Klein 4B.</small></span></div>
          <div className="p4-action-pair"><button className="p4-button p4-button-light p4-plan-meta-button" onClick={requestPlan} disabled={busy || !activeImage || !prompt.trim() || !boxPixels()}>{busy ? <LoaderCircle className="p4-spin" size={15} /> : <Scan size={15} />} Plan with Meta</button><span>Optional · analyzes only the selected freehand region. Contributor mode may use submitted images and prompts to improve Meta models.</span></div>
          {plan && <div className="p4-plan-card"><div className="p4-plan-head"><span className="p4-overline">META PLAN · {plan.status?.replaceAll("_", " ")}</span><span>{plan.plan ? `${Math.round(plan.plan.confidence * 100)}% confidence` : "Manual mode"}</span></div><p>{plan.message}</p>{plan.plan && <><div className="p4-plan-target"><span>INTENDED TARGET</span><strong>{plan.plan.target_phrase}</strong><small>Type: {plan.plan.edit_type} · {plan.plan.match_count} match(es)</small></div>{plan.plan.preserve?.length > 0 && <div className="p4-preserve-list"><span>KEEP UNCHANGED</span>{plan.plan.preserve.map((item, i) => <small key={i}><Check size={11} />{item}</small>)}</div>}</>}
            {currentQuestion && <div className="p4-question"><strong>{currentQuestion.question}</strong>{currentQuestion.options.map((option) => <label key={option} className={`p4-option ${answerChoice === option ? "chosen" : ""}`}><input type="radio" name="plan-answer" value={option} checked={answerChoice === option} onChange={() => setAnswerChoice(option)} /><span>{option}</span></label>)}<button className="p4-button p4-button-light p4-full" onClick={submitAnswer} disabled={!answerChoice || busy}>Submit answer <ArrowRight size={14} /></button></div>}
            {plan.warnings?.map((warning, i) => <p className="p4-plan-warning" key={i}>{warning}</p>)}
            {plan.status === "needs_manual_box" && <p className="p4-plan-warning">Select the target region manually. Model coordinates remain unverified and are not applied automatically.</p>}
          </div>}
          <button className="p4-button p4-button-orange p4-full p4-run-edit" onClick={runEdit} disabled={busy || !activeImage || !prompt.trim()}>{busy ? <LoaderCircle className="p4-spin" size={16} /> : <WandSparkles size={16} />} Apply edit <ArrowRight size={16} /></button>
          {selectionError && <p className="p4-selection-error" role="alert">{selectionError}</p>}
          <p className="p4-microcopy"><ShieldCheck size={13} /> Only the selected freehand region is composited into the output. A confirmed selection is required.</p>
        </section>
        <section className="p4-panel p4-canvas-panel"><div className="p4-canvas-title"><div><span className="p4-overline">CANVAS / TARGET SELECTION</span><h3>{sourceUrl ? "Choose the edit area" : "Start by uploading an image"}</h3></div><div className="p4-canvas-title-actions"><span className="p4-coordinate-tag">{dimensions.width ? `${dimensions.width} × ${dimensions.height}` : "NO IMAGE"}</span>{sourceUrl && <button type="button" className="p4-button p4-button-light p4-remove-image" onClick={removeSourceImage} disabled={busy} title="Remove this image and choose another"><X size={13} /> Remove image</button>}</div></div>{!sourceUrl ? <UploadDrop onFile={chooseSource} fileName={sourceFile?.name} /> : <><ImageCanvas imageSrc={sourceUrl} box={box} setBox={(next) => { setBox(next); if (next) setSelectionError(""); }} points={points} setPoints={(next) => { setPoints(next); if (next) setSelectionError(""); }} dimensions={dimensions} setDimensions={setDimensions} /><div className="p4-canvas-help"><span><i className="p4-help-dot" /> Drag freehand to outline target area</span><span><i className="p4-help-outline" /> Release to close contour</span><span><i className="p4-help-grid" /> Outside region preserved bit-for-bit</span></div></>}</section>
      </div>}

      {view === "results" && <div className="p4-results-layout">
        <section className="p4-panel p4-comparison-panel"><div className="p4-canvas-title"><div><span className="p4-overline">RESULT / BEFORE + AFTER</span><h3>Inspect the change</h3></div><button className={`p4-toggle ${showHeatmap ? "on" : ""}`} onClick={() => setShowHeatmap((v) => !v)} disabled={!heatmapUrl}><span />{showHeatmap ? "Heatmap on" : "Show heatmap"}</button></div>
          {sourceUrl && resultUrl ? <div className="p4-compare-stage"><img className="p4-compare-base" src={sourceUrl} alt="Before edit" /><img className="p4-compare-after" src={resultUrl} alt="After edit" style={{ clipPath: `inset(0 ${100 - comparePosition}% 0 0)` }} />{showHeatmap && heatmapUrl && <img className="p4-compare-heat" src={heatmapUrl} alt="Pixel difference heatmap" />}<div className="p4-compare-divider" style={{ left: `${comparePosition}%` }}><span>↔</span></div><span className="p4-compare-label before">BEFORE</span><span className="p4-compare-label after">{showHeatmap ? "HEATMAP" : "AFTER"}</span></div> : <div className="p4-output-empty p4-results-empty"><div className="p4-output-shape"><span /><span /><span /></div><strong>No completed edit yet</strong><span>Upload an image, select a region and apply an edit to see your result here.</span><button className="p4-button p4-button-orange" onClick={() => setView("edit")}>Go to editor <ArrowRight size={15} /></button></div>}
          {sourceUrl && resultUrl && <div className="p4-slider-control"><span>ORIGINAL</span><input aria-label="Before and after comparison slider" type="range" min="0" max="100" value={comparePosition} onChange={(e) => setComparePosition(Number(e.target.value))} /><span>RESULT</span></div>}
          {resultUrl && <div className="p4-download-actions"><button className="p4-button p4-button-orange" onClick={() => downloadImage(resultUrl, "pixl-result.png")}><Download size={15} /> Download result PNG</button><button className="p4-button p4-button-light" onClick={refineAgain} disabled={busy}><SlidersHorizontal size={15} /> Refine again</button></div>}
        </section>
        <aside className="p4-panel p4-receipt-panel"><ReceiptPanel receipt={currentReceipt} job={editJob} onDownloadPdf={editJob?.artifacts?.receipt ? downloadReceiptPdf : null} onDownloadJson={editJob?.artifacts?.receipt ? downloadReceiptJson : null} /><div className="p4-job-meta"><span>JOB STATUS</span><strong className={editJob?.status === "completed" ? "good" : ""}>{editJob?.status || "No job"}</strong>{editJob?.job_id && <small>{editJob.job_id}</small>}</div></aside>
      </div>}
      <footer className="p4-footer"><span><span className="p4-brand-mark mini">p<span>.</span></span> PIXL / IMAGE INTELLIGENCE</span><span>FEWER ITERATIONS. MORE INTENTIONAL IMAGES.</span><button onClick={resetWorkspace}>Reset workspace <RefreshCw size={12} /></button></footer>
    </main>
    {receiptPrinting && <div className="p4-receipt-print-overlay" role="status" aria-live="polite"><div className="p4-printer-scene"><div className="p4-printer-brand"><span className="p4-brand-mark mini">p<span>.</span></span><span>PIXL <small>RECEIPT SYSTEM</small></span></div><div className="p4-printer-slot" /><div className="p4-paper-feed"><div className="p4-paper-logo">pixl<span>.</span></div><div className="p4-paper-line wide" /><div className="p4-paper-line" /><div className="p4-paper-rule" /><div className="p4-paper-line wide" /><div className="p4-paper-line short" /><div className="p4-paper-stats"><b>CHANGE</b><b>VERIFIED</b></div><div className="p4-paper-barcode" /></div><div className="p4-printer-body"><span>PIX / THERMAL OUTPUT</span><i /><i /><i /></div></div><strong>Printing your change receipt</strong><p>Preparing visual evidence and edit statistics…</p><div className="p4-print-progress"><span /></div></div>}
  </div>;
}
