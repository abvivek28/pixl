"""Run the Phase 2 CLI orchestrator on a manifest of ten or more annotated photos.

Manifest JSON: [{"image":"photos/01.jpg", "instruction":"make the jacket blue",
"box":[x1,y1,x2,y2], "output":"data/batch/01"}, ...]
Paths are resolved relative to the manifest file. A box is required until Meta coordinates
have been empirically verified against five manually annotated images.
"""
import argparse, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.engine.deterministic import Box, ingest_image
from app.engine.orchestrator import run_edit, FakeHueShiftProvider
from app.providers.cloudflare_klein4b import CloudflareKlein4B
from app.planning.meta_planner import MetaPlanner

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("manifest"); parser.add_argument("--provider",choices=["fake","cloudflare"],default="fake")
    parser.add_argument("--plan-with-meta",action="store_true"); args=parser.parse_args()
    manifest_path=Path(args.manifest).resolve(); jobs=json.loads(manifest_path.read_text(encoding="utf-8"))
    if len(jobs)<10: print(f"Note: manifest has {len(jobs)} images; Phase 2 target is 10 photos.")
    provider=CloudflareKlein4B() if args.provider=="cloudflare" else FakeHueShiftProvider()
    planner=MetaPlanner() if args.plan_with_meta else None
    summary=[]
    for index, job in enumerate(jobs,1):
        image_path=(manifest_path.parent/job["image"]).resolve()
        out=(manifest_path.parent/job.get("output",f"data/batch/{index:02d}")).resolve(); out.mkdir(parents=True,exist_ok=True)
        master=out/"master.png"; ingest_image(image_path,master)
        raw_box=job.get("box")
        box=Box(*raw_box) if raw_box else None
        result=run_edit(master,job["instruction"],out,selected_box=box,provider=provider,planner=planner,
                        box_source="manual" if box else "meta")
        summary.append({"image":str(image_path),"status":result.get("status","completed"),
                        "outside_changed_pixels":result.get("metrics",{}).get("outside_changed_pixels"),
                        "receipt":result.get("artifacts",{}).get("receipt"),"message":result.get("message")})
        print(f"[{index}/{len(jobs)}] {image_path.name}: {summary[-1]['status']}")
    report=manifest_path.parent/"phase2_batch_report.json"; report.write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(f"Batch report: {report}")
if __name__=="__main__": main()
