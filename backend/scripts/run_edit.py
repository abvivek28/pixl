from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.engine.deterministic import Box, ingest_image
from app.engine.orchestrator import run_edit, FakeHueShiftProvider
from app.providers.cloudflare_klein4b import CloudflareKlein4B
from app.planning.meta_planner import MetaPlanner

def main():
    p = argparse.ArgumentParser(description="Pixl Phase 2 CLI: image + instruction -> plan and verified edit receipt")
    p.add_argument("image", help="Input image path")
    p.add_argument("instruction", help="Requested edit, quoted if it contains spaces")
    p.add_argument("--box", nargs=4, type=int, metavar=("X1","Y1","X2","Y2"), help="Manual pixel box; recommended until Meta coordinate mapping is verified")
    p.add_argument("--output", default="data/phase2-demo")
    p.add_argument("--provider", choices=["fake","cloudflare"], default="fake")
    p.add_argument("--plan-with-meta", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verified-meta-coordinates", action="store_true", help="Only use after 5-image coordinate verification")
    args = p.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    master = out / "master.png"
    ingest_image(args.image, master)
    box = Box(*args.box) if args.box else None
    provider = CloudflareKlein4B() if args.provider == "cloudflare" else FakeHueShiftProvider()
    planner = MetaPlanner() if args.plan_with_meta else None
    result = run_edit(master, args.instruction, out, selected_box=box, provider=provider,
                      planner=planner, seed=args.seed, box_source="manual" if box else "meta",
                      coordinate_format_verified=args.verified_meta_coordinates)
    print(json.dumps(result, indent=2))
    if result.get("status") in {"needs_choice","needs_manual_box","manual_fallback"}:
        raise SystemExit(2)

if __name__ == "__main__": main()
