from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.engine.deterministic import Box, ingest_image, run_fake_edit

def main():
    parser = argparse.ArgumentParser(description="Run Pixl's deterministic fake-edit engine.")
    parser.add_argument("image", help="Input image path (JPEG/PNG/WebP etc.)")
    parser.add_argument("--output", default="data/phase1-demo", help="Output directory")
    parser.add_argument("--box", nargs=4, type=int, metavar=("X1", "Y1", "X2", "Y2"),
                        required=True, help="Selected box in pixel coordinates")
    parser.add_argument("--prompt", default="Change the hue of the selected region")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    master = output / "master.png"
    info = ingest_image(args.image, master)
    receipt = run_fake_edit(master, Box(*args.box), output, prompt=args.prompt)
    print(f"Master: {master} ({info['width']}x{info['height']}, sha256={info['sha256']})")
    print(f"Result: {receipt['artifacts']['result']}")
    print(f"Heatmap: {receipt['artifacts']['heatmap']}")
    print(f"Receipt: {receipt['artifacts']['receipt']}")
    print(json.dumps(receipt["metrics"], indent=2))
    print(receipt["summary"])

if __name__ == "__main__":
    import json
    main()
