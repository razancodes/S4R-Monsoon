import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT / "src"))

from s4r.route_b.s1_fetch import fetch_village_stacks
from s4r.route_b.pseudo_labels import generate_pseudo_labels, write_pseudo_annotations

DATA_ROOT = Path(r"C:\Users\MRaza\.cache\kagglehub\competitions\anrf-aise-hack-2026-round-1-sar-crop-mapping-challenge")
NPZ_CACHE = ROOT / "data" / "processed" / "s1_stacks.npz"
ANNOTATIONS_OUT = ROOT / "data" / "weak_labels" / "annotations.csv"

def main():
    print("Fetching S1 stacks from Planetary Computer...")
    stacks = fetch_village_stacks(DATA_ROOT, NPZ_CACHE)
    print(f"Fetched {len(stacks)} village stacks.")

    print("Generating pseudo labels using OlmoEarth...")
    fractions, diagnostics = generate_pseudo_labels(stacks)
    print(f"Generated fractions for {len(fractions)} villages.")
    print(f"Diagnostics: {diagnostics}")

    out = write_pseudo_annotations(fractions, ANNOTATIONS_OUT)
    print(f"Wrote annotations to {out}")

if __name__ == "__main__":
    main()
