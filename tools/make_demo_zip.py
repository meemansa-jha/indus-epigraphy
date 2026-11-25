"""Builds dist/indus_demo.zip: everything `python demo/demo.py` needs, and nothing else (~35 MB).
  python tools/make_demo_zip.py
"""
import zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["demo", "src", "README.md", "DATA_SOURCES.md",
           "data/raw/m3_csv", "data/raw/cisi/features", "data/raw/IM_417_150_PNG",
           "data/raw/isd/1/1text.jpg", "data/processed/crosswalk.csv", "data/processed/m4/controls_test.json",
           "data/processed/engine_synthetic/images/syn_016.png",
           "runs/sign_detector/weights/best.pt", "runs/recognizer/best.pt", "runs/denoiser/best.pt"]
SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints", "outputs", "cisi_corpus"}
SKIP_FILES = {"cv_folds.json", "cv_folds_ensemble.json", ".DS_Store"}
SKIP_SUFFIX = {".ipynb", ".pyc"}
out = ROOT / "dist" / "indus_demo.zip"; out.parent.mkdir(exist_ok=True)
n = 0
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for item in INCLUDE:
        p = ROOT / item
        files = [p] if p.is_file() else sorted(f for f in p.rglob("*") if f.is_file())
        for f in files:
            rel = f.relative_to(ROOT)
            if SKIP_DIRS & set(rel.parts) or f.name in SKIP_FILES or f.suffix in SKIP_SUFFIX: continue
            if rel.parts[0] == "demo" and f.name == "README.md": continue      # replaced by FRIEND_README
            z.write(f, Path("indus_demo") / rel); n += 1
    z.write(ROOT / "demo" / "FRIEND_README.md", Path("indus_demo") / "START_HERE.md"); n += 1
print(f"{n} files -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
