"""Copy everything Overleaf needs into a clean folder and zip it.

    venv/Scripts/python paper/export_overleaf.py [destination]   (default: ~/Downloads/ieee_access)

The folder gets main.tex, references.bib, the class/style/font/logo files of the IEEE Access template, tables/ and figures/
(PDF only), plus main.pdf for reference. Build artifacts (*.aux, *.log, *.bbl, ...) are left out.
"""
import shutil
import sys
import zipfile
from pathlib import Path

SRC = Path(__file__).resolve().parent
DST = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "Downloads" / "ieee_access"

KEEP_EXT = {".cls", ".bst", ".sty", ".pfb", ".tfm", ".map", ".fd", ".png"}
TOP = ["main.tex", "references.bib", "main.pdf", "README.md"]

if DST.exists():
    shutil.rmtree(DST)
DST.mkdir(parents=True)
for f in sorted(SRC.iterdir()):
    if f.is_file() and (f.name in TOP or f.suffix in KEEP_EXT):
        shutil.copy2(f, DST / f.name)
(DST / "tables").mkdir()
for f in sorted((SRC / "tables").glob("*.tex")):
    shutil.copy2(f, DST / "tables" / f.name)
(DST / "figures").mkdir()
for f in sorted((SRC / "figures").glob("*.pdf")):
    shutil.copy2(f, DST / "figures" / f.name)
shutil.copy2(SRC / "figures" / "fig1_overview.tex", DST / "figures" / "fig1_overview.tex")

zip_path = DST.parent / (DST.name + ".zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(DST.rglob("*")):
        if f.is_file():
            z.write(f, f.relative_to(DST))
n = sum(1 for f in DST.rglob("*") if f.is_file())
print(f"copied {n} files to {DST}\nzip for Overleaf: {zip_path}")
