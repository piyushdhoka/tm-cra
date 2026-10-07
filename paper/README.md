# IEEE Access manuscript

`main.tex` is the manuscript, built on the official IEEE Access LaTeX template (May 2026): `ieeeaccess.cls`,
`IEEEtran.cls`, `IEEEtran.bst`, `spotcolor.sty`, the Formata/Times font files (`t1-*.pfb`, `*.tfm`, `*.map`, `*.fd`) and the
logos. Keep all of them next to `main.tex`; Overleaf and pdfLaTeX find them automatically.

## Regenerating tables and figures from the results

Every number in the tables and figures comes from `results/` through two scripts (run from the repository root):

```
venv/Scripts/python paper/make_tables.py     # writes paper/tables/*.tex (table bodies only)
venv/Scripts/python paper/make_figures.py    # writes paper/figures/fig2..fig9 (vector PDF, IEEE sizes)
cd paper/figures && pdflatex fig1_overview.tex   # the pipeline diagram (standalone TikZ)
```

`paper/model_facts.json` holds the checkpoint names, parameter counts (from the Hugging Face API) and hardware notes used in
Table I.

## Building the PDF

```
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

The IEEE Access class emits a handful of `Overfull \hbox (505pt)` / `(9.27pt)` warnings from its own page header; the
unmodified template produces the same warnings and they can be ignored.

## Overleaf

`venv/Scripts/python paper/export_overleaf.py` copies a clean set of files to `~/Downloads/ieee_access` and writes
`~/Downloads/ieee_access.zip`. Upload the zip as a new Overleaf project, set the compiler to pdfLaTeX and the main document
to `main.tex`.

## Before submission (checklist from ieeeaccess.ieee.org, verified 2026-10-07)

- Fill in the author affiliations, e-mails, the ORCID of the submitting author, and a 100--150-word biography per author.
- Keep the abstract at 150--250 words with no abbreviations; keep the AI-use disclosure in the Acknowledgment.
- Upload the LaTeX source *and* the PDF; the contents must match. Choose 3--10 keywords and the manuscript type
  "Research Article" in the portal.
