#!/usr/bin/env python
"""Export code cells from the full Colab notebook into a single script.

This is useful for auditing or debugging, but the recommended entry point is the
notebook itself or the modular package under src/.
"""
from pathlib import Path
import nbformat

nb_path = Path("notebooks/01_full_colab_workflow.ipynb")
out_path = Path("scripts/full_notebook_pipeline.py")
nb = nbformat.read(nb_path, as_version=4)
chunks = ["# Auto-extracted from notebooks/01_full_colab_workflow.ipynb\n"]
for i, cell in enumerate(nb.cells):
    if cell.cell_type == "code" and cell.source.strip():
        chunks.append(f"\n# %% [cell {i}]\n")
        chunks.append(cell.source.rstrip() + "\n")
out_path.write_text("".join(chunks))
print(f"Wrote {out_path}")
