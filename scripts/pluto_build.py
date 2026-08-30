#!/usr/bin/env python3
"""Assemble a Pluto.jl notebook file from a list of Julia code cell strings.

Not part of the demo content itself -- just a one-off authoring helper used
to hand-write valid Pluto notebook (.jl) files without a running Pluto
server. Each entry in `cells` becomes one reactive Pluto cell.
"""
import sys
import uuid


def build(cells, pluto_version="v0.19.45"):
    lines = []
    lines.append("### A Pluto.jl notebook ###")
    lines.append(f"# {pluto_version}")
    lines.append("")
    lines.append("using Markdown")
    lines.append("using InteractiveUtils")
    lines.append("")

    ids = [str(uuid.uuid4()) for _ in cells]

    for cid, code in zip(ids, cells):
        lines.append(f"# ╔═╡ {cid}")
        lines.append(code.rstrip("\n"))
        lines.append("")

    lines.append("# ╔═╡ Cell order:")
    for cid in ids:
        lines.append(f"# ╠═{cid}")

    return "\n".join(lines) + "\n"


def write_notebook(path, cells):
    text = build(cells)
    with open(path, "w") as f:
        f.write(text)
    print(f"wrote {path} ({len(cells)} cells)")
