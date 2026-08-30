#!/usr/bin/env julia
# Executes each Pluto notebook with PlutoSliderServer (baking in real outputs
# from a live Julia run), producing a static, self-contained HTML file per
# notebook. Converting that HTML to PDF is a separate step (see
# scripts/html_to_pdf.sh) using the container's headless Chromium, since
# Pluto's own PDF export path depends on Deno/puppeteer which isn't set up
# here.
using Pkg
Pkg.activate(dirname(@__DIR__))

using PlutoSliderServer

const ROOT = dirname(@__DIR__)
const NOTEBOOK_DIR = joinpath(ROOT, "notebooks")
const HTML_DIR = joinpath(ROOT, "html")

mkpath(HTML_DIR)

notebooks = filter(f -> endswith(f, ".jl"), readdir(NOTEBOOK_DIR; join=true))

for nb in sort(notebooks)
    @info "Exporting" nb
    PlutoSliderServer.export_notebook(
        nb;
        Export_output_dir = HTML_DIR,
        Export_baked_state = true,
        Export_offer_binder = false,
    )
end

@info "Done. Static HTML written to" HTML_DIR
