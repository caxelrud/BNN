# BNN — Bayesian Neural Network examples for Oil & Gas (Julia / Pluto / Turing.jl)

Two self-contained [Pluto.jl](https://plutojl.org/) notebooks demonstrating
Bayesian neural networks with [Turing.jl](https://turinglang.org/) on
synthetic (but petrophysically-motivated) oil & gas datasets.

## Notebooks

| Notebook | What it shows |
|---|---|
| [`notebooks/01_bnn_permeability_regression.jl`](notebooks/01_bnn_permeability_regression.jl) | A BNN (weights sampled with NUTS instead of backprop) predicting reservoir permeability from well-log curves (porosity, gamma ray, resistivity, water saturation), with full posterior-predictive uncertainty bands. |
| [`notebooks/02_additive_bayesian_updating.jl`](notebooks/02_additive_bayesian_updating.jl) | **Additive learning**: folding a newly-drilled batch of wells into an *existing* posterior (moment-matched Gaussian prior + NUTS on the new batch only), compared against a stale never-updated model and a from-scratch full retrain on all data. |

Rendered PDFs of both notebooks (with all outputs baked in) are in
[`pdfs/`](pdfs/).

## Why "additive" matters here

Notebook 2 is the direct answer to "can we add new data to an existing
model instead of retraining from scratch": Phase A wells build the initial
posterior; when Phase B wells (a geologically different step-out area)
arrive, that posterior is turned into an informative prior and NUTS runs
again over *only* the new wells. The notebook compares this against (a)
leaving the model stale and (b) a full retrain on the pooled data, on
accuracy, and reports the MCMC wall-time / wells-processed per update step
so the scaling argument for additive updates is visible, not just claimed.

## Results (from the shipped PDFs)

| | RMSE (pooled held-out test) | R² | wells processed *this step* |
|---|---|---|---|
| Stale (Phase A only, distribution-shifted Phase B never seen) | 0.313 | −0.18 | — |
| **Additive update** (Phase A posterior → prior, NUTS on Phase B only) | 0.246 | 0.27 | 19 |
| Full retrain (Phase A ∪ Phase B, from scratch) | 0.247 | 0.27 | 49 |

The additive update recovers essentially all of the full retrain's accuracy
gain over the stale model — while its MCMC step only ever touches the new
batch of wells, not the growing history.

## Running the notebooks interactively

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pluto; Pluto.run(notebook="notebooks/01_bnn_permeability_regression.jl")'
```

Then open the second notebook the same way, or from Pluto's own notebook
picker UI.

## Regenerating the PDFs

Requires Node with the `playwright` package available (globally or on
`NODE_PATH`) and its bundled Chromium, in addition to the Julia project.

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. scripts/export_pdfs.jl   # runs both notebooks headlessly via PlutoSliderServer -> html/
scripts/html_to_pdf.sh                     # vendors Pluto's frontend assets locally, serves html/ over
                                            # localhost, and prints each notebook to pdfs/*.pdf with headless
                                            # Chromium (Playwright), waiting for the notebook to finish
                                            # rendering before printing
```

`html/` is a regenerable build artifact (it also vendors a full copy of
Pluto's frontend assets so PDF rendering works offline) and is not tracked
in git — see `.gitignore`.

## Reference

Both notebooks follow the structure of Turing.jl's official [Bayesian
Neural Network tutorial](https://turinglang.org/docs/tutorials/03-bayesian-neural-network/):
a small MLP's weights are flattened into one vector, given an isotropic (or,
in notebook 2's update step, a moment-matched informative) Gaussian prior,
and sampled as a block with NUTS — trading a single trained network for a
posterior over networks.

## Repository layout

```
notebooks/    Pluto notebooks (plain .jl files — also valid standalone Julia scripts)
pdfs/         Rendered PDF exports of each notebook, outputs included
scripts/      Notebook-authoring helper + PDF export pipeline (see below)
Project.toml  Julia environment / dependencies
Manifest.toml
```

`scripts/`:
- `pluto_build.py`, `gen_notebook1.py`, `gen_notebook2.py` — the notebooks
  were authored by generating valid Pluto cell/UUID structure from these
  scripts rather than by hand; re-run them after editing a notebook's cell
  content in these files.
- `export_pdfs.jl` — runs both notebooks via `PlutoSliderServer.export_notebook`
  to bake their outputs into static HTML under `html/` (gitignored).
- `vendor_frontend_dist.sh` — copies Pluto's built frontend assets next to
  the HTML export and rewrites its CDN links to local paths (invoked
  automatically by `html_to_pdf.sh`).
- `html_to_pdf.sh` / `html_to_pdf.js` — serves `html/` over local HTTP and
  drives headless Chromium (via Playwright) to print each notebook to
  `pdfs/*.pdf`.
