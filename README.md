# BNN — Bayesian Neural Network examples for Oil & Gas (Julia / Pluto)

Four self-contained [Pluto.jl](https://plutojl.org/) notebooks demonstrating
Bayesian neural networks — via [Turing.jl](https://turinglang.org/) NUTS
sampling and via [LaplaceRedux.jl](https://github.com/JuliaTrustworthyAI/LaplaceRedux.jl)
post-hoc Laplace approximation — on synthetic (but petrophysically-motivated)
oil & gas datasets.

## Notebooks

| Notebook | What it shows |
|---|---|
| [`notebooks/01_bnn_permeability_regression.jl`](notebooks/01_bnn_permeability_regression.jl) | A BNN (weights sampled with NUTS instead of backprop) predicting reservoir permeability from well-log curves (porosity, gamma ray, resistivity, water saturation), with full posterior-predictive uncertainty bands. |
| [`notebooks/02_additive_bayesian_updating.jl`](notebooks/02_additive_bayesian_updating.jl) | **Additive learning**: folding a newly-drilled batch of wells into an *existing* posterior (moment-matched Gaussian prior + NUTS on the new batch only), compared against a stale never-updated model and a from-scratch full retrain on all data. Also sweeps a **prior-inflation factor τ** (a power prior) to show how to tune how much the new batch counts relative to the carried-forward prior. |
| [`notebooks/03_laplace_redux_permeability.jl`](notebooks/03_laplace_redux_permeability.jl) | The **same** permeability-regression task as notebook 1, but using [LaplaceRedux.jl](https://github.com/JuliaTrustworthyAI/LaplaceRedux.jl): an ordinary backprop-trained Flux.jl MLP with a post-hoc Laplace (Gaussian curvature) approximation over its weights instead of full NUTS sampling — trading some posterior fidelity for roughly an order-of-magnitude less wall-clock time. |
| [`notebooks/04_laplace_additive_updating.jl`](notebooks/04_laplace_additive_updating.jl) | The **same** additive-updating task as notebook 2 (same Phase A/B dataset, same stale/additive/full-retrain comparison, same τ knob), but via **online Laplace approximation**: Phase A's Laplace posterior becomes a quadratic (Elastic-Weight-Consolidation-style) penalty for continued training on Phase B only, then a fresh Laplace fit combines Phase A's carried-forward precision with Phase B's own curvature. |

Rendered PDFs of the notebooks (with all outputs baked in) are in
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

Notebook 1 (permeability regression): held-out-well **R² = 0.90**, RMSE =
0.19 in log10(mD) space.

| | RMSE (pooled held-out test) | R² | wells processed *this step* |
|---|---|---|---|
| Stale (Phase A only, distribution-shifted Phase B never seen) | 0.313 | −0.18 | — |
| **Additive update** (Phase A posterior → prior, NUTS on Phase B only) | 0.246 | 0.27 | 19 |
| Full retrain (Phase A ∪ Phase B, from scratch) | 0.247 | 0.27 | 49 |

The additive update recovers essentially all of the full retrain's accuracy
gain over the stale model — while its MCMC step only ever touches the new
batch of wells, not the growing history.

### How much should the new batch count? (notebook 2 §7, prior-inflation τ)

Notebook 2 also sweeps a prior-inflation factor `τ` (a power prior on the
carried-forward Phase A posterior) to make "how much do I trust the new
data" an explicit, tunable knob rather than a fixed choice (one run):

| τ | RMSE Phase A test | RMSE Phase B test | R² (pooled) | MCMC seconds |
|---|---|---|---|---|
| 1 | 0.283 | 0.186 | 0.24 | 7.2 |
| 2 | 0.282 | 0.183 | 0.25 | 9.1 |
| 4 | 0.278 | 0.166 | 0.29 | 12.8 |
| 8 | 0.260 | 0.165 | 0.37 | 21.3 |

The textbook expectation is a trade-off (better on new data, worse on old);
here both improve as τ grows, because Phase A's posterior mean is itself a
noisy estimate from only ~30 wells and loosening the prior gives NUTS room
to find a jointly better fit — at roughly 3x the MCMC cost by τ=8. See the
notebook for the full discussion.

## Turing/NUTS vs. LaplaceRedux (notebook 1 vs. notebook 3)

Same dataset, split and MLP architecture, two different routes to
weight-space uncertainty (measured on this machine — see notebook 3 §6 for
discussion). Notebook 3 re-trains from scratch on every run, so its exact
numbers drift a little run to run (floating-point non-determinism in
threaded BLAS/autodiff); notebook 1's NUTS numbers are fixed, from the
previously-shipped PDF:

| | RMSE (log10 K) | R² | wall time |
|---|---|---|---|
| Notebook 1 — Turing.jl, NUTS, 300 samples | 0.19 | 0.90 | ≈ 183 s (sampling only, post-warmup) |
| Notebook 3 — Flux + LaplaceRedux | ≈ 0.22–0.26 | ≈ 0.81–0.86 | ≈ 15–20 s (train + fit + `optimize_prior!`) |

NUTS explores the full weight posterior and gives the better-calibrated
uncertainty; LaplaceRedux fits a single Gaussian to the curvature around a
backprop-trained point estimate, which is much cheaper but only locally
approximates the posterior around that one mode.

## Additive updating via online Laplace approximation (notebook 4)

Notebook 4 repeats notebook 2's Phase A/B experiment with Flux +
LaplaceRedux instead of Turing/NUTS: Phase A's Laplace posterior mean and
curvature become a quadratic penalty (an Elastic-Weight-Consolidation-style
regularizer — EWC *is* a Laplace approximation to sequential Bayes) for
continued training on Phase B only, then a fresh Laplace fit around the
updated weights combines Phase A's carried-forward precision with Phase
B's own curvature (the standard *online Laplace* combination rule). One
run, this machine (see notebook 4 §10 for the full discussion, and note
the same run-to-run drift as notebook 3 applies here):

| | RMSE (pooled held-out test) | R² | wall time |
|---|---|---|---|
| Stale (Phase A only) | ≈ 0.30–0.33 | ≈ −0.29 to −0.07 | — |
| **Additive update** (online Laplace, backprop on Phase B only) | ≈ 0.23–0.24 | ≈ 0.32–0.37 | ≈ 1–1.2 s |
| Full retrain (Phase A ∪ Phase B, from scratch) | ≈ 0.23–0.24 | ≈ 0.34–0.35 | ≈ 0.1–0.15 s |

Accuracy-wise this is the same story as notebook 2: the additive update
recovers essentially all of the full retrain's gain over the stale model.
**The wall-clock story is not**, and that's worth calling out rather than
glossing over: NUTS's cost scales with how much data it samples over, so
notebook 2's additive update was clearly cheaper than its full retrain.
Plain backprop on a few dozen wells is already so cheap that a full
retrain here is not obviously slower than the additive update — sometimes
it measures faster, since the additive path does strictly more work (a
custom regularized training loop plus an extra Hessian fit). At this data
scale, the real benefit of the additive route is that the update step
never needs Phase A's raw wells again — only the fixed-size `(μ_A, Λ_A,
σ_A)` summary — not wall-clock time. That would start to change at much
larger historical datasets or networks, where backprop cost per epoch
stops being negligible.

## Repeated updates and the forgetting factor (notebook 4 §9)

Sections 4–7 of notebook 4 run the online-Laplace update once. Run it
every round, forever, with no discount (`τ=1`), and it breaks: posterior
precision after `n` rounds is `Λ_n = Λ_0 + H_1 + H_2 + ... + H_n`, and
since every `H_k` is positive semi-definite, `Λ_n` only ever grows. The
model gets more certain every round regardless of whether new wells still
agree with it, until the penalty anchoring it to old weights dwarfs the
pull from any new batch — it locks in and stops being able to learn,
exactly when a drifting field needs it to keep adapting.

**The fix** is to apply the same prior-inflation `τ` at *every* round
instead of once: `Λ_k = Λ_(k-1)/τ + H_k`. Applied repeatedly, precision no
longer grows without bound — it converges to a steady state. Solving for
that steady state gives the **effective memory window**:

```
N_eff = τ / (τ - 1)          ⟺          τ = N_eff / (N_eff - 1)
```

`N_eff` is the number of wells' worth of information the model behaves as
if it remembers, independent of how many it has actually seen — the same
idea as a recursive-least-squares forgetting factor or a Kalman filter's
process-noise injection. **Practical recipe: decide how many wells of
memory you want** (how much drift you expect the field to show over time),
then set `τ = N_eff / (N_eff - 1)`.

Notebook 4 §9 tests this by drilling 15 further phases beyond Phase A,
each a little further from Phase A's rock quality than the last
(continuous drift), running the online-Laplace update every round with
`τ=1` against `τ` set from `N_eff=15` (one run, this machine):

| | round 0 | round 5 | round 10 | round 15 |
|---|---|---|---|---|
| avg. posterior std. dev., `τ=1` (no forgetting) | 1.62 | 1.37 | 1.24 | 1.17 |
| avg. posterior std. dev., `τ` from `N_eff=15` | 1.62 | 1.60 | 1.66 | 1.80 |

Without forgetting, the model's uncertainty shrinks monotonically with no
floor. With `τ` from `N_eff=15`, it stabilizes instead (and even ticks
back up here, since each new round's own curvature is milder than the
discounted history it replaces) — precision stops accumulating without
bound, exactly as the `N_eff` derivation predicts. The practical
consequence shows up directly in how far the weights move each round
(`‖θ_k − θ_(k-1)‖`): it is larger with forgetting than without at **every
single one of the 15 rounds** in this run — the forgetting factor visibly
keeping the model able to move toward new evidence instead of gradually
freezing in place.

One honest limitation, reported rather than hidden: held-out RMSE on each
round's own test wells did *not* show a correspondingly clean improvement
between the two — with only a handful of test wells per round, that
signal is too noisy at this data scale to separate from run-to-run
variance. The precision and weight-movement effects are the direct,
mechanistically guaranteed consequences of the forgetting factor; a
visible predictive-accuracy payoff would likely need more rounds, a
longer sustained drift, or larger batches per round before it clears the
noise floor.

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
- `pluto_build.py`, `gen_notebook1.py`, `gen_notebook2.py`, `gen_notebook3.py`,
  `gen_notebook4.py` — the notebooks were authored by generating valid
  Pluto cell/UUID structure from these scripts rather than by hand; re-run
  them after editing a notebook's cell content in these files.
- `export_pdfs.jl` — runs both notebooks via `PlutoSliderServer.export_notebook`
  to bake their outputs into static HTML under `html/` (gitignored).
- `vendor_frontend_dist.sh` — copies Pluto's built frontend assets next to
  the HTML export and rewrites its CDN links to local paths (invoked
  automatically by `html_to_pdf.sh`).
- `html_to_pdf.sh` / `html_to_pdf.js` — serves `html/` over local HTTP and
  drives headless Chromium (via Playwright) to print each notebook to
  `pdfs/*.pdf`.
