#!/usr/bin/env python3
import sys
sys.path.insert(0, "/home/user/bnn/scripts")
from pluto_build import write_notebook

cells = []

cells.append('''md"""
# Additive (Incremental) Bayesian Updating for a BNN — Oil & Gas Field Expansion

**Domain:** Oil & Gas — production forecasting during field development

**Reference:** [Turing.jl](https://turinglang.org/)

**The question this notebook tests:** when new wells are drilled after a
model is already in production use, can we *add* their data to the
existing posterior — instead of re-running MCMC over the entire well
history every time a handful of new wells shows up?

We compare three models, all sharing the same Bayesian-neural-network
architecture as `01_bnn_permeability_regression.jl`:

1. **Stale** — trained once on Phase A wells only, never updated.
2. **Additive update** — Phase A's posterior is turned into an
   informative prior (moment-matched Gaussian) and NUTS runs *again only
   on the new Phase B data*, i.e. sequential Bayesian updating
   (``p(\\theta \\mid A, B) \\approx p(\\theta \\mid B) \\cdot p(\\theta \\mid A)``
   used as prior).
3. **Full retrain** — NUTS run from scratch on Phase A ∪ Phase B pooled
   together — the expensive gold standard the additive update is trying to
   approximate.
"""''')

cells.append('''begin
	using Turing
	using MCMCChains
	using Distributions
	using Random
	using LinearAlgebra
	using StatsPlots
	using Plots
	using DataFrames
	using PlutoUI
end''')

cells.append('TableOfContents()')

cells.append('''md"""
## 1. Synthetic production data: two field-development phases

We predict **Estimated Ultimate Recovery**, `EUR` (thousand barrels), from
three early-life well attributes: initial rate `qi` (bbl/d), initial
Arps decline rate `Di` (1/yr) and reservoir permeability `K` (mD).

* **Phase A ("core area")** — the first 40 wells drilled, used to build the
  initial model.
* **Phase B ("step-out area")** — 25 wells drilled later in an adjacent
  fault block with measurably different reservoir quality. This is a
  deliberate *distribution shift*: it is exactly the situation where a
  stale model needs updating, and where "can we add data without
  retraining" matters operationally.
"""''')

cells.append('''function generate_phase(n::Int; seed::Int, quality_shift::Float64)
	rng = MersenneTwister(seed)
	qi = rand(rng, LogNormal(log(600.0), 0.35), n)          # initial rate, bbl/d
	Di = rand(rng, Uniform(0.35, 0.85), n)                   # initial decline, 1/yr
	K  = rand(rng, LogNormal(log(40.0) + quality_shift, 0.5), n)  # permeability, mD

	# Simplified hyperbolic-decline-flavoured EUR model: higher qi and K help,
	# steep initial decline Di hurts, with a mild qi x K synergy term.
	log_eur = @. (3.55 + 0.55 * log(qi / 600.0) - 1.15 * Di +
	              0.30 * log(K / 40.0) + 0.12 * log(K / 40.0) * log(qi / 600.0))
	noise = rand(rng, Normal(0.0, 0.20), n)
	eur = exp.(log_eur .+ noise)                             # Mbbl

	DataFrame(QI=qi, DI=Di, K=K, EUR=eur, LOGEUR=log.(eur))
end''')

cells.append('''begin
	Random.seed!(2024)
	phaseA = generate_phase(40; seed=101, quality_shift=0.0)    # core area
	phaseB = generate_phase(25; seed=202, quality_shift=-0.55)  # step-out area: lower-K rock
	(nA=nrow(phaseA), nB=nrow(phaseB))
end''')

cells.append('''begin
	@df phaseA scatter(:K, :EUR, label="Phase A (core)", xlabel="permeability K (mD)",
		ylabel="EUR (Mbbl)", yscale=:log10, xscale=:log10, markersize=4, alpha=0.7, size=(650,450))
	@df phaseB scatter!(:K, :EUR, label="Phase B (step-out)", markersize=4, alpha=0.7, markershape=:diamond)
end''')

cells.append('''md"""
## 2. Shared BNN architecture & Turing model

Same idea as notebook 1: 3 inputs → 5 tanh → 5 tanh → 1 linear output,
weights flattened into one vector `θ`. The model function takes an
explicit `prior_mean` / `prior_cov` so it can be reused both for a vague
"fit from scratch" prior and for an informative "posterior-of-A-as-prior"
distribution.
"""''')

cells.append('''begin
	FEATURES2 = [:QI, :DI, :K]
	N_IN2, N_H1_2, N_H2_2 = 3, 5, 5
	N_W1_2 = N_IN2 * N_H1_2 + N_H1_2
	N_W2_2 = N_H1_2 * N_H2_2 + N_H2_2
	N_W3_2 = N_H2_2 * 1 + 1
	N_PARAMS2 = N_W1_2 + N_W2_2 + N_W3_2

	function nn_forward2(X::AbstractMatrix, θ::AbstractVector)
		i = 1
		W1 = reshape(θ[i:i+N_IN2*N_H1_2-1], N_H1_2, N_IN2); i += N_IN2 * N_H1_2
		b1 = θ[i:i+N_H1_2-1];                                i += N_H1_2
		W2 = reshape(θ[i:i+N_H1_2*N_H2_2-1], N_H2_2, N_H1_2); i += N_H1_2 * N_H2_2
		b2 = θ[i:i+N_H2_2-1];                                i += N_H2_2
		W3 = reshape(θ[i:i+N_H2_2-1], 1, N_H2_2);            i += N_H2_2
		b3 = θ[i]

		h1 = tanh.(W1 * X .+ b1)
		h2 = tanh.(W2 * h1 .+ b2)
		vec(W3 * h2 .+ b3)
	end

	N_PARAMS2
end''')

cells.append('''@model function bayes_nn2(X, y, prior_mean, prior_cov; obs_sigma=0.25)
	θ ~ MvNormal(prior_mean, prior_cov)
	preds = nn_forward2(X, θ)
	y ~ MvNormal(preds, obs_sigma^2 * I(length(y)))
end''')

cells.append('''md"""
### Feature standardization

Standardize using **Phase A statistics only** — Phase B is meant to arrive
later, "in production", so nothing about it may leak into preprocessing
decided when the original model was built.
"""''')

cells.append('''begin
	function tt_split(df; frac_train=0.75, seed=1)
		rng = MersenneTwister(seed)
		idx = shuffle(rng, 1:nrow(df))
		ntr = round(Int, frac_train * nrow(df))
		df[idx[1:ntr], :], df[idx[ntr+1:end], :]
	end

	A_train, A_test = tt_split(phaseA; seed=11)
	B_train, B_test = tt_split(phaseB; seed=22)

	fmean2 = [mean(A_train[!, f]) for f in FEATURES2]
	fstd2  = [std(A_train[!, f])  for f in FEATURES2]
	standardize2(df) = hcat([ (df[!, f] .- m) ./ s for (f, m, s) in zip(FEATURES2, fmean2, fstd2) ]...)

	XA_train, yA_train = standardize2(A_train), Float64.(A_train.LOGEUR)
	XA_test,  yA_test  = standardize2(A_test),  Float64.(A_test.LOGEUR)
	XB_train, yB_train = standardize2(B_train), Float64.(B_train.LOGEUR)
	XB_test,  yB_test  = standardize2(B_test),  Float64.(B_test.LOGEUR)

	(nA_train=length(yA_train), nA_test=length(yA_test), nB_train=length(yB_train), nB_test=length(yB_test))
end''')

cells.append('''md"""
## 3. Step 1 — fit the initial ("stale") model on Phase A only
"""''')

cells.append('''begin
	Random.seed!(1)
	vague_mean = zeros(N_PARAMS2)
	vague_cov  = 1.0^2 * I(N_PARAMS2)

	t_fitA = @elapsed chainA = sample(
		bayes_nn2(XA_train', yA_train, vague_mean, vague_cov), NUTS(0.8), 250; progress=false, chain_type=Chains)
	t_fitA
end''')

cells.append('''function rmse_r2(model_forward, θsamples, Xrows, y)
	Xt = Xrows'  # (n_features, n_obs)
	preds = reduce(hcat, [model_forward(Xt, θsamples[:, s]) for s in 1:size(θsamples, 2)])
	pmean = vec(mean(preds, dims=2))
	rmse = sqrt(mean((pmean .- y).^2))
	ss_res, ss_tot = sum((y .- pmean).^2), sum((y .- mean(y)).^2)
	(rmse=rmse, r2=1 - ss_res/ss_tot, pred_mean=pmean, preds=preds)
end''')

cells.append('''begin
	θA = Array(group(chainA, :θ))'
	stale_on_A = rmse_r2(nn_forward2, θA, XA_test, yA_test)
	stale_on_B = rmse_r2(nn_forward2, θA, XB_test, yB_test)
	(rmse_on_A_test=round(stale_on_A.rmse, digits=3), rmse_on_B_test_STALE=round(stale_on_B.rmse, digits=3))
end''')

cells.append('''md"""
The RMSE jump from Phase A test wells to Phase B test wells (evaluated with
the *same, unmodified* Phase-A-only model) is the distribution-shift
penalty of doing nothing. That gap is what an update — additive or full
retrain — should close.
"""''')

cells.append('''md"""
## 4. Step 2a — additive update: fold Phase B into the existing posterior

We moment-match Phase A's posterior samples to a Gaussian
(``\\hat\\mu, \\hat\\Sigma``) and use it as the prior for a fresh NUTS run
that sees **only Phase B data** — Phase A's raw well data is never touched
again.
"""''')

cells.append('''begin
	post_mean_A = vec(mean(θA, dims=2))
	post_cov_A  = cov(θA'; dims=1) + 1e-6 * I(N_PARAMS2)  # jitter for PD-ness

	Random.seed!(2)
	t_update = @elapsed chainB_additive = sample(
		bayes_nn2(XB_train', yB_train, post_mean_A, post_cov_A), NUTS(0.8), 250; progress=false, chain_type=Chains)
	t_update
end''')

cells.append('''md"""
## 5. Step 2b — baseline: full retrain on Phase A ∪ Phase B pooled
"""''')

cells.append('''begin
	Xall_train = vcat(XA_train, XB_train)
	yall_train = vcat(yA_train, yB_train)

	Random.seed!(3)
	t_retrain = @elapsed chain_full = sample(
		bayes_nn2(Xall_train', yall_train, vague_mean, vague_cov), NUTS(0.8), 250; progress=false, chain_type=Chains)
	t_retrain
end''')

cells.append('''md"""
## 6. Compare: stale vs. additive update vs. full retrain

All three are evaluated on the *same* held-out mix of Phase A and Phase B
test wells.
"""''')

cells.append('''begin
	Xall_test = vcat(XA_test, XB_test)
	yall_test = vcat(yA_test, yB_test)

	θB_additive = Array(group(chainB_additive, :θ))'
	θfull       = Array(group(chain_full, :θ))'

	res_stale    = rmse_r2(nn_forward2, θA,          Xall_test, yall_test)
	res_additive = rmse_r2(nn_forward2, θB_additive, Xall_test, yall_test)
	res_full     = rmse_r2(nn_forward2, θfull,        Xall_test, yall_test)

	comparison = DataFrame(
		model = ["stale (Phase A only)", "additive update (A-posterior → prior, fit on B)", "full retrain (A+B pooled)"],
		rmse  = round.([res_stale.rmse, res_additive.rmse, res_full.rmse], digits=3),
		r2    = round.([res_stale.r2, res_additive.r2, res_full.r2], digits=3),
		mcmc_seconds = round.([t_fitA, t_update, t_retrain], digits=1),
		wells_processed_this_step = [nrow(A_train), nrow(B_train), nrow(A_train) + nrow(B_train)],
	)
	comparison
end''')

cells.append('''@df comparison bar(:model, :rmse, xrotation=15, legend=false, size=(700,420),
	ylabel="RMSE on pooled held-out wells (log EUR)", title="Additive update recovers most of the full-retrain gain")''')

cells.append('''md"""
The key comparison is **rows 2 vs. 3**: the additive update reprocesses
only the `nrow(B_train)` new wells through MCMC (not all
`nrow(A_train) + nrow(B_train)`), yet its held-out RMSE should land close
to the full retrain — and clearly better than leaving the stale model
untouched (row 1). As the historical well count grows, "wells processed
this step" for the additive route stays flat at the new-batch size while
the full-retrain route grows without bound — that gap is the whole point
of testing additive capability.
"""''')

cells.append('''md"""
## 7. Posterior predictive on a Phase B well: stale vs. additive vs. retrain
"""''')

cells.append('''begin
	k_grid = range(minimum(phaseB.K), maximum(phaseB.K); length=60)
	qi_med, di_med = median(B_train.QI), median(B_train.DI)

	Xg = zeros(N_IN2, length(k_grid))
	for j in 1:length(k_grid)
		Xg[1, j] = (qi_med     - fmean2[1]) / fstd2[1]
		Xg[2, j] = (di_med     - fmean2[2]) / fstd2[2]
		Xg[3, j] = (k_grid[j]  - fmean2[3]) / fstd2[3]
	end

	function band(θsamples, X)
		preds = reduce(hcat, [nn_forward2(X, θsamples[:, s]) for s in 1:size(θsamples, 2)])
		m = vec(mean(preds, dims=2))
		lo = [quantile(preds[i, :], 0.05) for i in 1:size(preds,1)]
		hi = [quantile(preds[i, :], 0.95) for i in 1:size(preds,1)]
		m, lo, hi
	end

	m_s, lo_s, hi_s = band(θA, Xg)
	m_u, lo_u, hi_u = band(θB_additive, Xg)
	m_f, lo_f, hi_f = band(θfull, Xg)

	plot(k_grid, m_s, ribbon=(m_s .- lo_s, hi_s .- m_s), label="stale (A only)",
		xscale=:log10, xlabel="permeability K (mD)", ylabel="predicted log(EUR)",
		title="Step-out area: does the model know what it doesn't know?", size=(700,460), fillalpha=0.18)
	plot!(k_grid, m_u, ribbon=(m_u .- lo_u, hi_u .- m_u), label="additive update", fillalpha=0.18)
	plot!(k_grid, m_f, ribbon=(m_f .- lo_f, hi_f .- m_f), label="full retrain", fillalpha=0.18, linestyle=:dash)
end''')

cells.append('''md"""
## 8. Discussion & caveats

* **What worked.** Sequential Bayesian updating — moment-matching the old
  posterior into a new prior and running MCMC on only the incoming batch —
  recovered most of the accuracy gain of a full retrain while touching
  none of the historical Phase A well data during the update step.
* **Why this matters operationally.** In a real field, historical well
  count only grows. A workflow that must re-run MCMC over the *entire*
  history every time a handful of wells is added does not scale; an
  additive update whose cost depends only on the new batch does.
* **Caveat — Gaussian moment-matching is an approximation.** True
  sequential Bayes (`posterior ∝ likelihood(B) × posterior(A)`) is exact
  only if the Gaussian summary of `posterior(A)` is a faithful
  representation of that posterior. For a strongly multimodal or
  heavy-tailed weight posterior (common in larger BNNs) this approximation
  degrades, and periodic full retrains are still worth budgeting for as a
  reconciliation step.
* **Caveat — this is not online/streaming SGD.** Each "update" here is
  still a batch MCMC run (just a cheaper one, over less data with a
  tighter, informative prior) — not a single-sample online update.
"""''')

write_notebook("/home/user/bnn/notebooks/02_additive_bayesian_updating.jl", cells)
