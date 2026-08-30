#!/usr/bin/env python3
import sys
sys.path.insert(0, "/home/user/bnn/scripts")
from pluto_build import write_notebook

cells = []

cells.append('''md"""
# Bayesian Neural Network for Reservoir Permeability Prediction

**Domain:** Oil & Gas — petrophysics / formation evaluation

**Reference:** [Turing.jl](https://turinglang.org/) — probabilistic programming in Julia.
This notebook follows the structure of Turing's official *Bayesian Neural
Network* tutorial, applied to a synthetic well-log dataset.

**Goal.** Predict reservoir permeability ``K`` (mD) from standard well-log
curves — porosity (``\\phi``), gamma ray (``GR``), deep resistivity
(``R_t``) and water saturation (``S_w``) — using a small multilayer
perceptron whose *weights* are given priors and inferred with MCMC (NUTS)
instead of point-estimated by backpropagation. The payoff is a full
posterior predictive distribution over permeability, i.e. calibrated
uncertainty bands instead of a single number — directly useful for P10 /
P50 / P90 reserve estimation and risked decision-making.

A companion notebook (`02_additive_bayesian_updating.jl`) reuses this same
model class to demonstrate *additive* learning: folding new well data into
an existing posterior without re-fitting on the full history.
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
## 1. Synthetic well-log dataset

Real well-log/core-permeability pairs are commercially sensitive, so we
generate a synthetic but petrophysically-motivated dataset: permeability is
approximately log-normal and depends *nonlinearly* on porosity and water
saturation (a simplified stand-in for Kozeny–Carman-type behaviour), with
gamma ray acting as a shale/clay-content proxy and resistivity carrying
independent (noisy) information about pore-fluid and texture.
"""''')

cells.append('''function generate_wells(n::Int; seed::Int)
	rng = MersenneTwister(seed)
	phi = rand(rng, Uniform(0.04, 0.30), n)                # porosity (fraction)
	gr  = rand(rng, Uniform(20.0, 150.0), n)                # gamma ray (API)
	rt  = rand(rng, LogNormal(log(20.0), 0.8), n)           # deep resistivity (ohm.m)
	sw  = clamp.(rand(rng, Beta(2.0, 2.5), n), 0.05, 0.98)  # water saturation (fraction)

	# nonlinear, petrophysically-flavoured log-permeability model
	log10K = @. (-1.2 + 9.0 * phi - 12.0 * phi^2 - 0.014 * gr -
	             1.6 * sw + 0.9 * sw * phi + 0.35 * log10(rt))
	noise = rand(rng, Normal(0.0, 0.22), n)
	log10K = log10K .+ noise
	K = clamp.(10.0 .^ log10K, 0.01, 5000.0)                # permeability (mD)

	DataFrame(PHI=phi, GR=gr, RT=rt, SW=sw, K=K, LOGK=log10.(K))
end''')

cells.append('''begin
	Random.seed!(20240517)
	wells = generate_wells(320; seed=20240517)
	first(wells, 5)
end''')

cells.append('''@df wells corrplot([:PHI :GR :RT :SW :LOGK], grid=false, size=(700,700))''')

cells.append('''md"""
## 2. Train / test split and feature standardization

The neural-network weight prior is easiest to reason about (and NUTS mixes
better) when inputs are standardized to zero mean / unit variance.
"""''')

cells.append('''begin
	FEATURES = [:PHI, :GR, :RT, :SW]

	function train_test_split(df::DataFrame; frac_train=0.8, seed=1)
		rng = MersenneTwister(seed)
		n = nrow(df)
		idx = shuffle(rng, 1:n)
		ntrain = round(Int, frac_train * n)
		df[idx[1:ntrain], :], df[idx[ntrain+1:end], :]
	end

	wells_train, wells_test = train_test_split(wells; frac_train=0.8, seed=7)
	(n_train=nrow(wells_train), n_test=nrow(wells_test))
end''')

cells.append('''begin
	feat_mean = [mean(wells_train[!, f]) for f in FEATURES]
	feat_std  = [std(wells_train[!, f]) for f in FEATURES]

	standardize(df) = hcat([ (df[!, f] .- m) ./ s
	                          for (f, m, s) in zip(FEATURES, feat_mean, feat_std) ]...)

	Xtrain = standardize(wells_train)   # size: (n_train, n_features)
	Xtest  = standardize(wells_test)    # size: (n_test, n_features)
	ytrain = Float64.(wells_train.LOGK) # model the log-permeability
	ytest  = Float64.(wells_test.LOGK)

	size(Xtrain), size(Xtest)
end''')

cells.append('''md"""
## 3. Bayesian neural network (Turing.jl)

A small MLP — 4 inputs → 6 tanh → 6 tanh → 1 linear output — with a
`Normal(0, σ_prior)` prior on every weight and bias, flattened into a
single parameter vector `θ` so it can be sampled as one block by NUTS.
"""''')

cells.append('''begin
	N_IN, N_H1, N_H2 = 4, 6, 6
	N_W1 = N_IN * N_H1 + N_H1
	N_W2 = N_H1 * N_H2 + N_H2
	N_W3 = N_H2 * 1 + 1
	N_PARAMS = N_W1 + N_W2 + N_W3

	"Unpack a flat parameter vector into the MLP's weight matrices/biases and
	evaluate it on a (n_features, n_obs) design matrix."
	function nn_forward(X::AbstractMatrix, θ::AbstractVector)
		i = 1
		W1 = reshape(θ[i:i+N_IN*N_H1-1], N_H1, N_IN); i += N_IN * N_H1
		b1 = θ[i:i+N_H1-1];                            i += N_H1
		W2 = reshape(θ[i:i+N_H1*N_H2-1], N_H2, N_H1);  i += N_H1 * N_H2
		b2 = θ[i:i+N_H2-1];                            i += N_H2
		W3 = reshape(θ[i:i+N_H2-1], 1, N_H2);          i += N_H2
		b3 = θ[i]

		h1 = tanh.(W1 * X .+ b1)
		h2 = tanh.(W2 * h1 .+ b2)
		out = W3 * h2 .+ b3
		vec(out)
	end

	N_PARAMS
end''')

cells.append('''@model function bayes_nn(X, y; sigma_prior=1.0, obs_sigma=0.30)
	θ ~ MvNormal(zeros(N_PARAMS), sigma_prior^2 * I(N_PARAMS))
	preds = nn_forward(X, θ)
	y ~ MvNormal(preds, obs_sigma^2 * I(length(y)))
end''')

cells.append('''md"""
## 4. Posterior sampling (NUTS)

We sample the full weight-space posterior with the No-U-Turn Sampler. This
is the step an ordinary (non-Bayesian) neural net replaces with a single
backprop-trained point estimate — here we get a distribution over networks.
"""''')

cells.append('''begin
	Random.seed!(42)
	bnn_model = bayes_nn(Xtrain', ytrain)
	chain = sample(bnn_model, NUTS(0.8), 300; progress=false, chain_type=Chains)
end''')

cells.append('''md"""
## 5. Posterior predictive & uncertainty quantification
"""''')

cells.append('''begin
	θ_samples = Array(group(chain, :θ))'  # N_PARAMS x n_samples
	n_post = size(θ_samples, 2)

	preds_test = reduce(hcat, [nn_forward(Xtest', θ_samples[:, s]) for s in 1:n_post])
	# preds_test: n_test x n_post

	pred_mean = vec(mean(preds_test, dims=2))
	pred_lo   = [quantile(preds_test[i, :], 0.05) for i in 1:size(preds_test,1)]
	pred_hi   = [quantile(preds_test[i, :], 0.95) for i in 1:size(preds_test,1)]

	rmse = sqrt(mean((pred_mean .- ytest).^2))
	ss_res = sum((ytest .- pred_mean).^2)
	ss_tot = sum((ytest .- mean(ytest)).^2)
	r2 = 1 - ss_res / ss_tot

	(rmse_log10K=round(rmse, digits=3), r2=round(r2, digits=3))
end''')

cells.append('''begin
	scatter(ytest, pred_mean,
		yerror = (pred_mean .- pred_lo, pred_hi .- pred_mean),
		xlabel = "true log10(K)  [mD]", ylabel = "predicted log10(K)  [mD]",
		title = "Held-out wells: BNN posterior predictive (90% CI)",
		label = "test wells", markersize=4, alpha=0.7, legend=:topleft, size=(650,500))
	minv, maxv = extrema(vcat(ytest, pred_mean))
	plot!([minv, maxv], [minv, maxv], label="perfect prediction", linestyle=:dash, color=:black)
end''')

cells.append('''md"""
### Partial-dependence view: permeability vs. porosity

Holding gamma ray, resistivity and water saturation at their (training)
median, we sweep porosity across its observed range and propagate the full
posterior through the network. The widening/narrowing shaded band is the
network's own epistemic uncertainty — it is not hand-tuned, it falls out of
the posterior.
"""''')

cells.append('''begin
	phi_grid = range(minimum(wells.PHI), maximum(wells.PHI); length=60)
	gr_med, rt_med, sw_med = median(wells_train.GR), median(wells_train.RT), median(wells_train.SW)

	Xgrid = zeros(N_IN, length(phi_grid))       # (n_features, n_grid) — matches nn_forward's layout
	for k in 1:length(phi_grid)
		Xgrid[1, k] = (phi_grid[k] - feat_mean[1]) / feat_std[1]
		Xgrid[2, k] = (gr_med      - feat_mean[2]) / feat_std[2]
		Xgrid[3, k] = (rt_med      - feat_mean[3]) / feat_std[3]
		Xgrid[4, k] = (sw_med      - feat_mean[4]) / feat_std[4]
	end

	grid_preds = reduce(hcat, [nn_forward(Xgrid, θ_samples[:, s]) for s in 1:n_post])
	grid_mean = vec(mean(grid_preds, dims=2))
	grid_lo   = [quantile(grid_preds[i, :], 0.05) for i in 1:size(grid_preds,1)]
	grid_hi   = [quantile(grid_preds[i, :], 0.95) for i in 1:size(grid_preds,1)]

	plot(phi_grid, grid_mean, ribbon=(grid_mean .- grid_lo, grid_hi .- grid_mean),
		xlabel="porosity (fraction)", ylabel="predicted log10(K)  [mD]",
		label="posterior mean ± 90% CI", title="Permeability vs porosity (GR, Rt, Sw held at median)",
		size=(650,450), fillalpha=0.25, linewidth=2)
end''')

cells.append('''md"""
## 6. Discussion

* The BNN's posterior predictive band gives well-by-well permeability
  uncertainty, not just a point value — this maps directly onto reserve
  risking (P10/P50/P90) and completion-design decisions under geological
  uncertainty.
* `chain` (an `MCMCChains.Chains` object) can be inspected with the usual
  diagnostics (`summarystats`, trace/density plots via `plot(chain)`,
  `ess_rhat`, etc.) to check mixing before trusting the uncertainty bands.
* The next notebook, `02_additive_bayesian_updating.jl`, shows how to fold
  *new* wells into this posterior as they are drilled, without re-running
  MCMC over the full well history each time.
"""''')

write_notebook("/home/user/bnn/notebooks/01_bnn_permeability_regression.jl", cells)
