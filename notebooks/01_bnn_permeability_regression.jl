### A Pluto.jl notebook ###
# v0.19.45

using Markdown
using InteractiveUtils

# ╔═╡ 79b05842-2edb-4a62-a377-48498cd00a48
md"""
# Bayesian Neural Network for Reservoir Permeability Prediction

**Domain:** Oil & Gas — petrophysics / formation evaluation

**Reference:** [Turing.jl](https://turinglang.org/) — probabilistic programming in Julia.
This notebook follows the structure of Turing's official *Bayesian Neural
Network* tutorial, applied to a synthetic well-log dataset.

**Goal.** Predict reservoir permeability ``K`` (mD) from standard well-log
curves — porosity (``\phi``), gamma ray (``GR``), deep resistivity
(``R_t``) and water saturation (``S_w``) — using a small multilayer
perceptron whose *weights* are given priors and inferred with MCMC (NUTS)
instead of point-estimated by backpropagation. The payoff is a full
posterior predictive distribution over permeability, i.e. calibrated
uncertainty bands instead of a single number — directly useful for P10 /
P50 / P90 reserve estimation and risked decision-making.

A companion notebook (`02_additive_bayesian_updating.jl`) reuses this same
model class to demonstrate *additive* learning: folding new well data into
an existing posterior without re-fitting on the full history.
"""

# ╔═╡ c2af0614-3d12-4847-84ea-d9d806a28c1e
begin
	using Turing
	using MCMCChains
	using Distributions
	using Random
	using LinearAlgebra
	using StatsPlots
	using Plots
	using DataFrames
	using PlutoUI
end

# ╔═╡ 807aaa81-f117-4ad4-bb76-331db726a581
TableOfContents()

# ╔═╡ ccd5e512-7849-487b-b059-3f7465dbb3e7
md"""
## 1. Synthetic well-log dataset

Real well-log/core-permeability pairs are commercially sensitive, so we
generate a synthetic but petrophysically-motivated dataset: permeability is
approximately log-normal and depends *nonlinearly* on porosity and water
saturation (a simplified stand-in for Kozeny–Carman-type behaviour), with
gamma ray acting as a shale/clay-content proxy and resistivity carrying
independent (noisy) information about pore-fluid and texture.
"""

# ╔═╡ cc0379c6-0961-4d96-815b-1d4acefcb9fd
function generate_wells(n::Int; seed::Int)
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
end

# ╔═╡ df14e21c-f49d-45c6-b2b1-9c53a460c7c4
begin
	Random.seed!(20240517)
	wells = generate_wells(320; seed=20240517)
	first(wells, 5)
end

# ╔═╡ 3a7b73a9-f8b3-4841-8ff7-af7313c9d767
@df wells corrplot([:PHI :GR :RT :SW :LOGK], grid=false, size=(700,700))

# ╔═╡ 2f8796dc-1d8d-4529-9261-b7791fa4c1b0
md"""
## 2. Train / test split and feature standardization

The neural-network weight prior is easiest to reason about (and NUTS mixes
better) when inputs are standardized to zero mean / unit variance.
"""

# ╔═╡ 6f27f11f-aeaf-4483-a940-6a39ff6debe3
begin
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
end

# ╔═╡ eda63f1f-7943-4326-8762-badef2cd6eff
begin
	feat_mean = [mean(wells_train[!, f]) for f in FEATURES]
	feat_std  = [std(wells_train[!, f]) for f in FEATURES]

	standardize(df) = hcat([ (df[!, f] .- m) ./ s
	                          for (f, m, s) in zip(FEATURES, feat_mean, feat_std) ]...)

	Xtrain = standardize(wells_train)   # size: (n_train, n_features)
	Xtest  = standardize(wells_test)    # size: (n_test, n_features)
	ytrain = Float64.(wells_train.LOGK) # model the log-permeability
	ytest  = Float64.(wells_test.LOGK)

	size(Xtrain), size(Xtest)
end

# ╔═╡ d2f6149c-b7f4-4346-8f03-aea46997523b
md"""
## 3. Bayesian neural network (Turing.jl)

A small MLP — 4 inputs → 6 tanh → 6 tanh → 1 linear output — with a
`Normal(0, σ_prior)` prior on every weight and bias, flattened into a
single parameter vector `θ` so it can be sampled as one block by NUTS.
"""

# ╔═╡ 6171a5db-c5cc-47a0-8d56-871df61e7a7d
begin
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
end

# ╔═╡ 67e74d1b-c06d-4691-a9cc-8a55c6e7d344
@model function bayes_nn(X, y; sigma_prior=1.0, obs_sigma=0.30)
	θ ~ MvNormal(zeros(N_PARAMS), sigma_prior^2 * I(N_PARAMS))
	preds = nn_forward(X, θ)
	y ~ MvNormal(preds, obs_sigma^2 * I(length(y)))
end

# ╔═╡ 3437751c-d0a4-43a6-9144-04c5c8024bda
md"""
## 4. Posterior sampling (NUTS)

We sample the full weight-space posterior with the No-U-Turn Sampler. This
is the step an ordinary (non-Bayesian) neural net replaces with a single
backprop-trained point estimate — here we get a distribution over networks.
"""

# ╔═╡ 591274e6-2e29-4ea4-a08d-f1591647db26
begin
	Random.seed!(42)
	bnn_model = bayes_nn(Xtrain', ytrain)
	chain = sample(bnn_model, NUTS(0.8), 300; progress=false, chain_type=Chains)
end

# ╔═╡ 87a7d517-4882-4ccc-9e69-d680e71b6008
md"""
## 5. Posterior predictive & uncertainty quantification
"""

# ╔═╡ fb08a8d3-c403-4b8e-9fb4-b90e1a1b8509
begin
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
end

# ╔═╡ d2d183b8-01e2-4195-acf2-174449819724
begin
	scatter(ytest, pred_mean,
		yerror = (pred_mean .- pred_lo, pred_hi .- pred_mean),
		xlabel = "true log10(K)  [mD]", ylabel = "predicted log10(K)  [mD]",
		title = "Held-out wells: BNN posterior predictive (90% CI)",
		label = "test wells", markersize=4, alpha=0.7, legend=:topleft, size=(650,500))
	minv, maxv = extrema(vcat(ytest, pred_mean))
	plot!([minv, maxv], [minv, maxv], label="perfect prediction", linestyle=:dash, color=:black)
end

# ╔═╡ 557d0155-cb75-4091-93bf-864e367a5e12
md"""
### Partial-dependence view: permeability vs. porosity

Holding gamma ray, resistivity and water saturation at their (training)
median, we sweep porosity across its observed range and propagate the full
posterior through the network. The widening/narrowing shaded band is the
network's own epistemic uncertainty — it is not hand-tuned, it falls out of
the posterior.
"""

# ╔═╡ a2b9c584-221e-489b-bbf8-f7e5d1d05088
begin
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
end

# ╔═╡ 85ed025b-886a-429a-b94b-0affd76f7939
md"""
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
"""

# ╔═╡ Cell order:
# ╠═79b05842-2edb-4a62-a377-48498cd00a48
# ╠═c2af0614-3d12-4847-84ea-d9d806a28c1e
# ╠═807aaa81-f117-4ad4-bb76-331db726a581
# ╠═ccd5e512-7849-487b-b059-3f7465dbb3e7
# ╠═cc0379c6-0961-4d96-815b-1d4acefcb9fd
# ╠═df14e21c-f49d-45c6-b2b1-9c53a460c7c4
# ╠═3a7b73a9-f8b3-4841-8ff7-af7313c9d767
# ╠═2f8796dc-1d8d-4529-9261-b7791fa4c1b0
# ╠═6f27f11f-aeaf-4483-a940-6a39ff6debe3
# ╠═eda63f1f-7943-4326-8762-badef2cd6eff
# ╠═d2f6149c-b7f4-4346-8f03-aea46997523b
# ╠═6171a5db-c5cc-47a0-8d56-871df61e7a7d
# ╠═67e74d1b-c06d-4691-a9cc-8a55c6e7d344
# ╠═3437751c-d0a4-43a6-9144-04c5c8024bda
# ╠═591274e6-2e29-4ea4-a08d-f1591647db26
# ╠═87a7d517-4882-4ccc-9e69-d680e71b6008
# ╠═fb08a8d3-c403-4b8e-9fb4-b90e1a1b8509
# ╠═d2d183b8-01e2-4195-acf2-174449819724
# ╠═557d0155-cb75-4091-93bf-864e367a5e12
# ╠═a2b9c584-221e-489b-bbf8-f7e5d1d05088
# ╠═85ed025b-886a-429a-b94b-0affd76f7939
