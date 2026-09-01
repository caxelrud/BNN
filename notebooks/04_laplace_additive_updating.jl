### A Pluto.jl notebook ###
# v0.19.45

using Markdown
using InteractiveUtils

# ╔═╡ 719213de-a9ad-4606-82ca-b724351daec1
md"""
# Additive Updating via Online Laplace Approximation — Oil & Gas Field Expansion

**Domain:** Oil & Gas — production forecasting during field development

**Reference:** [LaplaceRedux.jl](https://github.com/JuliaTrustworthyAI/LaplaceRedux.jl) /
[Flux.jl](https://fluxml.ai/) — repeats notebook 2's question with a different
mechanism for "add new data without retraining from scratch".

**The question this notebook tests is the same as notebook 2's:** when new
wells are drilled after a model is already in production use, can we *add*
their data to the existing model — instead of re-fitting on the entire well
history every time a handful of new wells shows up? Notebook 2 answered this
with Turing/NUTS: moment-match the old posterior to a Gaussian and run MCMC
on only the new batch. Here we answer it with an ordinary backprop-trained
Flux network and a **Laplace approximation carried forward as a prior** —
this is the same idea sequential Bayesian updating uses, but via a quadratic
penalty during continued training instead of MCMC. It's also, from the
continual-learning literature's side, exactly [Elastic Weight
Consolidation](https://en.wikipedia.org/wiki/Overcoming_catastrophic_forgetting)-style
regularization, since EWC *is* a Laplace approximation to sequential Bayes.

We compare the same three models as notebook 2, on the *same* synthetic
Phase A / Phase B well dataset:

1. **Stale** — trained once on Phase A wells only, never updated.
2. **Additive update** — Phase A's Laplace posterior (mean + curvature)
   becomes a quadratic penalty for continued training on Phase B only,
   then a fresh Laplace fit around the updated weights combines Phase A's
   carried-forward precision with Phase B's own curvature.
3. **Full retrain** — a fresh network trained from scratch on Phase A ∪
   Phase B pooled — the expensive(?) gold standard the additive update is
   trying to approximate.
"""

# ╔═╡ a7c5079a-9517-497f-86aa-0b2044f55131
begin
	using Flux
	using LaplaceRedux
	using Distributions
	using Random
	using LinearAlgebra
	using Statistics
	using Logging
	using StatsPlots
	using Plots
	using DataFrames
	using PlutoUI
end

# ╔═╡ e5606e00-522e-4efb-8194-c58dbffc965e
TableOfContents()

# ╔═╡ a773c149-9974-4f45-ac90-0e3dd1af9a73
md"""
## 1. Synthetic production data: two field-development phases

Identical dataset to notebook 2: **Estimated Ultimate Recovery**, `EUR`
(thousand barrels), predicted from initial rate `qi` (bbl/d), initial Arps
decline rate `Di` (1/yr) and reservoir permeability `K` (mD).

* **Phase A ("core area")** — the first 40 wells drilled, used to build the
  initial model.
* **Phase B ("step-out area")** — 25 wells drilled later in an adjacent
  fault block with measurably different reservoir quality (a deliberate
  distribution shift).
"""

# ╔═╡ 494d7c90-608c-42f0-8af6-c978d7ae92c2
function generate_phase(n::Int; seed::Int, quality_shift::Float64)
	rng = MersenneTwister(seed)
	qi = rand(rng, LogNormal(log(600.0), 0.35), n)          # initial rate, bbl/d
	Di = rand(rng, Uniform(0.35, 0.85), n)                   # initial decline, 1/yr
	K  = rand(rng, LogNormal(log(40.0) + quality_shift, 0.5), n)  # permeability, mD

	log_eur = @. (3.55 + 0.55 * log(qi / 600.0) - 1.15 * Di +
	              0.30 * log(K / 40.0) + 0.12 * log(K / 40.0) * log(qi / 600.0))
	noise = rand(rng, Normal(0.0, 0.20), n)
	eur = exp.(log_eur .+ noise)                             # Mbbl

	DataFrame(QI=qi, DI=Di, K=K, EUR=eur, LOGEUR=log.(eur))
end

# ╔═╡ b199cd8f-9b3d-446b-812a-eb4dac0a7bbf
begin
	Random.seed!(2024)
	phaseA = generate_phase(40; seed=101, quality_shift=0.0)    # core area
	phaseB = generate_phase(25; seed=202, quality_shift=-0.55)  # step-out area: lower-K rock
	(nA=nrow(phaseA), nB=nrow(phaseB))
end

# ╔═╡ 017b9996-3e43-4470-b152-9f0fada3ea08
begin
	@df phaseA scatter(:K, :EUR, label="Phase A (core)", xlabel="permeability K (mD)",
		ylabel="EUR (Mbbl)", yscale=:log10, xscale=:log10, markersize=4, alpha=0.7, size=(650,450))
	@df phaseB scatter!(:K, :EUR, label="Phase B (step-out)", markersize=4, alpha=0.7, markershape=:diamond)
end

# ╔═╡ 2249e8f0-c5a8-42ae-9b15-ef8e67408767
md"""
## 2. Shared MLP architecture (Flux.jl) & feature standardization

Same architecture as notebook 2's Bayesian network — 3 inputs → 5 tanh → 5
tanh → 1 linear output — but here an ordinary Flux network trained by
backprop, and standardized using **Phase A statistics only** (Phase B
arrives later, "in production", so nothing about it may leak into
preprocessing decided when the original model was built).
"""

# ╔═╡ 315151cf-0b68-4491-b75f-1a069de93e7a
begin
	FEATURES2 = [:QI, :DI, :K]
	D2 = length(FEATURES2)
	n_hidden2 = 5
	make_nn() = Chain(Dense(D2, n_hidden2, tanh), Dense(n_hidden2, n_hidden2, tanh), Dense(n_hidden2, 1))

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

	Xall_test, yall_test = vcat(XA_test, XB_test), vcat(yA_test, yB_test)

	# Flux's column-major (n_features, n_obs) layout, as Float32
	XA_train_mat = Float32.(Matrix(XA_train')); yA_train32 = Float32.(yA_train)
	XB_train_mat = Float32.(Matrix(XB_train')); yB_train32 = Float32.(yB_train)
	XA_test_mat  = Float32.(Matrix(XA_test'));  XB_test_mat  = Float32.(Matrix(XB_test'))
	Xall_test_mat = Float32.(Matrix(Xall_test'))

	(nA_train=length(yA_train), nA_test=length(yA_test), nB_train=length(yB_train), nB_test=length(yB_test))
end

# ╔═╡ e2af23f6-e7a7-4312-b2bf-3280ea8b71db
function rmse_r2_pred(fmu, y)
	pmean = vec(fmu)
	rmse = sqrt(mean((pmean .- y) .^ 2))
	ss_res, ss_tot = sum((y .- pmean) .^ 2), sum((y .- mean(y)) .^ 2)
	(rmse=rmse, r2=1 - ss_res / ss_tot)
end

# ╔═╡ 54448d3b-322c-4298-8ffc-d213b7fd04fd
function train_bp!(nn, Xmat, y32; epochs=400, lr=1f-2, seed=1)
	Random.seed!(seed)
	opt_state = Flux.setup(Adam(lr), nn)
	# see notebook 3 §3 — silence Flux.train!'s per-call progress logging
	Logging.with_logger(Logging.NullLogger()) do
		for _ in 1:epochs
			Flux.train!(nn, [(Xmat, reshape(y32, 1, :))], opt_state) do m, x, y
				Flux.Losses.mse(m(x), y)
			end
		end
	end
end

# ╔═╡ ae40b735-d57d-4afd-967f-d606f1cd6bee
md"""
## 3. Step 1 — fit the initial ("stale") model on Phase A only

Train a Flux MLP on Phase A by backprop, then fit a Laplace approximation
around it: `la_A`'s posterior mean is the trained weights, and its
posterior *precision* — curvature of the loss around those weights — is
exactly what we carry forward as the "memory" of Phase A in the next
section.

(The cell below just runs Flux/Zygote/LaplaceRedux once on trivial data
first, so the `@elapsed` timings reported for stale/additive/retrain later
in this notebook are comparable to each other — otherwise whichever step
happens to run first would eat a one-off JIT-compilation cost that has
nothing to do with the actual comparison.)
"""

# ╔═╡ 54091370-7960-4f75-8753-0c4cd8eebdc2
let
	nn_warm = make_nn()
	train_bp!(nn_warm, XA_train_mat, yA_train32; epochs=2, seed=0)
	la_warm = LaplaceRedux.Laplace(nn_warm; likelihood=:regression, subset_of_weights=:all)
	fit!(la_warm, [(XA_train_mat[:, 1], [yA_train32[1]])])
	predict(la_warm, XA_test_mat; ret_distr=false)
	nothing
end

# ╔═╡ 0f7f38a9-6592-4530-90d0-c27924783163
begin
	nn_stale = make_nn()
	t_stale = @elapsed train_bp!(nn_stale, XA_train_mat, yA_train32; seed=1)

	la_A = LaplaceRedux.Laplace(nn_stale; likelihood=:regression, subset_of_weights=:all)
	fit!(la_A, [(XA_train_mat[:, i], [yA_train32[i]]) for i in 1:size(XA_train_mat, 2)])
	optimize_prior!(la_A; verbosity=0)

	mu_A = la_A.posterior.posterior_mean          # MAP weights, flattened
	Lambda_A = posterior_precision(la_A)           # curvature + prior precision, n_params x n_params
	sigma_A = la_A.prior.observational_noise
	theta_A, re_A = Flux.destructure(nn_stale)      # re_A: flat vector -> Chain, same order as mu_A

	t_stale
end

# ╔═╡ efba196b-836a-4b1f-9255-f903aba25e4e
begin
	fmu_stale_A, _ = predict(la_A, XA_test_mat; ret_distr=false)
	fmu_stale_B, _ = predict(la_A, XB_test_mat; ret_distr=false)
	(rmse_on_A_test = round(rmse_r2_pred(fmu_stale_A, yA_test).rmse, digits=3),
	 rmse_on_B_test_STALE = round(rmse_r2_pred(fmu_stale_B, yB_test).rmse, digits=3))
end

# ╔═╡ 5acaeacc-6cfb-4577-8dd0-34622f22a89a
md"""
The RMSE jump from Phase A test wells to Phase B test wells (evaluated with
the *same, unmodified* Phase-A-only model) is the distribution-shift
penalty of doing nothing — same diagnostic as notebook 2, same purpose:
this is the gap an update should close.
"""

# ╔═╡ c58eb869-2190-4d50-9370-2dbd4ae7e6cd
md"""
## 4. Step 2a — additive update: online Laplace approximation

This is the Laplace-approximation analogue of notebook 2's "moment-match
the posterior, use it as a prior for NUTS on the new batch": here, Phase
A's Laplace posterior `N(μ_A, Λ_A⁻¹)` becomes a **quadratic penalty**
added to Phase B's training loss —

```
loss(θ) = mse(nn_θ(X_B), y_B) + (θ - μ_A)ᵀ Λ_A (θ - μ_A) / n_B
```

— pulling the continued-training fit back toward Phase A's weights, with
exactly the strength justified by how well-identified each direction in
weight space was by Phase A's data (curvature). This *is* [Elastic Weight
Consolidation](https://en.wikipedia.org/wiki/Overcoming_catastrophic_forgetting):
EWC's penalty is precisely a Laplace approximation to the old task's
posterior used as a prior for the new one. Phase A's raw wells are never
touched again — only the compact summary `(μ_A, Λ_A, σ_A)`.

After continued training, we fit a **fresh** Laplace approximation around
the *updated* weights, but seed its prior precision with `Λ_A` (instead of
LaplaceRedux's default isotropic prior) so that the reported posterior
precision is the textbook *online Laplace* combination
`Λ_A + H_B` (Phase A's carried-forward curvature plus Phase B's own) — the
standard sequential-Gaussian-Bayes combination rule, applied here to
curvature instead of raw MCMC samples. We deliberately skip
`optimize_prior!` on this object: it assumes an isotropic prior precision,
which `Λ_A` no longer is.
"""

# ╔═╡ 413f95b4-163a-42ad-b7f6-b5ca259ff0c5
function additive_update(mu_A, Lambda_A, sigma_A, theta_init, re; tau=1.0, epochs=400, lr=1f-2, seed=2)
	Lambda_eff = Lambda_A ./ Float32(tau)
	nB = size(XB_train_mat, 2)

	function ewc_loss(θ)
		m = re(θ)
		preds = vec(m(XB_train_mat))
		mse_term = mean((preds .- yB_train32) .^ 2)
		dθ = θ .- mu_A
		ewc_term = dot(dθ, Lambda_eff * dθ) / nB
		mse_term + ewc_term
	end

	Random.seed!(seed)
	θ = copy(theta_init)
	opt_state = Flux.setup(Adam(lr), θ)
	Logging.with_logger(Logging.NullLogger()) do
		for _ in 1:epochs
			gs = Flux.gradient(ewc_loss, θ)[1]
			opt_state, θ = Flux.Optimisers.update!(opt_state, θ, gs)
		end
	end
	nn_updated = re(θ)

	la_updated = LaplaceRedux.Laplace(nn_updated; likelihood=:regression, subset_of_weights=:all,
		prior_precision_matrix=Lambda_eff, observational_noise=sigma_A)
	fit!(la_updated, [(XB_train_mat[:, i], [yB_train32[i]]) for i in 1:size(XB_train_mat, 2)])
	la_updated
end

# ╔═╡ f4a2b164-90b3-4bd0-a217-4eb9e8bc1087
begin
	t_additive = @elapsed la_additive = additive_update(mu_A, Lambda_A, sigma_A, theta_A, re_A; tau=1.0)
	t_additive
end

# ╔═╡ d9eb09ad-2a31-4551-be6d-9463cb218213
md"""
## 5. Step 2b — baseline: full retrain on Phase A ∪ Phase B pooled
"""

# ╔═╡ 9052ea75-1b37-42c7-b9bc-ffeec7862248
begin
	Xall_train = vcat(XA_train, XB_train)
	yall_train = vcat(yA_train, yB_train)
	Xall_train_mat = Float32.(Matrix(Xall_train'))
	yall_train32 = Float32.(yall_train)

	nn_full = make_nn()
	t_retrain = @elapsed train_bp!(nn_full, Xall_train_mat, yall_train32; seed=3)

	la_full = LaplaceRedux.Laplace(nn_full; likelihood=:regression, subset_of_weights=:all)
	fit!(la_full, [(Xall_train_mat[:, i], [yall_train32[i]]) for i in 1:size(Xall_train_mat, 2)])
	optimize_prior!(la_full; verbosity=0)

	t_retrain
end

# ╔═╡ 3e984ce2-e64b-4eba-a811-dd355f1e4d5f
md"""
## 6. Compare: stale vs. additive update vs. full retrain

All three are evaluated on the *same* held-out mix of Phase A and Phase B
test wells.
"""

# ╔═╡ cfb2fe8a-2ca3-4b87-a925-d0ad91c68098
begin
	fmu_stale_all, _ = predict(la_A, Xall_test_mat; ret_distr=false)
	fmu_additive_all, _ = predict(la_additive, Xall_test_mat; ret_distr=false)
	fmu_full_all, _ = predict(la_full, Xall_test_mat; ret_distr=false)

	res_stale    = rmse_r2_pred(fmu_stale_all, yall_test)
	res_additive = rmse_r2_pred(fmu_additive_all, yall_test)
	res_full     = rmse_r2_pred(fmu_full_all, yall_test)

	comparison = DataFrame(
		model = ["stale (Phase A only)", "additive update (online Laplace, fit on B)", "full retrain (A+B pooled)"],
		rmse  = round.([res_stale.rmse, res_additive.rmse, res_full.rmse], digits=3),
		r2    = round.([res_stale.r2, res_additive.r2, res_full.r2], digits=3),
		wall_seconds = round.([t_stale, t_additive, t_retrain], digits=2),
		wells_processed_this_step = [nrow(A_train), nrow(B_train), nrow(A_train) + nrow(B_train)],
	)
	comparison
end

# ╔═╡ 3b25c574-058e-4344-a351-0e4ad5e936a1
@df comparison bar(:model, :rmse, xrotation=15, legend=false, size=(700,420),
	ylabel="RMSE on pooled held-out wells (log EUR)", title="Additive update recovers most of the full-retrain gain")

# ╔═╡ fda4ffa3-18fa-43b3-a0d6-db0e2f0a7252
md"""
The key comparison is again **rows 2 vs. 3**: the additive update's
training step only ever touches the `nrow(B_train)` new wells (plus the
fixed-size `(μ_A, Λ_A)` summary — never Phase A's raw wells), yet its
held-out RMSE lands close to the full retrain, and clearly better than
leaving the stale model untouched. Unlike notebook 2, though, check
`wall_seconds` before assuming this also *saves time* here — see §9.
"""

# ╔═╡ bc934d8f-2303-4382-8a1f-781624da7277
md"""
## 7. Tuning how much the new data counts: prior inflation τ

Same knob as notebook 2 §7: `Lambda_A ./ τ` inflates Phase A's
carried-forward precision (a power prior, ``P_0 \propto \Lambda_A^{1/\tau}``)
before folding in Phase B. `τ = 1` reproduces the additive update above
exactly.
"""

# ╔═╡ 71f350a9-ac65-4c9c-8e3b-febc1dd87f46
begin
	τ_grid = [1.0, 2.0, 4.0, 8.0]

	τ_rows = map(τ_grid) do τ
		t_τ = @elapsed la_τ = additive_update(mu_A, Lambda_A, sigma_A, theta_A, re_A; tau=τ)
		fmu_A, _ = predict(la_τ, XA_test_mat; ret_distr=false)
		fmu_B, _ = predict(la_τ, XB_test_mat; ret_distr=false)
		fmu_all, _ = predict(la_τ, Xall_test_mat; ret_distr=false)
		res_A, res_B, res_all = rmse_r2_pred(fmu_A, yA_test), rmse_r2_pred(fmu_B, yB_test), rmse_r2_pred(fmu_all, yall_test)
		(τ=τ, rmse_A_test=round(res_A.rmse, digits=3), rmse_B_test=round(res_B.rmse, digits=3),
		 rmse_pooled=round(res_all.rmse, digits=3), r2_pooled=round(res_all.r2, digits=3),
		 wall_seconds=round(t_τ, digits=2))
	end

	tau_comparison = DataFrame(τ_rows)
end

# ╔═╡ d8abdd8d-3018-42bd-b025-d556263beb94
begin
	plot(tau_comparison.τ, tau_comparison.rmse_A_test, marker=:circle, linewidth=2,
		label="RMSE on Phase A test", xlabel="prior inflation τ", ylabel="RMSE (log EUR)",
		xscale=:log2, title="Prior inflation trades old-data fit for new-data fit", size=(650,450))
	plot!(tau_comparison.τ, tau_comparison.rmse_B_test, marker=:diamond, linewidth=2,
		label="RMSE on Phase B test")
end

# ╔═╡ 3b0ba3af-a9e0-47b5-800c-fa7392c2e097
md"""
**Reading the sweep.** Unlike notebook 2's τ sweep (where the trade-off did
*not* show up cleanly — see its §7 discussion), here it typically does:
as `τ` grows, `rmse_B_test` falls (the fit leans harder into the new,
shifted data) while `rmse_A_test` rises (less of Phase A survives) — the
textbook trade-off. Whether `r2_pooled` keeps improving all the way to
`τ=8` or peaks somewhere in between and turns back down (as Phase B's
small sample starts getting overfit) varies run to run; either way, `τ`
much larger than needed eventually costs Phase A accuracy for a Phase B
gain that stops paying for itself. A plausible reason the trade-off itself
is cleaner here than in notebook 2: `Λ_A` here comes from the Laplace
curvature of the actual loss
surface, a well-conditioned, informative precision matrix — where notebook
2's carried-forward prior came from the *empirical sample covariance* of
only 250 MCMC draws in a 56-dimensional space, a much noisier estimate of
"how identified" each weight direction really was. Practically: `τ`
controls how much the new batch counts, exactly as in notebook 2, and it
is worth checking `wall_seconds` here too — see §9.
"""

# ╔═╡ 0cab45d2-2b58-4b7e-85a0-ef7eed189cf0
md"""
## 8. Posterior predictive on a Phase B well: stale vs. additive vs. retrain
"""

# ╔═╡ d67b885a-018b-494a-9f84-7cf6d0a97a43
begin
	k_grid = range(minimum(phaseB.K), maximum(phaseB.K); length=60)
	qi_med, di_med = median(B_train.QI), median(B_train.DI)

	Xg = zeros(Float32, D2, length(k_grid))
	for j in 1:length(k_grid)
		Xg[1, j] = (qi_med     - fmean2[1]) / fstd2[1]
		Xg[2, j] = (di_med     - fmean2[2]) / fstd2[2]
		Xg[3, j] = (k_grid[j]  - fmean2[3]) / fstd2[3]
	end

	function band(la, X)
		fmu, fvar = predict(la, X; ret_distr=false)
		m = vec(fmu)
		sd = sqrt.(vec(fvar))
		dists = Normal.(m, sd)
		m, quantile.(dists, 0.05), quantile.(dists, 0.95)
	end

	m_s, lo_s, hi_s = band(la_A, Xg)
	m_u, lo_u, hi_u = band(la_additive, Xg)
	m_f, lo_f, hi_f = band(la_full, Xg)

	plot(k_grid, m_s, ribbon=(m_s .- lo_s, hi_s .- m_s), label="stale (A only)",
		xscale=:log10, xlabel="permeability K (mD)", ylabel="predicted log(EUR)",
		title="Step-out area: does the model know what it doesn't know?", size=(700,460), fillalpha=0.18)
	plot!(k_grid, m_u, ribbon=(m_u .- lo_u, hi_u .- m_u), label="additive update", fillalpha=0.18)
	plot!(k_grid, m_f, ribbon=(m_f .- lo_f, hi_f .- m_f), label="full retrain", fillalpha=0.18, linestyle=:dash)
end

# ╔═╡ 550f769a-ffbd-4ec7-86e0-c13aeb9ca84d
md"""
## 9. Repeated updates and the forgetting factor

Sections 4–7 ran the online-Laplace update *once*. In production, updates
happen every time a new batch of wells comes in — round after round, for
years. Naively reusing `τ=1` every round (no discount at all) has a
failure mode: the posterior precision after `n` rounds is

```
Λ_n = Λ_(n-1) + H_n = Λ_0 + H_1 + H_2 + ... + H_n
```

and since each `H_k` is positive semi-definite, `Λ_n` only ever grows —
the model gets more and more certain, forever, regardless of whether
reality is still cooperating with what it learned early on. Eventually the
quadratic penalty anchoring it to old weights dwarfs the gradient pull
from any new batch, and the model stops being able to move at all —
"locked in" to a belief it can no longer revise, exactly when a drifting
field most needs it to.

**The fix (a forgetting factor)** is to apply the same `τ` inflation at
*every* round: `Λ_k = Λ_(k-1)/τ + H_k`. Applied repeatedly, this no longer
lets precision grow without bound — it converges to a steady state. Setting
`Λ_(k-1)/τ = Λ_(k-1)` at that steady state and solving gives the
**effective memory window**:

```
N_eff = τ / (τ - 1)
```

`N_eff` is the number of wells' worth of information the model behaves as
if it remembers, no matter how many thousand it has actually seen —
exactly analogous to a recursive-least-squares "forgetting factor" or a
Kalman filter's process-noise injection, both of which solve the same
runaway-certainty problem. This inverts into a practical recipe: **decide
how many wells of memory you want** (how much drift you expect the field
to show), then set `τ = N_eff / (N_eff - 1)`.

We simulate this by drilling `K` further phases beyond Phase A, each a
little further from Phase A's rock quality than the last (continuous
drift, rather than the one-time step to Phase B above), and run the
online-Laplace update every round two ways: `τ=1` (no forgetting) against
`τ` set from a target `N_eff`.
"""

# ╔═╡ 5e4f3203-2a5f-47f2-a025-ff1d03d8d829
begin
	K_rounds = 15
	n_per_round = 15
	shift_per_round = -0.12   # cumulative drift: -0.12 * K_rounds by the last round

	Random.seed!(999)
	drift_phases = [generate_phase(n_per_round; seed=1000 + k, quality_shift=shift_per_round * k) for k in 0:K_rounds]
	drift_splits = [tt_split(drift_phases[k + 1]; seed=2000 + k) for k in 0:K_rounds]  # (train, test), k=0..K_rounds

	# standardize once, on round 0's stats only — mirrors "decided when the model first went into production"
	fmean_seq = [mean(drift_splits[1][1][!, f]) for f in FEATURES2]
	fstd_seq  = [std(drift_splits[1][1][!, f])  for f in FEATURES2]
	standardize_seq(df) = hcat([ (df[!, f] .- m) ./ s for (f, m, s) in zip(FEATURES2, fmean_seq, fstd_seq) ]...)

	seq_Xtrain = [Float32.(Matrix(standardize_seq(tr)')) for (tr, te) in drift_splits]
	seq_ytrain = [Float32.(tr.LOGEUR) for (tr, te) in drift_splits]
	seq_Xtest  = [Float32.(Matrix(standardize_seq(te)')) for (tr, te) in drift_splits]
	seq_ytest  = [Float64.(te.LOGEUR) for (tr, te) in drift_splits]

	(K_rounds=K_rounds, n_per_round=n_per_round, total_drift=shift_per_round * K_rounds)
end

# ╔═╡ 3acc1e06-e9e7-4d4e-900a-fb98788adf1e
function online_round(mu_prev, Lambda_prev, sigma_prev, theta_prev, re_prev, Xk_mat, yk32; tau=1.0, epochs=400, lr=1f-2, seed=1)
	Lambda_eff = Lambda_prev ./ Float32(tau)
	nk = size(Xk_mat, 2)

	function loss(θ)
		m = re_prev(θ)
		preds = vec(m(Xk_mat))
		mse_term = mean((preds .- yk32) .^ 2)
		dθ = θ .- mu_prev
		ewc_term = dot(dθ, Lambda_eff * dθ) / nk
		mse_term + ewc_term
	end

	Random.seed!(seed)
	θ = copy(theta_prev)
	opt_state = Flux.setup(Adam(lr), θ)
	Logging.with_logger(Logging.NullLogger()) do
		for _ in 1:epochs
			gs = Flux.gradient(loss, θ)[1]
			opt_state, θ = Flux.Optimisers.update!(opt_state, θ, gs)
		end
	end
	nn_k = re_prev(θ)

	la_k = LaplaceRedux.Laplace(nn_k; likelihood=:regression, subset_of_weights=:all,
		prior_precision_matrix=Lambda_eff, observational_noise=sigma_prev)
	fit!(la_k, [(Xk_mat[:, i], [yk32[i]]) for i in 1:nk])
	theta_k, re_k = Flux.destructure(nn_k)
	(la=la_k, mu=la_k.posterior.posterior_mean, Lambda=posterior_precision(la_k), theta=theta_k, re=re_k)
end

# ╔═╡ bc67e51d-2aa6-4455-b692-3d589340fa4b
function run_sequence(; tau=1.0, seed_offset=10)
	nn0 = make_nn()
	train_bp!(nn0, seq_Xtrain[1], seq_ytrain[1]; seed=1)
	la0 = LaplaceRedux.Laplace(nn0; likelihood=:regression, subset_of_weights=:all)
	fit!(la0, [(seq_Xtrain[1][:, i], [seq_ytrain[1][i]]) for i in 1:size(seq_Xtrain[1], 2)])
	optimize_prior!(la0; verbosity=0)
	sigma0 = la0.prior.observational_noise
	mu, Lambda = la0.posterior.posterior_mean, posterior_precision(la0)
	theta, re = Flux.destructure(nn0)

	avg_sd = Float64[mean(sqrt.(diag(inv(Lambda))))]
	step_norms = Float64[]

	for k in 1:K_rounds
		res = online_round(mu, Lambda, sigma0, theta, re, seq_Xtrain[k + 1], seq_ytrain[k + 1]; tau=tau, seed=seed_offset + k)
		push!(step_norms, norm(res.theta .- theta))
		mu, Lambda, theta, re = res.mu, res.Lambda, res.theta, res.re
		push!(avg_sd, mean(sqrt.(diag(inv(Lambda)))))
	end
	(avg_sd=avg_sd, step_norms=step_norms)
end

# ╔═╡ 8a46fcea-af5e-4ab3-a412-94e39144c18b
begin
	N_eff = 15.0
	tau_forget = N_eff / (N_eff - 1)

	run_sequence(tau=1.0)  # warm up JIT before timing/using the real runs below

	seq_no_forget = run_sequence(tau=1.0)
	seq_forget    = run_sequence(tau=tau_forget)

	(N_eff=N_eff, tau=round(tau_forget, digits=4))
end

# ╔═╡ 1d323fb3-217b-4038-b9b6-fd8f17a764a7
plot(0:K_rounds, seq_no_forget.avg_sd, marker=:circle, linewidth=2, label="τ=1 (no forgetting)",
	xlabel="update round", ylabel="avg. posterior std. dev. across weights",
	title="Without forgetting, precision only ever grows", size=(650,450))
plot!(0:K_rounds, seq_forget.avg_sd, marker=:diamond, linewidth=2, label="τ=$(round(tau_forget,digits=3)) (N_eff=$(Int(N_eff)))")

# ╔═╡ c121e6c8-64e1-4136-8d5c-1352035322e5
begin
	bar(["round $i" for i in 1:K_rounds], seq_no_forget.step_norms; label="τ=1 (no forgetting)",
		alpha=0.6, xrotation=90, size=(750,420), ylabel="‖θ_k - θ_(k-1)‖  (how far weights moved this round)",
		title="Forgetting keeps the model able to move")
	bar!(["round $i" for i in 1:K_rounds], seq_forget.step_norms; label="τ=$(round(tau_forget,digits=3))", alpha=0.6)
end

# ╔═╡ 7c7514ff-2a00-423b-926e-934b14af5617
md"""
**Reading the plots.** Without forgetting, the average posterior standard
deviation shrinks round after round with no floor — the model becomes
steadily, unboundedly more confident regardless of whether the incoming
wells still agree with it. With `τ` set from `N_eff=$(Int(N_eff))`, it
instead stabilizes (here it even settles slightly *higher*, since each new
round's own curvature is milder than the discounted history it replaces)
— precision stops accumulating without bound, exactly as the `N_eff`
derivation predicts.

The second plot makes the practical consequence direct: `‖θ_k - θ_(k-1)‖`
— how far the weights actually moved during each round's update — is
larger with forgetting than without at every single round in this run.
That is the forgetting factor doing its job: keeping the model *able* to
move toward new evidence, round after round, instead of gradually
freezing in place.

One honest limitation: we did **not** see a correspondingly clean
improvement in held-out RMSE on each round's own test wells between the
two — with only a handful of test wells per round, that signal is too
noisy at this scale to separate from run-to-run variance. The precision
and weight-movement effects are the direct, mechanistically guaranteed
consequences of the forgetting factor; a visible predictive-accuracy
payoff would likely need either many more rounds, a longer sustained
drift, or larger batches per round before it clears the noise floor.
"""

# ╔═╡ f5a43e06-4aa8-4e5b-bc48-de0dd13b7102
md"""
## 10. Discussion & caveats

* **What worked.** Online Laplace updating — carrying Phase A's posterior
  mean and curvature forward as a quadratic penalty for continued training
  on Phase B, then re-fitting a Laplace approximation around the result —
  recovered most of the full retrain's accuracy gain over the stale model,
  without ever touching Phase A's raw wells again during the update step.
  Same qualitative story as notebook 2, different mechanism.
* **The wall-clock story is genuinely different from notebook 2.** Check
  `wall_seconds` in §6: unlike NUTS (whose cost scales with the size of
  the *entire* dataset it samples over, which is why notebook 2's additive
  update was clearly cheaper than its full retrain), plain backprop on a
  few dozen wells is already so cheap that a full retrain here is not
  obviously slower than the additive update — sometimes it measures
  faster. **The real benefit of the additive route at this scale is not
  wall-clock time; it's that the update step never needs Phase A's raw
  wells again — only the fixed-size `(μ_A, Λ_A, σ_A)` summary.** That
  matters for data retention/locality (e.g. Phase A wells living on
  infrastructure you no longer have access to) even when it doesn't yet
  matter for speed. The wall-clock advantage would start to show up at
  much larger historical datasets or networks, where backprop cost per
  epoch stops being negligible.
* **Caveat — this is still an approximation.** The online-Laplace
  combination `Λ_A + H_B` assumes Phase A's posterior was well-summarized
  by a single Gaussian around its MAP (exactly notebook 2's "Gaussian
  moment-matching" caveat, transplanted from MCMC samples to curvature) —
  see [Ritter et al., *Online Structured Laplace Approximations for
  Overcoming Catastrophic
  Forgetting*](https://arxiv.org/abs/1805.07810) for the fuller
  treatment, including structured (Kronecker-factored) approximations
  that scale to much larger networks than the full Hessian used here.
* **Caveat — this is not online/streaming SGD either.** Each "update" is
  still a full continued-training run over Phase B (just regularized by a
  penalty derived from Phase A) plus one Hessian fit — not a single-sample
  online update.
* **Tuning old-vs-new trust, once.** §7's `τ` sweep is the same explicit
  knob as notebook 2's, for the same reason: pick `τ ≈ 1` when the new
  batch is small/noisy and shouldn't move the model much, larger when you
  trust it reflects a real, lasting shift.
* **Tuning old-vs-new trust, repeatedly.** If you run *many* rounds of
  updates over time — not just the one Phase A→B step above — §9 shows why
  `τ=1` every round eventually locks the model out of learning anything
  new, and how to set a *sustained* `τ` from a target effective memory
  window `N_eff = τ/(τ-1)` instead.
"""

# ╔═╡ Cell order:
# ╠═719213de-a9ad-4606-82ca-b724351daec1
# ╠═a7c5079a-9517-497f-86aa-0b2044f55131
# ╠═e5606e00-522e-4efb-8194-c58dbffc965e
# ╠═a773c149-9974-4f45-ac90-0e3dd1af9a73
# ╠═494d7c90-608c-42f0-8af6-c978d7ae92c2
# ╠═b199cd8f-9b3d-446b-812a-eb4dac0a7bbf
# ╠═017b9996-3e43-4470-b152-9f0fada3ea08
# ╠═2249e8f0-c5a8-42ae-9b15-ef8e67408767
# ╠═315151cf-0b68-4491-b75f-1a069de93e7a
# ╠═e2af23f6-e7a7-4312-b2bf-3280ea8b71db
# ╠═54448d3b-322c-4298-8ffc-d213b7fd04fd
# ╠═ae40b735-d57d-4afd-967f-d606f1cd6bee
# ╠═54091370-7960-4f75-8753-0c4cd8eebdc2
# ╠═0f7f38a9-6592-4530-90d0-c27924783163
# ╠═efba196b-836a-4b1f-9255-f903aba25e4e
# ╠═5acaeacc-6cfb-4577-8dd0-34622f22a89a
# ╠═c58eb869-2190-4d50-9370-2dbd4ae7e6cd
# ╠═413f95b4-163a-42ad-b7f6-b5ca259ff0c5
# ╠═f4a2b164-90b3-4bd0-a217-4eb9e8bc1087
# ╠═d9eb09ad-2a31-4551-be6d-9463cb218213
# ╠═9052ea75-1b37-42c7-b9bc-ffeec7862248
# ╠═3e984ce2-e64b-4eba-a811-dd355f1e4d5f
# ╠═cfb2fe8a-2ca3-4b87-a925-d0ad91c68098
# ╠═3b25c574-058e-4344-a351-0e4ad5e936a1
# ╠═fda4ffa3-18fa-43b3-a0d6-db0e2f0a7252
# ╠═bc934d8f-2303-4382-8a1f-781624da7277
# ╠═71f350a9-ac65-4c9c-8e3b-febc1dd87f46
# ╠═d8abdd8d-3018-42bd-b025-d556263beb94
# ╠═3b0ba3af-a9e0-47b5-800c-fa7392c2e097
# ╠═0cab45d2-2b58-4b7e-85a0-ef7eed189cf0
# ╠═d67b885a-018b-494a-9f84-7cf6d0a97a43
# ╠═550f769a-ffbd-4ec7-86e0-c13aeb9ca84d
# ╠═5e4f3203-2a5f-47f2-a025-ff1d03d8d829
# ╠═3acc1e06-e9e7-4d4e-900a-fb98788adf1e
# ╠═bc67e51d-2aa6-4455-b692-3d589340fa4b
# ╠═8a46fcea-af5e-4ab3-a412-94e39144c18b
# ╠═1d323fb3-217b-4038-b9b6-fd8f17a764a7
# ╠═c121e6c8-64e1-4136-8d5c-1352035322e5
# ╠═7c7514ff-2a00-423b-926e-934b14af5617
# ╠═f5a43e06-4aa8-4e5b-bc48-de0dd13b7102
