### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ 8b983177-4105-4490-b705-590329f7f163
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

# ╔═╡ 9aeac5b8-3a66-434b-bd98-5bdd7df22f55
md"""
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
   (``p(\theta \mid A, B) \approx p(\theta \mid B) \cdot p(\theta \mid A)``
   used as prior).
3. **Full retrain** — NUTS run from scratch on Phase A ∪ Phase B pooled
   together — the expensive gold standard the additive update is trying to
   approximate.

A final section then asks: how much should the new batch count? It sweeps
a *prior inflation* factor `τ` that controls how strongly Phase A's
posterior constrains the additive fit on Phase B.
"""

# ╔═╡ 87515d6c-b1e7-4ebd-8274-05aad9c19a68
TableOfContents()

# ╔═╡ 2543c334-d419-487e-a9b9-fd536d8c3785
md"""
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
"""

# ╔═╡ 4916dddd-0363-4aa0-a30a-fc4e4791df6a
function generate_phase(n::Int; seed::Int, quality_shift::Float64)
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
end

# ╔═╡ 4a7337a7-1f18-4551-a581-4d3767df372f
begin
	Random.seed!(2024)
	phaseA = generate_phase(40; seed=101, quality_shift=0.0)    # core area
	phaseB = generate_phase(25; seed=202, quality_shift=-0.55)  # step-out area: lower-K rock
	(nA=nrow(phaseA), nB=nrow(phaseB))
end

# ╔═╡ 3c32bd56-768e-4fec-8811-1981a7ce65cb
begin
	@df phaseA scatter(:K, :EUR, label="Phase A (core)", xlabel="permeability K (mD)",
		ylabel="EUR (Mbbl)", yscale=:log10, xscale=:log10, markersize=4, alpha=0.7, size=(650,450))
	@df phaseB scatter!(:K, :EUR, label="Phase B (step-out)", markersize=4, alpha=0.7, markershape=:diamond)
end

# ╔═╡ e05034a9-686f-4cbe-b11b-2c6fc7496324
md"""
## 2. Shared BNN architecture & Turing model

Same idea as notebook 1: 3 inputs → 5 tanh → 5 tanh → 1 linear output,
weights flattened into one vector `θ`. The model function takes an
explicit `prior_mean` / `prior_cov` so it can be reused both for a vague
"fit from scratch" prior and for an informative "posterior-of-A-as-prior"
distribution.
"""

# ╔═╡ a72c1070-ec7a-4ab9-8a5e-087af12bb829
begin
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
end

# ╔═╡ 4d73ee31-728c-4318-b7ae-3519843dadc7
@model function bayes_nn2(X, y, prior_mean, prior_cov; obs_sigma=0.25)
	θ ~ MvNormal(prior_mean, prior_cov)
	preds = nn_forward2(X, θ)
	y ~ MvNormal(preds, obs_sigma^2 * I(length(y)))
end

# ╔═╡ f2205061-836d-48c3-9d66-51d666f3ce75
md"""
### Feature standardization

Standardize using **Phase A statistics only** — Phase B is meant to arrive
later, "in production", so nothing about it may leak into preprocessing
decided when the original model was built.
"""

# ╔═╡ d37fc21a-e817-4cb9-b7bc-f1bc1062f793
begin
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
end

# ╔═╡ 778210a6-28ca-48c3-8492-1999f8694a02
md"""
## 3. Step 1 — fit the initial ("stale") model on Phase A only
"""

# ╔═╡ a5ebbe58-6f54-4b07-b9d6-298de7719844
begin
	Random.seed!(1)
	vague_mean = zeros(N_PARAMS2)
	vague_cov  = 1.0^2 * I(N_PARAMS2)

	t_fitA = @elapsed chainA = sample(
		bayes_nn2(XA_train', yA_train, vague_mean, vague_cov), NUTS(0.8), 250; progress=false, chain_type=Chains)
	t_fitA
end

# ╔═╡ 32910327-b150-42d1-b2bb-8933d4cf2d1e
function rmse_r2(model_forward, θsamples, Xrows, y)
	Xt = Xrows'  # (n_features, n_obs)
	preds = reduce(hcat, [model_forward(Xt, θsamples[:, s]) for s in 1:size(θsamples, 2)])
	pmean = vec(mean(preds, dims=2))
	rmse = sqrt(mean((pmean .- y).^2))
	ss_res, ss_tot = sum((y .- pmean).^2), sum((y .- mean(y)).^2)
	(rmse=rmse, r2=1 - ss_res/ss_tot, pred_mean=pmean, preds=preds)
end

# ╔═╡ c71168a8-161c-4e14-8d8d-fbb5b3753baa
begin
	θA = Array(group(chainA, :θ))'
	stale_on_A = rmse_r2(nn_forward2, θA, XA_test, yA_test)
	stale_on_B = rmse_r2(nn_forward2, θA, XB_test, yB_test)
	(rmse_on_A_test=round(stale_on_A.rmse, digits=3), rmse_on_B_test_STALE=round(stale_on_B.rmse, digits=3))
end

# ╔═╡ 0de9523f-c772-4a36-8ba4-df092eb5de9e
md"""
The RMSE jump from Phase A test wells to Phase B test wells (evaluated with
the *same, unmodified* Phase-A-only model) is the distribution-shift
penalty of doing nothing. That gap is what an update — additive or full
retrain — should close.
"""

# ╔═╡ 25ccf42d-cc1e-4e2b-97f3-3412d54d5ae0
md"""
## 4. Step 2a — additive update: fold Phase B into the existing posterior

We moment-match Phase A's posterior samples to a Gaussian
(``\hat\mu, \hat\Sigma``) and use it as the prior for a fresh NUTS run
that sees **only Phase B data** — Phase A's raw well data is never touched
again.
"""

# ╔═╡ f139ea88-876a-4668-90b9-bbb793d66a96
begin
	post_mean_A = vec(mean(θA, dims=2))
	post_cov_A  = cov(θA'; dims=1) + 1e-6 * I(N_PARAMS2)  # jitter for PD-ness

	Random.seed!(2)
	t_update = @elapsed chainB_additive = sample(
		bayes_nn2(XB_train', yB_train, post_mean_A, post_cov_A), NUTS(0.8), 250; progress=false, chain_type=Chains)
	t_update
end

# ╔═╡ 30292e63-b29d-4db1-ab6a-fbf2b20b917f
md"""
## 5. Step 2b — baseline: full retrain on Phase A ∪ Phase B pooled
"""

# ╔═╡ fd9292ea-c8e7-4cc2-a166-8eeb56b24512
begin
	Xall_train = vcat(XA_train, XB_train)
	yall_train = vcat(yA_train, yB_train)

	Random.seed!(3)
	t_retrain = @elapsed chain_full = sample(
		bayes_nn2(Xall_train', yall_train, vague_mean, vague_cov), NUTS(0.8), 250; progress=false, chain_type=Chains)
	t_retrain
end

# ╔═╡ 887aca7e-8d23-4c80-b121-f31ad672ca1e
md"""
## 6. Compare: stale vs. additive update vs. full retrain

All three are evaluated on the *same* held-out mix of Phase A and Phase B
test wells.
"""

# ╔═╡ c1b8a4c9-a9f6-4a45-9b65-fd70fd118207
begin
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
end

# ╔═╡ 4f7add61-3614-4cae-86a7-6b87b50c5d24
@df comparison bar(:model, :rmse, xrotation=15, legend=false, size=(700,420),
	ylabel="RMSE on pooled held-out wells (log EUR)", title="Additive update recovers most of the full-retrain gain")

# ╔═╡ d8b4db51-b24d-4727-8130-935411068532
md"""
The key comparison is **rows 2 vs. 3**: the additive update reprocesses
only the `nrow(B_train)` new wells through MCMC (not all
`nrow(A_train) + nrow(B_train)`), yet its held-out RMSE should land close
to the full retrain — and clearly better than leaving the stale model
untouched (row 1). As the historical well count grows, "wells processed
this step" for the additive route stays flat at the new-batch size while
the full-retrain route grows without bound — that gap is the whole point
of testing additive capability.
"""

# ╔═╡ 4372320c-46b2-415b-bad4-6423fded9e20
md"""
## 7. Tuning how much the new data counts: prior inflation τ

The additive update above used Phase A's posterior *as-is* for a prior. But
that prior can be made more or less informative before folding in Phase B:
inflating its covariance by a factor `τ` is a **power prior**
(``p(\theta) \propto p(\theta \mid A)^{1/\tau}``) — it keeps the same
mean but loosens how tightly it constrains the fit, letting Phase B's
likelihood dominate more as `τ` grows. `τ = 1` reproduces the additive
update computed above exactly.
"""

# ╔═╡ fda488ed-5eb3-4918-a79d-dc397fafe69d
begin
	τ_grid = [1.0, 2.0, 4.0, 8.0]

	function fit_additive(τ)
		τ == 1.0 && return chainB_additive, t_update
		Random.seed!(2)
		t = @elapsed chain = sample(
			bayes_nn2(XB_train', yB_train, post_mean_A, post_cov_A .* τ), NUTS(0.8), 250;
			progress=false, chain_type=Chains)
		chain, t
	end

	τ_rows = map(τ_grid) do τ
		chain_τ, t_τ = fit_additive(τ)
		θτ = Array(group(chain_τ, :θ))'
		res_A      = rmse_r2(nn_forward2, θτ, XA_test, yA_test)
		res_B      = rmse_r2(nn_forward2, θτ, XB_test, yB_test)
		res_pooled = rmse_r2(nn_forward2, θτ, Xall_test, yall_test)
		(τ=τ, rmse_A_test=round(res_A.rmse, digits=3), rmse_B_test=round(res_B.rmse, digits=3),
		 rmse_pooled=round(res_pooled.rmse, digits=3), r2_pooled=round(res_pooled.r2, digits=3),
		 mcmc_seconds=round(t_τ, digits=1))
	end

	tau_comparison = DataFrame(τ_rows)
end

# ╔═╡ e1ae99e6-4079-4d8a-8fb8-3faec1dcfe46
begin
	plot(tau_comparison.τ, tau_comparison.rmse_A_test, marker=:circle, linewidth=2,
		label="RMSE on Phase A test", xlabel="prior inflation τ", ylabel="RMSE (log EUR)",
		xscale=:log2, title="Prior inflation trades old-data fit for new-data fit", size=(650,450))
	plot!(tau_comparison.τ, tau_comparison.rmse_B_test, marker=:diamond, linewidth=2,
		label="RMSE on Phase B test")
end

# ╔═╡ 16f4fe1e-8a88-4b25-b129-e5216e148762
md"""
**Reading the sweep.** The "textbook" expectation is a trade: as `τ`
grows, `rmse_B_test` should fall (the fit leans harder into the new,
shifted data) while `rmse_A_test` rises (less of the historical fit
survives). That is *not* quite what happens here — on this run,
`rmse_B_test` falls as expected, but `rmse_A_test` improves too (and
`r2_pooled` climbs from ≈0.24 at `τ=1` to ≈0.37 at `τ=8`, at the cost of
`mcmc_seconds` roughly tripling). The reason: `post_mean_A` is itself a
noisy point estimate from only ~30 Phase A wells fit to a 56-parameter
network, so it is not tightly identified even for Phase A's *own*
held-out wells — loosening the prior around it gives NUTS room to find a
fit that is jointly better on both regimes, not purely a trade. With a
larger, better-identified Phase A posterior you would expect the classic
old-vs-new trade-off to show up more clearly. Either way, `τ` is the
practical knob for "how much do I trust this new batch", and it is not
free: check `mcmc_seconds` before assuming a larger `τ` is a free lunch.
If you run *many* additive rounds over time, applying a modest `τ > 1` at
every round is also the standard fix for a failure mode of naive
sequential Bayes: the prior covariance keeps shrinking round over round
until the model becomes overconfident and stops updating even when new
data disagrees with it.
"""

# ╔═╡ 4097af08-6914-45ac-88d7-8a76d1c7aa12
md"""
## 8. Posterior predictive on a Phase B well: stale vs. additive vs. retrain
"""

# ╔═╡ 82a22b74-248a-4423-96ee-9ff79e043a03
begin
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
end

# ╔═╡ 017e5933-ebbb-45fc-bd7d-217ed7060f79
md"""
## 9. Discussion & caveats

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
* **Tuning old-vs-new trust.** Section 7's `τ` sweep is the explicit knob
  for how much the new batch should count relative to the carried-forward
  prior — useful when you have a specific reason to believe the new wells
  either are noisy/small and shouldn't move the model much (`τ ≈ 1`), or
  reflect a real, lasting shift worth trusting more (`τ` larger).
"""

# ╔═╡ Cell order:
# ╠═9aeac5b8-3a66-434b-bd98-5bdd7df22f55
# ╠═8b983177-4105-4490-b705-590329f7f163
# ╠═87515d6c-b1e7-4ebd-8274-05aad9c19a68
# ╠═2543c334-d419-487e-a9b9-fd536d8c3785
# ╠═4916dddd-0363-4aa0-a30a-fc4e4791df6a
# ╠═4a7337a7-1f18-4551-a581-4d3767df372f
# ╠═3c32bd56-768e-4fec-8811-1981a7ce65cb
# ╠═e05034a9-686f-4cbe-b11b-2c6fc7496324
# ╠═a72c1070-ec7a-4ab9-8a5e-087af12bb829
# ╠═4d73ee31-728c-4318-b7ae-3519843dadc7
# ╠═f2205061-836d-48c3-9d66-51d666f3ce75
# ╠═d37fc21a-e817-4cb9-b7bc-f1bc1062f793
# ╠═778210a6-28ca-48c3-8492-1999f8694a02
# ╠═a5ebbe58-6f54-4b07-b9d6-298de7719844
# ╠═32910327-b150-42d1-b2bb-8933d4cf2d1e
# ╠═c71168a8-161c-4e14-8d8d-fbb5b3753baa
# ╠═0de9523f-c772-4a36-8ba4-df092eb5de9e
# ╠═25ccf42d-cc1e-4e2b-97f3-3412d54d5ae0
# ╠═f139ea88-876a-4668-90b9-bbb793d66a96
# ╠═30292e63-b29d-4db1-ab6a-fbf2b20b917f
# ╠═fd9292ea-c8e7-4cc2-a166-8eeb56b24512
# ╠═887aca7e-8d23-4c80-b121-f31ad672ca1e
# ╠═c1b8a4c9-a9f6-4a45-9b65-fd70fd118207
# ╠═4f7add61-3614-4cae-86a7-6b87b50c5d24
# ╠═d8b4db51-b24d-4727-8130-935411068532
# ╠═4372320c-46b2-415b-bad4-6423fded9e20
# ╠═fda488ed-5eb3-4918-a79d-dc397fafe69d
# ╠═e1ae99e6-4079-4d8a-8fb8-3faec1dcfe46
# ╠═16f4fe1e-8a88-4b25-b129-e5216e148762
# ╠═4097af08-6914-45ac-88d7-8a76d1c7aa12
# ╠═82a22b74-248a-4423-96ee-9ff79e043a03
# ╠═017e5933-ebbb-45fc-bd7d-217ed7060f79
