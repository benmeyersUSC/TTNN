# Parameter Movement Metrics
*A framework for measuring learning as movement — in parameter space and in output space*

---

## Motivation

Gradient descent is a physical process. Parameters move through a high-dimensional space under forces determined by the loss landscape, the architecture, and the update rule. Standard training diagnostics — loss curves, accuracy, gradient norms — are scalar projections of this movement that discard almost everything.

The goal of this framework is to instrument the *trajectory* itself: how far did each parameter travel, in what direction, driven by what forces, and how much of that travel was purposeful vs. wasted? And further: how much did each parameter actually *matter* — structurally, instantaneously, or in expectation over the space of possible networks?

Two lenses, two coordinate systems, one physics:

- **Parameter space.** Each `θ_i` is a real number; the update `Δθ_i(t)` at each step advances that coordinate. Movement is measured with plain Euclidean bookkeeping.
- **Output space.** Every `θ_i` casts an influence into the model's output shape via `∂F(x)/∂θ_i`. This vector *is* the parameter's causal influence — perturb `θ_i` by `ε`, and the output moves by `ε · ∂F/∂θ_i` to first order. Movement here is measured in the language the model actually produces.

The most informative signals live in the second lens. Parameter-space movement is the raw dynamics; output-space movement is what those dynamics *do*. The whole framework is built to interconvert cleanly between them and to expose collaboration, cancellation, and directedness.

---

## Historical Context

The idea that parameter space has geometry is not new. **Loss landscape visualization** (Li et al. 2018) revealed that loss surfaces vary dramatically with architecture and training choices — but examined statically, as snapshots. **Neural ODE / gradient flow** literature (Elkabetz & Cohen; various) treats training as a continuous dynamical system, deriving theoretical properties of the flow but rarely measuring the trajectory empirically. **Optimizer comparison** papers plot gradient norms but almost never net displacement or path efficiency.

The **mean field theory of deep networks** (Poole et al. 2016; Schoenholz et al. 2017) analyzes signal and gradient propagation through random networks as a function of architecture — deriving *expected* gradient magnitude before any training occurs. That is the intellectual ancestor of Metric III below.

**Elastic Weight Consolidation** (Kirkpatrick et al. 2017) uses the diagonal Fisher Information Matrix — the expected squared Jacobian — to measure parameter importance for continual learning. That is the closest prior work to Metric II here, though used as a regularizer rather than as a trajectory weight.

The **Riemannian framing** is the unifying idea: the natural metric on parameter space is not Euclidean but the Fisher Information Matrix `F = E[J^T J]`, where `J = ∂F(x)/∂θ`. Moving in a direction that strongly influences the output is *longer* in this geometry than moving in a behaviorally inert direction. **Natural gradient descent** moves in this geometry. The metrics below are empirical approximations to trajectory length, cross-parameter collaboration, and source attribution under this geometry.

To the best of current knowledge, two contributions in this framework are novel:

1. **Decomposition of parameter trajectory metrics by gradient source** — attributing gross path length and net displacement to specific loss functions or training objectives.
2. **Output-space accord** — treating each parameter's Jacobian column as a first-class object, measuring GROSS and NET *in the model's output shape*, and using their ratio as a live signal of cross-parameter collaboration.

---

## The Base Accumulation — Parameter-Space Gross Path & Net Displacement

Let `θ_i(t)` denote parameter element `i` at step `t`, and `Δθ_i(t) = θ_i(t) − θ_i(t−1)` the update applied at step `t`.

**Gross path length** — total distance parameter `i` has traveled:
```
G_i = Σ_t |Δθ_i(t)|
```
The odometer reading. Accumulates regardless of direction.

**Net displacement** — signed sum of all updates:
```
D_i = Σ_t Δθ_i(t) = θ_i(T) − θ_i(0)
```
Straight-line vector from initial to current position.

**Efficiency ratio** — the primary waste metric:
```
η = ‖D‖_2 / G       where G = Σ_i G_i and ‖D‖_2 = sqrt(Σ_i D_i²)
```
Ranges from 0 (pure churn, all movement cancels) to 1 (perfectly geodesic, every step points the same way). `1 − η` is waste.

Both `G_i` and `D_i` are per-parameter scalars. They cost nothing at training time: two extra tensors of the same shape as `value` on every `Param`, updated in-place after every Adam step.

---

## Per-Source Attribution

When training uses multiple gradient sources — a trunk loss `L_0` and auxiliary head losses `L_1, ..., L_H`, or TF and RL objectives blended — the total gradient at step `t` is
```
g_i(t) = Σ_{s=0}^{S-1} g_i^{(s)}(t)      (S = H + 1)
```
A single Adam step is applied using the combined gradient `g_i(t)`, preserving unified moment estimates and the "conflicted multi-mandate" loss landscape. The actual parameter update `Δθ_i(t)` is then attributed proportionally:
```
Δθ_i^{(s)}(t) = (g_i^{(s)}(t) / g_i(t)) · Δθ_i(t)
```

This proportional split is **exact under Adam** because Adam applies the same elementwise rescaling to all sources — pre-Adam and post-Adam source proportions are identical. Consequently
```
Σ_s Δθ_i^{(s)}(t) = Δθ_i(t)
```
holds to the last float, and per-source contributions to net displacement are additive.

Per-source gross path and net displacement:
```
G_i^{(s)} = Σ_t |Δθ_i^{(s)}(t)|
D_i^{(s)} = Σ_t Δθ_i^{(s)}(t)
```
Per-source efficiency `η^{(s)} = ‖D^{(s)}‖ / Σ_i G_i^{(s)}` follows directly.

**When the split is undefined** (both numerator and denominator ≈ 0): fall back to equal split `Δθ^{(s)} = Δθ / S`. Documented "no-signal step" convention.

---

## Metric II — Functional Influence in Output Space

The Jacobian at input `x`:
```
J = ∂F(x)/∂θ ∈ R^{p × P}
```
where `p` is the output dimension (`OutputTensor::Size`) and `P` is the total parameter count. Column `i`,
```
J_i = ∂F(x)/∂θ_i ∈ R^p
```
is a vector in output space. **That vector is the full weight of parameter `i`'s causal influence** — perturb only `θ_i` by `ε`, and the output moves by `ε · J_i` to first order. Nothing missing.

We keep `J_i` as an **output-shape tensor per parameter element** — not collapsed to its L2 norm. Every parameter then speaks the same language: its influence lives natively in the model's output shape. Scalarization is downstream.

### Output-space GROSS and NET

For a single training step with update `Δθ`:
```
NET_output_j (single step)   = Σ_i Δθ_i · J_ij = (J · Δθ)_j
GROSS_output_j (single step) = Σ_i |Δθ_i · J_ij|
```
`NET_output` is the actual output shift caused by the step. `GROSS_output` is the total unsigned output-space movement — the same shift with cancellations "unfolded."

Accumulated over training:
```
NET_output_j   = Σ_t Σ_i Δθ_i(t) · J_ij(t)
GROSS_output_j = Σ_t Σ_i |Δθ_i(t) · J_ij(t)|
```
Both are vectors in the model's output shape.

Per-parameter accumulators (for leverage-weighted downstream analysis):
```
net_output[i]   = Σ_t Δθ_i(t) · J_i(t)     (output-shape tensor)
gross_output[i] = Σ_t |Δθ_i(t)| · |J_i(t)|  (elementwise, output-shape tensor)
```

### Accord ratio

```
η_output = ‖NET_output‖ / ‖GROSS_output‖ ∈ [0, 1]
```
`= 1` iff every parameter's per-step output-space contribution `Δθ_i · J_i` points the same way. Falling short of `1` means destructive interference across parameters — the training did output-space work that cancelled.

**Per-output-dimension version.** `η_j = NET_j / GROSS_j` per output coordinate exposes *which output directions* got collaborative updates. On mod-113 classification, a rising `η_j` at a Fourier key frequency `j` would indicate that the network's trajectory is coordinated in exactly the direction that matters for generalization.

### Fisher, without materializing Fisher

The empirical Fisher Information Matrix is `F = J^T J ∈ R^{P × P}`. Its diagonal `F_ii = ‖J_i‖²` is what EWC and diagonal-Fisher approaches use. Its off-diagonal `F_ij = ⟨J_i, J_j⟩` encodes cross-parameter coupling:

- Diagonal treats each parameter as if it acts orthogonally in output space. It does not.
- Off-diagonal captures collaboration — how much two parameters push the output in overlapping directions.

The identity
```
‖NET_output‖² = ‖J · Δθ‖² = Δθ^T · F · Δθ
              = Σ_i Δθ_i² · F_ii  +  2 Σ_{i<j} Δθ_i · Δθ_j · F_ij
                (diagonal, unsigned)     (off-diagonal, signed — collaboration)
```
means the accord ratio implicitly probes the full off-diagonal structure of `F` without ever building the `P × P` matrix. For a 100K-parameter model this matters — `F` would be 10¹⁰ floats. `J` is only `p × P` (about 45 MB at `p = 113, P = 100K`); we compute it directly and use it once, then discard it.

### Cost and cadence

Computing `J` at a single `x` costs `p` backward passes: feed `e_j = one-hot(j)` as the upstream gradient, collect each `Param::grad` — that row of `J` is `∂F_j / ∂θ`. Repeat for `j ∈ [0, p)`, stack.

For Nanda-scale (`p = 113`, `P ≈ 10⁵`) this is cheap enough to run every step during grokking replication. For larger models, run at checkpoints and use the last `J` for the intervening interval.

**Storage discipline.** Never persist `J` itself — 45 MB per snapshot, unmanageable at 10⁵ steps. Persist only the running per-parameter output-space accumulators (`net_output[i]`, `gross_output[i]`) and network-level aggregates (`p`-dim `NET_output`, `GROSS_output`). All scalarizations derive from these.

---

## Metric I — Positional Leverage (Architecture-Only)

A parameter in layer 1 of an `L`-layer network has its contribution pass through `L − 1` subsequent transformations before reaching the output. Every downstream parameter must accommodate the representations it produces. Structural leverage is proportional to how many parameters come after it in the computational graph.

```
λ_i^{pos} = |{ j : j is downstream of i }| / P ∈ [0, 1]
```

Fully determined by the computational graph topology — no weight values needed. Computed once at construction (`constexpr` in TTTN, since block ordering and param counts are all compile-time).

**The vanishing-gradient counterpoint.** This metric and gradient magnitude point in *opposite* directions for early layers. Early parameters have maximum positional leverage but minimum gradient magnitude. Positional leverage measures *potential* influence regardless of whether current values allow it to propagate.

Weighted accumulation:
```
G̃_i^{I} = λ_i^{pos} · G_i
```

---

## Metric III — Structural Potential (Value-Free)

Both the training gradient and the Jacobian `J_i` are value-dependent — they measure influence at a specific point in weight space, coloured by whatever values the network currently holds. But influence is also a property of *architecture*.

You cannot do calculus without values — but you can average over values. Define:
```
λ_i^{str} = E_{θ ~ P_init}[ ‖∂F(x)/∂θ_i‖ ]
```
the expected Jacobian norm under the initialization distribution, integrated over all possible weight configurations.

Idiosyncratic value-dependent fluctuations cancel as you average over many random initializations. What persists is the *mean propagation gain* through the computational graph from position `i` to the output — determined by layer type, activation function, depth, and connectivity, not by specific weight values.

**Computation.** Monte Carlo estimate with `K` random Xavier initializations. Since this is value-free and architecture-dependent, it is computed **once** as a precomputation step and reused as a fixed scaling factor across all experiments in the family.

**Connection to mean field theory.** Poole et al. (2016) derived this analytically for fully-connected networks with specific activation functions — showing expected gradient magnitude decays exponentially with depth in the "ordered phase" and grows exponentially in the "chaotic phase," with a critical edge-of-chaos boundary. Metric III is the empirical version, applicable to arbitrary architectures including transformers where analytical derivation is intractable.

```
G̃_i^{III} = λ_i^{str} · G_i
```

---

## Triangulation — What Each Metric Sees

| Metric | Value-dep? | Loss-dep? | What it measures |
|---|---|---|---|
| Raw `G, D, η` | Yes | Yes (indirectly) | Actual movement in parameter space; waste vs. direction |
| I. Positional Leverage | No | No | Structural potential by graph topology |
| II. Functional Influence | Yes | No | Realized behavioral influence right now, output-native |
| III. Structural Potential | No (averaged out) | No | Architectural influence in expectation |

The metrics triangulate:

- A parameter that scores high on III but low on II is being *suppressed* by the current weight configuration — vanishing gradients preventing structural leverage from being realized.
- A parameter that scores high on II but low on III is *punching above its architectural weight* — likely a late-layer parameter sitting at a critical output bottleneck.
- Watching `λ_i^{fn}(t) / λ_i^{str}` — realized over potential — evolve over training, decomposed by gradient source, is the most informative derived quantity in the framework. It shows which training objectives recruit structurally important parameters and which spend gradient budget on structurally peripheral ones.

---

## Leverage-Weighted Source Attribution

Given leverage weights `λ_i` from any of Metrics I / II / III, the **leverage-weighted behavioural influence** of gradient source `s` is:
```
B(s) = Σ_i λ_i · G_i^{(s)}
```
Units: output-space displacement caused by source `s`, summed over all parameters it moved, scaled by how much each parameter matters to the output.

**Leverage-weighted efficiency of source `s`:**
```
D̃^{(s)} = ‖ λ ⊙ D^{(s)} ‖_2
η̃^{(s)} = D̃^{(s)} / B(s)
```
Of all the output-space movement driven by source `s`, what fraction was directed vs. wasted?

**The key research quantity:**
```
φ = B(structured heads) / Σ_s B(s)
```
the fraction of total leverage-weighted output-space movement attributable to structured auxiliary signal. If `φ` rises as `η` rises, the structured heads are the geodesic force.

---

## The Experimental Program

**Mod-113 modular arithmetic** — `(a + b) mod p` for `p = 113`. Full dataset `p² = 12,769` examples. Grokking phenomenon is well-known and mechanistically reverse-engineered (Nanda et al. 2023 — Fourier features on a circle, angle addition on key frequencies).

Two experiments in order:

### Experiment 1 — Vanilla Nanda replication with instrumentation

Single-source vanilla training (no auxiliary heads). One-layer transformer, AdamW with weight decay, full-batch training on 30 % of the dataset, ~10⁵ steps to grokking.

**Question:** Do the output-space efficiency metrics `η_output(t)`, per-dimension `η_j(t)`, and parameter-space `η(t)` cooccur with or predict the canonical grokking signature (train-loss plateau, delayed val-acc breakthrough)?

The runner and analysis pipeline for this experiment live in the [GrokkingMetrics](https://github.com/benmeyersUSC/GrokkingMetrics) repository, which pulls TTTN as a dependency. No source attribution — that's the point. If the *unweighted* accord ratio is already a grokking predictor, we have a useful signal. If it isn't, the negative result still narrows the hypothesis space.

### Experiment 2 — Structured-head family with source attribution

The family from `research/learning_mechanics_ideas.md`: same architecture, but a family of runs at gradient budget shares `{0, 25, 50, 75, 100}%` between the main output loss and auxiliary heads (Fourier features of `a`, `b`, and unwrapped sum `a+b` at prescribed key frequencies, per Nanda). Fixed-accuracy stopping criterion.

**Questions:**
- Higher structured-head share → higher `η`? Higher `φ`?
- Non-monotone optimum in the gradient share curve?
- Does `η_output` under structured supervision peak in the same output dimensions Nanda identifies as the key Fourier frequencies?

Runner uses `BranchTrainer` with `InstrumentedFit`; source attribution and leverage weighting are the whole point.

---

## Implementation Handles

| Concept | Code |
|---|---|
| Base accumulation `G_i, D_i` | [`src/NetworkUtil.hpp`](src/NetworkUtil.hpp) — `Param<T>::net_disp`, `Param<T>::gross_path`, `Param::update` |
| `η` scalar reductions | [`src/NetworkUtil.hpp`](src/NetworkUtil.hpp) — `TotalGrossPath`, `TotalNetL2`, `Efficiency` |
| Per-source attribution | [`src/SourceTrajectory.hpp`](src/SourceTrajectory.hpp) — `SourceTrajectory<NumSources>` |
| Instrumented training with heads | [`src/BranchTrainer.hpp`](src/BranchTrainer.hpp) — `InstrumentedFit`, `InstrumentedBatchFit` |
| Metric II Jacobian | [`src/FunctionalInfluence.hpp`](src/FunctionalInfluence.hpp) — `FunctionalInfluence`, `OutputSpaceGross`, `OutputSpaceNet`, `AccordRatio` |
| Metric III structural potential | [`src/StructuralPotential.hpp`](src/StructuralPotential.hpp) |
| Metric I positional leverage | [`src/BlockSequence.hpp`](src/BlockSequence.hpp) — `PositionalLeverage` `constexpr` array |
| Grokking runner (Experiment 1) | [`GrokkingMetrics`](https://github.com/benmeyersUSC/GrokkingMetrics) — `nanda_grokking.cpp` + analysis tools |
| Weight decay for grokking | [`src/NetworkUtil.hpp`](src/NetworkUtil.hpp) — `AdamState::wd` |
| Trajectory reset | `TrainableTensorNetwork::ResetTrajectory`, `EncoderDecoder::ResetTrajectory` |

---

## The Einsteinian Framing

In general relativity, mass-energy doesn't exert forces — it curves spacetime, and objects follow geodesics through the curved geometry. The "force of gravity" is the straightening of paths in a curved space.

The parameter-space analog: the architecture and loss function don't exert forces on parameters — they define a geometry (the Fisher-Riemannian metric), and gradient descent follows approximate geodesics through it. The structured solutions — algorithms, rules, Fourier features — are massive objects that warp the geometry, making paths toward them shorter. Memorization is the scenic route: longer under the natural metric, more costly, ultimately unstable.

Grokking, in this framing, is not a curiosity. It is a thermodynamic inevitability. Given sufficient training, the network is pulled into the basin of the structured solution because that basin is deeper and wider. The delayed generalization is the time it takes to find the top of the hill separating the memorization basin from the structure basin.

The metrics above are instruments for measuring this geometry empirically — not just whether parameters moved, but *how* they moved, *where* the space was curved, and *which forces* were responsible for bending the path.
