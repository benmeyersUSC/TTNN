# Parameter Movement Metrics
*A framework for measuring learning as movement in parameter space*

---

## Motivation

Gradient descent is a physical process. Parameters move through a high-dimensional space under forces determined by the loss landscape, the architecture, and the update rule. Standard training diagnostics — loss curves, accuracy, gradient norms — are projections of this movement onto a single scalar. They discard almost everything.

The goal of this framework is to instrument the *trajectory* itself: how far did each parameter travel, in what direction, driven by what forces, and how much of that travel was purposeful vs. wasted? And further: how much did each parameter actually *matter* — structurally, instantaneously, or in expectation over the space of possible networks?

Four metrics. One base accumulation, three ways to weight it. Each answers a different question about movement in parameter space.

---

## Historical Context

The idea that parameter space has geometry is not new. **Loss landscape visualization** (Li et al., 2018) revealed that the geometry of the loss surface varies dramatically with architecture and training choices — but examined it statically, as a snapshot, not as a trajectory. **Neural ODE / gradient flow** literature (Elkabetz & Cohen; various) treats training as a continuous dynamical system in parameter space, deriving theoretical properties of the flow — but rarely measures the trajectory empirically. **Optimizer comparison** papers plot gradient norm curves but almost never net displacement or path efficiency.

The **mean field theory of deep networks** (Poole et al. 2016; Schoenholz et al. 2017) analyzes signal and gradient propagation through random networks as a function of architecture and activation choice — deriving the *expected* gradient magnitude at each layer before any training occurs. This is the intellectual ancestor of Metric III below.

**Elastic Weight Consolidation** (Kirkpatrick et al., 2017) uses the diagonal Fisher Information Matrix — which is the expected squared Jacobian — to measure parameter importance for continual learning. This is the closest prior work to Metric II, but used as a regularizer rather than a trajectory weight.

The **Riemannian framing** is the unifying idea: the natural metric on parameter space is not Euclidean but is defined by the Fisher Information Matrix `F = E[J^T J]`, where `J = ∂f/∂θ` is the Jacobian of the model output with respect to parameters. Moving in a direction that strongly influences the output is *longer* in this geometry than moving in a behaviorally inert direction. **Natural gradient descent** moves in this geometry. The metrics below are empirical approximations to trajectory length and source attribution under this geometry.

To the best of current knowledge, the decomposition of parameter trajectory metrics **by gradient source** — attributing gross path length and net displacement to specific loss functions or training objectives — has not been done. This is the novel contribution of the framework.

---

## The Base Accumulation — Raw Displacement & Distance

**What it is:** For each parameter `θ_i`, accumulate two running quantities over training:

```
gross_path[i]     += |Δθ_i(t)|        // absolute update at each step — total distance walked
net_displacement[i] += Δθ_i(t)        // signed update at each step — net vector traveled
```

At any checkpoint, `gross_path[i]` is the total path length of `θ_i` through parameter space. `||net_displacement[i]||` is the straight-line distance from init to current position.

**Efficiency Ratio** (the key derived metric):
```
Efficiency = ||net_displacement|| / gross_path
```
Ranges from 0 (pure churn, all movement cancels) to 1 (perfectly geodesic, every step is purposeful). This is the *waste metric* — its inverse measures how tortured the path was relative to where it ended up.

**By gradient source:** Run separate backward passes per loss function before summation, snapshot grad buffers per source, then call `Update` once (preserving unified Adam moment estimates). This gives `gross_by_source[i][s]` and `net_by_source[i][s]` — the path length and displacement attributable to source `s`. Per-source efficiency follows directly.

**What it tells you:** How much did this parameter move, in what direction, and was that movement directed or wasteful? Which training objective drove the useful movement vs. the churn?

**Limitation:** Treats all parameter positions as equal. A step of size `ε` in layer 1 and a step of size `ε` in layer 10 are counted identically. The three scaling metrics below correct for this.

---

## Metric I — Positional Leverage (Coarse-Grained, Architecture-Dependent)

**Intuition:** A parameter in layer 1 of a 10-layer network has its contribution pass through 9 subsequent transformations before reaching the output. Every downstream parameter must accommodate the representations it produces. It has *structural leverage* — the potential to influence everything that follows. A parameter in the final layer influences only the output projection. The structural leverage of a parameter is proportional to how many parameters come after it in the computational graph.

**Definition:**
```
PositionalLeverage(θ_i) = |{θ_j : j is downstream of i in the computational graph}| / TotalParams
```
Or more simply: `(NumParams - LayerIndex) / NumParams`, normalized to [0, 1].

**Weighted accumulation:**
```
weighted_gross[i] += PositionalLeverage(i) · |Δθ_i(t)|
```

**Architecture-dependent:** Fully determined by the computational graph topology. No values needed. Computed once at network construction time.

**The vanishing gradient counterpoint:** This metric and gradient magnitude point in *opposite* directions for early layers. Early parameters have maximum positional leverage but minimum gradient magnitude (vanishing gradients collapse the chain rule product through long paths). This tension is the whole point — positional leverage measures *potential* influence regardless of whether the current values allow that influence to propagate. It is the structural prior on importance, uncorrupted by training dynamics.

**What it tells you:** How much *should* this parameter matter, by virtue of where it sits in the computation? A parameter with high positional leverage that barely moves (low raw displacement) is either well-initialized or being under-trained at a structurally critical site.

---

## Metric II — Instantaneous Functional Influence (Value & Architecture-Dependent)

**Intuition:** At any moment in training, how much does parameter `θ_i` actually bend the output function? Not how sensitive the loss is to it (that's the training gradient, which conflates parameter influence with loss geometry) — but how much does moving `θ_i` change what the network computes, right now, with its current values?

**Definition:**
```
FunctionalInfluence(θ_i, t) = ||∂f/∂θ_i||_F
```
The Frobenius norm of the Jacobian of the model output `f` with respect to parameter `θ_i`, evaluated at the current parameter values. For a network with output dimension `p` (e.g. `p=113` for modular arithmetic), this is a `p × 1` vector per scalar parameter, computed via one backward pass per output dimension (or estimated via Hutchinson's randomized estimator).

**Relationship to the training gradient:**
```
∂L/∂θ_i  =  (∂L/∂f) · (∂f/∂θ_i)
```
The training gradient is the Jacobian scaled by the loss gradient at the output. Vanishing gradients arise from either a flat loss surface (small `∂L/∂f`) or a small Jacobian (small `∂f/∂θ_i`). Metric II isolates the second factor — it is loss-free, measuring behavioral influence directly.

**Riemannian interpretation:** The matrix `F = E[J^T J]` is the Fisher Information Matrix — the natural Riemannian metric tensor on parameter space. Moving distance `ε` in the direction of `θ_i` has length `sqrt(ε^T F ε)` under this metric. Metric II is computing the diagonal of `F` evaluated at the current parameter values — a diagonal approximation to the natural Riemannian metric.

**Weighted accumulation:**
```
weighted_gross[i] += FunctionalInfluence(i, t) · |Δθ_i(t)|
```

**Computed:** Periodically (every N epochs) rather than every step — expensive but tractable for small networks. For `p=113` output classes and a small transformer, this is 113 backward passes per snapshot.

**What it tells you:** How much is this parameter actually doing right now? High influence + small movement = a critical parameter being moved carefully (or being constrained). Low influence + large movement = a parameter churning in a behaviorally inert direction. Comparing this to Metric III reveals how much the network's realized influence structure has diverged from its architectural prior.

---

## Metric III — Structural Potential Leverage (Architecture-Dependent, Value-Free)

**Motivation:** Both the training gradient and the Jacobian are value-dependent — they measure influence at a specific point in weight space, colored by whatever values the network currently holds. But influence is also a property of *architecture*: the operations that follow a parameter, the nonlinearities they pass through, the depth of the composition chain. Can we measure this architectural influence potential, completely disentangled from current parameter values?

**The key insight:** You cannot do calculus without values — but you can *average over values*. Define:

```
StructuralPotential(θ_i) = E_{θ ~ P_init}[ ||∂f/∂θ_i||_F ]
```

The expected Jacobian norm of parameter `θ_i` under the initialization distribution — integrated over all possible weight configurations. This is a property of the *architecture and nonlinearity choice*, not of any specific trained or randomly initialized network.

**Why this works:** For a given architecture, as you average over many random initializations, the idiosyncratic value-dependent fluctuations cancel and the structural signal remains. What persists is the *mean propagation gain* through the computational graph from position `i` to the output — determined by layer type, activation function, depth, and connectivity, not by specific weight values.

**Computation:** Monte Carlo estimate. Sample `K` random initializations (from the actual init distribution used for training), compute `||∂f/∂θ_i||_F` for each, average. Since this is architecture-dependent and value-free, it is computed **once** as a precomputation step, before any training begins. Heavy precomputation is acceptable — this is a fixed scaling factor reused across all experiments in the family.

For TTTN, since types encode architecture, `StructuralPotential` is a property of the *type* of a parameter position, not its instance. Precompute once per architecture type.

**Connection to mean field theory:** Poole et al. (2016) derived this analytically for fully connected networks with specific activation functions — showing that expected gradient magnitude decays exponentially with depth in the "ordered phase" and grows exponentially in the "chaotic phase," with a critical "edge of chaos" boundary. Metric III is the empirical version of this, applicable to arbitrary architectures including transformers where analytical derivation is intractable.

**Weighted accumulation:**
```
weighted_gross[i] += StructuralPotential(i) · |Δθ_i(t)|
```

**What it tells you:** How much *could* this parameter matter, by virtue of its architectural position and the operations that follow it — independent of what the network has actually learned? The ratio `FunctionalInfluence(i,t) / StructuralPotential(i)` over training is a measure of how much the network has *realized* its architectural potential at position `i`. Parameters where this ratio rises sharply are being actively recruited. Parameters where it stays near zero despite high structural potential are being wasted.

---

## Complementarity — What Each Metric Sees

| Metric | Value-dependent? | Loss-dependent? | What it measures |
|---|---|---|---|
| Raw Displacement | Yes | Yes (indirectly) | Actual movement; waste vs. direction |
| I. Positional Leverage | No | No | Structural potential by graph topology |
| II. Functional Influence | Yes | No | Realized behavioral influence right now |
| III. Structural Potential | No (averaged out) | No | Architectural influence in expectation |

The metrics triangulate. A parameter that scores high on III (high structural potential) but low on II (low realized influence) is being suppressed by the current weight configuration — perhaps vanishing gradients are preventing its structural leverage from being realized. A parameter that scores high on II but low on III is punching above its architectural weight — perhaps a late-layer parameter that happens to sit at a critical output bottleneck.

Watching **II / III** (realized vs. potential) evolve over training, per gradient source, is the most interesting derived quantity in the framework. It shows which training objectives are recruiting structurally important parameters vs. which are spending gradient budget on structurally peripheral ones.

---

## The Einsteinian Framing

In general relativity, mass-energy doesn't exert forces — it curves spacetime, and objects follow geodesics through that curved geometry. The "force of gravity" is the straightening of paths in a curved space.

The parameter space analog: the architecture and loss function don't exert forces on parameters — they define a geometry (the Fisher-Riemannian metric), and gradient descent follows approximate geodesics through that geometry. The structured solutions — algorithms, rules, Fourier features — are massive objects that warp the geometry, making paths toward them shorter. Memorization is the scenic route: longer under the natural metric, more costly, ultimately unstable.

The metrics above are instruments for measuring this geometry empirically — not just whether parameters moved, but *how* they moved, *where* the space was curved, and *which forces* were responsible for bending the path.
