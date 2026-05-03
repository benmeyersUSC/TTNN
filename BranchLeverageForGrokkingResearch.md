# Branch Leverage — Jacobian-Based Behavioral Influence
*TTTN add-on notes for the modular arithmetic grokking experiment*

---

## The Problem with Raw Attribution

`BranchTrainer::InstrumentedFit` accumulates `source_gross_[s][i]` and `source_net_[s][i]` — the total absolute and signed parameter displacement attributed to each gradient source. These are accurate measures of movement through parameter space, but they treat every parameter element as equally important. A unit of displacement in the first embedding weight and a unit of displacement in the final output bias are counted identically.

The correction is **leverage**: how much does a unit shift in parameter i actually change the model's output?

---

## The Leverage Measure

For parameter element `i`, define:

```
leverage_i  =  ||∂output / ∂θ_i||_2
```

Where `output` is the model's raw logit vector (shape `[p]` for mod-p arithmetic) and the norm is Euclidean. This is the **rate of output displacement per unit parameter movement**: if you shift `θ_i` by ε, the output vector moves by approximately `ε · leverage_i` in L2 distance.

**Why L2 norm, not normalisation:** normalising would discard magnitude and leave only the direction in output space that `θ_i` affects. Two parameters — one shifting the output by 0.001 and one by 10.0 — would both normalise to 1. The L2 length preserves exactly the quantity of interest.

---

## How to Compute It

The Jacobian `∂output / ∂θ` is an `[OutSize × TotalParamCount]` matrix. We want the **column norms** (one scalar per parameter element). This is a pure reverse-mode computation — `OutSize` backward passes, one per output dimension.

**Algorithm:**

```python
leverage = zeros(TotalParamCount)

for j in range(OutputSize):          # p = 113 for mod-113
    e_j = one_hot(j, OutputSize)     # unit vector at output dim j

    # backward from e_j — as if e_j were the loss gradient
    # this gives dOutput_j/dTheta for ALL parameters in one pass
    grads = backward(e_j)            # shape: [TotalParamCount]
    leverage += grads ** 2

leverage = sqrt(leverage)            # elementwise sqrt
```

In TTTN terms: call `ZeroGrad()`, then use the existing backward infrastructure with a one-hot gradient at the output. No loss function involved — the output IS the "loss" (one dimension at a time). Repeat for all `p` output dimensions, accumulating squared gradients, then take the elementwise square root.

**Cost:** `p` backward passes per measurement. For `p = 113` this is cheap. Run at checkpoints during training (e.g., every 10 epochs), not inside the training loop.

---

## Connecting Leverage to Source Attribution

`InstrumentedFit` tracks for each source `s` and parameter element `i`:

- `source_gross_[s][i]` — total `|Δθ_s_i|` attributed to source `s` (parameter space movement)
- `source_net_[s][i]`  — total `Δθ_s_i` signed (net displacement)

The **leverage-weighted behavioral influence** of source `s` is:

```
behavioral_influence(s)  =  Σ_i  leverage_i  ·  source_gross_[s][i]
```

Units: *output displacement caused by source s, summed over all parameters it moved*.

The **leverage-weighted efficiency** of source `s`:

```
weighted_net(s)   =  ||leverage_i · source_net_[s][i]||_2     (weighted displacement vector norm)
weighted_gross(s) =  Σ_i  leverage_i · source_gross_[s][i]
weighted_eff(s)   =  weighted_net(s) / weighted_gross(s)
```

This answers: of all the output-space movement driven by source `s`, what fraction was directed (geodesic) vs. wasted (churn)?

---

## Practical Structure in the Research Repo

The TTTN side already exposes everything needed:
- `BranchTrainer::SourceTrajectory()` returns `source_gross_[s]` and `source_net_[s]` as aggregate scalars
- The raw flat vectors `source_gross_[s]` and `source_net_[s]` are members — expose a getter or copy them out for the weighted computation

The leverage computation itself lives in the **research repo** as a client-side utility:

```cpp
// Pseudocode — research repo, not TTTN core
std::vector<float> ComputeLeverage(BranchTrainer& bt, const BatchInput& X) {
    constexpr size_t P = TrunkNet::TotalParamCount;
    std::vector<float> lev_sq(P, 0.f);

    auto& trunk = bt.trunk();
    TrunkNet::TrainingCache<1> cache;
    trunk.template ForwardAll<1>(X, cache);

    for (size_t j = 0; j < OutputSize; ++j) {
        trunk.ZeroGrad();
        OutputTensor e_j{}; e_j.flat(j) = 1.f;
        trunk.template BackwardRange<1, 0, TrunkNet::NumBlocks>(cache, e_j);

        size_t i = 0;
        std::apply([&](const auto&... ps) {
            ([&] {
                for (size_t k = 0; k < ps.Size; ++k)
                    lev_sq[i++] += ps.grad.flat(k) * ps.grad.flat(k);
            }(), ...);
        }, trunk.all_params());
    }

    for (auto& v : lev_sq) v = std::sqrt(v);
    return lev_sq;  // leverage_i for each parameter element
}
```

---

## Key Interpretive Points

**Leverage vs. gradient:** The gradient `∂L/∂θ_i` folds together structural leverage AND current loss curvature. A small gradient at an early layer can reflect co-adaptation by later layers (they have tuned themselves to compensate), not low structural importance. Leverage strips the loss curvature out and measures pure output sensitivity.

**Not the same as gradient magnitude:** At convergence, ALL gradients are small (near a local minimum). Leverage is nonzero everywhere — it measures architectural sensitivity, not training signal.

**Frequency of measurement:** Leverage is a property of the current parameter values and changes as training progresses. Measuring it every 10-20 epochs gives a trajectory of how structural influence shifts as the network grokks. The most interesting question: does leverage redistribute toward the structured heads as grokking occurs?

**The key research number:** `behavioral_influence(structured_heads) / behavioral_influence(total)` — the fraction of total output-space movement driven by the structured auxiliary signal. If this rises as efficiency rises, the structured heads are the geodesic force.
