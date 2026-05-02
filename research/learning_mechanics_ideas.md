# Ideas in Learning Mechanics — Working Notes
*Ben Meyers — April 2026*

---

## I. Structured Learning as Energetic Privilege

### The Core Intuition

Task-relevant structures — algorithms, syntactic rules, data regularities, logical operations — are not merely patterns that a neural network happens to fit. They are **low-entropy attractors that the loss landscape inherits from the data distribution**. Gradient descent is a physical process exploring a geometry that was shaped by those attractors before the network has any "knowledge" of them.

The conjecture: **reaching low loss is unreasonably difficult if you avoid these routes.** To memorize rather than generalize is not simply a different solution — it is an energetically expensive detour. The structured solution is the geodesic. Memorization is the scenic route: longer, more tortured, and ultimately unstable.

### Implications

- **Grokking is not a curiosity — it is a thermodynamic inevitability.** Given sufficient training, the network is eventually pulled into the basin of the structured solution because that basin is deeper and wider. The delayed generalization is the time it takes to find the top of the hill separating the memorization basin from the structure basin.
- **The principle of least action analog:** structured solutions extremize something in the learning process. If this can be made precise — defining "action" in parameter space — it would be a serious theoretical result.
- **Memorization as high-action path:** a model that has memorized rather than generalized has, in some precise sense, traveled *farther* in parameter space to achieve equivalent loss. This is empirically testable.

### Open Challenge

Pinning this down requires confronting the co-composition problem: the loss landscape is jointly determined by data distribution, architecture, loss function, and update granularity. Isolating the "gravitational source" — the task structure — from these confounds is the hard part. But the signal is there to be found.

---

## II. Massive Objects in Learning / Concept Space

### The Intuition

If learning is movement through parameter space, then task-relevant structures behave like **massive objects that warp the geometry of the space being traversed**. They exert a pull on the optimizer — not through any explicit mechanism, but because the loss landscape is shaped in their image.

Syntactic structures, merge operations, logical rules, arithmetic algorithms: these are not just targets. They are **attractors beckoning to be found**, because finding them collapses the entropy of the prediction problem dramatically.

### The Parameter-vs-Activation Problem

The hard version of this idea lives in raw parameter space, and the mapping from parameters to semantic objects is deeply unclear — parameters don't have semantic identity the way activations do. But there are softer, more tractable versions:

- **Function space / representation space:** The "mass" may be better located in function space or embedding space, where NTK and mean-field theory already partially formalize geometry. Massive objects in function space pull the model toward low-complexity, high-generalization solutions.
- **Embedding space as a middle ground:** The geometry of learned representations — their intrinsic dimensionality, isotropy, clustering — may be the right level of abstraction to observe the gravitational effects of task structure, without requiring parameter-level semantic assignment.
- **Grokking as empirical confirmation:** The network is pulled toward a generalizing algorithmic solution that exists as a deep attractor, found long after loss convergence. The delayed discovery is the optimizer wandering the foothills before falling into the basin.

### Framing

Syntactic and algorithmic regularities in data lower the entropy of the prediction problem *catastrophically* once grokked. The loss landscape inherits this entropy structure. The optimizer, agnostic to semantics, nonetheless finds these routes because they are downhill.

---

## III. Gradient Metrics — A Vocabulary for Movement

### Metric A: Gradient Budget (Net Absolute Update Mass)

**What it measures:** The total absolute update administered to model parameters by a given learning rule or training strategy, accumulated over training.

**The leverage premium:** Updates to earlier layers carry more downstream influence than updates to later layers — analogous to early splits in a decision tree. The Gradient Budget weights updates by their position in the network, awarding a premium for earlier-layer influence.

**Use case:** When training with mixed strategies (e.g., teacher forcing + autoregressive RL decode on a warm-up/down schedule), the Gradient Budget tracks the net influence each strategy has on the model as schedules shift. It answers: *how much did this rule move the model?*

**Connection to learning mechanics:** Directly operationalizes "learning as movement" — the Gradient Budget is a measure of total path length in parameter space attributable to a specific learning rule.

---

### Metric B: Displacement Attribution (Directed Credit Assignment)

**What it measures:** For each parameter, the net *signed* displacement from initialization to current position — then decomposed into fractional contributions from each update rule, epoch, or training example.

**The key insight:** This is backprop-style credit assignment, but one meta-level up. Rather than asking "which neuron contributed to this output," we ask: *"which training event contributed to this parameter being where it now is?"*

Example output: "RL contributed 80% of the net displacement of this parameter toward its current (better) setting. Cross-entropy loss contributed 20%."

**Aggregation hierarchy:**
- Rule-level: which learning rule did most of the useful work?
- Epoch-level: when in training did meaningful displacement occur?
- Batch-level: which batches drove the most directed movement?
- Example-level: which individual training examples had the most influence?

**The noisy update caveat:** Not every gradient step is a good update — small-batch training in particular produces high-variance, often counterproductive individual updates. Displacement Attribution does not treat every update as dogmatically correct. By aggregating signed displacements over time and over the network, noise cancels in expectation and systematic signal accumulates. The metric measures *net* contribution to the final (better) parameter setting, not momentary influence.

**Relation to influence functions:** Related to the interpretability literature on influence functions — attributing model behavior to training examples — but operating at the level of parameter update trajectories rather than loss curvature. More mechanistically direct.

---

### Metric C: Efficiency Ratio (Displacement / Gradient Work)

**What it measures:** The ratio of net signed displacement to total absolute gradient work (path length) for a given parameter or set of parameters, attributable to a given learning rule.

**Intuition:** 
- High ratio → the learning rule is pushing the model *directedly* toward better solutions. Movement is purposeful.
- Low ratio → the learning rule is generating a lot of churn — large absolute updates that cancel out, producing little net displacement. Movement is undirected, expensive, noisy.

**The memorization-vs-generalization test:** A memorizing network should have a **lower efficiency ratio** than a grokking network on the same task, assuming the grokking network's path through parameter space is more geodesic — shorter net displacement to reach equivalent loss, less wasted movement. If this holds empirically, it would directly support the conjecture in Section I.

**Formal statement:**

```
Efficiency(rule, params, t) = ||Σ Δθ_rule|| / Σ ||Δθ_rule||
```

Where the numerator is net signed vector displacement and the denominator is total absolute update mass. Ranges from 0 (pure churn) to 1 (perfectly directed, monotone movement).

---

## IV. Connecting the Metrics to the Conjecture

The three metrics form a coherent vocabulary for testing the core conjecture:

| Metric | What it captures |
|---|---|
| Gradient Budget | Total energy expended by a learning rule |
| Displacement Attribution | Credit assignment for net parameter improvement |
| Efficiency Ratio | Directedness of movement — geodesic vs. scenic route |

**The baseline experiment:** Train two networks on the same rule-generated task — one allowed to grok (sufficient data, long training), one forced toward memorization (limited data, early stopping, or architectural constraint). Compare their Efficiency Ratios over training. If the memorizing network has systematically lower efficiency — more gradient work per unit of useful displacement — that is direct empirical evidence that **structured solutions are the paths of least resistance**, and memorization is energetically costly.

---

## V. The Branch Training Experiment

### Motivation

A researcher in the Simon/Kunin group has proposed a related research direction: take datasets generated or governed by known structural rules or dynamics, train networks to learn them, and then search for the representations you *know in advance* to be useful — because the data generation process tells you what they should be. The question is whether a toolkit can be built to map from **rule/structure space → network representation space** for arbitrary datasets.

The Branch Training experiment is a controlled, empirical special case of this same hypothesis. Rather than searching for unknown representations post-hoc, it **incentivizes the network to build known intermediate representations during training**, then asks whether doing so changes the geometry of how the network learns.

### Setup

Train a family of networks on the same rule-generated dataset. Every network in the family shares the same trunk architecture. Each network differs only in its **gradient budget share** — the fraction of total gradient influence allocated to structured auxiliary heads vs. the main output head.

The family spans the full range:
- **0% structured** — pure task loss, no auxiliary heads. The network must find structure on its own or not at all.
- **25%, 50%, 75%** — intermediate mixtures. Structured heads are present and exerting increasing gradient influence on the trunk.
- **100% structured** — gradient flow is dominated entirely by auxiliary heads grading for intermediate rule representations. Main output head present but nearly silent.

The auxiliary heads branch from multiple layers of the trunk. Each head is trained to predict an intermediate structural feature that is *known from the data generation rules* — not discovered, but prescribed. For example, if the dataset is generated by a composition of logical rules, one head might grade for whether a given layer's representation has encoded rule A, another for rule B, etc.

### The Fixed-Accuracy Measurement Protocol

Rather than running all models for the same number of epochs (which would confound time with learning), **run every model until it reaches a fixed accuracy threshold** — e.g., 85% on a held-out validation set. This equalizes the outcome and makes the journey the object of study.

For each network at the moment it crosses the threshold, compute:

**1. Time to threshold** — how many epochs / gradient steps did it take? This is sample efficiency under the gradient mix.

**2. Gross path length** — total absolute parameter displacement accumulated over training: `Σ_t Σ_i |Δθ_i(t)|`. The total distance the model walked through parameter space.

**3. Net displacement** — the straight-line distance from initialization to final position: `||θ_final - θ_init||`. Where it ended up, regardless of the path taken.

**4. Efficiency Ratio** — net displacement / gross path length. The fraction of movement that was *useful* — not wasted on oscillation, churn, or corrected mistakes. Ranges from 0 (pure noise) to 1 (perfectly geodesic).

**5. Structured-head contribution to gross path length** — of the total distance walked, what fraction was driven by gradients from the structured auxiliary heads vs. the main output head? This is the Gradient Budget decomposition applied to the efficiency numerator.

### The Core Question

**Do networks where more of the gross path length is attributable to structured-head gradients have higher Efficiency Ratios?**

In other words: are the structure-seeking updates the *directed* ones — the ones that reduce waste, pull the model toward the attractor, and make the path more geodesic? And are the main-head updates more responsible for churn?

If yes, this would mean structured intermediate supervision doesn't just help performance — it **changes the geometry of the learning trajectory**. The structured heads are not merely adding signal; they are acting as attractors that bend the path of parameter movement toward the low-entropy solution.

### Expected Results and What They Would Mean

| Outcome | Interpretation |
|---|---|
| Higher structured-head share → higher Efficiency Ratio | Structure-seeking updates are the geodesic force. Confirms the attractor hypothesis. |
| Higher structured-head share → faster time to threshold | Structure supervision accelerates convergence by reducing wasted movement. |
| 0% structured network has lowest efficiency but eventually reaches threshold | Unguided networks find the attractor eventually — but via a longer, more tortured path. Confirms memorization-as-scenic-route. |
| 100% structured network has high efficiency but slow or poor performance | Over-constraining intermediate representations may prevent the trunk from finding its own compositional solutions. Upper bound on useful supervision. |

The most interesting result would be a **non-monotone relationship** — efficiency peaks at some intermediate gradient share, not at 100% structured. This would suggest that the optimal training regime is one where structured heads provide directional pull but leave the trunk enough freedom to compose representations in its own way.

### Connection to Known Literature

- **Grokking:** The 0% structured network on a rule-generated dataset is precisely the grokking setup. The experiment predicts it will have the lowest Efficiency Ratio and the longest path to threshold.
- **Auxiliary losses and representation learning:** Related to multi-task learning and intermediate supervision work, but reframed: rather than asking "does auxiliary loss improve final performance," we ask "does it change the *geometry of the path* to equivalent performance?"
- **The Simon/Kunin program:** This experiment is an empirical operationalization of the rule-space → representation-space mapping question, with a specific intervention (gradient share) and a specific geometric measurement (efficiency ratio). It produces a curve across the full family of networks, not a single finding.

### Implementation Notes (TTTN)

**What is already built and directly usable:**

- `BranchTrainer<TrunkNet, HeadBranch<TapIndex, HeadNet, HeadLoss>...>` is the exact object needed. The multi-head tapped architecture is fully implemented, including compile-time enforcement of non-decreasing tap indices and static assertion that each head's `InputTensor` matches the trunk activation tensor at its tap site.
- `HeadBranch<TapIndex, HeadNet, HeadLoss>` supports `NoLoss` to silence any head — useful for ablations where you want the head present architecturally but contributing zero gradient.
- Independent per-head learning rates are already supported via the `head_lrs` array in `BranchTrainer::Fit` and `BatchFit`. The gradient share experiment maps directly onto this: varying `head_lrs[i]` relative to `trunk_lr` controls the effective budget share.
- Each head's `TrainableTensorNetwork` carries its own `AdamState`. The trunk carries its own. Optimizer states are already cleanly separated by source — the right foundation for per-source budget tracking.
- `BackwardRange<Lo, Hi>` is the exact primitive for threading gradients through specific trunk segments. The `backward_heads` chain already uses this to route each head's gradient contribution through the correct trunk slice.
- `BlockSequence::ForwardAll` and `BatchedForwardAll` return full `Activations` objects exposing every intermediate activation — tap sites are already first-class objects.

**What needs to be added for the metrics:**

The core issue: `BranchTrainer::Fit` currently accumulates all head gradients additively into the trunk's parameter grad buffers via the `backward_heads` chain, then calls `Update`. By the time `Update` fires, attribution is lost — you cannot tell which portion of a parameter's accumulated gradient came from the trunk loss vs. head H1 vs. head H2.

The fix is **separate backward passes per source before summation**, with parameter delta snapshots taken between each. Concretely:

1. **Before any backward pass:** snapshot all trunk parameter values into `θ_before`.
2. **Run trunk backward pass alone** (trunk loss only, no head contributions), snapshot the resulting grad buffer → `g_trunk`.
3. **Run each head's backward pass alone** (one at a time, zeroing grad between), snapshot → `g_head[i]`.
4. **Sum the grads** and call `Update` once (preserving Adam's unified moment estimates).
5. **After Update:** snapshot `θ_after`. Compute:
   - Signed delta: `θ_after - θ_before` → net displacement this step
   - Per-source attribution: `g_source / Σ g_sources` → fractional gradient budget share
   - Absolute delta per source: `|g_source|` → contribution to gross path length

This requires a new instrumented training loop alongside (not replacing) the existing `Fit`/`BatchFit` — call it `InstrumentedFit`. It accumulates four running tensors over training:
- `gross_path` — running sum of `|Δθ|` per step (total distance walked)
- `net_displacement` — running sum of signed `Δθ` per step (net vector traveled)
- `gross_by_source[i]` — gross path length attributable to source `i`
- `net_by_source[i]` — signed displacement attributable to source `i`

Efficiency Ratio at any checkpoint: `||net_displacement|| / gross_path` (scalar). Per-source efficiency: `||net_by_source[i]|| / gross_by_source[i]`.

- The fixed-accuracy stopping criterion requires a validation loop with early termination — add a `RunUntilAccuracy<Loss, Batch>(target_acc, max_epochs)` method to `BranchTrainer` that calls `InstrumentedFit` per epoch and checks a validation pass. Returns the full metric history at the stopping point.
- The full family (5+ gradient mix levels × N seeds for variance) is computationally feasible given `p=113` → 12,769 examples total dataset size.

---

## VII. Dataset Design & Architecture — Modular Arithmetic as the Starting Point

### Why Modular Arithmetic

The ideal dataset for this experiment has four properties: (1) the data generation rules are known analytically, (2) the intermediate representations useful for solving the task are derivable from first principles, (3) the compositionality is deep enough to have meaningful intermediate stages, and (4) the grokking phenomenon is known to occur, giving a baseline and a literature to situate results against.

Modular arithmetic satisfies all four. The task is `(a + b) mod p → c` for a prime `p`. The full dataset is all `p²` ordered pairs — exhaustive, no sampling. For `p = 113` (the modulus used in the original grokking paper) this is 12,769 examples. Tiny, fast, and fully controlled.

### Input / Output Format

**Inputs:** Two values `a, b ∈ {0, ..., p-1}`, represented either as one-hot vectors of length `p` (concatenated to length `2p`) or as learned embeddings. Learned embeddings are preferred — this is where the interesting Fourier geometry emerges and where the auxiliary heads have the most to supervise.

**Output:** One-hot over `p` classes. Cross-entropy loss on the main head.

### The Known Intermediate Structure

The Nanda et al. mechanistic interpretability paper (*Progress measures for grokking via mechanistic interpretability*, 2023) reverse-engineered what the network actually learns when it generalizes. The discovered algorithm is:

1. **Embed `a` and `b` as points on a circle.** The network learns Fourier features — representing each integer as `(cos(2πka/p), sin(2πka/p))` for a small set of key frequencies `k`. This is the natural representation for modular arithmetic because addition in the original domain becomes angle addition on the circle.

2. **Add the angles.** Because `cos(2π(a+b)/p) = cos(2πa/p + 2πb/p)`, the combination step is just rotation composition — the network is doing trigonometric addition.

3. **Read off `(a+b) mod p`** from the resulting angle via the output embedding.

This means the intermediate representations are **analytically prescribed**, not guessed:

- After early layers: Fourier features of `a` and `b` separately
- After middle layers: Fourier features of the *unwrapped sum* `a + b` (before modular reduction)
- At the output: the modular result

The unwrapped sum is the key intermediate object. In the example `8 + 11 mod 3`: the unwrapped sum is `19`, but the supervision target is not the integer `19` — it is the **Fourier encoding of 19** at the key frequencies, i.e. `(cos(2π·19/p), sin(2π·19/p))` for each `k`. This is a regression target, not a classification target, which means auxiliary head losses are MSE rather than cross-entropy.

### Auxiliary Head Design

Four heads, branching from different depths of the trunk:

| Head | Branch point | Supervision target | Loss |
|---|---|---|---|
| H1 | After embedding / layer 1 | Fourier features of `a` | MSE |
| H2 | After embedding / layer 1 | Fourier features of `b` | MSE |
| H3 | After layer 2 (middle) | Fourier features of unwrapped sum `a+b` | MSE |
| H4 (main) | End of trunk | `(a+b) mod p` | Cross-entropy |

H1 and H2 can share a branch point — they are grading the same layer for two different things. H3 sits deeper, supervising the combination step. H4 is the standard task output.

The Fourier features to supervise are not all `p` frequencies — just the small subset of *key frequencies* that the network actually uses, which Nanda et al. identify empirically. Using all frequencies would over-constrain the representation. Using the key frequencies is prescribing the right inductive bias without being dictatorial about the rest.

### Architecture — Two Phases

**Phase 1: MLP (start here)**

A two- or three-layer MLP taking concatenated embeddings of `(a, b)` as input. No attention, no sequence — just `embed(a) ⊕ embed(b) → hidden layers → output`. This is the cleanest possible setting for the efficiency metrics: no attention geometry to complicate the parameter space story, unambiguous layer ordering for the leverage premium, and clear head branch points.

The Fourier features still emerge in this setting — they show up in the embedding weights and MLP activations. The MLP result is the controlled, interpretable baseline.

**Phase 2: Small Transformer (after MLP results are clean)**

Input sequence: `[a, op, b, =]` — four tokens, predict the result token. One or two layer transformer with a small number of attention heads. This is the architecture used in the original grokking paper and Nanda et al.

The transformer setting adds a new question: does structured head gradient share change *attention head specialization*? The Nanda et al. paper shows that in the unguided setting, specific attention heads learn to compute the Fourier combination. Does structured supervision cause this specialization to emerge earlier, more cleanly, or in a different head configuration? This connects directly to the mech-interp / learning mechanics interface that Simon et al. identify as a key frontier.

The transformer result is more complex to interpret but more exciting and more connected to the broader literature. It is also the natural bridge to replicating the experiment on more realistic architectures.

### Schedule Variants

Beyond static gradient budget shares, the scheduling dimension is its own experimental axis:

- **Structure-first, task-fine-tune:** Heavy structured head share early (carve the attractor basin during the chaotic initial phase), taper to pure task loss. Hypothesis: most efficient overall — structured heads do the geometric work when the landscape is most malleable.
- **Task-first, structure-late:** Pure task loss early, structured supervision introduced after partial convergence. Does late structural grading correct a wandering network, or is it too late to reshape the trajectory?
- **Accuracy-gated annealing:** Structured head share decays as a function of *validation accuracy* rather than epoch. The heads loosen their grip exactly as the network demonstrates internalization of the relevant representations. Most principled schedule — supervision is removed when it is no longer needed.
- **Warm oscillation:** Alternating emphasis on a cycle, analogous to the teacher-forcing schedule used in the compiler Enc-Dec project. Does periodic structural correction produce a different path geometry than monotone decay?

Each schedule produces a distinct Efficiency Ratio curve over training time. Comparing those curves across schedules — holding final accuracy fixed — is itself a rich result, independent of the static budget experiments.

### Key Reference

Nanda et al., *Progress measures for grokking via mechanistic interpretability* (2023). Read before building — it specifies exactly which Fourier frequencies are the key ones, what the attention heads are doing, and what clean generalization looks like in representation space. Your auxiliary head targets should be derived directly from this paper's findings.

---

## VIII. Open Questions

1. Can "path length" and "action" in parameter space be defined in an architecture-agnostic, meaningful way — or does it always depend on the parameterization?
2. Is there a natural coordinate system in parameter space where the attractor basins of structured solutions become geometrically legible?
3. What is the relationship between Efficiency Ratio and generalization gap? Does high efficiency during training predict robustness?
4. Can Displacement Attribution at the example level recover something like "which training examples taught the model the rule" — a mechanistic version of data attribution?
5. How does the leverage premium in the Gradient Budget interact with skip connections and attention — where the "early/late" distinction is less clean?
6. Is there a non-monotone optimum in the gradient share curve — an intermediate level of structured supervision that maximizes efficiency without over-constraining the trunk's representational freedom?
7. Does the Efficiency Ratio during training predict final generalization, independent of accuracy? Could it serve as an early-stopping signal for *quality* of learning, not just quantity?
