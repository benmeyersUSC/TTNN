# The Hot-Swap API Boundary Protocol
## Testing Strict Abstraction Barriers in Neural Networks via Surrogate Module Grafting

### 1. The Core Concept
The overarching hypothesis is that deep neural networks, through gradient descent, naturally factorize complex tasks into specialized sub-circuits. More importantly, we hypothesize that these sub-circuits are bounded by strict **API boundaries**—geometric and linear abstraction barriers that allow the rest of the network to treat the module as a black-box functional tool.

The **Hot-Swap API Boundary Protocol** is a proposed experiment to definitively prove the existence of these internal abstraction barriers. It involves extracting a known competent sub-circuit, mapping its input/output (I/O) manifold, training a structurally distinct *surrogate* module offline to replicate this I/O mapping, and hot-swapping the surrogate back into the frozen network. If the network successfully resumes its task without retraining the outer layers, it proves the network relies strictly on the geometric contract (the API) rather than entangled structural dependencies.

### 2. Motivation
* **Mechanistic Interpretability:** Inspired by the rigorous reverse-engineering of toy models (e.g., Neel Nanda's modular addition and double-angle trigonometric algorithms). We know networks learn beautiful, generalized algorithms. But we must test how modular and encapsulated these learned algorithms truly are.
* **Biological Competency (Michael Levin):** Biological systems exhibit scale-free competency, where higher-level tissues treat lower-level cells as goal-seeking black boxes, communicating via bioelectric/chemical "APIs." Deep learning models may exhibit a similar "morphogenetic" competency in weight space, where surrounding layers build translation pipelines around specialized modules.
* **Functional Isomorphism:** If a neural network only cares about the geometric mapping of a space rather than the path taken to compute it, we can separate *what* a network knows from *how* it stores it.

### 3. The Experimental Protocol

#### Phase 1: Identification & Grafting
1. Identify a model trained on a specific task (e.g., modular arithmetic, algorithmic reasoning).
2. Isolate a specific module (e.g., an Attention head + MLP combination) known to perform a generalizable subroutine (like calculating a half-angle).
3. Freeze the model's weights.

#### Phase 2: Offline I/O Manifold Generation
1. Pass a large, diverse dataset through the frozen, original network.
2. At the identified API boundary, record the exact activation vectors *entering* the module ($X_{in}$) and the exact activation vectors *exiting* the module ($Y_{out}$).
3. This creates a purely functional, localized dataset: $D = \{ (X_{in}^{(i)}, Y_{out}^{(i)}) \}_{i=1}^N$.

#### Phase 3: Surrogate Distillation
1. Initialize a **surrogate module** with a *completely different architecture* than the original module (e.g., swapping a Transformer block for a State Space Model, a Convolutional block, or a deeper, narrower MLP).
2. Train this surrogate offline using the $D$ dataset to perfectly replicate the geometric I/O mapping of the original module. 
3. Because the surrogate is trained strictly on this localized mapping, it learns to fulfill the exact API contract required by the rest of the network.

#### Phase 4: The Hot-Swap
1. Excise the original module from the frozen main network.
2. Stitch the newly trained surrogate module into the gap.
3. Keep the entire network's weights frozen.

#### Phase 5: Evaluation
1. Run the modified network on the global task.
2. **Success Metric:** If the network's global performance remains unperturbed, we have proven the existence of a strict API boundary. The outer network is blind to the internal mechanical overhaul of the module, proving the module acts purely as a composable, abstract function.

### 4. Broad Implications

* **Algorithmic Hard-Coding (The Holy Grail of Efficiency):** If API boundaries hold, we can locate computationally expensive, black-box modules (like massive MLP layers in LLMs), reverse-engineer their algorithms, and hot-swap them with perfect, $O(1)$ or $O(n)$ hand-written Python/C++ scripts. The network would dynamically query standard code libraries for subroutines.
* **Network Surgery & Stitching:** We could build Frankenstein models, dynamically mixing and matching optimized modules from different architectures without full end-to-end retraining, simply by training linear translation layers to satisfy the localized API contracts.
* **Causal Proof of Modularity:** This moves interpretability from "observational" (looking at weights and activations) to "causal" (performing structural interventions and observing system resilience).

---

### 5. Claude's read — running this on the nanda grokker (session notes, 2026-07-07)

**The boundary that matters is the MLP.** X_in = post-attention residual at ReadPos
(128-dim, carrying the half-angle points across the S planes); Y_out = the MLP's additive
contribution. The MLP's job is angle-*doubling* per plane — (x,y)→(x²−y², 2xy), complex
squaring — not literally ×2, but the point stands: **the true function is tiny** (S
independent quadratic gears in a 128-dim shell), so it has *many* legal surrogates.

**Two things make it a real experiment, not a tautology:**
1. **The margin is thin.** Our readout data says one k is correct while five are confident —
   the interference win is small. So the sharp question isn't "can a surrogate fit the MLP"
   (of course, MSE→0), it's **how much surrogate I/O error the frozen readout tolerates
   before accuracy collapses.** That's a resilience curve, and it's measurable. That's the
   paper.
2. **The contract includes phase/basis.** Distilling on recorded (X_in, Y_out) bakes in the
   trained net's basis, so a stitched surrogate *should* work — the interesting failure is a
   surrogate that fits in MSE but drifts in the directions the readout is most sensitive to.

**The cheap version needs no new training run.** If v3's snapshots dumped the **MLP-boundary
acts** (post-attention *and* post-MLP at ReadPos), the whole test is offline Python on the
grokked checkpoint:
- fit surrogate g ≈ MLP on (X_in, Y_out),
- recompute `logits = Unembed(X_in + g(X_in))`, measure val accuracy,
- sweep surrogate families and watch accuracy vs I/O error.

**Surrogate ladder (increasingly "not a network"):**
- linear layer → **should fail** (task is quadratic) — the control that proves the
  nonlinearity is load-bearing;
- degree-2 polynomial features (outer products) → should nail it cheaply;
- different-shape distilled MLP (narrow/deep) → Levin-style structural swap;
- **the hand-written O(1) circle-squarer** — project onto the known k-planes, apply
  (x²−y², 2xy), re-embed. If *that* stitches into the frozen net at full accuracy, you've
  hot-swapped a learned MLP for ~20 lines of C++. That's the holy-grail bullet in §4, made
  concrete on a model we fully own.

**Gating question before any of this:** does v3 actually snapshot the residual *at the MLP
boundary* (in and out), or only the final readout acts + logits? If in/out are there →
offline test today. If only final acts → we need one instrumented eval pass over all 12769
pairs on a grokked checkpoint (a forward-only dump, still no *training*).
