# Visualizing the Parameter Landscape

*A guide to rendering learning as movement in network space*

---

## The Core Idea

Every metric in the framework produces a scalar per parameter. The natural visualization of a scalar-per-parameter is a heatmap. But a flat heatmap indexed by parameter number throws away the most important information:
*where in the network that parameter lives*, what it connects to, and what computational role it plays.

The goal is to render the network as it actually is — a graph of operations and weights in 3D space, like a protein structure — and then drape every metric over that geometry as a dynamic color field. Not a list of parameters. Not a layer-by-layer bar chart. The actual topology, lit up.

---

## The Network as a 3D Object

### What "topology" means here

A
`TrainableTensorNetwork` is a directed acyclic graph. Nodes are operations (attention heads, MLP layers, embedding tables, layer norms, projection matrices). Edges are tensor flows between them. Parameters live
*on* nodes — they are the weights that define what a node does.

The 3D layout encodes:

- **Depth (z-axis):** position in the forward pass — input at bottom, output at top
- **Layer spread (x-axis):
  ** within a block, different operations fanned out horizontally (attention, MLP, residual, norm)
- **Width (y-axis):
  ** for multi-head attention, individual heads spread along y; for MLP layers, neuron groups spread along y

This gives you a structure that looks like a transformer *as it
is* — not a schematic, but a spatially embedded computational object. Blocks stack vertically. Heads fan out laterally. Residual connections are visible as skip-edges bypassing blocks. The embedding table sits at the bottom like a wide flat slab. The output projection sits at the top.

### Parameter geometry within nodes

Each node (e.g. an attention head, an MLP layer) is itself a small 3D object. Its parameters are embedded within it — weight matrix elements laid out as a grid, bias vectors as a row. The node's spatial extent is proportional to its parameter count. A large embedding table is a big slab; a small output projection is a thin plate.

This is the "protein" view: the network has secondary structure (blocks), tertiary structure (how blocks connect), and primary structure (individual weight elements within each block).

---

## The Metrics as Color Fields

Each metric produces a different coloring of the same 3D object. The visualization is a single interactive scene where you toggle which metric is driving the color.

### Base: Gross Path `G_i` and Net Displacement `D_i`

**Color:** `G_i` mapped to brightness — brighter = more total movement.
`D_i` mapped to hue — positive net displacement (moved "forward" from init) in warm tones, negative in cool tones.

**What you see:
** Which parameters actually moved during training, and in what direction. Parameters that moved a lot but in no consistent direction (high
`G_i`, low
`|D_i|`) will be bright but desaturated. Parameters that moved purposefully will be bright and strongly colored.

**Interesting view:
** Animate over training time. Watch the movement spread through the network — which parameters start moving first? Which stay frozen longest? Is there a wave of activation that propagates from output toward input as the network finds structure?

---

### Per-Parameter Efficiency `η_i = |D_i| / G_i`

**Color:** Cool-to-warm colormap, 0 (pure churn, blue) to 1 (geodesic, red).

**What you see:** A heatmap of
*waste* across the network. Blue regions are parameters that oscillated — large gross path, tiny net displacement. Red regions moved with purpose.

**The grokking transition:
** At the moment of grokking, you expect a discontinuous shift — certain parameter groups snap from blue (searching) to red (locked onto the structured solution and moving coherently toward it). This snap, rendered spatially, would be one of the most striking frames in the animation. You'd see which
*part* of the network grokks first.

**Per source:
** Toggle between "efficiency attributable to structured heads" and "efficiency attributable to task loss." The hypothesis: structured head gradients produce spatially coherent red regions in early/high-leverage layers. Task-only gradients produce diffuse, lower-efficiency coloring.

---

### Metric I: Positional Leverage `λ_i^pos`

**Color:
** Fixed (does not change during training). Deep purple at output layer (low leverage), bright yellow-green at input/embedding layer (maximum leverage).

**What you see:
** The structural skeleton of influence. This is the network's "bones" — the architecture's intrinsic gradient of potential influence, before any learning has occurred.

**Use:
** This view never animates. It's the reference frame against which all other metrics are compared. Keep it visible as a semi-transparent underlay while other metrics are foregrounded.

---

### Metric II: Instantaneous Functional Influence `λ_i^fn(t)`

**Color:** Black (zero influence) to bright white (maximum Jacobian norm), with intermediate values in gold.

**What you see:** Which parameters are currently *doing
work* — actively bending the output function. This is the network's functional activity map, analogous to a brain scan showing which regions are firing.

**The vanishing gradient paradox made visible:** Early layers will often be dark (low
`λ^fn`) despite being yellow-green in the positional leverage view. The contrast between these two colorings is the vanishing gradient problem, rendered spatially.

**Snapshot cadence:
** Computed every N epochs (expensive — p backward passes). Render as a flip-book of snapshots rather than smooth animation. Each flip shows how the functional activity map has reorganized.

---

### Metric III: Structural Potential `λ_i^str`

**Color:** Computed once via Monte Carlo over K random initializations. Same black-to-white-gold scale as Metric II.

**What you see:** The architectural prior on influence — what the network
*expects* each parameter to do, averaged over all possible weight configurations. This is not a training artifact. It is a property of the architecture itself.

**The interesting comparison:
** Render Metric III and Metric II side by side (or as a split-view of the same 3D object). Metric III is the prior; Metric II at any training checkpoint is the posterior. The difference is what learning has done to the functional geometry.

**What it looks like structurally:** For a transformer, you'd expect to see:

- The output projection and final layer norm: bright (short path to output, high structural potential)
- Middle MLP layers: medium
- Early embedding parameters: darker (long chain of nonlinearities between them and the output)
- Attention query/key weights: interesting — depends heavily on depth and head index
- Layer norm parameters: probably surprisingly bright, since they sit on the residual stream at every block

Seeing the actual shape of
`λ^str` across a real transformer architecture would itself be a novel and publishable figure.

---

### Realized-to-Potential Ratio `ρ_i(t) = λ_i^fn(t) / λ_i^str`

**Color:** Diverging colormap centered at 1. Blue: `ρ ≪ 1` (suppressed — structural potential not realized). White:
`ρ ≈ 1` (realized matches prior). Red: `ρ ≫ 1` (amplified — punching above architectural weight).

**What you see:
** Where the network has diverged from its architectural prior. Red parameters are being actively recruited beyond their structural expectation. Blue parameters are being suppressed — their potential leverage is being choked off by the learned weight configuration downstream.

**The central research visualization:** Animate
`ρ_i(t)` over training, decomposed by gradient source. The hypothesis: structured head gradients selectively redden early, high-leverage parameters (raising
`ρ` where
`λ^str` is high), while task-only gradients produce a more diffuse or output-biased pattern. If this is visible in the 3D render, it is a direct spatial confirmation that structured supervision is the geodesic force.

---

### Activations (Bonus: Dynamic Field)

**Color:
** Signed activation magnitude at each node, for a given input example. Positive activations warm, negative cool, near-zero transparent.

**What you see:
** The forward pass as a wave of activation propagating through the 3D network object. For a specific input
`(a, b)` from the modular arithmetic dataset, you watch the signal travel from the embedding slab at the bottom, through attention blocks, up through the MLP layers, arriving at the output projection.

**Combined view:** Overlay activation magnitude (transparency/brightness) with
`η_i` (hue). You see simultaneously: which parameters are active for this input, and which of those active parameters have been moving efficiently during training. Parameters that are both active and efficient are the network's "load-bearing" elements for this example.

---

## Implementation Path

### Phase 1: Parameter metadata extraction

At network construction time, emit a structured metadata object mapping every parameter element to:

- `(block_index, layer_type, position_within_layer, head_index_if_applicable)`
- `λ_i^pos` (computed from graph topology)
- `λ_i^str` (computed once via Monte Carlo precomputation)

This metadata is the coordinate system for all subsequent visualization. TTTN's type-indexed `BlockSequence` and
`all_params()` traversal make this tractable — the compile-time structure encodes the topology.

### Phase 2: 3D layout engine

Map the parameter metadata to 3D coordinates:

- Block depth → z
- Operation type within block → x offset (attention left, MLP right, norm center)
- Head index / neuron group → y
- Parameter element within weight matrix → fine-grained position within the node's spatial extent

This is essentially a graph layout problem with strong structural constraints. The transformer's regularity (repeated identical blocks) makes it tractable — lay out one block, then tile vertically.

### Phase 3: Metric overlay

For each metric, produce a per-parameter scalar array in the same order as
`all_params()`. Map to color. Render as a colored point cloud or volumetric object over the 3D layout.

**Libraries:
** Three.js for the web-based interactive version (point clouds, instanced geometry, animation). Python/matplotlib or Plotly for quick static snapshots during development. The interactive version is the target.

### Phase 4: Animation

For time-varying metrics (`G_i`, `D_i`, `η_i`, `λ_i^fn`,
`ρ_i`): store per-checkpoint scalar arrays during training. The visualization scrubs through checkpoints, re-coloring the fixed 3D geometry. The network's shape never changes — only its coloring does. This is the key design decision: the topology is the stable reference frame, and learning is visible as a
*changing color field on a fixed object*.

---

## The Gallery

| View | What you're seeing |
|---|---|
| `G_i` animated | The odometer — where did the network walk? |
| `η_i` at grokking transition | The snap — where does coherent movement crystallize? |
| `λ^pos` static | The bones — architectural gradient of potential |
| `λ^str` static | The prior — expected influence by position |
| `λ^fn` flip-book | The activity scan — what's doing work right now? |
| `ρ_i` animated by source | The recruitment map — what is each training objective doing to the functional geometry? |
| Activation overlay | The forward pass as a wave through the protein |
| `η_i` split by source | Which gradient source moves which part of the network, and how efficiently? |

---

## The Image

A transformer, rendered as a 3D object. Embedding table: a wide flat slab at the bottom, colored by
`λ^str` — medium gold, structurally important but not maximally so. Attention blocks stacking upward: heads fanning out along y, each head a small dense cube. The residual stream is visible as bright vertical lines connecting blocks, bypassing the attention and MLP operations. MLP sublayers: wider, denser blocks flanking each attention block. Layer norms: thin bright slices between operations, surprisingly high
`λ^str`. Output projection: a thin bright plate at the top, the highest `λ^str` in the network.

Then you press play. The color field shifts.
`ρ_i` starts uniform (white — realized matches prior, because training hasn't started). Over the first epochs, patches of blue appear in the early layers (vanishing gradients suppressing realization of structural potential). Then, as the network approaches grokking, a wave of red propagates
*downward* — from the output toward the embedding layer — as the Fourier algorithm crystallizes and early parameters are recruited to encode circular representations. That wave is learning, made visible as geometry.