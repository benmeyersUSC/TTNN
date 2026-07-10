# KV Cache & Autoregressive Decode — design brief

Goal: a clean TTTN abstraction for incremental autoregressive inference — generate one
token at a time, reusing cached K/V instead of re-running the full prefix through
attention every step. Training is untouched (teacher forcing, static shapes, existing
backward). This is inference-only machinery: no grads, no Params, no Adam.

## Why (the arithmetic)

Naive generation re-runs the whole prefix per new token: step t costs O(t²) attention →
O(T³) total. With a cache, step t computes only q_t, k_t, v_t and attends q_t against
the stored K[0..t] → O(t) per step, O(T²) total. The cache IS the prefix's attention
state; everything non-attention (embed, LN, FFN, unembed) is position-wise and needs no
memory at all.

## The philosophical tension (and why it dissolves)

TTTN says shape is type. Generation length is runtime. But notice what the cache stores:
**values, not structure**. The resolution that keeps the faith:

> **Capacity is type; occupancy is value.**

Two candidate embodiments — decide by feel + benchmark:
1. **Typed arena**: `Tensor<MaxLen, Heads, HeadDim>` per layer, compile-time MaxLen,
   runtime cursor `len`. Fully static types, zero allocation during decode, wasted memory
   past the cursor. Softmax/dots run bounded loops `[0, len)`.
2. **Dynamic vector** (your instinct): `std::vector`-backed rows, `reserve(MaxLen)`,
   grown per step. Honest about dynamism; types cover the per-step slice
   (`Tensor<Heads, HeadDim>`), the time axis is a container.
Either way the *per-step* objects are fully typed — the sequence axis is the only thing
that escapes the type system, and it escapes into a cursor or a container, not into
shape-unsafe code.

## The precedent already in the codebase

`TrainingCache<Batch>` is the pattern: caller-owned, typed, per-block scratch threaded
through a `BlockSequence` as a tuple, empty for blocks that don't need it. The decode
cache is its sibling:

```cpp
template<Block... Blocks> struct DecodeCache;   // tuple of per-block caches
// attention blocks: KV storage; Dense/LN/FFN/Residual: std::tuple<> (position-wise)
```

Keep the cache **user-owned and passed in** (`StepForward(x_t, cache) const`) — nets stay
const during inference, state stays explicit, same style as DataCursor/RLState.

## The new verb

Blocks currently speak `Forward<Batch>(X)` over full sequences. Incremental decode needs:

```cpp
// per block: consume ONE position, update cache, emit one position
Tensor<EmbDim> StepForward(const Tensor<EmbDim>& x_t, BlockCache& c) const;
```

- Position-wise blocks (Dense, MapDense, LN, FFN, Residual wrappers): trivial — apply the
  same weights to one row; empty cache. Consider a default/CRTP so they get StepForward
  for free.
- **MultiHeadAttentionBlock (Masked=true only — static_assert it)**: compute q_t,k_t,v_t
  from x_t; append k_t,v_t; scores = q_t·K[0..t] (per head); dynamic-length softmax
  (runtime loop — do NOT shoehorn the static axis-typed Softmax); out = Σ α_i·V[i]; then O.
- `BlockSequence::StepForward` chains blocks, threading the cache tuple — mirror of
  `forward_training_impl`.
- Top level: `Generate(prompt, max_new, cache, sampler)` — prefill phase (run prompt
  through step-forward token by token, or batched prefill later), then the decode loop.
  Sampler as a functor parameter (argmax / temperature / top-k), same spirit as Loss
  functors; RNG passed in, not owned.

## Correctness invariants (build the test FIRST)

1. **The golden test**: for random weights and a random sequence, full masked
   `Forward` on the prefix and incremental `StepForward` replay must produce
   **identical logits at every position** (elementwise |Δ| < 1e-5). This single test
   catches 90% of cache bugs. Automate it over several lengths.
2. **Absolute positions**: the positional embedding for step t must use cache position t,
   not "position 0 of this call." The cache owns the position counter.
3. **No cross-position statistics in the step path**: LN must be position-wise (it is —
   verify), and the only sequence-axis op should be the attention softmax over the cache.
4. Numerical drift: cached decode sums in a different order than full forward — expect
   1e-6-ish float divergence, not exactness; set the tolerance accordingly.

## Performance thoughts

- Memory layout for the hot loop: q_t·K^T wants K contiguous along HeadDim (minor axis)
  per head — `[Heads][T][HeadDim]` grown along T. Same for the α-weighted V sum.
  Benchmark against the interleaved `[T][Heads][HeadDim]` append-friendly layout.
- `reserve(MaxLen)` once; zero allocations inside the decode loop.
- Batch=1 first. Batched generation = ragged sequences = a different project; defer.
- Sanity benchmark: tokens/sec cached vs full-recompute at T = 128/256/512 — the win
  should scale ~linearly with T.

## Milestones

1. Golden equivalence harness (tiny causal net, full-vs-replay).
2. `KVCache` + `MultiHeadAttentionBlock::StepForward` (masked path only).
3. `DecodeCache` tuple + `BlockSequence::StepForward` + `TrainableTensorNetwork` passthrough.
4. `Generate(prompt, max_new, cache, sampler)`.
5. Benchmark + README documentation (dual-sig discipline: `// @doc:` lines + README prose).
6. Later: `EncoderDecoder` — cross-attention K/V computed **once** from the encoder
   (static, even easier than self-attention), self-attention cache in the decoder;
   ProCC/NeuralCompiler can then drop the fixed 384-step full-recompute decode.

## Open questions to sit with

- Arena vs vector (or: vector now, arena if the allocator ever shows up in a profile)?
- Does `StepForward` belong on the Block concept itself (making it wider) or as a
  separate `AutoregressiveBlock` concept that only some blocks satisfy — and
  `DecodeCache` only exists for nets whose blocks all satisfy it? (Concepts, not
  inheritance — the second feels more TTTN.)
- Sliding-window / ring-buffer eviction: skip until a real need exists.
- Who owns sampling temperature schedules — Generate's caller, always.
