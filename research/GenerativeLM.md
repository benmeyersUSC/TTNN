# Generative Language Model — Plan

## Vision

Build a decoder-only language model entirely within TTTN — no external libraries, no borrowed
weights, trained on real data on local hardware. End goal: a talkable English LM, extended with
RL-with-verifiable-reward to produce a math-reasoning model.

---

## Phase 1 — Decoder-Only Implementation + TinyStories

**Next code milestone.**

### Architecture

- Decoder-only transformer (causal self-attention, no cross-attention)
- Tied input/output embeddings (same weight matrix for token embed and logit projection)
- Causal mask baked into attention — upper triangle = -inf before softmax
- Template shape target: ~30M params to start
  - ~8-10 layers, 512 dim, 8 heads, 2048 FFN, 512 context length

### KV Cache (Inference Only)

- Training uses teacher forcing with a static sequence length — no dynamic cache needed,
  same kernel shapes as current encoder/decoder training
- Inference gets its own `KVCache<Heads, HeadDim>` struct backed by `std::vector`
  - Grows one step at a time; pre-allocated to max context
  - Attention at step `t` reads positions `0..t-1` from cache, writes position `t`
- The rigid templated kernels stay untouched; dynamicity is isolated to the cache struct
- `Generate(prompt_tokens, max_new_tokens, cache)` as the inference entry point

### Dataset

**TinyStories** — Eldan & Li 2023

- ~475MB, ~475M characters, simple English stories
- Purpose: fast pipeline validation — days to first coherent output, not weeks
- Preprocessing: run through `BytePairTokenizer` to build a domain vocab, serialize to binary
  subsets in the format `DataIO` already expects
- Target vocab size: 8192–16384 (BPE from TinyStories corpus)

### Success Criterion

Coherent multi-sentence English generation from a short prompt, decoded greedily or with
top-p sampling. The model should feel like it knows what a sentence is.

---

## Phase 2 — Scale to Real English

Once Phase 1 pipeline is validated:

- **Model**: 50–85M params (10-12 layers, 768 dim, 12 heads, 3072 FFN, 1024 context)
- **Dataset**: OpenWebText (~8GB processed, ~9B tokens) — the open reconstruction of GPT-2's
  actual training corpus. Diverse, clean, real web text.
- **Vocab**: retrain BPE on OpenWebText slice, 32768–65536 tokens
- **Training**: same CE teacher-forcing loop as Phase 1, longer ramp schedules, larger subsets
- **Goal**: a genuinely talkable general English LM running on local metal

Realistic timeline on Apple Silicon: weeks to months depending on subset size and batch throughput.

---

## Phase 3 — Math Reasoning via RL with Verifiable Reward

After a working generative base exists:

- **Dataset**: GSM8K — 8.5K grade school math word problems, final answer always an integer
- **Reward signal**: `reward = (extracted_answer == ground_truth)` — no human labels, no reward
  model, just a verifier. This is the DeepSeek-R1-Zero approach.
- **Method**: the existing RL infrastructure in `TransformerTrainer` maps directly onto this.
  Generate K candidate solutions per problem, score each, compute advantages, run policy gradient.
- **Why this works**: verifiable reward eliminates label noise; GSM8K answers are unambiguous;
  grade school math is nontrivial enough that improvement is meaningful but tractable enough
  that a small model can learn.

This is the most ambitious step and potentially the most interesting result — a math reasoning
model trained entirely from scratch on local hardware with no external frameworks.

---

## Key Design Principles (Carry Forward)

- **Rigid kernels, dynamic shell**: the templated compute kernels stay fixed-shape and maximally
  optimized. KV cache and generation loops are the only dynamic surfaces.
- **BPE as first-class citizen**: `BytePairTokenizer` handles all corpus preprocessing;
  the trained vocab feeds directly into the model's `VocabSize` template parameter.
- **RL as a second stage**: CE teacher forcing builds the base; RL refines toward a specific
  capability. Mixed CE+RL (as in NeuralCompiler) is also on the table.
- **No libraries**: this is the whole point.