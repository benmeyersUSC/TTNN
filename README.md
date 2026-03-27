# TTTN : Trainable Template Tensor Network

A header-only C++20 neural network library built around a fully type-safe, arbitrary-rank
`Tensor` type. Network topology, tensor shapes, and layer connectivity are all encoded at compile time.

Used in my [AlphaToe](https://github.com/benmeyersUSC/AlphaToe) library.

**Namespace:** `TTTN`
**Umbrella include:** `#include "src/TTTN.hpp"`

---

## Templated Tensors

- This library treats
  `Tensor` as a statically-shaped, dimension-typed functor over scalar values, paired with deeply generalized `Zip`,
  `Map`, and `Reduce` operations. Leveraging **C++** templates and the type system, we enforce shape correctness at compile time while enabling aggressive precomputation of traversal structure. Runtime execution becomes a planned walk over constant topologies - fully fused, vectorizable, and allocation-free.

- The dimension-typed `Tensor` unlocks a unified
  `Contraction` abstraction, parameterized by three orthogonal components:
    - `Align`: how two `Tensor`s are brought into correspondence (which elements meet)
    - `Map`: how aligned elements interact (`Map :: (T, T) -> T`)
    - `Reduce`: how mapped results are aggregated along contracted axes (`Reduce :: (T, T) -> T`)

- Under this formulation, a contraction is:
    - `Contraction = Reduce ∘ zipWith(Map) ∘ Align`

- Classical tensor operations such as `Einsum`, `ΣΠ` (generalized *sum of products*), `Matmul`, and
  `Dot` arise as specializations of this pattern. By choosing different `Align`, `Map`, and
  `Reduce` components, one can express and efficiently execute a wide range of computations (from linear algebra to activation functions, loss functions, and full training pipelines) within a single fused kernel.

- This decomposition mirrors the algebraic structure of `Functor` (`Map`), `Applicative` (`zipWith`), and `Foldable` (
  `Reduce`), ensuring that fusion is not an optimization trick, but a consequence of the underlying laws.

## Demonstration

- Given
    - `A ∈ Tensor<M, K>`
    - `B ∈ Tensor<K, N>`
- We want:
    - `C ∈ Tensor<M, N>`
- `Contraction`:
    - `Align`:
        - Match the last axis of `A` (`K`) with the first axis of `B` (`K`)
        - Each output element, `C[i,j]` sees row `i` of `A` and column `j` of `B`
    - `Map`:
        - Multiply elementwise:
            - `map(a, b) = a * b`
    - `Reduce`:
        - Reduce with summation along `K`:
            - `reduce(acc, x) = acc + x`
    - Result
        - `C[i, j] = Σ_k (A[i, k] * B[k, j])`
- In [`TTTN`](src/TTTN.hpp), we have:
  ```cpp
    auto C = InnerContract<N, Mul, Add>(A, B);  // Map=Mul, Reduce=Add
    ```

    - Different contractions on the same type:
        - **Dot product / Frobenius inner product**
          ```cpp
          float sim = Collapse<Mul, Add>(A, B); // Collapse helper: contract all axes
          ```
        - **L1 Distance**
          ```cpp
          float dist = Collapse<AbsDiff, Add>(A, B); // Collapse with AbsDiff -> Add for L1
          ```
        - **Max product** (no identity for Max over products — use lambda form)
          ```cpp
          auto C = InnerContract<1>(A, B,
              -std::numeric_limits<float>::infinity(),
              Mul{},
              Max{});
          ```

---

### Visualizing Contraction (Matrix Multiply)

We align the shared axis `K`, then map + reduce:

            A (M×K)                     B (K×N)
        
               k →                        k ↓
        ┌───────────────┐         ┌────────────────┐
      i │ aᵢ₀ aᵢ₁ … aᵢₖ │          │ b₀ⱼ b₁ⱼ … bₖⱼ  │ j
        └───────────────┘         └────────────────┘
                │                   │
                └─────── zip ───────┘
                          │
                       map(a, b)
                          │       
            [ aᵢ₀·b₀ⱼ, aᵢ₁·b₁ⱼ, ... aᵢₖ·bₖⱼ]
                          │
                       reduce(+)
                          │
                       C[i,j]
            
               Result: C ∈ Tensor<M, N>

**Shape rules the day**:

    A: Tensor<M, K>
    B: Tensor<K, N>
    
    Align:   A[..., k] ↔ B[k, ...]
    Map:     f(a, b)
    Reduce:  ⊕ over k
    
    C[i,j] = ⊕ₖ f(A[i,k], B[k,j])

## Future Directions

### Interpretability API

This is where shape-as-type pays its most interesting dividends.

The goal is a general **`BrainSaladSurgery`** protocol: any block or network that opts in
returns a structured snapshot of its internal activations, formatted and labeled for
downstream exploration.  Think of it as a typed, self-describing introspection packet —
not a raw dump, but something with enough compile-time shape information baked in that a
visualization tool can consume it without guessing layout.

The attention pattern work already in `main` is the proof of concept.  The head-weight
matrices, the pre-softmax scores, the value-weighted outputs — all of those are already
`Tensor<...>` objects with shapes fully known at compile time.  The step is to make that
exposure **formal and uniform**: define a concept (`BrainSaladProvider` or similar) and
require that conforming blocks expose a `peek()` method returning a `BrainSaladSurgery`
struct whose members are themselves typed tensors.  No shape ambiguity, no runtime
reinterpretation — the graphical tool on the other end knows exactly what it is receiving
because the type says so.

#### Cross-Network Comparison

The shape-as-type model opens another powerful avenue: **comparing networks of the same
type**.  Because topology is encoded in the C++ type, the compiler enforces that two
networks being compared are actually structurally identical.  This makes the following
operations trivially safe to express:

- **Same random seed, different data direction** — train the same architecture on
  transposed or permuted data, then compare learned representations layer by layer.
  Because both networks are `SomeNet<...>` with identical template parameters, a
  `compare(net_a, net_b)` function can zip their weight tensors together without any
  runtime shape checking.

- **Checkpoint-to-checkpoint drift** — save a typed snapshot at each epoch; compare
  corresponding weight tensors across training time using Frobenius cosine similarity or
  any other metric.  The type guarantees you are comparing the same layer in the same
  position, not accidentally swapping heads or layers.

- **Head alignment across runs** — for attention-based networks, identify which heads in
  run A correspond most closely (by value similarity or by what they attend to) to heads
  in run B.  With shape-as-type, the per-head slices are well-typed objects and can be
  passed directly into alignment routines.

The common thread: **the type system is doing the bookkeeping** that in most frameworks
requires careful string-matching on layer names or fragile index arithmetic.

### SDL Visualization Plugin

A companion C++ graphics library (in progress) will consume `BrainSaladSurgery` packets
and render them live.  Planned views:

- Rank-1 tensors as bar charts or histograms
- Rank-2 tensors as heat maps (weight matrices, attention score grids)
- Rank-3 tensors as stacked slices or volume renders
- Network topology graphs with live activation overlays during a forward pass

Because the visualization library will also be typed against the same `Tensor<...>`
template, it can validate at compile time that the tensors it receives are within its
renderable rank range — no runtime surprises when you accidentally pass a rank-5 weight
blob to a heat-map view.

### GPU Backend

Replace the parallel CPU dispatch (`std::execution::par_unseq`) with CUDA or Metal
kernels. The contraction and elementwise op layers are the natural insertion point;
higher-level code does not need to change because the `Tensor` API stays the same. **Apple AMX** is currently used for a specialization of `Contract` where `Map=Mul` and `Reduce=Add`, but other operations could still benefit further from GPUs. 

---

## Table of Contents

1. [TensorPrereqs.hpp: Compile-Time Fundamentals](#tensorprereqshpp--compile-time-fundamentals)
2. [TensorStorage.hpp: Storage Policy](#tensorstoragehpp--storage-policy)
3. [Tensor.hpp: The Foundational Object](#tensorhpp--the-foundational-object)
4. [TensorShapeOps.hpp: Tensor Type Algebra](#tensorshapeopshpp--tensor-type-algebra)
5. [TensorOps.hpp: Op Tags and Element-wise Primitives](#tensoropshpp--op-tags-and-element-wise-primitives)
6. [TensorFunctions.hpp: Functional Helpers](#tensorfunctionshpp--functional-helpers)
7. [TensorContract.hpp: Contraction](#tensorcontracthpp--contraction)
8. [TensorReduce.hpp: Reduction and Broadcast](#tensorreducehpp--reduction-and-broadcast)
9. [TensorUtil.hpp: Tensor Layer Umbrella](#tensorutilhpp--tensor-layer-umbrella)
10. [TTTN_ML.hpp: ML Primitives](#tttn_mlhpp--ml-primitives)
11. [Dense.hpp: Fully-Connected Layer](#densehpp--fully-connected-layer)
12. [Attention.hpp: Multi-Head Self-Attention](#attentionhpp--multi-head-self-attention)
13. [ChainBlock.hpp: Sequential Block Composition](#chainblockhpp--sequential-block-composition)
14. [NetworkUtil.hpp: Concepts, Types, and Utilities](#networkutilhpp--concepts-types-and-utilities)
15. [Params.hpp: Parameter Storage and Optimizer](#paramshpp--parameter-storage-and-optimizer)
16. [TrainableTensorNetwork.hpp: The Network](#trainabletensornetworkhpp--the-network)
17. [Snapshot.hpp: Activation Snapshots](#snapshothpp--activation-snapshots)
18. [DataIO.hpp: Data Loading and Batching](#dataiohpp--data-loading-and-batching)

---

## [TensorPrereqs.hpp](src/TensorPrereqs.hpp): Compile-Time Fundamentals

Concepts, compile-time dimension arithmetic, stride computation, and the parallel loop helper. Everything `Tensor.hpp` needs before it can define the `Tensor` class itself.

- ***TensorDimsProduct*** — [`struct TensorDimsProduct<size_t... Ds>`](src/TensorPrereqs.hpp)
    - Template recursion to store product of `size_t...` variadic in `static constexpr size_t value` 

- ***SizeTemplateGet*** — [`struct SizeTemplateGet<size_t N, size_t... Ds>`](src/TensorPrereqs.hpp)
    - Template recursion to store `N`-th element of `size_t...` variadic in `static constexpr size_t value`

- ***ComputeStrides*** — [`struct ComputeStrides<size_t... Ds>`](src/TensorPrereqs.hpp)
    - Template recursion to store `Tensor<Ds...>` stride array in `static constexpr std::array<size_t, sizeof...(Ds)> value`

- ***ParForEach*** — [`template<std::invocable<size_t> F> void ParForEach(size_t n, F f)`](src/TensorPrereqs.hpp)
    - Helper function used throughout library to run `std::invocable<size_t> F` `n` times using `std::execution::par_unseq` policy

- ***FloatUnaryOp*** — [`template<typename F> concept FloatUnaryOp`](src/TensorPrereqs.hpp)
    - `concept` to enforce `F :: float -> float` operations on `Tensor`s

- ***FloatBinaryOp*** — [`template<typename F> concept FloatBinaryOp`](src/TensorPrereqs.hpp)
    - `concept` to enforce `F :: float -> float -> float` operations on two `Tensor`s


---

## [TensorStorage.hpp](src/TensorStorage.hpp): Storage Policy

Two specializations of `TensorStorage<S, bool Small>` selected at compile time: inline `alignas(64) float[S]` for small tensors (≤ 16 elements) and 64-byte aligned heap allocation for large ones. `Tensor` owns one instance as `storage_`.

We 64-byte-align here so `Tensor`s' data (when grabbed to cache from RAM) starts at beginning of cache. Also makes vectorization optimizations and specialized `cblas_sgemm` call in `TensorContract.hpp` as fast as possible. 

- ***TensorStorage*** — [`template<size_t S, bool Small = (S <= 16)> struct TensorStorage`](src/TensorStorage.hpp)
    - Struct wrapper for storage of `Tensor`'s `float[]`
    - `Tensor` owns one instance as `storage_`
    - Specialized for `bool Small = true` (inline 64-byte-aligned stack `float[]`) and `bool Small = false` (64-byte-aligned heap `float[]`)

- ***TensorStorage (STO)*** — [`template<size_t S> struct TensorStorage<S, true>`](src/TensorStorage.hpp)
    - Specialization for *small `Tensor` optimization* (**STO**)
    - Member array is defined as: `alignas(64) float data[S]{}`

- ***TensorStorage (heap)*** — [`template<size_t S> struct TensorStorage<S, false>`](src/TensorStorage.hpp)
    - Specialization for larger `Tensor` to be allocated on the heap at a 64-byte aligned address
    - Member array is defined as: `std::unique_ptr<float[], AlignedDeleter> heap_`




---

## [Tensor.hpp](src/Tensor.hpp): The Foundational Object

The impetus and core data structure of the library is the variadic-dimension-templated `Tensor<size_t...Dims>`. In
`TTTN`, ***shapes*** are ***types*** and ***types*** denote ***shapes***. At compile-time, a
`Tensor` is declared and its `std::tuple<size_t>` of ***dimensions*** (shape), ***rank***, and ***size*** are all known.
`Tensor`s' data are all stored in flat heap arrays; to index these efficiently and in accordance with rank, we also compute a
***strides***
vector (row-major) for the type at compile-time which compose in a dot-product with the N-rank index provided to locate the requested index in the flat array.

### Compile-Time Shape Metaprogramming

- ***TensorDimsProduct*** — [`struct TensorDimsProduct<size_t... Ds>`](src/Tensor.hpp)
    - Template-specialization-based recursion to collapse variadic template `<size_t...Ds>` into single
      `size_t`, stored statically as `TensorDimsProduct<size_t...Ds>::value`
    - Used to compute `Tensor::Shape`, used in [`struct ComputeStrides<size_t... Ds>`](src/Tensor.hpp) to compute
      `Tensor::Strides`


- ***ComputeStrides*** — [`struct ComputeStrides<size_t... Ds>`](src/Tensor.hpp)
    - Template-specialization-based recursion to compute `Tensor::Strides` array
        - The `Tensor::Strides` array is vital to mapping from indices into
          `Tensor::Shape` to flat indices for the backing array
        - In general, for a `Tensor` with `Tensor::Shape = [A, B, ..., N]`, its
          `Tensor::Strides = [A * B * ... * N, B * ... * N, ..., 1]`
    - Specialize for `<>` and `<size_t D>`
        - `value = ` `[]` and `[1]`, respectively
    - Specialize for `<size_t First, size_t Second, size_t... Rest>`
        - recursively compute `tail = ComputeStrides<Second, Rest...>::value`
        - `value[0] = TensorDimsProduct<Second, Rest...>::value`
        - `value[i] = tail[i + 1]`

### `class Tensor<size_t... Dims>`

The primary data structure. `Dims...` encodes the full shape at compile time, informs statics `Shape`, `Rank`, `Size`,
`Strides`, and several static functions.

**Static members:**

- ***Rank*** — [`static constexpr size_t Rank`](src/Tensor.hpp)
    - Number of dimensions
- ***Size*** — [`static constexpr size_t Size`](src/Tensor.hpp)
    - Product of all dimensions = total distinct values in `Tensor`
- ***Shape*** — [`static constexpr std::array<size_t, Rank> Shape`](src/Tensor.hpp)
    - `<size_t... Dims>` captured into an array
- ***Strides*** — [`static constexpr std::array<size_t, Rank> Strides`](src/Tensor.hpp)
    - Uses `ComputeStrides` to create array
    - The `Tensor::Strides` array is vital to mapping from indices into
      `Tensor::Shape` to flat indices for the backing array
    - In general, for a `Tensor` with `Tensor::Shape = [A, B, ..., N]`, its
      `Tensor::Strides = [A * B * ... * N, B * ... * N, ..., 1]`

**Static methods:**

- ***FlatToMulti*** — [`static constexpr std::array<size_t, Rank> FlatToMulti(const size_t flat)`](src/Tensor.hpp)
    - Inverse of `MultiToFlat`; map a flat index `[0, Size)` to its `Rank`-term index
    - Pattern matches a `std::index_sequence` parameterized by `<size_t... Rank>` (=
      `[0, ..., Rank]`) and unpacks into an array:
      `[(flat / Strides[0]) % Shape[0], ..., (flat / Strides[Rank]) % Shape[Rank]]`

- ***MultiToFlat*** — [`static constexpr size_t MultiToFlat(const std::array<size_t, Rank> &multi)`](src/Tensor.hpp)
    - Inverse of `FlatToMulti`; map a `Rank`-term index to its flat index `[0, Size)`
    - Pattern matches a `std::index_sequence` parameterized by `<size_t... Rank>` (= `[0, ..., Rank]`) and
      `+`-folds terms `multi[0] * Strides[0] + ... + multi[Rank] * Strides[Rank]`
    - Dot product of given `multi` index and `Strides`

**Constructors / Rule of Five:**

- ***Tensor*** — [`Tensor()`](src/Tensor.hpp)
    - Default constructor
    - Initialize `float[Size]` on the heap

- ***Tensor*** — [`Tensor(std::initializer_list<float> init)`](src/Tensor.hpp)
    - Initializer list constructor
    - Fill the first `Size` elements of `std::initializer_list<float> init` to flat indices of backing array

- ***Tensor*** — [`~Tensor() = default`](src/Tensor.hpp)
    - Default destructor
    - RAII: destructs `std::unique_ptr` to `float[Size]` on heap

- ***Tensor*** — [`Tensor(const Tensor& other)`](src/Tensor.hpp)
    - Deep copy constructor
    - Allocate new `float[Size]` on heap and `std::memcpy` from `other.data()`

- ***operator=*** — [`Tensor& operator=(const Tensor& other)`](src/Tensor.hpp)
    - Deep copy assignment operator
    - `std::memcpy` from `other.data()`

- ***Tensor*** — [`Tensor(Tensor&&) noexcept = default`](src/Tensor.hpp)
    - Default move constructor
    - `std::unique_ptr` to data handles this already

- ***operator=*** — [`Tensor &operator=(Tensor&&) noexcept = default`](src/Tensor.hpp)
    - Default move assigment operator
    - `std::unique_ptr` to data handles this already

**Data access:**

- ***fill*** — [`void fill(const float v) const`](src/Tensor.hpp)
    - Fill a `Tensor`'s underlying array with some `float v`

- ***data*** — [`float* data()`](src/Tensor.hpp)
    - Returns pointer to `Tensor`'s underlying array

- ***data*** — [`const float* data() const`](src/Tensor.hpp)
    - Returns `const` pointer to `Tensor`'s underlying array

- ***flat*** — [`float& flat(size_t idx)`](src/Tensor.hpp)
    - Returns `float&` reference to item at `idx` in underlying array

- ***flat*** — [`float flat(size_t idx) const`](src/Tensor.hpp)
    - Returns `const float&` reference to item at `idx` in underlying array

**Functional transform primitives:**

- ***zip*** — [`template<FloatBinaryOp F> Tensor zip(const Tensor &other, F f) const`](src/Tensor.hpp)
    - Use `std::execution::par_unseq` to `std::transform` two `Tensor`s' underlying data by `float -> float -> float`
      map `f`, returning a new `Tensor`

- ***apply*** — [`template<typename F> void apply(F f)`](src/Tensor.hpp)
    - Use `std::execution::par_unseq` + `std::for_each` to apply `float -> float` map `f` to
      `Tensor`'s underlying data in-place

- ***map*** - [`template<FloatUnaryOp F> Tensor map(F f) const`](src/Tensor.hpp)
    - Copy-constructs `Tensor result` from `*this`, calls `apply`, returns copy 

- ***zip_apply*** — [`template<FloatBinaryOp F> void zip_apply(const Tensor &other, F f)`](src/Tensor.hpp)
    - In-place binary transform: `self[i] = f(self[i], other[i])` for all `i`. Mutating counterpart to `zip`. Accepts any `FloatBinaryOp` including op tags: `zip_apply(b, Add{})`

**Indexing operators:**

- ***operator()*** — [`float& operator()(Indices... idxs)`](src/Tensor.hpp)
    - Variadic multi-index access, returns reference
    - Uses compile-time-templated `MultiToFlat` for efficient access

- ***operator()*** — [`float operator()(Indices... idxs) const`](src/Tensor.hpp)
    - Variadic multi-index access, returns copy
    - Uses compile-time-templated `MultiToFlat` for efficient access

- ***operator()*** — [`float& operator()(const std::array<size_t, Rank>& multi)`](src/Tensor.hpp)
    - Array-based multi-index access, returns reference
    - Uses compile-time-templated `MultiToFlat` for efficient access

- ***operator()*** — [`float operator()(const std::array<size_t, Rank>& multi) const`](src/Tensor.hpp)
    - Array-based multi-index access, returns copy
    - Uses compile-time-templated `MultiToFlat` for efficient access

**Serialization:**

- ***Save*** — [`void Save(std::ofstream& f) const`](src/Tensor.hpp)
    - Writes entirety of flat backing array (`float[Size]`) to binary file

- ***Load*** — [`void Load(std::ifstream& f)`](src/Tensor.hpp)
    - Reads binary file into flat backing array (`float[Size]`)

---

### Type Traits / Concepts

- ***is_tensor*** — [`struct is_tensor<T>`](src/Tensor.hpp)
    - SFINAE type traits for verifying that a type is a `Tensor`
        - Specialize `<size_t...Dims>`: matches into `<Tensor<Dims...>`, inherits from `std::true_type`
        - Specialize <>: inherits from `std::false_type`, backup when former fails


- ***IsTensor*** — [`concept IsTensor<T>`](src/Tensor.hpp)
    - Wrapper `concept` around `is_tensor` type trait, satisfied if `T` is a `Tensor`

- **`FloatUnaryOp`** — `std::regular_invocable<F, float>` with return type `float`. Used by `Tensor::map`, `Map`.
- **`FloatBinaryOp`** — `std::regular_invocable<F, float, float>` with return type `float`. Used by `ReduceApply`,
  `BroadcastApply`, `BroadcastReduce`, `InnerContract`, `Contract`, `Collapse`, `TensorIndexApply`.

---

---

### Tensor Type Algebra

Shape-only metaprogramming. No data, no runtime — purely compile-time type-level operations on
`Tensor` dimension packs. Available to any code that includes `Tensor.hpp`.


### Tensor Demonstration

```cpp
// shapes are types: different shapes are different types
Tensor<3, 4> mat;                                  // 3x4 matrix
Tensor<2, 3, 4> cube;                              // rank-3 tensor

// compile-time statics
static_assert(mat.GetRank() == 2);
static_assert(mat.GetSize() == 12);
static_assert(mat.GetShape()[0] == 3);
static_assert(mat.GetStrides()[0] == 4);     // row-major: stride[0] = cols

// multi-index and flat access (same element)
mat(1, 2) = 9.27f;
mat.flat(6) == 9.27f;                              // 1*4 + 2 = 6

// index conversion
auto multi = Tensor<3,4>::FlatToMulti(6);          // -> [1, 2]
size_t f = Tensor<3,4>::MultiToFlat({1, 2});       // -> 6

// initializer list
Tensor<3> vec = {1.0f, 2.0f, 3.0f};

// functional transforms (all parallel under the hood)
auto tripled = vec.map([](float x){ return 3.0f * x; });
auto summed  = vec.zip(tripled, [](float a, float b){ return a + b; });
vec.apply([](float x){ return -x; });              // in-place

```

---

## [TensorShapeOps.hpp](src/TensorShapeOps.hpp): Tensor Type Algebra

Shape-only metaprogramming — no data, no runtime. Purely compile-time type-level operations on `Tensor` dimension packs: concatenation, axis removal/insertion, slicing, and permutation type computation.

- ***TensorConcat*** — [`struct TensorConcat<typename T1, typename T2>`](src/TensorShapeOps.hpp)
    - #########

- ***ArrayToTensor*** — [`struct ArrayToTensor<typename KeptIdxs, typename Iota>`](src/TensorShapeOps.hpp)
    - #########

- ***RemoveAxis*** — [`struct RemoveAxis<size_t Skip, size_t... Dims>`](src/TensorShapeOps.hpp)
    - #########

- ***PermutedTensorType*** — [`struct PermutedTensorType<typename T, size_t... Perm>`](src/TensorShapeOps.hpp)
    - #########

---

## [TensorOps.hpp](src/TensorOps.hpp): Op Tags and Element-wise Primitives

Operation tags, element-wise operations, arithmetic operators, and `Permute`. Base layer included by `TensorContract.hpp` and `TensorReduce.hpp`.

### Operation Tags

Default-constructible callable structs satisfying `FloatBinaryOp` or
`FloatUnaryOp`. Pass as type template parameters instead of lambdas — one template instantiation per
`(Axis, Op, Dims...)`, zero runtime overhead, fully visible to the optimizer. Monoid tags (those with
`identity`) unlock the no-init overloads of `ReduceApply`, `BroadcastReduce`, etc.

**Binary tags:**

| Tag | Expression | `identity` |
|-----|-----------|------------|
| `Add` | `a + b` | `0.f` |
| `Mul` | `a * b` | `1.f` |
| `Max` | `std::max(a, b)` | `-∞` |
| `Sub` | `a - b` | — |
| `Div` | `a / b` | — |
| `AbsDiff` | `std::abs(a - b)` | — |

**Unary tags** — used with `Map<Op>` or `tensor.map(Op{})`:

| Tag | Expression |
|-----|-----------|
| `Log` | `std::log(x)` |
| `Exp` | `std::exp(x)` |
| `Neg` | `-x` |
| `Sq` | `x * x` |
| `Abs` | `std::abs(x)` |
| `OneMinus` | `1.f - x` |
| `Clamp<Lo, Hi>` | `std::min(std::max(x, Lo), Hi)` — float NTTPs; `Hi` defaults to `+∞` for one-sided use |
| `Step<T>` | `x < T ? 1.f : 0.f` — float NTTP threshold; useful for counting elements below a threshold |

- ***Compose*** - [`template<typename F, typename G> struct Compose`](src/TensorOps.hpp)
    - Chain application of `FloatUnaryOp` with either another `FloatUnaryOp` or a `FloatBinaryOp`
    - Struct has two specialized overloads of `operator()` for the two following cases:
        - Unary ∘ Unary → Unary:  `Compose<Log, Abs>{}(x) == log(|x|)`
        - Unary ∘ Binary → Binary: `Compose<Exp, Sub>{}(a, b) == exp(a - b)`

---

### Arithmetic Operators

- ***operator+*** — [
  `template<size_t... Dims> Tensor<Dims...> operator+(const Tensor<Dims...>& a, const Tensor<Dims...>& b)`](src/TensorOps.hpp)
    - Element-wise add, uses parallel functional `zip`

- ***operator-*** — [
  `template<size_t... Dims> Tensor<Dims...> operator-(const Tensor<Dims...>& a, const Tensor<Dims...>& b)`](src/TensorOps.hpp)
    - Element-wise subtract, uses parallel functional `zip`

- **operator*** — [
  `template<size_t... Dims> Tensor<Dims...> operator*(const Tensor<Dims...>& a, float s)`](src/TensorOps.hpp)
    - Scalar multiply, uses parallel functional `map`

- **operator*** — [
  `template<size_t... Dims> Tensor<Dims...> operator*(float s, const Tensor<Dims...>& a)`](src/TensorOps.hpp)
    - Scalar multiply, uses parallel functional `map`

- **operator*** — [
  `template<size_t... Dims> Tensor<Dims...> operator*(const Tensor<Dims...>& a, const Tensor<Dims...>& b)`](src/TensorOps.hpp)
    - Hadamard (element-wise) product, uses parallel functional `zip`

- ***operator/*** — [
  `template<size_t... Dims> Tensor<Dims...> operator/(const Tensor<Dims...>& a, const Tensor<Dims...>& b)`](src/TensorOps.hpp)
    - Element-wise) division, uses parallel functional `zip`

```cpp
Tensor<3, 4> A, B;
Tensor<3, 4> C = A + B;                            // element-wise add
Tensor<3, 4> D = A * B;                            // Hadamard product — NOT matmul
Tensor<3, 4> E = A * 2.0f;                         // scalar scale
A += B;                                            // in-place accumulate

// every operator preserves shape — and REQUIRES matching shape:
// A + Tensor<4, 3>{};                             // ✗ won't compile: Tensor<3,4> + Tensor<4,3>
```

---

### Permutation

- ***MoveToLastPerm*** — [`struct MoveToLastPerm<size_t Src, size_t Rank>`](src/TensorContract.hpp)
    - Compile-time helper to rearrange `Tensor`'s shape such that `Src` is at index
      `[Rank - 1]` and all others are kept in order

- ***MoveToFirstPerm*** — [`struct MoveToFirstPerm<size_t Src, size_t Rank>`](src/TensorContract.hpp)
    - Compile-time helper to rearrange `Tensor`'s shape such that `Src` is at index
      `[0]` and all others are kept in order

- ***PermuteFromHolder*** — [
  `template<typename PermHolder, size_t... I, size_t... Dims> auto PermuteFromHolder(const Tensor<Dims...>& t, std::index_sequence<I...>)`](src/TensorContract.hpp)
    - Unpack a `constexpr` permutation indices array into a proper `Permute`-given `Tensor` type
    - `PermHolder` is an array of permutation indices, typically the result of `MoveToLastPerm` or `MoveToFirstPerm`
    - Call with `PermHolder` as template arg, `Tensor<Dims...>` as first arg,
      `std::make_index_sequence<sizeof...(Dims)>{}` as second arg

```cpp
Tensor<3, 4, 5> cube;

// arbitrary axis reordering — the result type is computed at compile time
auto p = Permute<2, 0, 1>(cube);
static_assert(std::is_same_v<decltype(p), Tensor<5, 3, 4>>);

// Transpose reverses all axes
auto t = Transpose(cube);
static_assert(std::is_same_v<decltype(t), Tensor<5, 4, 3>>);

// rank-2 transpose is classical matrix transpose
Tensor<3, 4> W;
auto Wt = Transpose(W);
static_assert(std::is_same_v<decltype(Wt), Tensor<4, 3>>);

// the permutation indices are template args — invalid permutations are compile errors:
// Permute<0, 0, 1>(cube);                         // ✗ repeated axis
// Permute<0, 1>(cube);                            // ✗ wrong number of axes
```

---

## [TensorFunctions.hpp](src/TensorFunctions.hpp): Functional Helpers

Free-function wrappers for the core tensor transforms: `Map`, `MapMove`, `Zip`, `ZipMove`, `Permute`, `Transpose`, `TensorIndex`, and `TensorIndexApply`. Also the canonical home of all op-tag structs (`Add`, `Mul`, `Max`, `Exp`, `Compose`, etc.) — these are the types you pass as template arguments to `BroadcastReduce`, `ReduceApply`, `Map`, and friends.

- ***TensorIndex*** — [
  `template<size_t Axis, size_t... Dims> typename RemoveAxis<Axis, Dims...>::type TensorIndex(const Tensor<Dims...>& src, size_t idx)`](src/TensorFunctions.hpp)
    - #########

---

## [TensorContract.hpp](src/TensorContract.hpp): Contraction

Contraction, named specializations (`ΣΠ`, `Dot`, `Matmul`, `Outer`, `Einsum`), axis-permutation helpers, generalized `Contract`, and `Collapse`. Includes `TensorOps.hpp`.

### Tensor Contraction

Tensor contraction is the unified `Reduce ∘ zipWith(Map) ∘ Align` operation on `Tensor`s. Every named operation —
`ΣΠ`, `Dot`, `Matmul`, `Outer`, `Einsum`, `Collapse` — is a specialization.

- ***ContractionKernel*** — [`template<size_t N, typename TA, typename TB> struct ContractionKernel`](src/TensorContract.hpp)
    - Unified compile-time index kernel. Specialized for `<N, Tensor<ADims...>, Tensor<BDims...>>`.
    - Compile-time `static constexpr`:
        - `RankA`, `RankB` — ranks of `A` and `B`
        - Asserts `N <= RankA && N <= RankB` and last `N` dims of `A` match first `N` dims of `B`
        - `A_Free = TensorSlice<0, RankA-N, ADims...>::type`
        - `B_Free = TensorSlice<N, RankB-N, BDims...>::type`
        - `Contracted = TensorSlice<RankA-N, N, ADims...>::type`
        - `ResultType = TensorConcat<A_Free, B_Free>::type`
        - `struct { std::array<size_t, Contracted::Size> a, b; } offsets` — flat-index offset into `A` and
          `B` for every contracted position; precomputed once per `(N, ADims, BDims)` and shared across all
          `(Map, Reduce)` variants
        - `b_free_size`, `contracted_size` — compile-time constants used by
          `InnerContract` to compute per-output base offsets as `O(1)` arithmetic (
          `base_a = (o / b_free_size) * contracted_size`, `base_b = o % b_free_size`)
          rather than a precomputed table — the compiler strength-reduces these to multiply-shift at `-O2`
    - **These pay real dividends for [TrainableTensorNetwork](./src/TrainableTensorNetwork.hpp) training schedules. Any
      weight `Tensor`'s `Dot`s, `Matmul`s, and `Outer`s (*in forward and backward passes*) are saved structs, and the
      runtime computations are parallelized and vectorized, following known, saved paths**

- ***InnerContract*** — [
  `template<size_t N, size_t... ADims, size_t... BDims, FloatBinaryOp Map, FloatBinaryOp Reduce> auto InnerContract(const Tensor<ADims...>& A, const Tensor<BDims...>& B, float init, Map map, Reduce reduce)`](src/TensorContract.hpp)
    - N-inner-axis contraction with custom `Map` and `Reduce` (lambda form)
    - Aligns the last `N` axes of `A` with the first `N` axes of `B`
    - `result.flat(o) = Reduce_c map(A[A_Free(o), c], B[c, B_Free(o)])`, for `o ∈ [0, ResultType::Size)`
    - **Tag-param overload**: `InnerContract<N, Map, Reduce>(A, B)` — `Reduce::identity` used as init; requires monoid
      `Reduce`

- ***ΣΠ*** / ***SigmaPi*** — [
  `template<size_t N, size_t... ADims, size_t... BDims> auto ΣΠ(const Tensor<ADims...>& A, const Tensor<BDims...>& B)`](src/TensorContract.hpp)
    - `InnerContract<N, Mul, Add>` — classical sum-of-products
    - `result.flat(o) = Σ_c A[A_Free(o), c] * B[c, B_Free(o)]`
    - `SigmaPi<N>(A, B)` is an ASCII alias for `ΣΠ<N>(A, B)`

- ***Einsum*** — [
  `template<size_t I, size_t J, size_t... ADims, size_t... BDims> auto Einsum(const Tensor<ADims...>& A, const Tensor<BDims...>& B)`](src/TensorContract.hpp)
    - `ΣΠ`-contracts over single selected indices `I` and `J` from `A` and `B`, respectively
    - Permutes `A` to move axis `I` last, `B` to move axis `J` first, then calls `ΣΠ<1>`

- ***Dot*** — [`template<size_t N> auto Dot(const Tensor<N>& A, const Tensor<N>& B)`](src/TensorContract.hpp)
    - `ΣΠ<1>` on two `Tensor<N>`s — returns `Tensor<>` (rank-0 scalar)

- ***Matmul*** — [
  `template<size_t M, size_t K, size_t N> auto Matmul(const Tensor<M,K>& A, const Tensor<K,N>& B)`](src/TensorContract.hpp)
    - `ΣΠ<1>` on rank-2 tensors — returns `Tensor<M,N>`

- ***Outer*** — [
  `template<size_t... ADims, size_t... BDims> auto Outer(const Tensor<ADims...>& A, const Tensor<BDims...>& B)`](src/TensorContract.hpp)
    - `ΣΠ<0>`: contract nothing — returns `Tensor<ADims..., BDims...>`

- ***Contract*** — [
  `template<AxisList AAxes, AxisList BAxes, ..., FloatBinaryOp Map, FloatBinaryOp Reduce> auto Contract(const Tensor<ADims...>& A, const Tensor<BDims...>& B, float init, Map map, Reduce reduce)`](src/TensorContract.hpp)
    - Grand-generalized contraction over arbitrary axis sets (lambda form)
    - Permutes `A` and `B` to align the selected axes, then delegates to `InnerContract<N>`
    - **Tag-param overload**: `Contract<AAxes, BAxes, Map, Reduce>(A, B)` — `Reduce::identity` used as init

- ***Collapse*** — [
  `template<size_t... Dims, FloatBinaryOp M, FloatBinaryOp R> float Collapse(const Tensor<Dims...>& A, const Tensor<Dims...>& B, float init, M m, R r)`](src/TensorContract.hpp)
    - Full-rank same-shape scalar reduction: `Reduce_i map(A[i], B[i])`
    - Direct `std::transform_reduce` over flat data — no index tables needed
    - **Tag-param overload**: `Collapse<M, R>(A, B)` — `R::identity` used as init
    - `Collapse<Mul, Add>(A, B)` == Frobenius inner product; `Collapse<AbsDiff, Add>(A, B)` == L1 distance

`InnerContract<N>` is the core primitive. `N` controls how many trailing dims of `A` contract with leading dims of
`B`. Every named operation is a specialization — and every result shape is resolved at compile time from the input shapes.

```cpp
Tensor<3> u, v;
Tensor<3, 4> W;
Tensor<5, 4> M;

// ── dot product: three equivalent spellings ──────────────────────────────────
auto d1 = Dot(u, v);                               // named alias
auto d2 = ΣΠ<1>(u, v);                             // "contract 1 dim"
auto d3 = Einsum<0, 0>(u, v);                      // "contract axis 0 of A with axis 0 of B"
static_assert(std::is_same_v<decltype(d1), Tensor<>>);  // all three → rank-0 scalar
static_assert(std::is_same_v<decltype(d2), Tensor<>>);
static_assert(std::is_same_v<decltype(d3), Tensor<>>);

// ── matmul ───────────────────────────────────────────────────────────────────
auto mm = Matmul(W, Transpose(M));                 // Tensor<3,4> × Tensor<4,5> → Tensor<3,5>
static_assert(std::is_same_v<decltype(mm), Tensor<3, 5>>);

// Einsum picks arbitrary axes — no Transpose needed:
auto d4 = Einsum<1, 1>(W, M);                      // contract axis 1 of W with axis 1 of M
static_assert(std::is_same_v<decltype(d4), Tensor<3, 5>>);  // same result, different path

// ── any-rank generalization ──────────────────────────────────────────────────
Tensor<4> x;
auto Wx = ΣΠ<1>(W, x);                             // Tensor<3,4> × Tensor<4> → Tensor<3>
static_assert(std::is_same_v<decltype(Wx), Tensor<3>>);

Tensor<3, 5, 4> W3;
auto W3x = ΣΠ<1>(W3, x);                           // contract last 1 dim → Tensor<3,5>
static_assert(std::is_same_v<decltype(W3x), Tensor<3, 5>>);

Tensor<5, 4, 2> K;
auto out = ΣΠ<2>(W3, K);                            // contract last 2 of W3 with first 2 of K
static_assert(std::is_same_v<decltype(out), Tensor<3, 2>>);

// ── outer product and full contraction: the two extremes ─────────────────────
auto outer = Outer(u, v);                           // ΣΠ<0>: contract nothing → Tensor<3,3>
static_assert(std::is_same_v<decltype(outer), Tensor<3, 3>>);

float frob = Collapse<Mul, Add>(W, W);              // Frobenius inner product — tag form, no lambdas

// ── what the type system rejects ─────────────────────────────────────────────
// ΣΠ<1>(x, W);                                    // ✗ last dim of Tensor<784> ≠ first dim of Tensor<128,784>
// ΣΠ<1>(W3, W);                                   // ✗ last dim 4 ≠ first dim 3
// Dot(u, Tensor<5>{});                             // ✗ Tensor<3> · Tensor<5> — dimension mismatch
```

---

## [TensorReduce.hpp](src/TensorReduce.hpp): Reduction and Broadcast

Axis-reduction kernel, `ReduceApply`, `Expand`, `BroadcastApply`, `BroadcastReduce`, and indexed slice access. Includes `TensorOps.hpp`.

### Reduction and Broadcast

- ***ReduceKernel*** — [`template<size_t Axis, size_t... Dims> struct ReduceKernel`](src/TensorReduce.hpp)
    - Shared kernel for all axis-reduction and broadcast operations
    - Compile-time `static constexpr`:
        - `axis_dim = SizeTemplateGet<Axis, Dims...>::value`
        - `axis_stride = Source::Strides[Axis]`
        - `std::array<size_t, Result::Size> bases` — flat index in `Source` for each output index with axis set to 0
        - `static constexpr size_t project(size_t i)` — flat index in `Result` for source flat index
          `i` (axis contribution stripped); closed-form `i - ((i / axis_stride) % axis_dim) * axis_stride`; no table,
          `axis_stride` compile-time so division compiles to multiply-shift


- ***Expand*** — [
  `template<size_t Axis, size_t N, size_t... Dims> InsertAxis<Axis, N, Dims...>::type Expand(const Tensor<Dims...>& src)`](src/TensorReduce.hpp)
    - Broadcasts a reduced tensor back up by repeating it `N` times along `Axis`
    - `Expand<0, 5>(Tensor<3>)` → `Tensor<5, 3>` — 5 copies stacked along axis 0

- ***BroadcastApply*** — [
  `template<size_t Axis, FloatBinaryOp F, size_t... Dims> Tensor<Dims...> BroadcastApply(const Tensor<Dims...>& A, const typename RemoveAxis<Axis, Dims...>::type& b, F f)`](src/TensorReduce.hpp)
    - Apply binary `f(a_elem, b_elem)` element-wise between `A` and `b` broadcast along `Axis`
    - **Tag-param overload**: `BroadcastApply<Axis, F>(A, b)` — default-constructs `F`
    - `BroadcastApply<0, Add>(Z, bias)` adds bias to every row

- ***BroadcastReduce*** — [
  `template<size_t Axis, FloatBinaryOp ApplyFn, FloatBinaryOp ReduceFn, size_t... Dims> Tensor<Dims...> BroadcastReduce(const Tensor<Dims...>& src, float init, ApplyFn afn, ReduceFn rfn)`](src/TensorReduce.hpp)
    - Reduce along `Axis` then broadcast the result back with a second op —
      `BroadcastApply<Axis>(src, ReduceApply<Axis>(src, init, rfn), afn)`
    - **Tag-param overload**: `BroadcastReduce<Axis, ApplyOp, ReduceOp>(src)` —
      `ReduceOp::identity` as init; requires monoid `ReduceOp`
    - Powers `Softmax`: `BroadcastReduce<Axis, Compose<Exp,Sub>, Max>(x)` then `BroadcastReduce<Axis, Div, Add>(exps)`

Every operation takes
`Axis` as a compile-time argument. Output shape, stride arithmetic, and projection are all resolved at compile time — the runtime loop is a flat parallel sweep with zero shape logic.

```cpp
Tensor<32, 10> logits;
Tensor<10>     bias;

// ── tag-param reductions — no init arg, no lambda ────────────────────────────
auto col_sum = ReduceApply<0, Add>(logits);         // sum over batch → Tensor<10>
auto row_max = ReduceApply<1, Max>(logits);         // max per sample → Tensor<32>
static_assert(std::is_same_v<decltype(col_sum), Tensor<10>>);
static_assert(std::is_same_v<decltype(row_max), Tensor<32>>);

// ── Expand: dual of reduction ─────────────────────────────────────────────────
auto stacked = Expand<0, 32>(bias);                 // Tensor<10> → Tensor<32, 10>
static_assert(std::is_same_v<decltype(stacked), Tensor<32, 10>>);

// ── tag-param broadcast ───────────────────────────────────────────────────────
auto biased = BroadcastApply<0, Add>(logits, bias); // add bias to every row
auto scaled = BroadcastApply<1, Div>(logits, row_max); // divide each row by its max
static_assert(std::is_same_v<decltype(biased), Tensor<32, 10>>);

// ── BroadcastReduce: reduce + broadcast in one call ───────────────────────────
// tag form — both ops from tags, identity automatic:
auto normed = BroadcastReduce<1, Div, Add>(logits); // divide each element by its row sum

auto centered = BroadcastReduce<1, Compose<Exp, Sub>, Max>(logits); // exp(x - row_max) — numerically stable
static_assert(std::is_same_v<decltype(centered), Tensor<32, 10>>);

// ── higher-rank ───────────────────────────────────────────────────────────────
Tensor<8, 16, 64> activations;
auto per_token = ReduceApply<2, Add>(activations);  // → Tensor<8, 16>
auto restored  = Expand<0, 8>(ReduceApply<0, Max>(activations)); // → Tensor<8, 16, 64>
static_assert(std::is_same_v<decltype(per_token), Tensor<8, 16>>);
static_assert(std::is_same_v<decltype(restored),  Tensor<8, 16, 64>>);
```

---

### Indexed Slice Access

`TensorIndex` and `TensorIndexApply` are the gather/scatter primitives. The axis is compile-time — the compiler knows the slice shape and stride layout. Only the index into that axis is runtime.

```cpp
Tensor<16, 64> seq;                                 // 16 tokens, 64-dim embeddings

// gather: extract a single token's embedding
auto tok5 = TensorIndex<0>(seq, 5);                 // seq[5, :] → Tensor<64>
static_assert(std::is_same_v<decltype(tok5), Tensor<64>>);

// scatter: accumulate a gradient into one token's slot
Tensor<16, 64> grad_seq;
Tensor<64>     grad_tok;
TensorIndexApply<0>(grad_seq, 5, grad_tok,
    [](float a, float b) { return a + b; });        // grad_seq[5, :] += grad_tok[:]

// the op is generic — overwrite instead of accumulate:
TensorIndexApply<0>(seq, 5, grad_tok,
    [](float, float b) { return b; });              // seq[5, :] = grad_tok[:]

// higher-rank: gather along a middle axis
Tensor<8, 16, 64> batch_seq;                        // batch × seq × embed
auto col3 = TensorIndex<1>(batch_seq, 3);           // batch_seq[:, 3, :] → Tensor<8, 64>
static_assert(std::is_same_v<decltype(col3), Tensor<8, 64>>);
```

---

---

### End-to-End: Forward and Backward by Hand

Everything above composes into a complete training step — contractions, reductions, broadcasts, outer products — all with shapes verified at compile time. No runtime shape checks, no asserts, no "expected shape [128] but got [10]" at 3 AM.

```cpp
// ── 2-layer feed-forward network (single sample, raw tensors) ────────────────
Tensor<784>      x;                                 // input: flattened 28×28 image
Tensor<128, 784> W1;  Tensor<128> b1;               // layer 1: 784 → 128
Tensor<10,  128> W2;  Tensor<10>  b2;               // layer 2: 128 → 10
float            lr = 0.01f;

// ── forward ──────────────────────────────────────────────────────────────────
auto z1 = ΣΠ<1>(W1, x) + b1;                       // Tensor<128,784> × Tensor<784> + Tensor<128> → Tensor<128>
auto a1 = z1.map([](float v) {
    return v > 0.f ? v : 0.f; });                   // ReLU → Tensor<128>
auto z2 = ΣΠ<1>(W2, a1) + b2;                      // Tensor<10,128> × Tensor<128> + Tensor<10> → Tensor<10>

// softmax output (two BroadcastReduce calls — stable, parallel, one line each):
auto exps  = BroadcastReduce<0, Compose<Exp, Sub>, Max>(z2);
auto probs = BroadcastReduce<0, Div, Add>(exps);

static_assert(std::is_same_v<decltype(z1), Tensor<128>>);
static_assert(std::is_same_v<decltype(probs), Tensor<10>>);

// ── backward ─────────────────────────────────────────────────────────────────
Tensor<10> target;                                  // one-hot label
auto dz2 = probs.zip(target,                       // softmax + CEL combined gradient = pred − target
    [](float p, float t) { return p - t; });

auto dW2 = Outer(dz2, a1);                         // Tensor<10> ⊗ Tensor<128> → Tensor<10, 128>
auto da1 = ΣΠ<1>(Transpose(W2), dz2);              // Tensor<128,10> × Tensor<10> → Tensor<128>
auto dz1 = da1 * z1.map([](float v) {
    return v > 0.f ? 1.f : 0.f; });                // ⊙ relu' → Tensor<128>
auto dW1 = Outer(dz1, x);                          // Tensor<128> ⊗ Tensor<784> → Tensor<128, 784>

static_assert(std::is_same_v<decltype(dW2), Tensor<10, 128>>);
static_assert(std::is_same_v<decltype(dW1), Tensor<128, 784>>);

W1 += dW1 * lr;   W2 += dW2 * lr;                  // SGD update
b1 += dz1 * lr;   b2 += dz2 * lr;

// ── what the type system rejects ─────────────────────────────────────────────
// every one of these is a compile error — not a runtime crash, not a wrong answer.

// ΣΠ<1>(x, W1);                  // ✗ Tensor<784> vs Tensor<128,784>: 784 ≠ 128
// ΣΠ<1>(W1, x) + b2;             // ✗ Tensor<128> + Tensor<10>: wrong layer's bias
// ΣΠ<1>(W2, dz2);                // ✗ forgot Transpose: 128 ≠ 10
// W2 += Outer(a1, dz2);          // ✗ Tensor<128,10> into Tensor<10,128>: order flipped
// W1 + W2;                       // ✗ Tensor<128,784> + Tensor<10,128>: mixing layers
```

---

## [TensorUtil.hpp](src/TensorUtil.hpp): Tensor Layer Umbrella

Thin include-only header that pulls in `TensorContract.hpp` and `TensorReduce.hpp` together. This is the last of the `Tensor___` headers; `TTTN.hpp` includes `TensorUtil.hpp` transitively via `TrainableTensorNetwork.hpp`. No public declarations — nothing to sentinel here.

---

## [TTTN_ML.hpp](src/TTTN_ML.hpp): ML Primitives

Activation functions, their derivatives, loss functions, and the `SoftmaxBlock` layer. Depends on `TensorContract.hpp` and `TensorReduce.hpp`.

**Constants:**

- ***EPS*** — [`static constexpr float EPS`](src/TTTN_ML.hpp)
  -

### Activation Op Tags

Defined in [TTTN_ML.hpp](src/TTTN_ML.hpp). Each tag satisfies both `FloatUnaryOp` and `ActivationOp`. Use directly with
`Map<Act>(z)` for forward and `Act::prime(a)` for the derivative (in terms of post-activation output).

| Tag | Forward `operator()(x)` | `prime(a)` |
|-----|--------------------------|------------|
| `Linear` | `x` | `1.f` |
| `ReLU` | `x > 0 ? x : 0` | `a > 0 ? 1 : 0` |
| `Sigmoid` | `1 / (1 + exp(-x))` | `a * (1 - a)` |
| `Tanh` | `std::tanh(x)` | `1 - a*a` |

- **`ActivationOp`** — concept: `FloatUnaryOp<T>` + `T::prime(float) -> float`

---

### Free Functions

- ***CrossEntropyLoss*** — [
  `template<size_t N> float CrossEntropyLoss(const Tensor<N>& output, const Tensor<N>& target)`](src/TTTN_ML.hpp)
    -

- ***XavierInitMD*** — [
  `template<size_t... Dims> void XavierInitMD(Tensor<Dims...>& W, const size_t fan_in, const size_t fan_out)`](src/TTTN_ML.hpp)
    -

---

### Softmax (Axis-Generalized)

- ***Softmax*** — [
  `template<size_t Axis, size_t... Dims> Tensor<Dims...> Softmax(const Tensor<Dims...>& x)`](src/TTTN_ML.hpp)
    -

- ***SoftmaxPrime*** — [
  `template<size_t Axis, size_t... Dims> Tensor<Dims...> SoftmaxPrime(const Tensor<Dims...>& grad, const Tensor<Dims...>& a)`](src/TTTN_ML.hpp)
    -

---

### `class SoftmaxBlock<size_t Axis, typename TensorT>`

A shape-preserving, parameter-free block. `all_params()` returns
`std::tuple<>{}` — TTN's bulk helpers become no-ops automatically.

- ***Forward*** — [`OutputTensor Forward(const InputTensor& x) const`](src/TTTN_ML.hpp)
  -

- ***Backward*** — [
  `InputTensor Backward(const OutputTensor& delta_A, const OutputTensor& a, const InputTensor& a_prev)`](src/TTTN_ML.hpp)
    -

- ***BatchedForward*** — [
  `template<size_t Batch> Tensor<Batch, Dims...> BatchedForward(const Tensor<Batch, Dims...>& X) const`](src/TTTN_ML.hpp)
    -

- ***BatchedBackward*** — [
  `template<size_t Batch> Tensor<Batch, Dims...> BatchedBackward(const Tensor<Batch, Dims...>& delta_A, const Tensor<Batch, Dims...>& a, const Tensor<Batch, Dims...>& a_prev)`](src/TTTN_ML.hpp)
    -

- ***all_params*** — [`auto all_params()`](src/TTTN_ML.hpp)
    - Returns `std::tuple<>{}` — no parameters; TTN bulk helpers become no-ops automatically

---

### `struct SoftmaxLayer<size_t Axis>` *(Block recipe)*

- ***Resolve*** — [`template<typename InputT> using Resolve = SoftmaxBlock<Axis, InputT>`](src/TTTN_ML.hpp)
  -

---

### Loss Functions

- ***LossFunction*** — [`concept LossFunction<typename L, typename TensorT>`](src/TTTN_ML.hpp)
  -

#### `struct MSE`

- ***Loss*** — [
  `template<size_t... Dims> static float Loss(const Tensor<Dims...>& pred, const Tensor<Dims...>& target)`](src/TTTN_ML.hpp)
    - #########
- ***Grad*** — [
  `template<size_t... Dims> static Tensor<Dims...> Grad(const Tensor<Dims...>& pred, const Tensor<Dims...>& target)`](src/TTTN_ML.hpp)
    -

#### `struct BinaryCEL`

- ***Loss*** — [
  `template<size_t... Dims> static float Loss(const Tensor<Dims...>& pred, const Tensor<Dims...>& target)`](src/TTTN_ML.hpp)
    -
- ***Grad*** — [
  `template<size_t... Dims> static Tensor<Dims...> Grad(const Tensor<Dims...>& pred, const Tensor<Dims...>& target)`](src/TTTN_ML.hpp)
    -

#### `struct CEL`

- ***Loss*** — [
  `template<size_t... Dims> static float Loss(const Tensor<Dims...>& pred, const Tensor<Dims...>& target)`](src/TTTN_ML.hpp)
    -
- ***Grad*** — [
  `template<size_t... Dims> static Tensor<Dims...> Grad(const Tensor<Dims...>& pred, const Tensor<Dims...>& target)`](src/TTTN_ML.hpp)
    -

#### `BatchAccuracy`

- ***BatchAccuracy*** — [
  `template<size_t Batch, size_t N> float BatchAccuracy(const Tensor<Batch, N>& pred, const Tensor<Batch, N>& labels)`](src/TTTN_ML.hpp)
    - Returns the percentage of correctly classified samples in a batch. `labels` must be one-hot. Correct iff
      `argmax(pred[b]) == argmax(labels[b])`, computed via
      `ReduceApply<1, Add>(pred ⊙ labels)` (probability assigned to the true class) vs
      `ReduceApply<1, Max>(pred)` (highest predicted probability) — no explicit argmax loop required.

---

## [Dense.hpp](src/Dense.hpp): Fully-Connected Layer

Implements the general multidimensional dense layer (`DenseMDBlock`) and its recipe types. Weights have shape
`Tensor<OutDims..., InDims...>`; forward pass is a generalized matrix-vector product via
`ΣΠ`. Includes Adam optimizer state.

- ***WTBlockSwapPerm*** — [`struct WTBlockSwapPerm<size_t N_out, size_t N_in>`](src/Dense.hpp)
  -

---

### `class DenseMDBlock<typename InT, typename OutT, ActivationOp Act_>`

The concrete fully-connected block. `W = Tensor<OutDims..., InDims...>`, `b = Tensor<OutDims...>`.

- ***DenseMDBlock*** — [`DenseMDBlock()`](src/Dense.hpp)
    - Xavier-initializes `W`
- ***Forward*** — [`OutputTensor Forward(const InputTensor& x) const`](src/Dense.hpp)
    - #########
- ***Backward*** — [
  `InputTensor Backward(const OutputTensor& delta_A, const OutputTensor& a, const InputTensor& a_prev)`](src/Dense.hpp)
    - #########
- ***BatchedForward*** — [
  `template<size_t Batch> Tensor<Batch, OutDims...> BatchedForward(const Tensor<Batch, InDims...>& X) const`](src/Dense.hpp)
    -
- ***BatchedBackward*** — [`template<size_t Batch> Tensor<Batch, InDims...> BatchedBackward(...)`](src/Dense.hpp)
  -
- ***all_params*** — [`auto all_params()`](src/Dense.hpp)
    - Returns `std::tie(W_, b_)`; TTN drives `ZeroGrad`, `Update`, `Save`, `Load` from this

---

### `struct DenseMD<typename OutT, ActivationOp Act_>` *(Block recipe)*

- ***Resolve*** — [`template<typename InputT> using Resolve = DenseMDBlock<InputT, OutT, Act_>`](src/Dense.hpp)
  -

### `template<size_t N, ActivationOp Act_> using Dense`

- ***Dense*** — [`using Dense = DenseMD<Tensor<N>, Act_>`](src/Dense.hpp)
    - `Dense<128, ReLU>`, `Dense<10, Sigmoid>`, `Dense<10>` (defaults to `Linear`)

---

## [Attention.hpp](src/Attention.hpp): Multi-Head Self-Attention

Implements scaled dot-product multi-head self-attention over sequences of arbitrary-rank token embeddings. Forward-pass cache is stored as
`mutable` members. All four weight matrices (`W_Q`, `W_K`, `W_V`, `W_O`) are updated with Adam.

### `class MultiHeadAttentionBlock<size_t SeqLen, size_t Heads, size_t... EmbDims>`

`InputTensor = OutputTensor = Tensor<SeqLen, EmbDims...>`. Constraint: `EmbSize % Heads == 0`.

- ***MultiHeadAttentionBlock*** — [`MultiHeadAttentionBlock()`](src/Attention.hpp)
    - Xavier-initializes `WQ`, `WK`, `WV`, `WO`
- ***Forward*** — [`OutputTensor Forward(const InputTensor& X) const`](src/Attention.hpp)
    - #########
- ***Backward*** — [
  `InputTensor Backward(const OutputTensor& delta_A, const OutputTensor& a, const InputTensor& a_prev)`](src/Attention.hpp)
    - #########
- ***BatchedForward*** — [
  `template<size_t Batch> Tensor<Batch, SeqLen, EmbDims...> BatchedForward(...)`](src/Attention.hpp)
    - #########
- ***BatchedBackward*** — [
  `template<size_t Batch> Tensor<Batch, SeqLen, EmbDims...> BatchedBackward(...)`](src/Attention.hpp)
    -
- ***all_params*** — [`auto all_params()`](src/Attention.hpp)
    - Returns `std::tie(WQ_, WK_, WV_, WO_)`; TTN drives `ZeroGrad`, `Update`, `Save`, `Load` from this

---

### `struct TensorFirstDim<typename T>`

- ***value*** — [`static constexpr size_t value`](src/Attention.hpp)
  -

---

### `struct MHAttention<size_t Heads, size_t... EmbDims>` *(Block recipe)*

- ***Resolve*** — [
  `template<typename InputT> using Resolve = MultiHeadAttentionBlock<TensorFirstDim<InputT>::value, Heads, EmbDims...>`](src/Attention.hpp)
    -

---

## [ChainBlock.hpp](src/ChainBlock.hpp): Sequential Block Composition

`ChainBlock<Blocks...>` composes an arbitrary sequence of `ConcreteBlock`s into a single block satisfying the `ConcreteBlock` concept. `InputTensor` is the first block's input; `OutputTensor` is the last block's output. Forward and backward threads through the chain in order; `all_params()` aggregates all sub-block parameters into one tuple.

- ***Forward*** — [`OutputTensor Forward(const InputTensor& x) const`](src/ChainBlock.hpp)
    - #########

---

## [NetworkUtil.hpp](src/NetworkUtil.hpp): Concepts, Types, and Utilities

Defines the two block concepts that gate the type system, the chain-resolution machinery used by
`NetworkBuilder`, and the `ActivationsWrap` safety wrapper.

### Concepts

- ***ConcreteBlock*** — [`concept ConcreteBlock<T>`](src/NetworkUtil.hpp)
    - Requires `InputTensor`, `OutputTensor` (both `IsTensor`), `Forward`, `Backward`, and
      `all_params()` (const + non-const).
    - `Update`, `ZeroGrad`, `Save`, `Load` are **not** in the concept — TTN derives them from
      `all_params()` via the bulk helpers in `Params.hpp`. Blocks only declare what they own.

- ***Block*** — [`concept Block<B>`](src/NetworkUtil.hpp)
  -

---

### Chain Resolution

- ***BuildChain*** — [`struct BuildChain<typename In, Block... Recipes>`](src/NetworkUtil.hpp)
  -

- ***Input*** — [`struct Input<size_t... Dims>`](src/NetworkUtil.hpp)
  -

- ***PrependBatch*** — [`struct PrependBatch<size_t Batch, typename T>`](src/NetworkUtil.hpp)
  -

---

### `class ActivationsWrap<typename TupleT>`

Thin owning wrapper around an activations tuple. Deletes the rvalue
`get()` overload at compile time to prevent dangling references.

- ***ActivationsWrap*** — [`explicit ActivationsWrap(TupleT t)`](src/NetworkUtil.hpp)
  -

- ***get*** — [`template<size_t N> auto get() const& -> const std::tuple_element_t<N, TupleT>&`](src/NetworkUtil.hpp)
  -

- ***get*** — [`template<size_t N> auto get() & -> std::tuple_element_t<N, TupleT>&`](src/NetworkUtil.hpp)
  -

- ***get*** — [`template<size_t N> auto get() && -> std::tuple_element_t<N, TupleT>&& = delete`](src/NetworkUtil.hpp)
  -

- ***tuple*** — [`const TupleT& tuple() const`](src/NetworkUtil.hpp)
  -

---

## [Params.hpp](src/Params.hpp): Parameter Storage and Optimizer

Defines the `Param<T>` template, the
`AdamState` struct, and bulk helpers. No block ever writes an optimizer loop — everything routes through here.

### `struct AdamState`

All Adam hyperparameters and per-network bias-correction state in one place. TTN owns one instance (
`mAdam_`) and passes it by const-ref to `UpdateAll` each step.

| Member | Default | Meaning |
|--------|---------|---------|
| `beta1` | `0.9` | first-moment decay |
| `beta2` | `0.999` | second-moment decay |
| `eps` | `1e-8` | denominator stabilizer |
| `mCorr` | `1` | `1 / (1 - β1^t)`, updated by `step()` |
| `vCorr` | `1` | `1 / (1 - β2^t)`, updated by `step()` |
| `t` | `0` | timestep counter |

- ***step*** — [`void step()`](src/Params.hpp)
    - Increments `t`, recomputes `mCorr = 1/(1-β1^t)` and `vCorr = 1/(1-β2^t)`. Call once per `Update()`.

### `template<typename TensorT> struct Param`

Single trainable tensor bundled with gradient and Adam moments (`value`, `grad`, `m`, `v`).

- ***update*** — [`void update(const AdamState& adam, float lr)`](src/Params.hpp)
    - One Adam step: updates `m`, `v`, then applies bias-corrected weight update to `value`

- ***zero_grad*** — [`void zero_grad()`](src/Params.hpp)
    - Zeroes `grad` — called by `ZeroAllGrads` at the start of each training step

- ***save*** — [`void save(std::ofstream& f) const`](src/Params.hpp)
    - Serializes `value` to binary file

- ***load*** — [`void load(std::ifstream& f)`](src/Params.hpp)
    - Deserializes `value` from binary file

### Bulk Helpers

Operate over the `std::tuple<Param<T>&...>` returned by `all_params()`.

- ***ZeroAllGrads*** — [`template<typename Tuple> void ZeroAllGrads(Tuple&& params)`](src/Params.hpp)
    - Calls `zero_grad()` on every `Param` in the tuple

- ***UpdateAll*** — [
  `template<typename Tuple> void UpdateAll(Tuple&& params, const AdamState& adam, float lr)`](src/Params.hpp)
    - Calls `update(adam, lr)` on every `Param` in the tuple

- ***SaveAll*** — [`template<typename Tuple> void SaveAll(Tuple&& params, std::ofstream& f)`](src/Params.hpp)
    - Calls `save(f)` on every `Param` in the tuple

- ***LoadAll*** — [`template<typename Tuple> void LoadAll(Tuple&& params, std::ifstream& f)`](src/Params.hpp)
    - Calls `load(f)` on every `Param` in the tuple

---

## [TrainableTensorNetwork.hpp](src/TrainableTensorNetwork.hpp): The Network

The top-level network class and the `NetworkBuilder` factory. Owns all blocks in a
`std::tuple` and one `AdamState` instance. Orchestrates forward/backward passes; drives `ZeroGrad`, `Update`, `Save`,
`Load` on every block via `all_params()` — no block implements these directly.

### `class TrainableTensorNetwork<ConcreteBlock... Blocks>`

**Type aliases and constants:**

- ***InputTensor*** — [`using InputTensor`](src/TrainableTensorNetwork.hpp)
    - Tensor type of the first block's input
- ***OutputTensor*** — [`using OutputTensor`](src/TrainableTensorNetwork.hpp)
    - Tensor type of the last block's output
- ***InSize*** — [`static constexpr size_t InSize`](src/TrainableTensorNetwork.hpp)
  -
- ***OutSize*** — [`static constexpr size_t OutSize`](src/TrainableTensorNetwork.hpp)
  -
- ***TotalParamCount*** — [`static constexpr size_t TotalParamCount`](src/TrainableTensorNetwork.hpp)
    - Derived from `TupleParamCount` over each block's `all_params()` — no `ParamCount` member required on blocks
- ***Activations*** — [`using Activations`](src/TrainableTensorNetwork.hpp)
  -
- ***BatchedActivations*** — [`template<size_t Batch> using BatchedActivations`](src/TrainableTensorNetwork.hpp)
  -

**Single-sample interface:**

- ***BackwardAll*** — [
  `void BackwardAll(const Activations& A, const OutputTensor& grad)`](src/TrainableTensorNetwork.hpp)
    -

- ***Update*** — [`void Update(float lr)`](src/TrainableTensorNetwork.hpp)
    - Calls `mAdam_.step()` then `UpdateAll(block.all_params(), mAdam_, lr)` for every block

- ***ZeroGrad*** — [`void ZeroGrad()`](src/TrainableTensorNetwork.hpp)
    - Calls `ZeroAllGrads(block.all_params())` for every block

- ***TrainStep*** — [
  `void TrainStep(const InputTensor& x, const OutputTensor& grad, float lr)`](src/TrainableTensorNetwork.hpp)
    -

- ***Fit*** — [
  `template<typename Loss> float Fit(const InputTensor& x, const OutputTensor& target, float lr)`](src/TrainableTensorNetwork.hpp)
    -

**Batched interface:**

- ***BatchedBackwardAll*** — [
  `template<size_t Batch> void BatchedBackwardAll(const BatchedActivations<Batch>& A, const typename PrependBatch<Batch, OutputTensor>::type& grad)`](src/TrainableTensorNetwork.hpp)
    -

- ***BatchTrainStep*** — [
  `template<size_t Batch> void BatchTrainStep(const typename PrependBatch<Batch, InputTensor>::type& X, const typename PrependBatch<Batch, OutputTensor>::type& grad, float lr)`](src/TrainableTensorNetwork.hpp)
    -

- ***BatchFit*** — [
  `template<typename Loss, size_t Batch> float BatchFit(const typename PrependBatch<Batch, InputTensor>::type& X, const typename PrependBatch<Batch, OutputTensor>::type& Y, float lr)`](src/TrainableTensorNetwork.hpp)
    -

**Serialization:**

- ***Save*** — [`void Save(const std::string& path) const`](src/TrainableTensorNetwork.hpp)
  -

- ***Load*** — [`void Load(const std::string& path)`](src/TrainableTensorNetwork.hpp)
  -

---

### Free Functions

- ***RunEpoch*** — [
  `template<typename Loss, size_t Batch, ConcreteBlock... Blocks, size_t N, size_t... DataDims, typename PrepFn> float RunEpoch(TrainableTensorNetwork<Blocks...>& net, const Tensor<N, DataDims...>& dataset, std::mt19937& rng, float lr, PrepFn prep)`](src/TrainableTensorNetwork.hpp)
    -

---

### `struct NetworkBuilder<typename In, Block... Recipes>`

- ***type*** — [`using type`](src/TrainableTensorNetwork.hpp) -- the fully resolved `TrainableTensorNetwork<...>` type
  -

---

### `struct CombineNetworks<typename NetA, typename NetB>`

Type-level composition of two networks into one. Concatenates the block lists of `NetA` and `NetB`
into a single `TrainableTensorNetwork`. A compile-time `static_assert` enforces that
`NetA::OutputTensor == NetB::InputTensor`. No shared weight state -- the result is an independent network whose parameter count equals
`NetA::TotalParamCount + NetB::TotalParamCount`. All three types (`NetA`,
`NetB`, and the combined type) can be instantiated and trained independently.

```cpp
using Encoder     = NetworkBuilder<Input<784>, Dense<128, ReLU>, Dense<32>>::type;
using Decoder     = NetworkBuilder<Input<32>,  Dense<128, ReLU>, Dense<784>>::type;
using Autoencoder = CombineNetworks<Encoder, Decoder>::type;

Encoder     enc;   // train for representations
Decoder     dec;   // train for generation
Autoencoder ae;    // train end-to-end -- all blocks update together
```

- ***type*** — [`using type`](src/TrainableTensorNetwork.hpp) -- the combined
  `TrainableTensorNetwork<BlocksA..., BlocksB...>`
    - Result of splicing the block lists of `NetA` and
      `NetB`; a complete network supporting all single-sample and batched interfaces

---

## [DataIO.hpp](src/DataIO.hpp) -- Data Loading and Batching

Utilities for loading datasets from disk, drawing random mini-batches, and displaying terminal progress bars. Shapes are compile-time parameters: the type
*is* the schema.

### `class ProgressBar`

Lightweight terminal progress bar. Construct with a total step count and optional label; call `tick()` each step.

- ***ProgressBar*** — [`explicit ProgressBar(size_t total, std::string label = "")`](src/DataIO.hpp)
  -

- ***tick*** — [`void tick(const std::string& suffix = "", size_t n = 1)`](src/DataIO.hpp)
    - Advances by `n` steps and redraws. `suffix` is printed to the right of the bar (e.g. `"loss=0.312"`).

- ***set_label*** — [`void set_label(const std::string& label)`](src/DataIO.hpp)
  -

- ***reset*** — [`void reset()`](src/DataIO.hpp)
  -

---

- ***LoadCSV*** — [
  `template<size_t Rows, size_t Cols> Tensor<Rows, Cols> LoadCSV(const std::string& path, bool skip_header = false)`](src/DataIO.hpp)
    - Parses a CSV into a `Tensor<Rows, Cols>`. On first call shows a progress bar, then writes a binary cache at
      `<path>.<Rows>x<Cols>.bin`; subsequent calls load that file directly (pure binary read, no CSV parsing). Delete the
      `.bin` file if the underlying CSV changes.

- ***RandomBatch*** — [
  `template<size_t Batch, size_t N, size_t... RestDims> Tensor<Batch, RestDims...> RandomBatch(const Tensor<N, RestDims...>& ds, std::mt19937& rng)`](src/DataIO.hpp)
    -

---

## [Snapshot.hpp](src/Snapshot.hpp): Activation Snapshots

Runtime-typed storage for capturing named activation tensors. `SnapshotEntry` holds a shape vector and a flat `float` copy — erasing the compile-time type so snapshots can be stored in a uniform `SnapshotMap` (`unordered_map<string, SnapshotEntry>`). Used by visualization and debugging tools.

- ***snap_add*** — [
  `template<size_t... Dims> void snap_add(SnapshotMap& out, const std::string& key, const Tensor<Dims...>& t)`](src/Snapshot.hpp)
    - #########
