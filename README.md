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
  `Map`, and `Reduce` operations. Leveraging **C++
  ** templates and the type system, we enforce shape correctness at compile time while enabling aggressive precomputation of traversal structure. Runtime execution becomes a planned walk over constant topologies - fully fused, vectorizable, and allocation-free.

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
    auto C = MinorContract<N, Mul, Add>(A, B);  // Map=Mul, Reduce=Add
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
          auto C = MinorContract<1>(A, B,
              -std::numeric_limits<float>::infinity(),
              Mul{},
              Max{});
          ```

---

### Visualizing Contraction (Matrix Multiply)

We align the shared axis `K`, then map + reduce:

            A (M×K)                     B (K×N)
        
               k ->                        k ↓
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

## Why Strictly Typed Tensors for ML?

*Why strictly typed ML? You'll need everything specified at runtime!*
**Exactly!** Machine learning is not about dynamic architectures or programmatic changing of
***shape***. We certainly have values to dynamically compute and change at runtime; this is
*everything*. But in machine learning, the
***shape*** of what you're doing is something that should be known before the program runs. This is because **the
program you're really running is *gradient descent*. You have to write a program that type checks in the language of
gradient descent via linear algebra.** You are instantiating an infrastructure into which can be seeded and nurtured a
***phenomenon***.

You have to just build the antenna to tap into this latent dynamical homeostatic pattern given to us by mathematics.

With ML, you have to represent a decision to be made numerically. *It is surprisingly easy.*
But it makes sense that it's easy because mathematics is a formalism of clear thinking and strict-but-dynamic representation. It is highly expressive and its representational capability is second to none.
**It is second to no other formalism.**

You represent your decision numerically, you marry mappings together so that gradients can communicate. You define a measure of success that is differentiated directly back to the output of the model, which, internally, aligns its gradients. And there you have it.

Because you have represented your problem space through numbers and the choice through arithmetic, you inherit the analytical techniques that have already grown into the formalism given by numbers. You can use mathematics to tell you exactly how to let signal flow and collect in a large network which constitutes your choice-making engine. Calculus is the lens you use to see the essence of the decision being modeled as made take root. You literally see the competency you seek to master represented as arterial subnetworks which carry information magnitude through the network.

The point is: you build an antenna, you make sure all the wires are connected forward and backward locally, then you invite the latent space to do its natural dance.

You must simply build the flower bed, plant the seeds, ensure intermittent water access and sunshine, and the latent chemistry takes it from there. The
***flowerbed*** is the `TrainableTensorNetwork` ***type***, the ***water*** may be the
***optimizer*** and its flow rate is the ***learning rate***, and the ***sunshine*** is the set of
***target variables***, shining down through the `LossFunction`.

## Future Directions

### Interpretability API

This is where shape-as-type pays its most interesting dividends.

The goal is a general **`BrainSaladSurgery`
** protocol: any block or network that opts in returns a structured snapshot of its internal activations, formatted and labeled for downstream exploration. Think of it as a typed, self-describing introspection packet — not a raw dump, but something with enough compile-time shape information baked in that a visualization tool can consume it without guessing layout.

The attention pattern work already in
`main` is the proof of concept. The head-weight matrices, the pre-softmax scores, the value-weighted outputs — all of those are already
`Tensor<...>` objects with shapes fully known at compile time. The step is to make that exposure **formal and uniform
**: define a concept (`BrainSaladProvider` or similar) and require that conforming blocks expose a
`peek()` method returning a `BrainSaladSurgery`
struct whose members are themselves typed tensors. No shape ambiguity, no runtime reinterpretation — the graphical tool on the other end knows exactly what it is receiving because the type says so.

#### Cross-Network Comparison

The shape-as-type model opens another powerful avenue: **comparing networks of the same type
**. Because topology is encoded in the C++ type, the compiler enforces that two networks being compared are actually structurally identical. This makes the following operations trivially safe to express:

- **Same random seed, different data direction
  ** — train the same architecture on transposed or permuted data, then compare learned representations layer by layer. Because both networks are
  `SomeNet<...>` with identical template parameters, a
  `compare(net_a, net_b)` function can zip their weight tensors together without any runtime shape checking.

- **Checkpoint-to-checkpoint drift
  ** — save a typed snapshot at each epoch; compare corresponding weight tensors across training time using Frobenius cosine similarity or any other metric. The type guarantees you are comparing the same layer in the same position, not accidentally swapping heads or layers.

- **Head alignment across runs
  ** — for attention-based networks, identify which heads in run A correspond most closely (by value similarity or by what they attend to) to heads in run B. With shape-as-type, the per-head slices are well-typed objects and can be passed directly into alignment routines.

The common thread: **the type system is doing the bookkeeping
** that in most frameworks requires careful string-matching on layer names or fragile index arithmetic.

### SDL Visualization Plugin

A companion C++ graphics library (in progress) will consume
`BrainSaladSurgery` packets and render them live. Planned views:

- Rank-1 tensors as bar charts or histograms
- Rank-2 tensors as heat maps (weight matrices, attention score grids)
- Rank-3 tensors as stacked slices or volume renders
- Network topology graphs with live activation overlays during a forward pass

Because the visualization library will also be typed against the same `Tensor<...>`
template, it can validate at compile time that the tensors it receives are within its renderable rank range — no runtime surprises when you accidentally pass a rank-5 weight blob to a heat-map view.

### GPU Backend

Replace the parallel CPU dispatch (
`std::execution::par_unseq`) with CUDA or Metal kernels. The contraction and elementwise op layers are the natural insertion point; higher-level code does not need to change because the
`Tensor` API stays the same. **Apple AMX** is currently used for a specialization of `Contract` where `Map=Mul` and
`Reduce=Add`, but other operations could still benefit further from GPUs.

---

## Table of Contents

1. [TensorPrereqs.hpp: Compile-Time Fundamentals](#tensorprereqshpp--compile-time-fundamentals)
2. [TensorStorage.hpp: Storage Policy](#tensorstoragehpp--storage-policy)
3. [Tensor.hpp: The Foundational Object](#tensorhpp--the-foundational-object)
4. [TensorShapeOps.hpp: Tensor Type Algebra](#tensorshapeopshpp--tensor-type-algebra)
5. [TensorFunctions.hpp: Functional Helpers](#tensorfunctionshpp--functional-helpers)
6. [TensorOps.hpp: Op Tags and Element-wise Primitives](#tensoropshpp--op-tags-and-element-wise-primitives)
7. [TensorContract.hpp: Contraction](#tensorcontracthpp--contraction)
8. [TensorReduce.hpp: Reduction and Broadcast](#tensorreducehpp--reduction-and-broadcast)
9. [TensorUtil.hpp: Tensor Layer Umbrella](#tensorutilhpp--tensor-layer-umbrella)
10. [TTTN_ML.hpp: ML Primitives](#tttn_mlhpp--ml-primitives)
11. [NetworkUtil.hpp: Concepts, Types, and Utilities](#networkutilhpp--concepts-types-and-utilities)
12. [TrainableTensorNetwork.hpp: The Network](#trainabletensornetworkhpp--the-network)
13. [Params.hpp: Parameter Storage and Optimizer](#paramshpp--parameter-storage-and-optimizer)
14. [NetworkComposition.hpp: Sequential Block Composition](#BlockCompositionhpp--sequential-block-composition)
15. [Snapshot.hpp: Activation Snapshots](#snapshothpp--activation-snapshots)
16. [Dense.hpp: Fully-Connected Layer](#densehpp--fully-connected-layer)
17. [Attention.hpp: Multi-Head Self-Attention](#attentionhpp--multi-head-self-attention)
18. [MoreNets.hpp: More helper Block types](#morenetshpp)
19. [TransformerBlock.hpp: Pre-norm Transformer Block](#transformerblockhpp--pre-norm-transformer-block)
20. [BranchTrainer.hpp: Shared-Trunk Multi-Head Training](#branchtrainerhpp--shared-trunk-multi-head-training)
21. [DataIO.hpp: Data Loading and Batching](#dataiohpp--data-loading-and-batching)

---

## [TensorPrereqs.hpp](src/TensorPrereqs.hpp): Compile-Time Fundamentals

Concepts, compile-time dimension arithmetic, stride computation, and the parallel loop helper. Everything
`Tensor.hpp` needs before it can define the `Tensor` class itself.

- ***TensorDimsProduct*** — [`struct TensorDimsProduct<size_t... Ds>`](src/TensorPrereqs.hpp)
    - Template recursion to store product of `size_t...` variadic in `static constexpr size_t value`

- ***SizeTemplateGet*** — [`struct SizeTemplateGet<size_t N, size_t... Ds>`](src/TensorPrereqs.hpp)
    - Template recursion to store `N`-th element of `size_t...` variadic in `static constexpr size_t value`

- ***ComputeStrides*** — [`struct ComputeStrides<size_t... Ds>`](src/TensorPrereqs.hpp)
    - Template recursion to store `Tensor<Ds...>` stride array in
      `static constexpr std::array<size_t, sizeof...(Ds)> value`

- ***ParForEach*** — [`template<std::invocable<size_t> F> void ParForEach(size_t n, F f)`](src/TensorPrereqs.hpp)
    - Helper function used throughout library to run `std::invocable<size_t> F` `n` times using
      `std::execution::par_unseq` policy

- ***FloatUnaryOp*** — [`template<typename F> concept FloatUnaryOp`](src/TensorPrereqs.hpp)
    - `concept` to enforce `F :: float -> float` operations on `Tensor`s

- ***FloatBinaryOp*** — [`template<typename F> concept FloatBinaryOp`](src/TensorPrereqs.hpp)
    - `concept` to enforce `F :: float -> float -> float` operations on two `Tensor`s

---

## [TensorStorage.hpp](src/TensorStorage.hpp): Storage Policy

Two specializations of `TensorStorage<S, bool Small>` selected at compile time: inline
`alignas(64) float[S]` for small tensors (≤ 16 elements) and 64-byte aligned heap allocation for large ones.
`Tensor` owns one instance as `storage_`.

We 64-byte-align here so
`Tensor`s' data (when grabbed to cache from RAM) starts at beginning of cache. Also makes vectorization optimizations and specialized
`cblas_sgemm` call in `TensorContract.hpp` as fast as possible.

- ***TensorStorage*** — [`template<size_t S, bool Small = (S <= 16)> struct TensorStorage`](src/TensorStorage.hpp)
    - Struct wrapper for storage of `Tensor`'s `float[]`
    - `Tensor` owns one instance as `storage_`
    - Specialized for `bool Small = true` (inline 64-byte-aligned stack `float[]`) and
      `bool Small = false` (64-byte-aligned heap `float[]`)

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
    - In-place binary transform: `self[i] = f(self[i], other[i])` for all `i`. Mutating counterpart to
      `zip`. Accepts any `FloatBinaryOp` including op tags: `zip_apply(b, Add{})`

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
  `BroadcastApply`, `BroadcastReduce`, `MinorContract`, `Contract`, `Collapse`, `TensorIndexApply`.

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

Shape-only metaprogramming — no data, no runtime. Purely compile-time type-level operations on
`Tensor` dimension packs: concatenation, axis removal/insertion, slicing, and permutation type computation.

- ***TensorConcat*** — [
  `template<size_t... Ds1, size_t... Ds2> struct TensorConcat<Tensor<Ds1...>, Tensor<Ds2...> >`](src/TensorShapeOps.hpp)
    - One layer unpacking of two `Tensor`s' dimensions
    - Two variadic lists as template parameters match with the shape arrays of the two input `Tensor`s

- ***KeptDimsHolder*** - [`template<size_t Skip, size_t... Dims> struct KeptDimsHolder`](src/TensorShapeOps.hpp)
    - Helper for `RemoveAxis`
    - Takes a dimension/axis to skip and variadic `size_t...` for existing dims and defines
      `std::array<size_t, sizeof...(Dims)> value` filled with `Dims...` sans `Skip`

- ***ArrayToTensor*** — [
  `template<typename KeptIdxs, size_t... Iota> requires requires { KeptIdxs::value; } struct ArrayToTensor<KeptIdxs, std::index_sequence<Iota...>`](src/TensorShapeOps.hpp)
    - Take in a holder of a `size_t...` dimensions pack (which must have static `value`) and a
      `std::index_sequence` of the `Rank` of the `Tensor`-to-be and unpack dimensions into new `Tensor` type

- ***RemoveAxis*** — [`struct RemoveAxis<size_t Skip, size_t... Dims>`](src/TensorShapeOps.hpp)
    - Helper to create new `Tensor` type from given `size_t...Dims` and an axis to `Skip`
    - Calls `ArrayToTensor<KeptDimsHolder<...>>`

- ***PermutedTensorType*** — [
  `template<size_t... Dims, size_t... Perm> struct PermutedTensorType<Tensor<Dims...>, Perm...>`](src/TensorShapeOps.hpp)
    - Helper to define a `Tensor` type, whose shape is `size_t...Dims` reorganized according to the indices specified by
      `size_t...Perm`

- ***InsertAxisHolder*** - [
  `template<size_t Axis, size_t N, size_t... Dims> struct InsertAxisHolder`](src/TensorShapeOps.hpp)
    - Like `KeptDimsHolder`, a helper holder of an array of `size_t` dimensions to be used to construct a new
      `Tensor` type
    - Used in `InsertAxis`

- ***InsertAxis*** - [`template<size_t Axis, size_t N, size_t... Dims> struct InsertAxis`](src/TensorShapeOps.hpp)
    - Like `RemoveAxis`, insert a dimension at a specified `size_t Axis`, with a specified magnitude `size_t N`
    - Calls `ArrayToTensor<InsertAxisHolder<...>>`


- ***SliceDimsHolder*** - [
  `template<size_t Start, size_t Len, size_t... Dims> struct SliceDimsHolder`](src/TensorShapeOps.hpp)
    - Like `InsertAxisHolder` and `KeptDimsHolder`, helper class to define an array of `size_t`, of length
      `Len`, containing the axes from `size_t...Dims` that start at `size_t Start`
    - Used in `TensorSlice`

- ***TensorSlice*** - [`template<size_t Start, size_t Len, size_t... Dims> struct TensorSlice`](src/TensorShapeOps.hpp)
    - Specify `size_t Start`, `size_t Len`, and a starting set of `size_t...Dims` and construct a
      `Tensor` whose shape only contains the `Len` dimensions starting at `Start`
    - Calls `ArrayToTensor<SliceDimsHolder<...>>`

- ***SwapNDims*** - [`template<size_t N_out, size_t N_in> struct SwapNDims`](src/TensorShapeOps.hpp)
    - Define permutation-ready `std::array<size_t, N_out + N_in>` that moves the final
      `N_out` axes to the front and the first `N_in` axes after

- ***PrependBatch*** - [`template<size_t Batch, size_t... Dims> struct PrependBatch`](src/TensorShapeOps.hpp)
    - Prepend a `Batch` axis, define `type = Tensor<Batch, Dims...>`


- ***TensorFirstDim*** - [
  `template<size_t D0, size_t... Rest> struct TensorFirstDim<Tensor<D0, Rest...> >`](src/TensorShapeOps.hpp)
    - Get the first axis size from a `Tensor`

- ***ConcatSeqs*** - [
  `template<size_t... As, size_t... Bs> struct ConcatSeqs<std::integer_sequence<size_t, As...>, std::integer_sequence<size_t, Bs...> >`](src/TensorShapeOps.hpp)
    - Given two `std::integer_sequence`s, concatenate their values into one `std::integer_sequence`

- ***SeqToTensor*** - [
  `template<size_t... Ds> struct SeqToTensor<std::integer_sequence<size_t, Ds...> >`](src/TensorShapeOps.hpp)
    - Given a `std::integer_sequence`, forward its values to a new `Tensor` type

- ***SeqProduct*** - [
  `template<> struct SeqProduct<std::integer_sequence<size_t> >`](src/TensorShapeOps.hpp)
    - Fold a `std::integer_sequence` with multiplication to compute the product of all values
    - Base case for empty sequence: `value = 1`

- ***SeqProduct*** - [
  `template<size_t D0, size_t... Rest> struct SeqProduct<std::integer_sequence<size_t, D0, Rest...> >`](src/TensorShapeOps.hpp)
    - Fold a `std::integer_sequence` with multiplication to compute the product of all values
    - Recursive case: `value = D0 * SeqProduct<std::integer_sequence<size_t, Rest...> >::value`

- ***SplitAt*** - [
  `template<size_t N, size_t... Dims> struct SplitAt`](src/TensorShapeOps.hpp)
    - Given axes `size_t... Dims` and `size_t N`, internally store `head` and `tail` `std::integer_sequence`s, where
      `head` contains the first `N` values of `Dims...` and `tail` the rest
    - Useful to define weight `Tensor`

- ***SplitAtImpl*** - [
  `template<size_t... Collected, size_t... Remaining> struct SplitAtImpl<0, std::integer_sequence<size_t, Collected...>, std::integer_sequence<size_t, Remaining...> >`](src/TensorShapeOps.hpp)
    - Recursive implementation of `SplitAt`
    - Base case, `N == 0`: `Collected...` and `Rest...` are forwarded into `std::integer_sequence head, tail`

- ***SplitAtImpl*** - [
  `template<size_t N, size_t... Collected, size_t D0, size_t... Rest> requires (N > 0) struct SplitAtImpl<N, std::integer_sequence<size_t, Collected...>, std::integer_sequence<size_t, D0, Rest...> >`](src/TensorShapeOps.hpp)
    - Recursive implementation of `SplitAt`
    - Recursive case: peel off leftmost value of `Rest...`, append to `Collected...`, call on `N - 1`

- ***PrependOnes*** - [
  `template<size_t... Dims> struct PrependOnes<0, Tensor<Dims...> >`](src/TensorShapeOps.hpp)
    - Prepend `N` `1`s to a `size_t...Dims`, creating a new `Tensor` type
    - Base case: `type = Tensor<Dims...>`

- ***PrependOnes*** - [
  `template<size_t N, size_t... Dims> struct PrependOnes<N, Tensor<Dims...> >`](src/TensorShapeOps.hpp)
    - Prepend `N` `1`s to a `size_t...Dims`, creating a new `Tensor` type
    - Recursive case: `type = PrependOnes<N - 1, Tensor<1, Dims...> >::type`

---

## [TensorFunctions.hpp](src/TensorFunctions.hpp): Functional Helpers

Free-function wrappers for the core tensor transforms: `Map`, `MapMove`, `Zip`, `ZipMove`, `Permute`, `Transpose`,
`TensorIndex`, and `TensorIndexApply`. Also, the canonical home of all op-tag structs (`Add`, `Mul`, `Max`, `Exp`,
`Compose`, etc.) — these are the types you pass as template arguments to `BroadcastReduce`, `ReduceApply`,
`Map`, and friends.

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

--- 

### Functions

- ***Compose*** - [`template<typename F, typename G> struct Compose`](src/TensorFunctions.hpp)
    - Compose a `FloatUnaryOp` with either another `FloatUnaryOp` or a
      `FloatBinaryOp`, creating a new operation that can be passed as a template tag

- ***TensorIndex*** — [
  `template<size_t Axis, size_t Index, size_t... Dims> RemoveAxis<Axis, Dims...>::type TensorIndex(const Tensor<Dims...> &src)`](src/TensorFunctions.hpp)
    - Get a `Tensor` slice from `Tensor<Dims...> src` by specifying a `size_t Axis` and a `size_t Index`
      *from* that axis
    - If we have a `Tensor<3, 2, 4>` and call `TensorIndex<0, 1>()`, we will get the `1-th` (second)
      `Tensor<2, 4>` that lives on the first axis

- ***TensorIndexApply*** - [
  `template<size_t Axis, FloatBinaryOp F, size_t... Dims> void TensorIndexApply(Tensor<Dims...> &dst, size_t idx, const typename RemoveAxis<Axis, Dims...>::type &src, F f)`](src/TensorFunctions.hpp)
    - Using indexing conventions described in `TensorIndex`, apply a `FloatBinaryOp` on a sub-`Tensor`, combining
      `src` elements with those of `dst`, a sub-`Tensor` of the same type as the slice

- ***TensorGet*** — [
  `template<size_t Axis, size_t... Dims> RemoveAxis<Axis, Dims...>::type TensorGet(const Tensor<Dims...> &src, size_t idx)`](src/TensorFunctions.hpp)
    - Runtime-index read: extract the slice at position `idx` along `Axis`, returning a `Tensor` with that axis removed
    - Runtime counterpart to compile-time `TensorIndex`

- ***TensorSet*** — [
  `template<size_t Axis, size_t... Dims> void TensorSet(Tensor<Dims...> &dst, size_t idx, const RemoveAxis<Axis, Dims...>::type &src)`](src/TensorFunctions.hpp)
    - Runtime-index write: assign `src` into the slice at position `idx` along `Axis` of `dst`
    - Plain-assignment counterpart to `TensorIndexApply` (no binary op needed)

- ***Map*** - [
  `template<typename Op, size_t... Dims> requires FloatUnaryOp<Op> && std::default_initializable<Op> Tensor<Dims...> Map(const Tensor<Dims...> &src)`](src/TensorFunctions.hpp)
    - Apply a `FloatUnaryOp` to every element of `Tensor<Dims...>& src` and return a copy

- ***Apply*** - [
  `template<typename Op, size_t... Dims> requires FloatUnaryOp<Op> && std::default_initializable<Op> void Apply(Tensor<Dims...> &&src)`](src/TensorFunctions.hpp)
    - Apply a `FloatUnaryOp` inplace

- ***MapMove*** - [
  `template<typename Op, size_t... Dims> requires FloatUnaryOp<Op> && std::default_initializable<Op> Tensor<Dims...> MapMove(Tensor<Dims...> &&src)`](src/TensorFunctions.hpp)
    - `Apply` a `FloatUnaryOp` inplace on `src`, return moved version
    - Memory efficient way to call `Apply` on a `Tensor` that is part of a composition or nested call

- ***Zip*** - [
  `template<typename Op, size_t... Dims> requires FloatBinaryOp<Op> && std::default_initializable<Op> Tensor<Dims...> Zip(const Tensor<Dims...> &A, const Tensor<Dims...> &B)`](src/TensorFunctions.hpp)
    - Create copy of `A`, call `FloatBinaryOp` taking in elements from `A` and `B`, return new copy

- ***ZipMove*** - [
  `template<typename Op, size_t... Dims> requires FloatBinaryOp<Op> && std::default_initializable<Op> Tensor<Dims...> ZipMove(Tensor<Dims...> &&A, const Tensor<Dims...> &B)`](src/TensorFunctions.hpp)
    - Call `zip_apply` inplace on `A`, taking in `B` values, return moved-from version

- ***Permute*** - [
  `template<size_t... Perm, size_t... Dims> Tensor<SizeTemplateGet<Perm, Dims...>::value...> Permute(const Tensor<Dims...> &src)`](src/TensorFunctions.hpp)
    - Given a `size_t...Perm`, the same length as `Dims...`, representing a new ordering of
      `Tensor<Dims...> src`'s axes, perform a permutation and return a copy
    - `Permute<1, 2, 0>(Tensor<8, 4, 7>)` returns a `Tensor<4, 7, 8>`

- ***Transpose*** - [`template<size_t... Dims> auto Transpose(const Tensor<Dims...> &src)`](src/TensorFunctions.hpp)
    - Call `Permute` on `Tensor<Dims...>` with permutation indices:
      `<Tensor<Dims...>::Rank - 1, ..., 0>` (reverse all axes of `Tensor<Dims...>`)
    - `Transpose(Tensor<8, 4, 7>)` returns a `Tensor<7, 4, 8>`


- ***PermuteFromArray*** - [
  `template<auto Perm, size_t... I, size_t... Dims> auto PermuteFromArray(const Tensor<Dims...> &t, std::index_sequence<I...>)`](src/TensorFunctions.hpp)
    - Takes a `std::array<size_t, Rank>` representing requested permutation ordering and unpacks them with a
      `std::index_sequence` into a call to `Permute`
    - Returns a `Tensor` of new permuted shape

- ***ConditionalPermute*** - [
  `template<auto Perm, size_t... I, size_t... Dims> decltype(auto) ConditionalPermute(const Tensor<Dims...> &t, std::index_sequence<I...>)`](src/TensorFunctions.hpp)
    - Zero-cost identity check: if `Perm` is the identity permutation (all `Perm[i] == i`), returns `t` as a
      `const&` with no allocation
    - Otherwise falls back to `Permute`, returning a permuted copy
    - Used internally by `Contract` and
      `BatchContract` to skip the permutation step when incoming tensors are already in minor-aligned layout


- ***Reshape*** - [
  `template<size_t... NewDims, size_t... OldDims> Tensor<NewDims...> Reshape(const Tensor<OldDims...> &src)`](src/TensorFunctions.hpp)
    - Reinterpret `Tensor<OldDims...>` as `Tensor<NewDims...>` — total size must match
    - Same flat data, new shape; copies via `std::copy`

- ***Flatten*** - [
  `template<size_t... Dims> Tensor<Tensor<Dims...>::Size> Flatten(const Tensor<Dims...> &src)`](src/TensorFunctions.hpp)
    - Collapse `Tensor<Dims...>` to rank-1 `Tensor<Size>`
    - Convenience wrapper around `Reshape<Size>`


- ***MoveToLastPerm*** - [`template<size_t Src, size_t Rank> struct MoveToLastPerm`](src/TensorFunctions.hpp)
    - Create member `std::array<size_t, Rank> value`, representing `Rank` dimensions, permuted such that `src` is the
      *last* index
    - (Pass to `PermuteFromArray`)

- ***MoveToFirstPerm*** - [`template<size_t Src, size_t Rank> struct MoveToFirstPerm`](src/TensorFunctions.hpp)
    - Create member `std::array<size_t, Rank> value`, representing `Rank` dimensions, permuted such that `src` is the
      *first* index
    - (Pass to `PermuteFromArray`)


- ***BatchMap*** - [
  `template<size_t N, size_t... Dims, typename Fn> auto BatchMap(const Tensor<N, Dims...> &src, Fn fn)`](src/TensorFunctions.hpp)
    - Using a map `Fn` from `Tensor<Dims...>` to some other `Tensor` shape, map
      `Tensor<N, Dims...> -> PrependBatch<N, OutSlice>::type`, where `OutSlice` is the return `Tensor`
      type from `Fn`

- ***BatchZip*** - [
  `template<size_t N, size_t... Dims, typename Fn> auto BatchZip(const Tensor<N, Dims...> &A, const Tensor<N, Dims...> &B, Fn fn)`](src/TensorFunctions.hpp)
    - Using a map `Fn` from `(Tensor<Dims...>, Tensor<Dims...>)` to some other `Tensor` shape, map
      `(Tensor<N, Dims...>, Tensor<N, Dims...>) -> PrependBatch<N, OutSlice>::type`, where `OutSlice` is the return
      `Tensor`
      type from `Fn`

- ***AddPositionalEncoding*** - [
  `template<size_t SeqAxis = 0, size_t... Dims> void AddPositionalEncoding(Tensor<Dims...> &X)`](src/TensorFunctions.hpp)
    - Add sinusoidal positional encoding in-place to `X` along `SeqAxis` (default 0)
    - For each position `pos` along `SeqAxis` and embedding index `j` (row-major over the remaining axes):
        - `freq = 1 / 10000^(2*(j/2) / EmbSize)`
        - `PE[pos][j] = sin(pos * freq)` if `j` even, `cos(pos * freq)` if `j` odd
    - PE tensor depends only on template parameters — computed once as a `static` local and added via `zip_apply`

---

## [TensorOps.hpp](src/TensorOps.hpp): Element-wise Primitives

Operation tags, element-wise operations, arithmetic operators, and `Permute`. Base layer included by
`TensorContract.hpp` and `TensorReduce.hpp`.

### Arithmetic Operators

- ***operator+=*** - [
  `template<size_t... Dims> Tensor<Dims...> &operator+=(Tensor<Dims...> &a, const Tensor<Dims...> &b)`](src/TensorOps.hpp)
    - `A += B` (inplace)

- ***operator-=*** - [
  `template<size_t... Dims> Tensor<Dims...> &operator-=(Tensor<Dims...> &a, const Tensor<Dims...> &b)`](src/TensorOps.hpp)
    - `A -= B` (inplace)

- ***operatorx=*** - [
  `template<size_t... Dims> Tensor<Dims...> &operator*=(Tensor<Dims...> &a, const Tensor<Dims...> &b)`](src/TensorOps.hpp)
    - `A *= B` (inplace)

- ***operator/=*** - [
  `template<size_t... Dims> Tensor<Dims...> &operator/=(Tensor<Dims...> &a, const Tensor<Dims...> &b)`](src/TensorOps.hpp)
    - `A /= B` (inplace)


- ***operator+*** - [
  `template<size_t... Dims> Tensor<Dims...> operator+(const Tensor<Dims...> &a, const Tensor<Dims...> &b)`](src/TensorOps.hpp)
    - `A + B` (returns copy)

- ***operator-*** - [
  `template<size_t... Dims> Tensor<Dims...> operator-(const Tensor<Dims...> &a, const Tensor<Dims...> &b)`](src/TensorOps.hpp)
    - `A - B` (returns copy)

- ***operatorx*** - [
  `template<size_t... Dims> Tensor<Dims...> operator*(const Tensor<Dims...> &a, const Tensor<Dims...> &b)`](src/TensorOps.hpp)
    - `A * B` (returns copy)

- ***operator/*** - [
  `template<size_t... Dims> Tensor<Dims...> operator/(const Tensor<Dims...> &a, const Tensor<Dims...> &b)`](src/TensorOps.hpp)
    - `A / B` (returns copy)


- ***operator+=*** - [
  `template<size_t... Dims> Tensor<Dims...> &operator+=(Tensor<Dims...> &a, float s)`](src/TensorOps.hpp)
    - `A += b` (inplace, `b` is scalar)

- ***operator-=*** - [
  `template<size_t... Dims> Tensor<Dims...> &operator-=(Tensor<Dims...> &a, float s)`](src/TensorOps.hpp)
    - `A -= b` (inplace, `b` is scalar)

- ***operatorx=*** - [
  `template<size_t... Dims> Tensor<Dims...> &operator*=(Tensor<Dims...> &a, float s)`](src/TensorOps.hpp)
    - `A *= b` (inplace, `b` is scalar)

- ***operator/=*** - [
  `template<size_t... Dims> Tensor<Dims...> &operator/=(Tensor<Dims...> &a, float s)`](src/TensorOps.hpp)
    - `A /= b` (inplace, `b` is scalar)


- ***operator+*** - [
  `template<size_t... Dims> Tensor<Dims...> operator+(const Tensor<Dims...> &a, float s)`](src/TensorOps.hpp)
    - `A + b` (returns copy, `b` is scalar)

- ***operator-*** - [
  `template<size_t... Dims> Tensor<Dims...> operator-(const Tensor<Dims...> &a, float s)`](src/TensorOps.hpp)
    - `A - b` (returns copy, `b` is scalar)

- ***operatorx*** - [
  `template<size_t... Dims> Tensor<Dims...> operator*(const Tensor<Dims...> &a, float s)`](src/TensorOps.hpp)
    - `A * b` (returns copy, `b` is scalar)

- ***operator/*** - [
  `template<size_t... Dims> Tensor<Dims...> operator/(const Tensor<Dims...> &a, float s)`](src/TensorOps.hpp)
    - `A / b` (returns copy, `b` is scalar)

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

## [TensorContract.hpp](src/TensorContract.hpp): Generalized Tensor Contraction

Tensor contraction is the unified `Reduce ∘ zipWith(Map) ∘ Align` operation on `Tensor`s. Sum of products (
`ΣΠ`) is the common use-case (and is specialized here to utilize `Apple Accelerate`'s highly optimized
`cblas_sgemm` function for matrix multiplication) but any `FloatBinaryOp` can be used for `Map` and any
`FloatBinaryOp` can be used for `Reduce`.

`Alignment` (choosing which axes to `Batch` or `Contract` is determined by which contraction function is chosen. NOTE:
`Batch` dimensions must be identical in two source `Tensor`s. This is the essence of batching: whereas
*free* indices from `A` and `B` are concatenated and indexed as a Cartesian product, the
`Batch` axes are independent lanes upon which contractions will occur. We do not say
`for a in A_Batch_Axes: {for b in B_Batch_Axes: {use A[a] and B[b]}}`, but instead we say
`for batch in Batches: {use A[batch] and B[batch]}`. This necessary for things like **multi-headed attention** in *
*Transformers**.

All contractions eventually become a `BatchMinorContraction`, which takes in four arrays of axes: `A`'s `Batch` and
`Contract` axes and `B`'s `Batch` and `Contract` axes. Non-batched `Contract` calls become
`Batch=0` calls to the former. `Dot`, `Matmul`, `ΣΠ`, `Outer`, and other variants all have
`Batch` versions, but, again are all wrappers around `BatchMinorContraction`.

- ***AxisList*** - [`template<size_t... Axes> struct AxisList`](src/TensorContract.hpp)
    - Necessary wrapper around variadic `size_t` axis packs used in `Contract` and `BatchContract` to specify specific
      `Batch` and/or `Contract` axis indices, rather than passing the *count* of left (for `Batch`) or right (for
      `Contract`) axes
    - `static_assert`s no duplicate axis indices

- ***BC_Permute*** - [
  `template<size_t Rank, AxisList BatchAxes, AxisList ContractAxes> struct BC_Permute`](src/TensorContract.hpp)
    - Compile-time helper to generate permutation indices for a `Tensor`'s axes in `BatchMinorContract` form:
      `[Batch..., Free..., Minor...]`
    - `static_assert`s that:
        - `Batch` and `Contract` axes are `disjoint`
        - no indices in `Batch` or `Contract` lists are greater than `Rank`
        - `Batch` and `Contract` axes have no duplicates
    - Intended to allow for `Permute`-`Contract` fusion, obviating many calls (in [
      `TrainableTensorNetwork`](src/TrainableTensorNetwork.hpp) blocks, for example) which allocate temporary `Tensor`s
    - `BatchMinorContract` form is part conventional and part performance-informed:
        - `Batch` being left-aligned adopts common convention for `Tensor` shapes in ML
        - `Minor` (contracted) axes being right-aligned reflects that `Tensor`s in `TTTN` are backed by ***row-major***
          `float` arrays. Only the rightmost (minor) axes are stored contiguously in memory. To maximize vectorization optimizations for
          `Reduce ∘ zipWith(Map)` operations, we want the loops over contracted indices to be traversing contiguous memory. Detailed comments on this subject are resident in the code.

- ***BatchedContractionKernel*** — [
  `template<size_t M_Batched, size_t N_Contracted, size_t... A_Dims, size_t... B_Dims> struct BatchedContractionKernel<M_Batched, N_Contracted, Tensor<A_Dims...>, Tensor<B_Dims...> >`](src/TensorContract.hpp)
    - Unified contraction bookkeeping kernel, used compile-time compute convenient shapes and values used in two versions of
      `BatchMinorContract` (the functions through which every contraction operation are routed)

- ***BatchMinorContract*** - [
  `template<size_t M, size_t N, size_t... ADims, size_t... BDims, FloatBinaryOp Map, FloatBinaryOp Reduce> auto BatchMinorContract(const Tensor<ADims...> &A, const Tensor<BDims...> &B, const float init, Map map, Reduce reduce)`](src/TensorContract.hpp)'s core primitive; all contraction routes through here
    - Core primitive: all contractions become `BatchMinorContract`
    - See `BatchContractionKernel` for more details on implementation

- ***BatchMinorContract*** - [
  `template<size_t M, size_t N, typename Map, typename Reduce, size_t... ADims, size_t... BDims> requires FloatBinaryOp<Map> && FloatBinaryOp<Reduce> && std::default_initializable<Map> && std::default_initializable<Reduce> && requires { { Reduce::identity } -> std::convertible_to<float>; } auto BatchMinorContract(const Tensor<ADims...> &A, const Tensor<BDims...> &B)`](src/TensorContract.hpp)
    - Tag-parameter specialization of `BatchMinorContract`; calls `BatchMinorContract`

- ***BatchMinorContract*** - [
  `template<size_t M, size_t N, size_t... ADims, size_t... BDims> auto BatchMinorContract(const Tensor<ADims...> &A, const Tensor<BDims...> &B, float /*init*/, Mul, Add)`](src/TensorContract.hpp)
    - Specialized version of generalized `BatchMinorContract` for `Map=Mul` and `Reduce=Add` (most common use-case)
    - Uses `Apple Accelerate`'s
      `cblas_sgemm` function to unlock aggressive vectorization optimization for matrix multiplication
    - Extensive commenting in code

- ***BatchΣΠ*** - [
  `template<size_t M, size_t N, size_t... ADims, size_t... BDims> auto BatchΣΠ(const Tensor<ADims...> &A, const Tensor<BDims...> &B)`](src/TensorContract.hpp)
    - Convenience wrapper for sum of product specialization of `BatchMinorContract`

- ***BatchSigmaPi*** - [
  `template<size_t M, size_t N, size_t... ADims, size_t... BDims> auto BatchSigmaPi(const Tensor<ADims...> &A, const Tensor<BDims...> &B)`](src/TensorContract.hpp)
    - ASCII overload of `BatchΣΠ`

- ***MinorContract*** - [
  `template<size_t N, size_t... ADims, size_t... BDims, FloatBinaryOp Map, FloatBinaryOp Reduce> auto MinorContract(const Tensor<ADims...> &A, const Tensor<BDims...> &B, float init, Map map, Reduce reduce)`](src/TensorContract.hpp)
    - Convenience wrapper for non-batched calls to generalized `BatchMinorContract`

- ***MinorContract*** - [
  `template<size_t N, typename Map, typename Reduce, size_t... ADims, size_t... BDims> requires FloatBinaryOp<Map> && FloatBinaryOp<Reduce> && std::default_initializable<Map> && std::default_initializable<Reduce> && requires { { Reduce::identity } -> std::convertible_to<float>; } auto MinorContract(const Tensor<ADims...> &A, const Tensor<BDims...> &B)`](src/TensorContract.hpp)
    - tag-param specialization of `MinorContract`

- ***ΣΠ*** - [
  `template<size_t N, size_t... ADims, size_t... BDims> auto ΣΠ(const Tensor<ADims...> &A, const Tensor<BDims...> &B)`](src/TensorContract.hpp)
    - Convenience wrapper for non-batched sum of products contraction

- ***SigmaPi*** - [
  `template<size_t N, size_t... ADims, size_t... BDims> auto SigmaPi(const Tensor<ADims...> &A, const Tensor<BDims...> &B)`](src/TensorContract.hpp)
    - ASCII convenience wrapper for `ΣΠ`

- ***Contract*** - [
  `template<AxisList AAxes, AxisList BAxes, size_t... ADims, size_t... BDims, FloatBinaryOp Map, FloatBinaryOp Reduce> auto Contract(const Tensor<ADims...> &A, const Tensor<BDims...> &B, float init, Map map, Reduce reduce)`](src/TensorContract.hpp)
    - Convenience wrapper for `BatchContract` (and
      `BatchMinorContract`) for non-Batched, arbitrary-axes contractions
    - Second-most general function in [TensorContract.hpp](src/TensorContract.hpp)

- ***Contract*** - [
  `template<AxisList AAxes, AxisList BAxes, typename Map, typename Reduce, size_t... ADims, size_t... BDims> requires FloatBinaryOp<Map> && FloatBinaryOp<Reduce> && std::default_initializable<Map> && std::default_initializable<Reduce> && requires { { Reduce::identity } -> std::convertible_to<float>; } auto Contract(const Tensor<ADims...> &A, const Tensor<BDims...> &B)`](src/TensorContract.hpp)
    - tag-param specialization of `Contract`

- ***Einsum*** [
  `template<size_t I, size_t J, size_t... ADims, size_t... BDims> auto Einsum(const Tensor<ADims...> &A, const Tensor<BDims...> &B)`](src/TensorContract.hpp)
    - Variant of `ΣΠ` for single-axis sum of product contractions (specified by `I` and `J` for `A` and
      `B`, respectively)

- ***BatchEinsum*** - [
  `template<AxisList ABatchAxes, AxisList BBatchAxes, size_t I, size_t J, size_t... ADims, size_t... BDims> auto BatchEinsum(const Tensor<ADims...> &A, const Tensor<BDims...> &B)`](src/TensorContract.hpp)
    - Batch version of `Einsum`

- ***Dot*** - [`template<size_t N> auto Dot(const Tensor<N> &A, const Tensor<N> &B)`](src/TensorContract.hpp)
    - Convenience wrapper for sum of product full-rank contraction (dot product) of two Rank-1 `Tensor`s

- ***Matmul*** - [
  `template<size_t M, size_t K, size_t N> auto Matmul(const Tensor<M, K> &A, const Tensor<K, N> &B)`](src/TensorContract.hpp)
    - Convenience wrapper for sum of product contraction (matrix multiplication) of two Rank-2 `Tensor`s
    - NOTE: expects Axis 1 of `A` to be contracted with Axis 0 of `B`, per `Matmul` convention

- ***Outer*** - [
  `template<size_t... ADims, size_t... BDims> auto Outer(const Tensor<ADims...> &A, const Tensor<BDims...> &B)`](src/TensorContract.hpp)
    - Convenience wrapper for sum of product no-rank contraction (outer product) of two Rank-1 `Tensor`s

- ***BatchContract*** - [
  `template<AxisList ABatchAxes, AxisList BBatchAxes, AxisList AContractAxes, AxisList BContractAxes, size_t... ADims, size_t... BDims, FloatBinaryOp Map, FloatBinaryOp Reduce> auto BatchContract(const Tensor<ADims...> &A, const Tensor<BDims...> &B, float init, Map map, Reduce reduce)`](src/TensorContract.hpp)
    - Most general function in [TensorContract.hpp](src/TensorContract.hpp)
    - Arbitrary `Align` (specified by `Batch` and `Contract` axes), arbitrary `Map` (to zip aligned elements of `A` and
      `B`), arbitrary `Reduce` (to fold down `Map` results along contracted axes)
    - Utilizes right-alignment convention of contracted axes (explained more in [
      `BC_Permute`](src/TensorContract.hpp) docs and code) and
      ***tiling*** to utilize vectorization of contiguous reads and computations

- ***BatchContract*** - [
  `template<AxisList ABatchAxes, AxisList BBatchAxes, AxisList AContractAxes, AxisList BContractAxes, typename Map, typename Reduce, size_t... ADims, size_t... BDims> requires FloatBinaryOp<Map> && FloatBinaryOp<Reduce> && std::default_initializable<Map> && std::default_initializable<Reduce> && requires { { Reduce::identity } -> std::convertible_to<float>; } auto BatchContract(const Tensor<ADims...> &A, const Tensor<BDims...> &B)`](src/TensorContract.hpp)
    - Tag-param specialization of `BatchContract`


- ***Collapse*** - [
  `template<size_t... Dims, FloatBinaryOp M, FloatBinaryOp R> float Collapse(const Tensor<Dims...> &A, const Tensor<Dims...> &B, float init, M map, R reduce)`](src/TensorContract.hpp)
    - Specialization for contraction over *all* axes
    - Known as ***Frobenius Inner Product***, it is a generalization of `Dot` or inner product for arbitrarily-shaped
      `Tensor`s

- ***Collapse*** - [
  `template<typename M, typename R, size_t... Dims> requires FloatBinaryOp<M> && FloatBinaryOp<R> && std::default_initializable<M> && std::default_initializable<R> && requires { { R::identity } -> std::convertible_to<float>; } float Collapse(const Tensor<Dims...> &A, const Tensor<Dims...> &B)`](src/TensorContract.hpp)
    - Tag-param specialization of `Collapse`

```cpp
Tensor<3> u, v;
Tensor<3, 4> W;
Tensor<5, 4> M;

// ── dot product: three equivalent spellings ──────────────────────────────────
auto d1 = Dot(u, v);                               // named alias
auto d2 = ΣΠ<1>(u, v);                             // "contract 1 dim"
auto d3 = Einsum<0, 0>(u, v);                      // "contract axis 0 of A with axis 0 of B"
static_assert(std::is_same_v<decltype(d1), Tensor<>>);  // all three -> rank-0 scalar
static_assert(std::is_same_v<decltype(d2), Tensor<>>);
static_assert(std::is_same_v<decltype(d3), Tensor<>>);

// ── matmul ───────────────────────────────────────────────────────────────────
auto mm = Matmul(W, Transpose(M));                 // Tensor<3,4> × Tensor<4,5> -> Tensor<3,5>
static_assert(std::is_same_v<decltype(mm), Tensor<3, 5>>);

// Einsum picks arbitrary axes — no Transpose needed:
auto d4 = Einsum<1, 1>(W, M);                      // contract axis 1 of W with axis 1 of M
static_assert(std::is_same_v<decltype(d4), Tensor<3, 5>>);  // same result, different path

// ── any-rank generalization ──────────────────────────────────────────────────
Tensor<4> x;
auto Wx = ΣΠ<1>(W, x);                             // Tensor<3,4> × Tensor<4> -> Tensor<3>
static_assert(std::is_same_v<decltype(Wx), Tensor<3>>);

Tensor<3, 5, 4> W3;
auto W3x = ΣΠ<1>(W3, x);                           // contract last 1 dim -> Tensor<3,5>
static_assert(std::is_same_v<decltype(W3x), Tensor<3, 5>>);

Tensor<5, 4, 2> K;
auto out = ΣΠ<2>(W3, K);                            // contract last 2 of W3 with first 2 of K
static_assert(std::is_same_v<decltype(out), Tensor<3, 2>>);

// ── outer product and full contraction: the two extremes ─────────────────────
auto outer = Outer(u, v);                           // ΣΠ<0>: contract nothing -> Tensor<3,3>
static_assert(std::is_same_v<decltype(outer), Tensor<3, 3>>);

float frob = Collapse<Mul, Add>(W, W);              // Frobenius inner product — tag form, no lambdas

// ── what the type system rejects ─────────────────────────────────────────────
// ΣΠ<1>(x, W);                                    // ✗ last dim of Tensor<784> ≠ first dim of Tensor<128,784>
// ΣΠ<1>(W3, W);                                   // ✗ last dim 4 ≠ first dim 3
// Dot(u, Tensor<5>{});                             // ✗ Tensor<3> · Tensor<5> — dimension mismatch
```

---

## [TensorReduce.hpp](src/TensorReduce.hpp): Reduction and Broadcast

Axis-reduction kernel, `ReduceApply`, `Expand`, `BroadcastApply`, `BroadcastReduce`, and indexed slice access. Includes
`TensorOps.hpp`.

### Reduction and Broadcast

- ***ReduceKernel*** — [`template<size_t Axis, size_t... Dims> struct ReduceKernel`](src/TensorReduce.hpp)
    - Shared kernel for all axis-reduction and broadcast operations
    - Compile-time computes convenient types/shapes, values, and `constexpr` functions for `offset` and
      `base` flat indexing required in `Broadcast` or `Reduce` functions

- ***Reduce*** - [
  `template<size_t Axis, typename Op, size_t... Dims> requires FloatBinaryOp<Op> && std::default_initializable<Op> && requires { { Op::identity } -> std::convertible_to<float>; } RemoveAxis<Axis, Dims...>::type Reduce(const Tensor<Dims...> &src)`](src/TensorReduce.hpp)
    - `Reduce` a `Tensor` along some `Axis` using a `FloatBinaryOp`

- ***Expand*** - [
  `template<size_t Axis, size_t N, size_t... Dims> InsertAxis<Axis, N, Dims...>::type Expand(const Tensor<Dims...> &src)`](src/TensorReduce.hpp)
    - `Expand` a `Tensor`, copying `N` times over the `Axis` passed as a template argument
    - Identity `Broadcast`

- ***BroadcastMap*** - [
  `template<size_t Axis, typename Op, size_t... Dims> requires FloatBinaryOp<Op> && std::default_initializable<Op> Tensor<Dims...> BroadcastMap(const Tensor<Dims...> &A, const typename RemoveAxis<Axis, Dims...>::type &b)`](src/TensorReduce.hpp)
    - `Broadcast` a `Tensor` of type `RemoveAxis<Axis, Dims...>` across a specified `Axis` of
      `Tensor<Dims...> A` using specified `FloatBinaryOp`
    - Copies `A` and returns copy

- ***BroadcastMapMove*** - [
  `template<size_t Axis, typename Op, size_t... Dims> requires FloatBinaryOp<Op> && std::default_initializable<Op> Tensor<Dims...> BroadcastMapMove(Tensor<Dims...> &&A, const typename RemoveAxis<Axis, Dims...>::type &b)`](src/TensorReduce.hpp)
    - `Broadcast` a `Tensor` of type `RemoveAxis<Axis, Dims...>` across a specified `Axis` of
      `Tensor<Dims...> A` using specified `FloatBinaryOp`
    - Moves `A`, overwrites its data, returns moved/overwritten `A`

- **BroadcastApply*** - [
  `template<size_t Axis, typename Op, size_t... Dims> requires FloatBinaryOp<Op> && std::default_initializable<Op> void BroadcastApply(Tensor<Dims...> &A, const typename RemoveAxis<Axis, Dims...>::type &b)`](src/TensorReduce.hpp)
    - `Broadcast` a `Tensor` of type `RemoveAxis<Axis, Dims...>` across a specified `Axis` of
      `Tensor<Dims...> A` using specified `FloatBinaryOp`
    - Overwrites `A` inplace, no return

- ***BroadcastReduce*** - [
  `template<size_t Axis, typename ApplyOp, typename ReduceOp, size_t... Dims> requires FloatBinaryOp<ApplyOp> && FloatBinaryOp<ReduceOp> && std::default_initializable<ApplyOp> && std::default_initializable<ReduceOp> && requires { { ReduceOp::identity } -> std::convertible_to<float>; } Tensor<Dims...> BroadcastReduce(const Tensor<Dims...> &src)`](src/TensorReduce.hpp)
    - Fused composition of `Broadcast` and `Reduce`; "`Broadcast` after `Reduce`"
    - `Reduce` with `ReduceOp`, then `Broadcast` that result back onto `Tensor<Dims...> src`
    - Copies `Tensor<Dims...> src` and returns copy

- ***BroadcastReduce*** - [
  `template<size_t Axis, typename ApplyOp, typename ReduceOp, size_t... Dims> requires FloatBinaryOp<ApplyOp> && FloatBinaryOp<ReduceOp> && std::default_initializable<ApplyOp> && std::default_initializable<ReduceOp> && requires { { ReduceOp::identity } -> std::convertible_to<float>; } Tensor<Dims...> BroadcastReduceMove(Tensor<Dims...> &&src)`](src/TensorReduce.hpp)
    - Fused composition of `Broadcast` and `Reduce`; "`Broadcast` after `Reduce`"
    - `Reduce` with `ReduceOp`, then `Broadcast` that result back onto `Tensor<Dims...> src`
    - Calls `Reduce` on `Tensor<Dims...> src`, moves and overwrites `Tensor<Dims...> src`, returns moved version

- ***BroadcastReduceInplace*** - [
  `template<size_t Axis, typename ApplyOp, typename ReduceOp, size_t... Dims> requires FloatBinaryOp<ApplyOp> && FloatBinaryOp<ReduceOp> && std::default_initializable<ApplyOp> && std::default_initializable<ReduceOp> && requires { { ReduceOp::identity } -> std::convertible_to<float>; } void BroadcastReduceInplace(Tensor<Dims...> &src)`](src/TensorReduce.hpp)
    - Fused composition of `Broadcast` and `Reduce`; "`Broadcast` after `Reduce`"
    - `Reduce` with `ReduceOp`, then `Broadcast` that result back onto `Tensor<Dims...> src`
    - Calls `Reduce` on `Tensor<Dims...> src`, overwrites `Tensor<Dims...> src` inplace, no return

Every operation takes
`Axis` as a compile-time argument. Output shape, stride arithmetic, and projection are all resolved at compile time — the runtime loop is a flat parallel sweep with zero shape logic.

```cpp
Tensor<32, 10> logits;
Tensor<10>     bias;

// ── tag-param reductions — no init arg, no lambda ────────────────────────────
auto col_sum = ReduceApply<0, Add>(logits);         // sum over batch -> Tensor<10>
auto row_max = ReduceApply<1, Max>(logits);         // max per sample -> Tensor<32>
static_assert(std::is_same_v<decltype(col_sum), Tensor<10>>);
static_assert(std::is_same_v<decltype(row_max), Tensor<32>>);

// ── Expand: dual of reduction ─────────────────────────────────────────────────
auto stacked = Expand<0, 32>(bias);                 // Tensor<10> -> Tensor<32, 10>
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
auto per_token = ReduceApply<2, Add>(activations);  // -> Tensor<8, 16>
auto restored  = Expand<0, 8>(ReduceApply<0, Max>(activations)); // -> Tensor<8, 16, 64>
static_assert(std::is_same_v<decltype(per_token), Tensor<8, 16>>);
static_assert(std::is_same_v<decltype(restored),  Tensor<8, 16, 64>>);
```

---

### Indexed Slice Access

`TensorIndex`, `TensorGet`, `TensorSet`, and
`TensorIndexApply` are the gather/scatter primitives. The axis is always compile-time — the compiler knows the slice shape and stride layout. The index into that axis is compile-time for
`TensorIndex`, runtime for the others.

```cpp
Tensor<16, 64> seq;                                 // 16 tokens, 64-dim embeddings

// compile-time gather: index must be known at compile time
auto tok1 = TensorIndex<0, 1>(seq);                 // seq[1, :] -> Tensor<64>

// runtime gather: index is a runtime value
size_t i = 5;
auto tok5 = TensorGet<0>(seq, i);                   // seq[i, :] -> Tensor<64>
static_assert(std::is_same_v<decltype(tok5), Tensor<64>>);

// runtime set: assign a slice directly
Tensor<64> embedding;
TensorSet<0>(seq, i, embedding);                    // seq[i, :] = embedding[:]

// scatter with binary op: accumulate a gradient into one token's slot
Tensor<16, 64> grad_seq;
Tensor<64>     grad_tok;
TensorIndexApply<0>(grad_seq, i, grad_tok,
    [](float a, float b) { return a + b; });        // grad_seq[i, :] += grad_tok[:]

// higher-rank: gather along a middle axis
Tensor<8, 16, 64> batch_seq;                        // batch × seq × embed
auto col3 = TensorGet<1>(batch_seq, 3);             // batch_seq[:, 3, :] -> Tensor<8, 64>
static_assert(std::is_same_v<decltype(col3), Tensor<8, 64>>);
```

---

---

### End-to-End: Forward and Backward by Hand

Everything above composes into a complete training step — contractions, reductions, broadcasts, outer products — all with shapes verified at compile time. No runtime shape checks, no asserts, no "expected shape [128] but got [10]" at 3 AM.

```cpp
// ── 2-layer feed-forward network (single sample, raw tensors) ────────────────
Tensor<784>      x;                                 // input: flattened 28×28 image
Tensor<128, 784> W1;  Tensor<128> b1;               // layer 1: 784 -> 128
Tensor<10,  128> W2;  Tensor<10>  b2;               // layer 2: 128 -> 10
float            lr = 0.01f;

// ── forward ──────────────────────────────────────────────────────────────────
auto z1 = ΣΠ<1>(W1, x) + b1;                       // Tensor<128,784> × Tensor<784> + Tensor<128> -> Tensor<128>
auto a1 = z1.map([](float v) {
    return v > 0.f ? v : 0.f; });                   // ReLU -> Tensor<128>
auto z2 = ΣΠ<1>(W2, a1) + b2;                      // Tensor<10,128> × Tensor<128> + Tensor<10> -> Tensor<10>

// softmax output (two BroadcastReduce calls — stable, parallel, one line each):
auto exps  = BroadcastReduce<0, Compose<Exp, Sub>, Max>(z2);
auto probs = BroadcastReduce<0, Div, Add>(exps);

static_assert(std::is_same_v<decltype(z1), Tensor<128>>);
static_assert(std::is_same_v<decltype(probs), Tensor<10>>);

// ── backward ─────────────────────────────────────────────────────────────────
Tensor<10> target;                                  // one-hot label
auto dz2 = probs.zip(target,                       // softmax + CEL combined gradient = pred − target
    [](float p, float t) { return p - t; });

auto dW2 = Outer(dz2, a1);                         // Tensor<10> ⊗ Tensor<128> -> Tensor<10, 128>
auto da1 = ΣΠ<1>(Transpose(W2), dz2);              // Tensor<128,10> × Tensor<10> -> Tensor<128>
auto dz1 = da1 * z1.map([](float v) {
    return v > 0.f ? 1.f : 0.f; });                // ⊙ relu' -> Tensor<128>
auto dW1 = Outer(dz1, x);                          // Tensor<128> ⊗ Tensor<784> -> Tensor<128, 784>

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

Thin include-only header that pulls in `TensorContract.hpp` and `TensorReduce.hpp` together. This is the last of the
`Tensor___` headers; `TTTN.hpp` includes `TensorUtil.hpp` transitively via
`TrainableTensorNetwork.hpp`. No public declarations — nothing to sentinel here.

---

## [TTTN_ML.hpp](src/TTTN_ML.hpp): ML Primitives

Activation functions, their derivatives, loss functions, and the `SoftmaxBlock` layer. Depends on
`TensorContract.hpp` and `TensorReduce.hpp`.

- ***EPS*** — [`static constexpr float EPS`](src/TTTN_ML.hpp)
    - Error constant used throughout ML file to allow divisions by `0`

### Activation Op Tags

Defined in [TTTN_ML.hpp](src/TTTN_ML.hpp). Each tag satisfies both `FloatUnaryOp` and `ActivationOp`. Use directly with
`Map<Act>(z)` for forward and `Act::prime(a)` for the derivative (in terms of post-activation output).

- ***ActivationOp*** - [
  `template<typename T> concept ActivationOp = FloatUnaryOp<T> && requires(float a)`](src/TTTN_ML.hpp)
    - `concept` requiring:
        - `constexpr float operator()(float x)`
        - `constexpr float prime(float a)`

- ***ReLU*** - [`struct ReLU`](src/TTTN_ML.hpp)
    - `ActivationOp` for ***Rectified Linear Unit*** (`ReLU`)
    - `operator()` -> `[0, infinity)`
    - `prime` -> `1.0f || 0.0f`

- ***Sigmoid*** - [`struct Sigmoid`](src/TTTN_ML.hpp)
    - `ActivationOp` for `Sigmoid`
    - `operator()` -> `[0, 1.0f]`
    - `prime` -> `(0.0f, 0.25f]`

- ***Tanh*** - [`struct Tanh`](src/TTTN_ML.hpp)
    - `ActivationOp` for ***Hyperbolic Tangent*** (`Tanh`)
    - `operator()` -> `[-1.0f, 1.0f]`
    - `prime` -> `(0.0f, 1.0f]`

- ***Liner*** - [`struct Linear`](src/TTTN_ML.hpp)
    - `ActivationOp` for `Linear` (no activation)
    - `operator()` -> `(-infinity, infinity)`
    - `prime` -> `(-infinity, infinity)`

---

### Free Functions

- ***CrossEntropyLoss*** - [
  `template<size_t N> float CrossEntropyLoss(const Tensor<N> &output, const Tensor<N> &target)`](src/TTTN_ML.hpp)
    - Computes ***Cross Entropy*** between two `Tensor`s, `output` and `target`, and returns `float`
    - Calls `Collapse<Mul, Add>(target, Map<Compose<Log, Clamp<EPS>>>(output)) * -1.f`

- ***XavierInitMD*** - [
  `template<size_t... Dims> void XavierInitMD(Tensor<Dims...> &W, const size_t fan_in, const size_t fan_out)`](src/TTTN_ML.hpp)
    - ***Xavier Initializes*** a `Tensor` inplace, given `fan_in` and
      `fan_out` values denoting net size of input and output to a neural network layer

---

### Softmax (Axis-Generalized)

- ***Softmax*** — [
  `template<size_t Axis, float Temp = 1.f, size_t... Dims> Tensor<Dims...> Softmax(const Tensor<Dims...> &x)`](src/TTTN_ML.hpp)
    - Given an `Axis` on which to normalize, perform `Softmax` normalization with optional temperature `Temp` (default
      `1.f`)
    - When `Temp != 1`, input is scaled by `1/Temp` before the standard softmax: `softmax(x / Temp)`
    - Elegantly calls
      `BroadcastReduceMove<Axis, Div, Add>(BroadcastReduce<Axis, Compose<Exp, Sub>, Max>(scaled))` to first map to
      `a = e^(x - max)` and then to `b = a / sum(a)`
    - Shape-preserving

- ***SoftmaxPrime*** — [
  `template<size_t Axis, float Temp = 1.f, size_t... Dims> Tensor<Dims...> SoftmaxPrime(const Tensor<Dims...> &grad, const Tensor<Dims...> &a)`](src/TTTN_ML.hpp)
    - Computes derivative of `Softmax` with optional temperature `Temp` (default `1.f`)
    - By the chain rule, `d softmax(x/T)/dx = (1/T) * J_softmax(x/T)`, so the result is scaled by `1/Temp`
    - Calls (efficient equivalent of) `a * BroadcastMap<Axis, Sub>(grad, BroadcastReduce<Axis, Add, Mul>(a, grad))`
        - Generalization of `a * (g - (g . a))`, scaled by `1/Temp`
    - Shape-preserving

- ***SoftmaxBlock*** - [
  `template<size_t Axis, float Temp, size_t... Dims> class SoftmaxBlock<Axis, Temp, Tensor<Dims...> >`](src/TTTN_ML.hpp)
    - Class representing the concrete block of a `Softmax` layer in a `TrainableTensorNetwork`, satisfying the
      `Block` `concept`; `Temp` is the softmax temperature (default `1.f` via `SoftmaxLayer`)

- ***SoftmaxBlock::Forward*** — [`OutputTensor SoftmaxBlock::Forward(const InputTensor &x) const`](src/TTTN_ML.hpp)
    - Calls `Softmax<Axis, Temp>(x)`

- ***SoftmaxBlock::Backward*** — [
  `InputTensor SoftmaxBlock::Backward(const OutputTensor &delta_A, const OutputTensor &a, const InputTensor & /*a_prev*/)`](src/TTTN_ML.hpp)
    - Calls `SoftmaxPrime<Axis, Temp>(delta_A, a)`

- ***SoftmaxBlock::BatchedForward*** — [
  `template<size_t Batch> Tensor<Batch, Dims...> SoftmaxBlock::BatchedForward(const Tensor<Batch, Dims...> &X) const`](src/TTTN_ML.hpp)
    - Calls `Softmax<Axis + 1, Temp>(X)`
    - NOTE: assumes first axis is `Batch` axis

- ***SoftmaxBlock::BatchedBackward*** — [
  `template<size_t Batch> Tensor<Batch, Dims...> SoftmaxBlock::BatchedBackward(const Tensor<Batch, Dims...> &delta_A, const Tensor<Batch, Dims...> &a, const Tensor<Batch, Dims...> & /*a_prev*/)`](src/TTTN_ML.hpp)
    - Calls `SoftmaxPrime<Axis + 1, Temp>(delta_A, a)`
    - NOTE: assumes first axis is `Batch` axis

- ***SoftmaxBlock::all_params*** — [`auto all_params()`](src/TTTN_ML.hpp)
    - Returns `std::tuple<>{}` (no parameters)


- ***SoftmaxLayer*** - [`template<size_t Axis, float Temp = 1.f> struct SoftmaxLayer`](src/TTTN_ML.hpp)
    - `BlockRecipe`-compliant recipe struct to create `Block SoftmaxBlock`
    - Pass in `Axis` of normalization and optional `Temp` (softmax temperature, default
      `1.f`); tensor shape is deduced from `InputT`

---

### Loss Function Concept

- ***LossFunction*** — [`template<typename L, typename TensorT> concept LossFunction`](src/TTTN_ML.hpp)
    - `concept` to define `LossFunction` structs
    - Requires:
        - `Loss(Tensor<Dims...>, Tensor<Dims...>) -> Tensor<>` (rank-0 scalar; implicitly converts to `float`)
        - `Grad(Tensor<Dims...>, Tensor<Dims...>) -> Tensor<Dims...>`

#### `struct MSE`

- ***MSE*** - [`struct MSE`](src/TTTN_ML.hpp)
    - `LossFunction` struct for ***Mean Squared Error*** (`MSE`)

- ***MSE::Loss*** - [
  `template<size_t... Dims> static Tensor<> MSE::Loss(const Tensor<Dims...> &pred, const Tensor<Dims...> &target)`](src/TTTN_ML.hpp)
    - Sum of squares of difference between `target` and `pred`
    - Calls `Collapse<Compose<Sq, Sub>, Add>(pred, target) * Inv`

- ***MSE::Grad*** - [
  `template<size_t... Dims> static Tensor<Dims...> MSE::Grad(const Tensor<Dims...> &pred, const Tensor<Dims...> &target)`](src/TTTN_ML.hpp)
    - Derivative of `MSE` loss
    -
  `2(pred - target) / Tensor<Dims...>::Size` (standard power rule derivative, scaled by how many elements composed the original sum)

#### `struct BinaryCEL`

- ***BinaryCEL*** - [`struct BinaryCEL`](src/TTTN_ML.hpp)
    - `LossFunction` struct for ***Binary Cross Entropy Loss*** (`BinaryCEL`)
    - Helper for binary cases, but is just a specialization of `struct CEL`

- ***BinaryCEL::Loss*** - [
  `template<size_t... Dims> static Tensor<> BinaryCEL::Loss(const Tensor<Dims...> &pred, const Tensor<Dims...> &target)`](src/TTTN_ML.hpp)
    - `-log(pred[true])` (negative log of the predicted value for `true` answer, whose target value is `1.0f`)

- ***BinaryCEL::Grad*** - [
  `template<size_t... Dims> static Tensor<Dims...> BinaryCEL::Grad(const Tensor<Dims...> &pred, const Tensor<Dims...> &target)`](src/TTTN_ML.hpp)
    - `(p - t) / (p * (1.f - p) + EPS)`

#### `struct CEL`

- ***CEL*** - [`struct CEL`](src/TTTN_ML.hpp)
    - `LossFunction` struct for ***Cross Entropy Loss*** (`CEL`)

- ***CEL::Loss*** - [
  `template<size_t... Dims> static Tensor<> CEL::Loss(const Tensor<Dims...> &pred, const Tensor<Dims...> &target)`](src/TTTN_ML.hpp)
    - `-log(pred[true])` (negative log of the predicted value for `true` answer, whose target value is `1.0f`)
    - Elegantly calls `Collapse<Mul, Add>(target, Map<Compose<Log, Clamp<EPS> > >(pred)) * -1.f`

- ***CEL::Grad*** - [
  `template<size_t... Dims> static Tensor<Dims...> CEL::Grad(const Tensor<Dims...> &pred, const Tensor<Dims...> &target)`](src/TTTN_ML.hpp)
    - Elegantly calls `Zip<Compose<Neg, Div> >(target, Map<Clamp<EPS> >(pred))`

#### `template<size_t PadId> struct SequenceCEL`

- ***SequenceCEL*** - [`template<size_t PadId> struct SequenceCEL`](src/TTTN_ML.hpp)
    - `LossFunction`-shaped struct for ***Sequence Cross Entropy*** with PAD masking — designed for autoregressive / encoder-decoder transformer training where target sequences are padded out to a fixed length
    - `PadId` is a compile-time template parameter; positions whose target one-hot is `1` at index `PadId` are excluded from both the loss value and the gradient
    - Loss is normalized by the count of non-PAD positions in the sample (not by `SeqLen`), so each sample contributes equally regardless of how much of its sequence is real

- ***SequenceCEL::Loss*** - [
  `template<size_t SeqLen, size_t Vocab> static Tensor<> SequenceCEL::Loss(const Tensor<SeqLen, Vocab> &pred, const Tensor<SeqLen, Vocab> &target)`](src/TTTN_ML.hpp)
    - Builds `mask: Tensor<SeqLen>` as `1 - target[t, PadId]`, computes per-position CE `-sum_v target[t,v] * log(clamp(pred[t,v]))` via `Reduce<1, Add>(target * Map<Compose<Log, Clamp<EPS>>>(pred)) * -1`, then sums `per_pos * mask` and divides by the non-PAD count (clamped to `>= 1` to avoid division by zero on all-PAD samples)

- ***SequenceCEL::Grad*** - [
  `template<size_t SeqLen, size_t Vocab> static Tensor<SeqLen, Vocab> SequenceCEL::Grad(const Tensor<SeqLen, Vocab> &pred, const Tensor<SeqLen, Vocab> &target)`](src/TTTN_ML.hpp)
    - Same `mask`/`n_nonpad` setup as `Loss`. Computes base grad `-target / clamp(pred)` via `Zip<Compose<Neg, Div>>`, scales by `1 / n_nonpad`, then zeros out PAD-position rows by `BroadcastApply<1, Mul>(g, mask)` (broadcasting the per-position mask along the `Vocab` axis)

#### `BatchAccuracy`

- ***BatchAccuracy*** - [
  `template<size_t Batch, size_t N> float BatchAccuracy(const Tensor<Batch, N> &pred, const Tensor<Batch, N> &labels)`](src/TTTN_ML.hpp)
    - Computes accuracy for `Tensor`s organized in a batched `Tensor`
    - Takes any `Tensor<Batch, Dims...>` and flattens `Dims...` internally
    - NOTE: assumes ***one-hot encoding*** for labels

- ***BatchAccuracy (ActivationsWrap overload)*** - [
  `template<typename TupleT> float BatchAccuracy(const ActivationsWrap<TupleT> &A, const std::tuple_element_t<std::tuple_size_v<TupleT> - 1, TupleT> &labels)`](src/TTTN_ML.hpp)
    - Overload that takes a full `ActivationsWrap` and automatically uses the final activation
    - Eliminates the need to manually index with `.get<N>()`

---

## [NetworkUtil.hpp](src/NetworkUtil.hpp): Concepts, Types, and Utilities

Defines the foundational block concepts that gate the type system and the `ActivationsWrap` safety wrapper.

### Concepts

- ***Block*** - [`template<typename T> concept Block`](src/NetworkUtil.hpp)
    - Any block in a `TrainableTensorNetwork` must satisfy `Block`:
        - Defined `InputTensor` and `OutputTensor` types which are `Tensor` objects
        - `OutputTensor Forward(InputTensor)`
        - `InputTensor Backward(OutputTensor, OutputTensor, InputTensor)`
        - `auto all_params()` and `auto all_params() const`
    - `TrainableTensorNetwork` blocks need not belong to a specific hierarchy; just satisfy this `concept`

### Block Operators

- ***operator>>*** - [
  `template<Block B, size_t... InDims> auto operator>>(const Tensor<InDims...> &x, const B &b)`](src/NetworkUtil.hpp)
    - Forward pass: `InDims...` deduced from `x`, `requires B::InputTensor == Tensor<InDims...>`; routes to
      `b.Forward(x)`
    - Single-sample overload — batched overload selected automatically when input has a prepended batch dim

- ***operator>> (batched)*** - [
  `template<Block B, size_t Batch, size_t... InDims> auto operator>>(const Tensor<Batch, InDims...> &X, const B &b)`](src/NetworkUtil.hpp)
    - Batched forward pass: `Batch` and `InDims...` deduced directly from `X`,
      `requires B::InputTensor == Tensor<InDims...>`; routes to `b.BatchedForward<Batch>(X)`

- ***BackwardArgs*** - [
  `template<typename DeltaT, typename AT, typename APrevT> struct BackwardArgs`](src/NetworkUtil.hpp)
    - Carrier struct bundling the three arguments to `Backward`: `delta_A`, `a`, `a_prev`
    - Used as the right-hand operand of `operator<<` on `Block`s
    - Single-sample callsite: `block << BackwardArgs{delta_A, a, a_prev}` (implicit — no type name needed once
      `B` is deduced from block)
    - Batched callsite: `block << BackwardArgs{delta_batched, a_batched, a_prev_batched}` (explicit type name required;
      `Batch` deduced from tensor dims)

- ***operator<<*** - [
  `template<Block B> auto operator<<(B &b, const BackwardArgs<B::OutputTensor, B::OutputTensor, B::InputTensor> &args)`](src/NetworkUtil.hpp)
    - Single-sample backward pass: routes to `b.Backward(args.delta_A, args.a, args.a_prev)`

- ***operator<< (batched)*** - [
  `template<Block B, size_t Batch, size_t... OutDims, size_t... InDims> auto operator<<(B &b, const BackwardArgs<Tensor<Batch, OutDims...>, Tensor<Batch, OutDims...>, Tensor<Batch, InDims...>> &args)`](src/NetworkUtil.hpp)
    - Batched backward pass: `Batch`, `OutDims...`, `InDims...` deduced from the `BackwardArgs` tensor types;
      `requires` enforces match against `B::OutputTensor`/`B::InputTensor`; routes to `b.BatchedBackward<Batch>(...)`

- ***PeekableBlock*** - [`template<typename T> concept PeekableBlock`](src/NetworkUtil.hpp)
    - Opt-in `concept` for `Block`s to be able to expose their internal activations to an owning
      `TrainableTensorNetwork`
    - Compliant `Block`s must implement `void peek(SnapshotMap& m, const std::string& s)`


- ***ActivationsWrap*** - [`template<typename TupleT> class ActivationsWrap`](src/NetworkUtil.hpp)
    - Wrapper around a `std::tuple` of `Tensor`s representing intermediate activations of a `TrainableTensorNetwork`
    - Internally stores `std::tuple` and provides safe access to elements and entire `std::tuple` via overloaded
      `get` methods

- ***ActivationsWrap::ActivationsWrap*** - [`explicit ActivationsWrap::ActivationsWrap(TupleT t)`](src/NetworkUtil.hpp)
    - `explicit` constructor which `move`s incoming `std::tuple` into member `data_`

- ***ActivationsWrap::get*** - [
  `template<size_t N> auto ActivationsWrap::get() const & -> const std::tuple_element_t<N, TupleT> &`](src/NetworkUtil.hpp)
    - `const &` getter, valid as long as `ActivationsWrap` object exists

- ***ActivationsWrap::get*** - [
  `template<size_t N> auto ActivationsWrap::get() & -> std::tuple_element_t<N, TupleT> &`](src/NetworkUtil.hpp)
    - `&` getter, valid as long as `ActivationsWrap` object exists


- ***ActivationsWrap::get*** - [
  `template<size_t N> auto ActivationsWrap::get() && -> std::tuple_element_t<N, TupleT> &&`](src/NetworkUtil.hpp)
    - Explicitly `delete`d function!
    - Getting temporary activation `Tensor` from temporary `ActivationsWrap` is a compile error, you must bind the
      `ActivationsWrap` object to a variable and get a reference
    - This is because it would return a dangling reference to a soon-deleted `ActivationsWrap` object
    - Instead of:
        - `auto& act = net.BatchedForwardAll(X).get<2>();`
    - You must do:
        - `auto& wrap = net.BatchedForwardAll(X);`
        - `auto& act = wrap.get<2>();`

- ***ActivationsWrap::tuple*** - [`const TupleT &ActivationsWrap::tuple() const`](src/NetworkUtil.hpp)
    - `const &` to raw `std::tuple`

- ***TensorTupleBuilder*** - [`template<typename Last> struct TensorTupleBuilder<Last>`](src/NetworkUtil.hpp)
    - Recursively build `std::tuple` of
      `Tensor` objects representing intermediate activations of the network, wrapped by `ActivationsWrap`
    - Base case: one single `Block` left, whose `InputTensor` and `OutputTensor` are wrapped in a `std::tuple`

- ***TensorTupleBuilder*** - [
  `template<typename First, typename... Rest> struct TensorTupleBuilder<First, Rest...>`](src/NetworkUtil.hpp)
    - Recursively build `std::tuple` of
      `Tensor` objects representing intermediate activations of the network, wrapped by `ActivationsWrap`
    - Recursive case: `std::tuple_cat` of `First` `InputTensor` object and `TensorTupleBuilder<Rest...>`


- ***BatchedTensorTupleBuilder*** - [
  `template<size_t Batch, typename Last> struct BatchedTensorTupleBuilder<Batch, Last>`](src/NetworkUtil.hpp)
    - For `Batched` functions and use-cases, create a `Batched` version of a `std::tuple` of activations by passing
      `PrependBatch<Batch, ...>` on all `Tensor`s that `TensorTupleBuilder` adds raw
    - Base case: one single `Block` left, whose `InputTensor` and `OutputTensor` are wrapped in
      `PrependBatch<Batch, ...>` and then in a `std::tuple`

- ***BatchedTensorTupleBuilder*** - [
  `template<size_t Batch, typename First, typename... Rest> struct BatchedTensorTupleBuilder<Batch, First, Rest...>`](src/NetworkUtil.hpp)
    - For `Batched` functions and use-cases, create a `Batched` version of a `std::tuple` of activations by passing
      `PrependBatch<Batch, ...>` on all `Tensor`s that `TensorTupleBuilder` adds raw
    - Recursive case: `std::tuple_cat` of `First` `PrependBatch<Batch, InputTensor>` object and
      `BatchedTensorTupleBuilder<Rest...>`

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

- ***AdamState::step*** - [`void AdamState::step()`](src/NetworkUtil.hpp)
    - Increment `t`
    - Recompute `mCorr`, `vCorr`

### Param

| Member          | Default | Meaning                      |
|-----------------|---------|------------------------------|
| `TensorT value` | `{}`    | parameter `Tensor` itself    |
| `TensorT grad`  | `{}`    | gradient `Tensor` of `value` |
| `TensorT m`     | `{}`    | `value`'s first Adam moment  |
| `TensorT v`     | `{}`    | `value`'s second Adam moment |

- ***struct Param*** - [`template<typename TensorT> struct Param`](src/NetworkUtil.hpp)
    - `struct` layer around a `Block` to abstract away management, Adam updates

- ***Param::Size*** - [`static constexpr size_t Param::Size`](src/NetworkUtil.hpp)
    - Size of parameter `Tensor`

- ***Param::zero_grad*** - [`void Param::zero_grad()`](src/NetworkUtil.hpp)
    - Fill `grad` with `0.f`

- ***Param::update*** - [`void Param::update(const AdamState &adam, float lr)`](src/NetworkUtil.hpp)
    - For each `float` parameter in `value`, use Adam moments and gradient to update


- ***Param::save*** - [`void Param::save(std::ofstream &f) const`](src/NetworkUtil.hpp)
    - Calls `Tensor::Save` on `value`, `m`, and
      `v` — full Adam state round-trip so incremental training resumes cleanly across runs

- ***Param::load*** - [`void Param::load(std::ifstream &f)`](src/NetworkUtil.hpp)
    - Calls `Tensor::Load` on `value`, `m`, and
      `v` — full Adam state round-trip so incremental training resumes cleanly across runs

### Free Concepts and Functions on Param

- ***IsParam*** - [`template<typename T> concept IsParam`](src/NetworkUtil.hpp)
    - Concept to verify that a type `T` is a `Param`

- ***IsParamTuple*** - [`template<typename Tuple> concept IsParamTuple`](src/NetworkUtil.hpp)
    - Concept to verify that a type `Tuple` is a `std::tuple` of `Param` objects

- ***ZeroAllGrads*** - [
  `template<IsParamTuple Tuple> void ZeroAllGrads(Tuple &&params)`](src/NetworkUtil.hpp)
    - Calls `Param::zero_grad` on each `Param` in the `std::tuple` of `Param`s


- ***UpdateAll*** - [
  `template<IsParamTuple Tuple> void UpdateAll(Tuple &&params)`](src/NetworkUtil.hpp)
    - Calls `Param::update` on each `Param` in the `std::tuple` of `Param`s


- ***SaveAll*** - [
  `template<IsParamTuple Tuple> void SaveAll(Tuple &&params)`](src/NetworkUtil.hpp)
    - Calls `Param::save` on each `Param` in the `std::tuple` of `Param`s

- ***LoadAll*** - [
  `template<IsParamTuple Tuple> void LoadAll(Tuple &&params)`](src/NetworkUtil.hpp)
    - Calls `Param::load` on each `Param` in the `std::tuple` of `Param`s

- ***TotalParamSize*** - [`template<IsParam... Params> constexpr size_t TotalParamSize`](src/NetworkUtil.hpp)
    - Sum of all `Param` sizes in variadic list of `Param`s
    - `(Params::Size + ...)`

- ***tuple_param_count_impl*** - [
  `template<IsParamTuple Tuple, size_t... Is> constexpr size_t tuple_param_count_impl(std::index_sequence<Is...>)`](src/NetworkUtil.hpp)
    - Unpacks `IsParamTuple` and sums each `Param::Size`, giving the net size of a `std::tuple` of `Param`s


- ***TupleParamCount*** - [`template<IsParamTuple Tuple> constexpr size_t TupleParamCount`](src/NetworkUtil.hpp)
    - Sum of all `Param` sizes in a `std::tuple` of `Param`s
    - Calls `tuple_param_count_impl`

### `struct LearnedContraction<InputTensor, OutputTensor, NumFree>`

- ***LearnedContraction*** - [
  `template<size_t... InDims, size_t... OutDims, size_t NumFree> struct LearnedContraction<Tensor<InDims...>, Tensor<OutDims...>, NumFree>`](src/NetworkUtil.hpp)
    - Abstraction of learned weight `Tensor` component to any `Block`, handling forward and backward pass internally
    - Templated by `Tensor<InDims...>`,
      `Tensor<OutDims...>`, and the number of leading (from the left) free axes you want to pass through the transformation, the internal
      `WeightTensor` type is deduced, initialized, stored, and updated internally
    - Learned weight `Tensor`s are as easy as simply declaring what your desired input and output shapes are.

### Type Aliases

- ***LearnedContraction::InputTensor*** - [`using LearnedContraction::InputTensor`](src/NetworkUtil.hpp)
    - Alias for input type: `Tensor<InDims...>`

- ***LearnedContraction::OutputTensor*** - [`using LearnedContraction::OutputTensor`](src/NetworkUtil.hpp)
    - Alias for output type: `Tensor<OutDims...>`

- ***LearnedContraction::N_contract_in*** - [
  `static constexpr size_t LearnedContraction::N_contract_in`](src/NetworkUtil.hpp)
    - Deduced number of contracted axes of `InputTensor`
    - `InputTensor::Rank - NumFree`

- ***LearnedContraction::N_contract_out*** - [
  `static constexpr size_t LearnedContraction::N_contract_out`](src/NetworkUtil.hpp)
    - Deduced number of contracted axes of `OutputTensor`
    - `OutputTensor::Rank - NumFree`


- ***LearnedContraction::InSplit*** - [`using LearnedContraction::InSplit`](src/NetworkUtil.hpp)
    - Split `InDims...` into first `NumFree` free axes (`InSplit::head`) and latter `N_contract_in` contracted axes of
      `InputTensor` (`InSplit::tail`)

- ***LearnedContraction::OutSplit*** - [`using LearnedContraction::OutSplit`](src/NetworkUtil.hpp)
    - Split `OutDims...` into first `NumFree` free axes (`OutSplit::head`) and latter
      `N_contract_out` contracted axes of
      `OutputTensor` (`OutSplit::tail`)

- ***LearnedContraction::WeightSeq*** - [`using LearnedContraction::WeightSeq`](src/NetworkUtil.hpp)
    - `std::integer_sequence` of two splits fused together:
        - `[OutSplit::tail, InSplit::tail]`
    - `WeightTensor` needs to minor-contract with `InSplit::tail` (the contracted axes of `InputTensor`)

- ***LearnedContraction::WeightTensor*** - [`using LearnedContraction::WeightTensor`](src/NetworkUtil.hpp)
    - `Tensor` type created from `WeightSeq` using `SeqToTensor`

### Members

- ***LearnedContraction::W_*** - [`Param<WeightTensor> LearnedContraction::W_`](src/NetworkUtil.hpp)
    - `Param` type, holding a `WeightTensor`

- ***LearnedContraction::X_cache_*** - [`mutable InputTensor LearnedContraction::X_cache_`](src/NetworkUtil.hpp)
    - `mutable` cache for `InputTensor`, used for efficient backward pass

- ***LearnedContraction::bX_buf_*** - [`mutable std::vector<float> LearnedContraction::bX_buf_`](src/NetworkUtil.hpp)
    - `mutable` cache for batched input `Tensor`, used for efficient backward pass
    - Uses `std::vector<float>` instead of `Tensor` type because `Batch` is only templated in `>>` and
      `<<` functions (i.e. `LearnedContraction` and its `WeightTensor` are oblivious to `Batch` dimensions)

### Methods

- ***LearnedContraction::LearnedContraction*** - [`LearnedContraction::LearnedContraction()`](src/NetworkUtil.hpp)
    - Default construct, call `XavierInitMD` on `W_.value`

- ***LearnedContraction::all_params*** - [`auto LearnedContraction::all_params()`](src/NetworkUtil.hpp)
    - Return `std::tuple` of `Param&` to weight parameter

- ***LearnedContraction::all_params*** - [`auto LearnedContraction::all_params() const`](src/NetworkUtil.hpp)
    - Return `std::tuple` of `const Param&` to weight parameter


- ***LearnedContraction::forward*** - [
  `OutputTensor LearnedContraction::forward(const InputTensor &X) const`](src/NetworkUtil.hpp)
    - Forward pass implementation, called by `>>`
    - Executes general pattern, implementation heavily documented in code

- ***LearnedContraction::backward*** - [
  `InputTensor LearnedContraction::backward(const OutputTensor &deltaO)`](src/NetworkUtil.hpp)
    - Backward pass implementation, called by `<<`
    - Executes general pattern, implementation heavily documented in code

- ***LearnedContraction::batched_forward*** - [
  `template<size_t Batch> Tensor<Batch, OutDims...> LearnedContraction::batched_forward(const Tensor<Batch, InDims...> &X) const`](src/NetworkUtil.hpp)
    - Batched forward pass implementation, called by `>>`
    - Executes general pattern, implementation heavily documented in code

- ***LearnedContraction::batched_backward*** - [
  `template<size_t Batch> Tensor<Batch, InDims...> LearnedContraction::batched_backward(const Tensor<Batch, OutDims...> &deltaO)`](src/NetworkUtil.hpp)
    - Batched backward pass implementation, called by `<<`
    - Executes general pattern, implementation heavily documented in code

### Operators

- ***operator>>*** - [
  `auto operator>>(const Tensor<InDims...> &X, const LearnedContraction<Tensor<InDims...>, Tensor<OutDims...>, NumFree> &lc)`](src/NetworkUtil.hpp)
    - Terse operator for forward pass (takes `const Tensor<InDims...> &X` on the left of the operator,
      `LearnedContraction` on the right)
    - *Propagate signal forward through `W_`*

- ***operator<<*** - [
  `auto operator<<(LearnedContraction<Tensor<InDims...>, Tensor<OutDims...>, NumFree> &lc, const Tensor<OutDims...> &deltaO)`](src/NetworkUtil.hpp)
    - Terse operator for backward pass (takes `LearnedContraction` on the left of the operator,
      `const Tensor<OutDims...> &deltaO` on the right)
    - *Propagate gradient backward through `W_`*

- ***operator>> (batched)*** - [
  `template<size_t Batch, size_t... InDims, size_t... OutDims, size_t NumFree> auto operator>>(const Tensor<Batch, InDims...> &X, const LearnedContraction<Tensor<InDims...>, Tensor<OutDims...>, NumFree> &lc)`](src/NetworkUtil.hpp)
    - Terse operator for batched forward pass (takes `const Tensor<Batch, InDims...> &X` on the left of the operator,
      `LearnedContraction` on the right)
    - *Propagate signal forward through `W_`*

- ***operator<< (batched)*** - [
  `template<size_t Batch> auto operator<<(LearnedContraction<Tensor<InDims...>, Tensor<OutDims...>, NumFree> &lc, const Tensor<Batch, OutDims...> &deltaO)`](src/NetworkUtil.hpp)
    - Terse operator for backward pass (takes `LearnedContraction` on the left of the operator,
      `const Tensor<Batch, OutDims...> &deltaO` on the right)
    - *Propagate gradient backward through `W_`*

---

## [BlockSequence.hpp](src/BlockSequence.hpp): The Sequence Core

The unified sequential core shared by `TrainableTensorNetwork` and `ComposeBlocks`. Owns a `std::tuple` of
`Block`s and a mutable activation cache. Satisfies `Block` itself, so a
`BlockSequence` can nest inside any other block (e.g. as an arm of
`Parallel`). Also, in compliance with the `Block` `concept`, exposes an explicit activation API (`ForwardAll`,
`BackwardFrom`, etc.) for top-level use by
`TrainableTensorNetwork`.

### `class BlockSequence<Block... Blocks>`

- ***BlockSequence*** - [
  `template<Block... Blocks> class BlockSequence`](src/BlockSequence.hpp)
    - Unified sequential core: wraps a shape-compliant chain of `Block`s and provides both the
      `Block` interface (for nesting) and the explicit activation API (for top-level training)

- ***BlockSequence::check_connected()*** - [
  `static constexpr bool BlockSequence::check_connected()`](src/BlockSequence.hpp)
    - Immediate `static_assert` function to ensure that `Block... Blocks` have compliant shapes:
      `std::is_same_v<typename std::tuple_element_t<Is, BlockTuple>::OutputTensor, typename std::tuple_element_t<Is + 1, BlockTuple>::InputTensor> && ...)`

- ***BlockSequence::BlockSequence*** - [`BlockSequence::BlockSequence()`](src/BlockSequence.hpp)
    - Default construct `mBlocks` and `mActs`

### Type Aliases and Public Members

- ***BlockSequence::NumBlocks*** - [`BlockSequence::NumBlocks`](src/BlockSequence.hpp)
    - `static constexpr size_t NumBlocks = sizeof...(Blocks)`

- ***BlockSequence::BlockTuple*** - [`using BlockSequence::BlockTuple`](src/BlockSequence.hpp)
    - Type alias for `std::tuple<Blocks...>`

- ***BlockSequence::InputTensor*** - [`BlockSequence::InputTensor`](src/BlockSequence.hpp)
    - Extract `InputTensor` type from first element of `BlockTuple`

- ***BlockSequence::OutputTensor*** - [`BlockSequence::OutputTensor`](src/BlockSequence.hpp)
    - Extract `OutputTensor` type from last element of `BlockTuple`

- ***BlockSequence::InSize*** - [`BlockSequence::InSize`](src/BlockSequence.hpp)
    - Convenience member for total size of `InputTensor` type

- ***BlockSequence::OutSize*** - [`BlockSequence::OutSize`](src/BlockSequence.hpp)
    - Convenience member for total size of `OutputTensor` type

- ***BlockSequence::TotalParamCount*** - [
  `static constexpr size_t BlockSequence::TotalParamCount`](src/BlockSequence.hpp)
    - Sum of parameter counts of all elements of `Blocks...`

- ***BlockSequence::block*** - [
  `template<size_t I> const auto &BlockSequence::block() const`](src/BlockSequence.hpp)
    - Get a `const &` to the `I`-th `Block` in `BlockSequence::mBlocks`

- ***BlockSequence::ActivationsTuple*** - [
  `using BlockSequence::ActivationsTuple`](src/BlockSequence.hpp)
    - Type alias around `TensorTupleBuilder<Blocks...>::type`

- ***BlockSequence::Activations*** - [
  `using BlockSequence::Activations`](src/BlockSequence.hpp)
    - Access-safe `ActivationsWrap` wrapper around `ActivationsTuple`

- ***BlockSequence::BatchedActivationsTuple*** - [
  `template<size_t Batch> using BlockSequence::BatchedActivationsTuple`](src/BlockSequence.hpp)
    - Type alias around `BatchedTensorTupleBuilder<Batch, Blocks...>::type`

- ***BlockSequence::BatchedActivations*** - [
  `template<size_t Batch> using BlockSequence::BatchedActivations`](src/BlockSequence.hpp)
    - Access-safe `ActivationsWrap` wrapper around `BatchedActivationsTuple`

### Private Members

- ***BlockSequence::mBlocks*** - [`BlockSequence::mBlocks`](src/BlockSequence.hpp)
    - Default-constructed `BlockTuple` containing actual `Block` objects

- ***BlockSequence::mActs*** - [`mutable BlockSequence::mActs`](src/BlockSequence.hpp)
    - Mutable `ActivationsTuple` cache used by the `Block` interface (`Forward`/`Backward`) so that
      `BlockSequence` can be used as a nested block without the caller managing activations

### Inference

- ***BlockSequence::ForwardAll*** - [
  `[[nodiscard]] Activations BlockSequence::ForwardAll(const InputTensor &x) const`](src/BlockSequence.hpp)
    - Forward pass returning full `ActivationsWrap Activations` object
    - Calls `BlockSequence::forward_impl` on `InputTensor &x`

- ***BlockSequence::Forward*** - [
  `[[nodiscard]] OutputTensor BlockSequence::Forward(const InputTensor &x) const`](src/BlockSequence.hpp)
    - Forward pass returning `OutputTensor`
    - Delegates to `ForwardAll` and extracts back element
    - Satisfies `Block` interface; uses `mActs` cache so a caller can follow with `Backward`

- ***BlockSequence::forward_impl*** - [
  `template<size_t I = 0> void BlockSequence::forward_impl(ActivationsTuple &A) const`](src/BlockSequence.hpp)
    - Private implementation; recursively fills `ActivationsTuple &A` by calling each
      `Block::Forward` in order and storing result

- ***BlockSequence::BatchedForwardAll*** - [
  `template<size_t Batch> [[nodiscard]] BatchedActivations<Batch> BlockSequence::BatchedForwardAll(const PrependBatch<Batch, InputTensor>::type &X) const`](src/BlockSequence.hpp)
    - Batched forward pass returning full `ActivationsWrap BatchedActivations` object

- ***BlockSequence::BatchedForward*** - [
  `template<size_t Batch> [[nodiscard]] PrependBatch<Batch, OutputTensor>::type BlockSequence::BatchedForward(const PrependBatch<Batch, InputTensor>::type &X) const`](src/BlockSequence.hpp)
    - Batched forward pass returning `PrependBatch<Batch, OutputTensor>`; extracted from back of `BatchedForwardAll`
    - Satisfies `Block` interface

- ***BlockSequence::batched_forward_impl*** - [
  `template<size_t Batch, size_t I = 0> void BlockSequence::batched_forward_impl(BatchedActivationsTuple<Batch> &A) const`](src/BlockSequence.hpp)
    - Private implementation; recursively fills `BatchedActivationsTuple &A` by calling each
      `Block::BatchedForward` in order

### Backward

- ***BlockSequence::BackwardFrom*** - [
  `template<size_t I, typename Delta> void BlockSequence::BackwardFrom(const Activations &A, const Delta &grad)`](src/BlockSequence.hpp)
    - Start the backward sweep at activation index `I` (rather than always `NumBlocks`)
    - Flexible primitive to backpropagate from anywhere, provided the correctly-shaped gradient coming in

- ***BlockSequence::BackwardAll*** - [
  `void BlockSequence::BackwardAll(const Activations &A, const OutputTensor &grad)`](src/BlockSequence.hpp)
    - Delegates to `BackwardFrom<NumBlocks>` - full backward from output to input
    - Gradients are accumulated into `Block` `Param` members

- ***BlockSequence::Backward*** - [
  `InputTensor BlockSequence::Backward(const OutputTensor &delta, const OutputTensor &, const InputTensor &)`](src/BlockSequence.hpp)
    - Uses cached `mActs` to run `backward_impl` and returns
      `InputTensor` gradient for the first block

- ***BlockSequence::backward_impl*** - [
  `template<size_t I, typename Delta> requires IsTensor<Delta> && std::is_same_v<Delta, std::tuple_element_t<I, ActivationsTuple>> auto BlockSequence::backward_impl(const ActivationsTuple &A, const Delta &delta)`](src/BlockSequence.hpp)
    - Starts with `Delta` (derivative of loss w.r.t. activation `I`), recurses down to `I == 1`, returning
      `InputTensor` gradient
    - At each `I`, calls `Block::Backward(delta, A[I], A[I-1])` or
      `Block::Backward(gradient wrt this block's output, this block's output, this block's input / previous block's output)`

- ***BlockSequence::BatchedBackwardFrom*** - [
  `template<size_t Batch, size_t I, typename Delta> void BlockSequence::BatchedBackwardFrom(const BatchedActivations<Batch> &A, const Delta &grad)`](src/BlockSequence.hpp)
    - Batched counterpart to `BackwardFrom` - start batched backward sweep at activation index `I`

- ***BlockSequence::BatchedBackwardAll*** - [
  `template<size_t Batch> void BlockSequence::BatchedBackwardAll(const BatchedActivations<Batch> &A, const PrependBatch<Batch, OutputTensor>::type &grad)`](src/BlockSequence.hpp)
    - Delegates to `BatchedBackwardFrom<Batch, NumBlocks>` - full batched backward from output to input

- ***BlockSequence::BatchedBackward*** - [
  `template<size_t Batch> PrependBatch<Batch, InputTensor>::type BlockSequence::BatchedBackward(const PrependBatch<Batch, OutputTensor>::type &delta, const PrependBatch<Batch, OutputTensor>::type &, const PrependBatch<Batch, InputTensor>::type &a_prev)`](src/BlockSequence.hpp)
    - Re-runs `BatchedForward` from
      `a_prev` to reconstruct activation cache, then runs `batched_backward_impl`, returning batched
      `InputTensor` gradient

- ***BlockSequence::batched_backward_impl*** - [
  `template<size_t Batch, size_t I, typename Delta> requires IsTensor<Delta> && std::is_same_v<Delta, std::tuple_element_t<I, BatchedActivationsTuple<Batch>>> auto BlockSequence::batched_backward_impl(const BatchedActivationsTuple<Batch> &A, const Delta &delta)`](src/BlockSequence.hpp)
    - Same logic as `backward_impl` but calls `Block::BatchedBackward` at each step

- ***BlockSequence::ZeroGrad*** - [`void BlockSequence::ZeroGrad()`](src/BlockSequence.hpp)
    - Calls `ZeroAllGrads` on each `Block`'s `all_params()`

### Serialization and Snapshot

- ***BlockSequence::Save*** - [
  `void BlockSequence::Save(const std::string &path) const`](src/BlockSequence.hpp)
    - Calls `SaveAll` on each `Block::all_params()`, which calls `Tensor` binary serialization

- ***BlockSequence::Load*** - [
  `void BlockSequence::Load(const std::string &path)`](src/BlockSequence.hpp)
    - Calls `LoadAll` on each `Block::all_params()`, which calls `Tensor` binary deserialization

- ***BlockSequence::peek*** - [
  `void BlockSequence::peek(SnapshotMap &out, const std::string &prefix) const`](src/BlockSequence.hpp)
    - Forwards peek calls from each `PeekableBlock` in `mBlocks` into `out`, keyed by `prefix + "block_N."`

- ***BlockSequence::Snap*** - [
  `[[nodiscard]] SnapshotMap BlockSequence::Snap() const`](src/BlockSequence.hpp)
    - Creates and fills a `SnapshotMap` (see [`Snapshot.hpp`](src/Snapshot.hpp)), calling `peek()` on any
      `PeekableBlock`s in `mBlocks`

---

## [TrainableTensorNetwork.hpp](src/TrainableTensorNetwork.hpp): The Network

Thin wrapper around `BlockSequence<Blocks...>` that adds an
`AdamState` and the loss-aware training API. All forward/backward/serialization calls delegate to the inner
`BlockSequence mSeq_`. Only `Update`, `TrainStep`, `BatchTrainStep`, `Fit`, `BatchFit`, and `RunEpoch` are
`TrainableTensorNetwork`-exclusive.

### `class TrainableTensorNetwork<Block... Blocks>`

- ***TrainableTensorNetwork*** - [
  `template<Block... Blocks> class TrainableTensorNetwork::TrainableTensorNetwork`](src/TrainableTensorNetwork.hpp)
    - Capstone object of the library; owns a `BlockSequence<Blocks...> mSeq_` and an `AdamState mAdam_`
    - All type aliases (`InputTensor`, `OutputTensor`,
      `Activations`, etc.) and inference/backward/serialization methods delegate directly to `mSeq_`
    - Exclusively owns the optimizer state and loss-parameterized training entry points

### Private Types and Members

- ***TrainableTensorNetwork::Seq*** - [`using TrainableTensorNetwork::Seq`](src/TrainableTensorNetwork.hpp)
    - `using Seq = BlockSequence<Blocks...>` - internal shorthand for the inner sequence type

- ***TrainableTensorNetwork::mSeq_*** - [`Seq TrainableTensorNetwork::mSeq_`](src/TrainableTensorNetwork.hpp)
    - The inner `BlockSequence<Blocks...>` that owns all blocks and activation caches

- ***TrainableTensorNetwork::mAdam_*** - [`AdamState TrainableTensorNetwork::mAdam_`](src/TrainableTensorNetwork.hpp)
    - `AdamState` instance; stepped once per `Update` call

### Type Aliases and Public Members

- ***TrainableTensorNetwork::InputTensor*** - [
  `using TrainableTensorNetwork::InputTensor`](src/TrainableTensorNetwork.hpp)
    - Delegates to `BlockSequence::InputTensor`

- ***TrainableTensorNetwork::OutputTensor*** - [
  `using TrainableTensorNetwork::OutputTensor`](src/TrainableTensorNetwork.hpp)
    - Delegates to `BlockSequence::OutputTensor`

- ***TrainableTensorNetwork::InSize*** - [
  `static constexpr size_t TrainableTensorNetwork::InSize`](src/TrainableTensorNetwork.hpp)
    - Delegates to `BlockSequence::InSize`

- ***TrainableTensorNetwork::OutSize*** - [
  `static constexpr size_t TrainableTensorNetwork::OutSize`](src/TrainableTensorNetwork.hpp)
    - Delegates to `BlockSequence::OutSize`

- ***TrainableTensorNetwork::NumBlocks*** - [
  `static constexpr size_t TrainableTensorNetwork::NumBlocks`](src/TrainableTensorNetwork.hpp)
    - Delegates to `BlockSequence::NumBlocks`

- ***TrainableTensorNetwork::TotalParamCount*** - [
  `static constexpr size_t TrainableTensorNetwork::TotalParamCount`](src/TrainableTensorNetwork.hpp)
    - Delegates to `BlockSequence::TotalParamCount`

- ***TrainableTensorNetwork::Activations*** - [
  `using TrainableTensorNetwork::Activations`](src/TrainableTensorNetwork.hpp)
    - Delegates to `BlockSequence::Activations`

- ***TrainableTensorNetwork::BatchedActivations*** - [
  `template<size_t Batch> using TrainableTensorNetwork::BatchedActivations`](src/TrainableTensorNetwork.hpp)
    - Delegates to `BlockSequence::BatchedActivations<Batch>`

- ***TrainableTensorNetwork::TrainableTensorNetwork()*** - [
  `TrainableTensorNetwork::TrainableTensorNetwork()`](src/TrainableTensorNetwork.hpp)
    - Default constructor `= default`

- ***TrainableTensorNetwork::block*** - [
  `template<size_t I> const auto &TrainableTensorNetwork::block() const`](src/TrainableTensorNetwork.hpp)
    - Delegates to `BlockSequence::block<I>()`

### Inference

- ***TrainableTensorNetwork::ForwardAll*** - [
  `[[nodiscard]] Activations TrainableTensorNetwork::ForwardAll(const InputTensor &x) const`](src/TrainableTensorNetwork.hpp)
    - Delegates to `BlockSequence::ForwardAll`

- ***TrainableTensorNetwork::Forward*** - [
  `[[nodiscard]] OutputTensor TrainableTensorNetwork::Forward(const InputTensor &x) const`](src/TrainableTensorNetwork.hpp)
    - Delegates to `BlockSequence::Forward`

- ***TrainableTensorNetwork::BatchedForwardAll*** - [
  `template<size_t Batch> [[nodiscard]] BatchedActivations<Batch> TrainableTensorNetwork::BatchedForwardAll(const PrependBatch<Batch, InputTensor>::type &X) const`](src/TrainableTensorNetwork.hpp)
    - Delegates to `BlockSequence::BatchedForwardAll<Batch>`

- ***TrainableTensorNetwork::BatchedForward*** - [
  `template<size_t Batch> [[nodiscard]] PrependBatch<Batch, OutputTensor>::type TrainableTensorNetwork::BatchedForward(const PrependBatch<Batch, InputTensor>::type &X) const`](src/TrainableTensorNetwork.hpp)
    - Delegates to `BlockSequence::BatchedForward<Batch>`

### Backward

- ***TrainableTensorNetwork::BackwardFrom*** - [
  `template<size_t I, typename Delta> void TrainableTensorNetwork::BackwardFrom(const Activations &A, const Delta &grad)`](src/TrainableTensorNetwork.hpp)
    - Delegates to `BlockSequence::BackwardFrom<I>`

- ***TrainableTensorNetwork::BackwardAll*** - [
  `void TrainableTensorNetwork::BackwardAll(const Activations &A, const OutputTensor &grad)`](src/TrainableTensorNetwork.hpp)
    - Delegates to `BlockSequence::BackwardAll`

- ***TrainableTensorNetwork::BatchedBackwardFrom*** - [
  `template<size_t Batch, size_t I, typename Delta> void TrainableTensorNetwork::BatchedBackwardFrom(const BatchedActivations<Batch> &A, const Delta &grad)`](src/TrainableTensorNetwork.hpp)
    - Delegates to `BlockSequence::BatchedBackwardFrom<Batch, I>`

- ***TrainableTensorNetwork::BatchedBackwardAll*** - [
  `template<size_t Batch> void TrainableTensorNetwork::BatchedBackwardAll(const BatchedActivations<Batch> &A, const PrependBatch<Batch, OutputTensor>::type &grad)`](src/TrainableTensorNetwork.hpp)
    - Delegates to `BlockSequence::BatchedBackwardAll<Batch>`

- ***TrainableTensorNetwork::ZeroGrad*** - [`void TrainableTensorNetwork::ZeroGrad()`](src/TrainableTensorNetwork.hpp)
    - Delegates to `BlockSequence::ZeroGrad`

### Optimizer

- ***TrainableTensorNetwork::Update*** - [
  `void TrainableTensorNetwork::Update(const float lr)`](src/TrainableTensorNetwork.hpp)
    - Calls `mAdam_.step()`, then `UpdateAll` (from [`NetworkUtil.hpp`](src/NetworkUtil.hpp)) on
      `mSeq_.all_params()` with `mAdam_` and `lr`

### Training

- ***TrainableTensorNetwork::TrainStep*** - [
  `void TrainableTensorNetwork::TrainStep(const InputTensor &x, const OutputTensor &grad, const float lr)`](src/TrainableTensorNetwork.hpp)
    - `ForwardAll` -> `ZeroGrad` -> `BackwardAll` -> `Update`
    - `grad` is `dLoss/dOutputTensor`, computed externally

- ***TrainableTensorNetwork::BatchTrainStep*** - [
  `template<size_t Batch> void TrainableTensorNetwork::BatchTrainStep(const PrependBatch<Batch, InputTensor>::type &X, const PrependBatch<Batch, OutputTensor>::type &grad, const float lr)`](src/TrainableTensorNetwork.hpp)
    - Batched `ForwardAll` -> `ZeroGrad` -> `BatchedBackwardAll` -> `Update`
    - `grad` is batched `dLoss/dOutputTensor`, computed externally

- ***TrainableTensorNetwork::Fit*** - [
  `template<typename Loss> float TrainableTensorNetwork::Fit(const InputTensor &x, const OutputTensor &target, const float lr)`](src/TrainableTensorNetwork.hpp)
    - Parameterized by `Loss` (satisfying
      `LossFunction<Loss, OutputTensor>`)
    - Computes loss and grad internally, then runs
      `TrainStep`-equivalent; returns loss value

- ***TrainableTensorNetwork::BatchFit*** - [
  `template<typename Loss, size_t Batch> float TrainableTensorNetwork::BatchFit(const PrependBatch<Batch, InputTensor>::type &X, const PrependBatch<Batch, OutputTensor>::type &Y, const float lr)`](src/TrainableTensorNetwork.hpp)
    - Batched counterpart to
      `Fit`
    - Averages per-sample gradients across the batch before backpropagating, returns mean loss

- ***TrainableTensorNetwork::RunEpoch*** - [
  `template<typename Loss, size_t Batch, size_t N, size_t... InDims, size_t... OutDims> float TrainableTensorNetwork::RunEpoch(const Tensor<N, InDims...> &X_data, const Tensor<N, OutDims...> &Y_data, std::mt19937 &rng, const float lr)`](src/TrainableTensorNetwork.hpp)
    - Run one full epoch: `Steps = N / Batch` rounds of `BatchFit`, returning average loss per step
    - `X_data` and `Y_data` must already be in network shape (`Tensor<InDims...> == InputTensor`,
      `Tensor<OutDims...> == OutputTensor`), enforced by `static_assert`
    - Creates temporary batch `Tensor<Batch, InDims...>`/`Tensor<Batch, OutDims...>` pairs where all `Batch` indices
    -
    - and samples `Batch` indices per step from `[0, N)` using `rng`, applied to both `Tensor`s in the same loop

### Serialization and Snapshot

- ***TrainableTensorNetwork::Save*** - [
  `void TrainableTensorNetwork::Save(const std::string &path) const`](src/TrainableTensorNetwork.hpp)
    - Delegates block params (including per-`Param` Adam moments `m`, `v`) to
      `BlockSequence::Save`, then reopens the file in append mode and writes the `mAdam_` POD trailer (`beta1`, `beta2`,
      `eps`, `mCorr`, `vCorr`, `t`)
    - Single-file format: `[block0 params][block1 params]...[blockN params][AdamState]`
    - Full optimizer round-trip — incremental training across runs resumes exactly where it left off

- ***TrainableTensorNetwork::Load*** - [
  `void TrainableTensorNetwork::Load(const std::string &path)`](src/TrainableTensorNetwork.hpp)
    - Delegates block params to
      `BlockSequence::Load` (compile-time shapes determine exact byte count consumed), then seeks to
      `end - sizeof(AdamState)` and reads the trailer back into `mAdam_`
    - Inverse of `Save` — restores weights, per-parameter Adam moments, and global Adam state in one call

- ***TrainableTensorNetwork::Snap*** - [
  `[[nodiscard]] SnapshotMap TrainableTensorNetwork::Snap() const`](src/TrainableTensorNetwork.hpp)
    - Delegates to `BlockSequence::Snap`

- ***is_trainable_network_v*** - [
  `template<typename T> inline constexpr bool is_trainable_network_v`](src/TrainableTensorNetwork.hpp)
    - ######

- ***IsTrainableNetwork*** - [`template<typename T> concept IsTrainableNetwork`](src/TrainableTensorNetwork.hpp)
    - ######

---

## [NetworkComposition.hpp](src/NetworkComposition.hpp): Network Creation and Composition

Defines the
`BlockRecipe` concept, the chain-resolution machinery, tuple-unpacking primitives, and composition helpers for creating and combining
`TrainableTensorNetwork`s.

### BlockRecipe Concept and Chain Resolution

- ***BlockRecipe*** - [`template<typename B> concept BlockRecipe`](src/NetworkComposition.hpp)
    - Declarable recipe to define a `Block` in a `TrainableTensorNetwork` template argument list
    - `BlockRecipe`s must define an `OutputTensor` type and alias a `Block` as `Resolve`
    - `BlockRecipe` argument lists passed to `NetworkBuilder` will be resolved into full `Block`s with chained
      `InputTensor` attributes

- ***BuildChain*** - [
  `template<typename Prev, BlockRecipe Last> struct BuildChain<Prev, Last>`](src/NetworkComposition.hpp)
    - Build `std::tuple` of `Block`s from a variadic argument list of `BlockRecipe`s
    - Base case for recursive `BuildChain`

- ***BuildChain*** - [
  `template<typename Prev, BlockRecipe Next, BlockRecipe... Rest> struct BuildChain<Prev, Next, Rest...>`](src/NetworkComposition.hpp)
    - Build `std::tuple` of `Block`s from a variadic argument list of `BlockRecipe`s
    - Recursive case: `std::tuple_cat` of
        - first `BlockRecipe`'s `Block` as given by its `Resolve` member
        - next `BlockRecipe`s' `Block`s
    - Used by `ApplyBuildChain`

- ***Input*** - [`template<size_t... Dims> struct Input`](src/NetworkComposition.hpp)
    - `BlockRecipe` type which begins and allows a variadic argument list of `BlockRecipe`s to be processed by
      `BuildChain` via
      `ApplyBuildChain`
    - Defines `OutputTensor = Tensor<Dims...>` to begin chain

- ***ApplyBuildChain*** - [
  `template<typename In, BlockRecipe... Rs> struct ApplyBuildChain<In, std::tuple<Rs...> >`](src/NetworkComposition.hpp)
    - Expects an `BlockRecipe<Input>` first and a trailing variadic list of `BlockRecipe`s passes them to `BuildChain`
    - Used in `NetworkBuilder` to define `BlockTuple`, a `TrainableTensorNetwork`'s tuple of `Block`s

### Tuple-Unpacking Primitives

- ***ConcretesToSequence*** - [`template<typename Tuple> struct ConcretesToSequence`](src/NetworkComposition.hpp)
    - Specialization: `template<Block... Bs> struct ConcretesToSequence<std::tuple<Bs...>>`
    - Unpacks a `std::tuple<Block...>` into `BlockSequence<Bs...>`
    - Used by `ComposeBlocks::ResolveImpl`

- ***ConcretesToNetwork*** - [`template<typename Tuple> struct ConcretesToNetwork`](src/NetworkComposition.hpp)
    - Specialization: `template<Block... Bs> struct ConcretesToNetwork<std::tuple<Bs...>>`
    - Unpacks a `std::tuple<Block...>` into `TrainableTensorNetwork<Bs...>`
    - Used by `NetworkBuilder`

### `struct ComposeBlocks`

Extremely useful `BlockRecipe`, taking in a variadic template list of *other* `BlockRecipe`s and creating a
`Block<BlockSequence>` from them via `ConcretesToSequence`

- ***ComposeBlocks*** - [
  `template<typename... Recipes> requires (BlockRecipe<Recipes> && ...) struct ComposeBlocks`](src/NetworkComposition.hpp)
    - Struct containing `ResolveChain` methods (several specializations) which unpack variadic lists of
      `BlockRecipe`s into a
      `BlockSequence` type, saved in `ComposeBlocks::type`

- ***ComposeBlocks::ResolveChain*** - [
  `template<typename In, typename Last> struct ComposeBlocks::ResolveChain<In, Last>`](src/NetworkComposition.hpp)
    - Wraps a variadic list of `BlockRecipe`s into a `std::tuple` of `Block`s
    - Used exclusively in `ResolveImpl` to unpack `BlockRecipe`s into a `BlockSequence`
    - Base case, resolving the last `BlockRecipe`'s `OutputTensor` by passing in the penultimate `BlockRecipe`'s
      `OutputTensor` and wrapping `Resolved` in a `std::tuple`

- ***ComposeBlocks::ResolveChain*** - [
  `template<typename In, typename First, typename... Rest> struct ComposeBlocks::ResolveChain<In, First, Rest...>`](src/NetworkComposition.hpp)
    - Wraps a variadic list of `BlockRecipe`s into a `std::tuple` of `Block`s
    - Used exclusively in `ResolveImpl` to unpack `BlockRecipe`s into a `BlockSequence`
    - Recursive case, resolving `First` by passing in `In` (starts as `IsTensor<InputT>` in `ResolveImpl`), then defines
      `Tail` by recursing on `Resolve`, finally wrapping `Resolved` and `Tail` in `std::tuple_cat`

- ***ComposeBlocks::LastOutputTensor*** - [
  `template<typename Last> struct ComposeBlocks::LastOutputTensor<Last>`](src/NetworkComposition.hpp)
    - Custom last-in-variadic getter, assuming that template args are `BlockRecipe`s, recursing until the
      `Last` is reached and finally grabbing `Last::OutputTensor`
    - Base case: `type = Last::OutputTensor`

- ***ComposeBlocks::LastOutputTensor*** - [
  `template<typename First, typename... Rest> struct ComposeBlocks::LastOutputTensor<First, Rest...>`](src/NetworkComposition.hpp)
    - Custom last-in-variadic getter, assuming that template args are `BlockRecipe`s, recursing until the
      `Last` is reached and finally grabbing `Last::OutputTensor`
    - Recursive case: `type = LastOutputTensor<Rest...>::type`

- ***ComposeBlocks::OutputTensor*** - [`using ComposeBlocks::OutputTensor`](src/NetworkComposition.hpp)
    - Type alias for `OutputTensor` of last `BlockRecipe` in `Recipes...`

- ***ComposeBlocks::ResolveImpl*** - [
  `template<typename InputT> requires IsTensor<InputT> struct ComposeBlocks::ResolveImpl`](src/NetworkComposition.hpp)
    - Implementation helper to take in `IsTensor<InputT>` and `Recipes...`
    - Runs `ResolveChain` to get a `std::tuple` of `Block`s, then calls `ConcretesToSequence` to produce
      `BlockSequence`, stored in `type`

- ***ComposeBlocks::Resolve*** - [
  `template<typename InputT> requires IsTensor<InputT> using ComposeBlocks::Resolve`](src/NetworkComposition.hpp)
    - Culmination: in compliance with `BlockRecipe`, `ComposeBlocks::Resolve` takes in
      `IsTensor<InputT>` and resolves to a
      `Block<BlockSequence>`

### `struct ComposeNetworks`

Thin convenience wrapper to take `Block...` lists from two `TrainableTensorNetwork`s and fuse them into one.

- ***Block*** - [
  `template<Block... BlocksA, Block... BlocksB> struct ComposeNetworks<TrainableTensorNetwork<BlocksA...>, TrainableTensorNetwork<BlocksB...> >`](src/NetworkComposition.hpp)
    - `static_assert` that `std::is_same_v<
    typename TrainableTensorNetwork<BlocksA...>::OutputTensor,
    typename TrainableTensorNetwork<BlocksB...>::InputTensor>`
    - Simple unpack: `type = TrainableTensorNetwork<BlocksA..., BlocksB...>`

### `struct NetworkBuilder`

Key builder helper to take variadic list of `BlockRecipe`s and create a
`TrainableTensorNetwork` from them. Preferred way to define a
`TrainableTensorNetwork` type because sizes can be written in `PyTorch` and
`TensorFlow` style, only writing output sizes, with intermediate connections deduced (here, at compile-time!).

- ***NetworkBuilder*** - [
  `template<typename In, typename... Recipes> requires requires { typename In::OutputTensor; } && IsTensor<typename In::OutputTensor> && (BlockRecipe<Recipes> && ...) struct NetworkBuilder`](src/NetworkComposition.hpp)
    - Takes variadic list of `BlockRecipe`s and creates a `TrainableTensorNetwork`
    - Calls `ApplyBuildChain` to get a `std::tuple` of `Block`s, then calls `ConcretesToNetwork`

## [Snapshot.hpp](src/Snapshot.hpp): Activation Snapshots

Runtime-typed storage for capturing named activation tensors. `SnapshotEntry` holds a shape vector and a flat
`float` copy — erasing the compile-time type so snapshots can be stored in a uniform `SnapshotMap` (
`unordered_map<string, SnapshotEntry>`). Used by visualization and debugging tools.

### `struct SnapShotEntry`

- ***SnapshotEntry***  - [`struct SnapshotEntry`](src/Snapshot.hpp)
    - `struct` to hold `data` and `shape` from `PeekableBlock`s' activation snapshots

- ***SnapshotEntry::shape*** - [`@doc: std::vector<size_t> SnapshotEntry::shape`](src/Snapshot.hpp)
    - `std::vector<size_t>` to hold an activation `Tensor`'s shape

- ***SnapshotEntry::data*** - [`@doc: std::vector<float> SnapshotEntry::data`](src/Snapshot.hpp)
    - `std::vector<float>` to hold an activation `Tensor`'s data

- ***SnapshotEntry*** - [`[[nodiscard]] size_t SnapshotEntry::total() const    `](src/Snapshot.hpp)
    - Getter for total size

- ***SnapshotEntry*** - [
  `[[nodiscard]] size_t SnapshotEntry::rows(const size_t outer_axis = 0) const`](src/Snapshot.hpp)
    - Getter for number of rows

- ***SnapshotEntry*** - [`[[nodiscard]] size_t SnapshotEntry::cols() const`](src/Snapshot.hpp)
    - Getter for number of columns


- ***SnapshotMap*** - [`using SnapshotMap`](src/Snapshot.hpp)
    - Type alias for `std::unordered_map<std::string, SnapshotEntry>`


- ***snap_add*** - [
  `template<size_t... Dims> void snap_add(SnapshotMap &out, const std::string &key, const Tensor<Dims...> &t)`](src/Snapshot.hpp)
    - Take a `Tensor`, a `std::string` key, and an existing `SnapshotMap` and add the `Tensor` as a `SnapshotEntry`

--- 

## [Dense.hpp](src/Dense.hpp): Fully-Connected Layer

Implements the general multidimensional dense layer (`DenseMDBlock`) and its recipe types. Weights have shape
`Tensor<OutDims..., InDims...>`; forward pass is a generalized matrix-vector product via
`ΣΠ`. Includes Adam optimizer state.

### Dense Block

- ***DenseMDBlock*** - [
  `template<size_t... InDims, size_t... OutDims, ActivationOp Act_> class DenseMDBlock<Tensor<InDims...>, Tensor<OutDims...>, Act_>`](src/Dense.hpp)
    - `Block` implementation for a generalized, multidimensional **Dense** layer
    - Input and output dimensions are variadic `size_t...`, letting **Dense
      ** contraction be far more general than the typical Rank-2 matrix multiplication

- ***DenseMBBlock::InputTensor*** - [`using DenseMDBlock::InputTensor`](src/Dense.hpp)
    - Alias: `InputTensor = Tensor<InDims...>`
- ***DenseMBBlock::OutputTensor*** - [`using DenseMDBlock::OutputTensor`](src/Dense.hpp)
    - Alias: `OutputTensor = Tensor<OutDims...>`
- ***DenseMBBlock::Act*** - [`using DenseMDBlock::Act`](src/Dense.hpp)
    - Alias: `Act = Act_`
        - `Act_` is a `DenseMDBlock` template argument, forced to be a valid `ActivationOp`

- ***DenseMDBlock::N_in*** - [`static constexpr size_t DenseMDBlock::N_in`](src/Dense.hpp)
    - Convenience member: `N_in = sizeof...(InDims)`
- ***DenseMDBlock::N_out*** - [`static constexpr size_t DenseMDBlock::N_out`](src/Dense.hpp)
    - Convenience member: `N_out = sizeof...(OutDims)`

- ***DenseMDBlock::W_*** - [`mutable LearnedContraction<InputTensor, OutputTensor, 0> DenseMDBlock::W__`](src/Dense.hpp)
    - `LearnedContraction` of all axes of `InputTensor` with all axes of `OutputTensor`

- ***DenseMDBlock::b_*** - [`Param<OutputTensor> DenseMDBlock::b_`](src/Dense.hpp)
    - `Param` whose underlying `Tensor` has type `OutputDims` (because `b_` is added to the output of the
      `W_` contraction)

- ***DenseMDBlock::all_params*** - [`auto DenseMDBlock::all_params()`](src/Dense.hpp)
    - Returns `std::tuple` of references to `W_` and `b_`
- ***DenseMDBlock::all_params*** - [`auto DenseMDBlock::all_params() const`](src/Dense.hpp)
    - Returns `std::tuple` of const references to `W_` and `b_`

- ***DenseMDBlock::DenseMDBlock*** - [`DenseMDBlock::DenseMDBlock()`](src/Dense.hpp)
    - Default constructor


- ***DenseMDBlock::Forward*** - [`OutputTensor DenseMDBlock::Forward(const InputTensor &x) const`](src/Dense.hpp)
    - Contracts `W_` and `InputTensor& x` using `ΣΠ`, adds `b_`, maps with `Act`

- ***DenseMDBlock::Backward*** - [
  `InputTensor DenseMDBlock::Backward(const OutputTensor &delta_A, const OutputTensor &a, const InputTensor &a_prev)`](src/Dense.hpp)
    - Computes `delta_z` from `delta_A`, propagates to `W_` and `b_`, passes `InputTensor delta_Input` upstream

- ***DenseMDBlock::BatchedForward*** - [
  `template<size_t Batch> Tensor<Batch, OutDims...> DenseMDBlock::BatchedForward(const Tensor<Batch, InDims...> &X) const`](src/Dense.hpp)
    - Contracts `W_` and `Tensor<Batch, InDims...>& X` using `ΣΠ`, adds `b_`, maps with `Act`

- ***DenseMDBlock::BatchedBackward*** - [
  `template<size_t Batch> Tensor<Batch, InDims...> DenseMDBlock::BatchedBackward(const Tensor<Batch, OutDims...> &delta_A, const Tensor<Batch, OutDims...> &a, const Tensor<Batch, InDims...> &a_prev)`](src/Dense.hpp)
    - Computes `delta_z` from `delta_A`, propagates to `W_` and `b_` (scaled by `Batch` count), passes
      `Tensor<Batch, InDims...> delta_Input` upstream

- ***DenseMD*** - [
  `template<typename OutT, ActivationOp Act_ = Linear> requires IsTensor<OutT> struct DenseMD`](src/Dense.hpp)
    - `BlockRecipe` for `DenseMDBlock`
    - Customarily, takes in a `Tensor` to be `OutputTensor` as well as an `ActivationOp`
    - `Resolve` takes an `IsTensor<InputT>` and defines the correct full-fledged `DenseMDBlock` type


- ***Dense*** - [`template<size_t N, ActivationOp Act_ = Linear> using Dense`](src/Dense.hpp)
    - Convenience wrapper around `BlockRecipe DenseMD` for the case where `OutT::Rank == 1`
    - User specifies how large the outgoing vector is to be and gets the corresponding `DenseMD` under the hood
    - Simple as: `using Dense = DenseMD<Tensor<N>, Act_>`

### Map Dense Block

DenseMD block which contracts the trailing (non-map) axes, doing so separately along specified leading 'map' axes. Almost like a generalization of a Batched Dense, but (A) for arbitrarily many batch axes and (B) native to the
`Block` (ie not just its `Batched` versions of functions). Motivation for inclusion is the **FFN** layers in
***TransformerBlock***s, where tokens in sequence (`[SeqLen, EmbDims...]`) are independently mapped through **FFN
** layers (becoming `[SeqLen, SomeProjectedEmbDims...]`). The same weights are used along each independent path.

- ***MapDenseMDBlock*** - [
  `template<size_t... InDims, size_t... PartOutDims, size_t N_map, ActivationOp Act_> class MapDenseMDBlock<Tensor<InDims...>, Tensor<PartOutDims...>, N_map, Act_>`](src/Dense.hpp)
    - `DenseMDBlock` that preserves first `N_map` axes, using shared weights to independently map the last
      `Rank - N_map` axes of `InDims...` to `PartOutDims...`

- ***MapDenseMDBlock::Split_*** - [`using MapDenseMDBlock::Split_`](src/Dense.hpp)
    - `SplitAt` resulting object, to be extracted from

- ***MapDenseMDBlock::MapSeq_*** - [`using MapDenseMDBlock::MapSeq_`](src/Dense.hpp)
    - `std::integer_sequence` with the first `N_map` axes of `InDims...`

- ***MapDenseMDBlock::ContractSeq_*** - [`using MapDenseMDBlock::ContractSeq_`](src/Dense.hpp)
    - `std::integer_sequence` with trailing axes to be contracted

- ***MapDenseMDBlock::OutDimSeq_*** - [`using MapDenseMDBlock::OutDimSeq_`](src/Dense.hpp)
    - `std::integer_sequence` representing final output shape: `[MapSeq_..., PartOutDims...]`

- ***MapDenseMDBlock::W_DimSeq_*** - [`using MapDenseMDBlock::W_DimSeq_`](src/Dense.hpp)
    - `std::integer_sequence` representing weight matrix for each transformation: `[PartOutDims_..., ContractSeq_...]`

- ***MapDenseMDBlock::InputTensor*** - [`using MapDenseMDBlock::InputTensor`](src/Dense.hpp)
    - Unpack `InDims...` to `Tensor<InDims...>`

- ***MapDenseMDBlock::OutputTensor*** - [`using MapDenseMDBlock::OutputTensor`](src/Dense.hpp)
    - Unpack `OutDimSeq_` to `Tensor` using `SeqToTensor`

- ***MapDenseMDBlock::W_Type*** - [`using MapDenseMDBlock::W_Type`](src/Dense.hpp)
    - Unpack `W_Dim_Seq_` to `Tensor` using `SeqToTensor`

- ***MapDenseMDBlock::BiasType*** - [`using MapDenseMDBlock::BiasType`](src/Dense.hpp)
    - Unpack `PartOutDims...` to `Tensor<PartOutDims...>`

- ***MapDenseMDBlock::Act*** - [`using MapDenseMDBlock::Act`](src/Dense.hpp)
    - Alias for `ActivationOp Act_`

- ***MapDenseMDBlock::N_contract*** - [`static constexpr size_t MapDenseMDBlock::N_contract`](src/Dense.hpp)
    - `N_contract = sizeof...(InDims) - N_map`

- ***MapDenseMDBlock::N_out_part*** - [`static constexpr size_t MapDenseMDBlock::N_out_part`](src/Dense.hpp)
    - `N_out_part = sizeof...(PartOutDims)`

- ***MapDenseMDBlock::MapVolume*** - [`static constexpr size_t MapDenseMDBlock::MapVolume`](src/Dense.hpp)
    - How many independent sub-`Tensor`s will we contract (one for each **map** index product)
    - `MapVolume = SeqProduct<MapSeq_>::value`

- ***MapDenseMDBlock::PartOutSize*** - [`static constexpr size_t MapDenseMDBlock::PartOutSize`](src/Dense.hpp)
    - Size of each projected copy at each **map** index product
    - `PartOutSize = (PartOutDims * ...)`

- ***MapDenseMDBlock::ContractSize*** - [`static constexpr size_t MapDenseMDBlock::ContractSize`](src/Dense.hpp)
    - Size contracted in each copy at each **map** index product
    - `ContractSize = SeqProduct<ContractSeq_>::value`

- ***MapDenseMDBlock::lc_*** - [
  `mutable LearnedContraction<InputTensor, OutputTensor, N_map> MapDenseMDBlock::lc_`](src/Dense.hpp)
    - Wrap `Tensor` of type `W_Type` into `Param` object

- ***MapDenseMDBlock::b_*** - [`Param<BiasType> MapDenseMDBlock::b_`](src/Dense.hpp)
    - Wrap `Tensor` of type `BiasType` into `Param` object

- ***MapDenseMDBlock::all_params()*** - [`auto MapDenseMDBlock::all_params()`](src/Dense.hpp)
    - Returns `std::tuple` of `Param&`s

- ***MapDenseMDBlock::all_params() const*** - [`auto MapDenseMDBlock::all_params() const`](src/Dense.hpp)
    - Returns `std::tuple` of `const Param&`s

- ***MapDenseMDBlock::MapDenseMDBlock()*** - [`MapDenseMDBlock::MapDenseMDBlock()`](src/Dense.hpp)
    - Default constructor, calls `XavierInitMD` on `W_`

- ***MapDenseMDBlock::Forward*** - [`OutputTensor MapDenseMDBlock::Forward(const InputTensor &x) const`](src/Dense.hpp)
    - Compute standard `ΣΠ<N_contract>(x, W_.value)`, add `b_`, wrap in `Act`

- ***MapDenseMDBlock::Backward*** - [
  `InputTensor MapDenseMDBlock::Backward(const OutputTensor &delta_A, const OutputTensor &a, const InputTensor &a_prev)`](src/Dense.hpp)
    - Backward pass, fills `W_.grad` and `b_.grad`
    - Similar logic as `DenseMD::Backward`

- ***MapDenseMDBlock::BatchedForward*** - [
  `template<size_t Batch> auto MapDenseMDBlock::BatchedForward(const PrependBatch<Batch, InputTensor>::type &X) const -> PrependBatch<Batch, OutputTensor>::type`](src/Dense.hpp)
    - Same internal logic as `MapDenseMDBlock::Forward`, but adds `Batch` axis upfront

- ***MapDenseMDBlock::BatchedBackward*** - [
  `template<size_t Batch> auto MapDenseMDBlock::BatchedBackward(const PrependBatch<Batch, OutputTensor>::type &delta_A, const PrependBatch<Batch, OutputTensor>::type &a, const PrependBatch<Batch, InputTensor>::type &a_prev) -> PrependBatch<Batch, InputTensor>::type`](src/Dense.hpp)
    - Batched backward pass, fills `W_.grad` and `b_.grad`
    - Similar logic as `DenseMD::BatchedBackward`

- ***MapDense*** - [
  `template<size_t N_map, typename PartOutT, ActivationOp Act_ = Linear> requires IsTensor<PartOutT> struct MapDense`](src/Dense.hpp)
    - `BlockRecipe` for `MapDenseMDBlock`
    - Takes in `N_map` and `Tensor<PartOutDims...> PartOutT`

---

## [Attention.hpp](src/Attention.hpp): Multi-Head Self-Attention

Implements scaled dot-product **mult-head self-attention
** over sequences of arbitrary-rank token embeddings. Forward-pass cache is stored as
`mutable` members. All four weight matrices (`W_Q`, `W_K`, `W_V`, `W_O`) are updated with Adam.

- ***MultiHeadAttentionBlock*** - [
  `template<size_t SeqLen, size_t Heads, bool Masked, size_t... EmbDims> class MultiHeadAttentionBlock`](src/Attention.hpp)
    - `Block` implementation for **mult-head self-attention** over sequences of arbitrary-rank token embeddings
    - Parameterized by `size_t SeqLen`, `size_t Heads`, `bool Masked` (causal mask), and `size_t...EmbDims`
    - When `Masked = true`, positions `k > q` in the attention score matrix are set to `-inf` before softmax

- ***MultiHeadAttentionBlock::InputTensor*** - [`using MultiHeadAttentionBlock::InputTensor`](src/Attention.hpp)
    - `InputTensor = Tensor<SeqLen, EmbDims...>`

- ***MultiHeadAttentionBlock::OutputTensor*** - [`using MultiHeadAttentionBlock::OutputTensor`](src/Attention.hpp)
    - `OutputTensor = Tensor<SeqLen, EmbDims...>`

- ***MultiHeadAttentionBlock::N_emb*** - [`static constexpr size_t MultiHeadAttentionBlock::N_emb`](src/Attention.hpp)
    - Embeddings rank (`sizeof...(EmbDims)`)

- ***MultiHeadAttentionBlock::EmbSize*** - [
  `static constexpr size_t MultiHeadAttentionBlock::EmbSize`](src/Attention.hpp)
    - Embeddings net size (`TensorDimsProduct<EmbDims...>::value`)

- ***MultiHeadAttentionBlock::HeadDim*** - [
  `static constexpr size_t MultiHeadAttentionBlock::HeadDim`](src/Attention.hpp)
    - `HeadDim = EmbSize / Heads`
    - `static_assert` that `EmbSize % Heads == 0`

- ***MultiHeadAttentionBlock::W_QKV_Type*** - [`using MultiHeadAttentionBlock::W_QKV_Type`](src/Attention.hpp)
    - Shape of `W_Q`, `W_K`, `W_V`
    - `[Heads, HeadDim, EmbDims...]`
    - For each head, contract `EmbDims...` from Input to get `QKV_Type`
        - `[Heads, HeadDim, EmbDims...] x [SeqLen, EmbDims...] -> [SeqLen, Heads, HeadDim]`

- ***MultiHeadAttentionBlock::W_O_Type*** - [`using MultiHeadAttentionBlock::W_O_Type`](src/Attention.hpp)
    - Shape of `W_O`
    - `[EmbDims..., Heads, HeadDim]`
    - Scores x `W_O` -> `OutputTensor`:
        - `[Heads, SeqLen, HeadDim] x [EmbDims..., Heads, HeadDim] -> [SeqLen, EmbDims...]`

- ***MultiHeadAttentionBlock::QKV_Type*** - [`using MultiHeadAttentionBlock::QKV_Type`](src/Attention.hpp)
    - Shape of `Q`, `K`, `V`
    - `[SeqLen, Heads, HeadDim]`
        - Right now, at each head, each element of the sequence is represented by `HeadDim`

- ***MultiHeadAttentionBlock::Scores_Type*** - [`using MultiHeadAttentionBlock::Scores_Type`](src/Attention.hpp)
    - Attention pattern matrix
    - `[Heads, SeqLen, SeqLen]`

- ***MultiHeadAttentionBlock::Attended_Type*** - [`using MultiHeadAttentionBlock::Attended_Type`](src/Attention.hpp)
    - Attended values after softmax-weighted sum (between attention pattern and `W_O` transformation)
    - `[Heads, SeqLen, HeadDim]`

- ***MultiHeadAttentionBlock::lc_Q_*** - [
  `mutable LearnedContraction<InputTensor, QKV_Type, 1> MultiHeadAttentionBlock::lc_Q_`](src/Attention.hpp)
    - Query projection `LearnedContraction`; weight accessed via `lc_Q_.W_`

- ***MultiHeadAttentionBlock::lc_K_*** - [
  `mutable LearnedContraction<InputTensor, QKV_Type, 1> MultiHeadAttentionBlock::lc_K_`](src/Attention.hpp)
    - Key projection `LearnedContraction`; weight accessed via `lc_K_.W_`

- ***MultiHeadAttentionBlock::lc_V_*** - [
  `mutable LearnedContraction<InputTensor, QKV_Type, 1> MultiHeadAttentionBlock::lc_V_`](src/Attention.hpp)
    - Value projection `LearnedContraction`; weight accessed via `lc_V_.W_`

- ***MultiHeadAttentionBlock::lc_O_*** - [
  `mutable LearnedContraction<QKV_Type, OutputTensor, 1> MultiHeadAttentionBlock::lc_O_`](src/Attention.hpp)
    - Output projection `LearnedContraction`; weight accessed via `lc_O_.W_`

- ***MultiHeadAttentionBlock::X_cache_*** - [`mutable InputTensor MultiHeadAttentionBlock::X_cache_`](src/Attention.hpp)
    - Cached `mutable` `Tensor` for `InputTensor x`, used by `Backward`

- ***MultiHeadAttentionBlock::Q_*** - [`mutable QKV_Type MultiHeadAttentionBlock::Q_`](src/Attention.hpp)
    - Cached `mutable` `Tensor` for `Q`, used by `Backward`

- ***MultiHeadAttentionBlock::K_*** - [`mutable QKV_Type MultiHeadAttentionBlock::K_`](src/Attention.hpp)
    - Cached `mutable` `Tensor` for `K`, used by `Backward`

- ***MultiHeadAttentionBlock::V_*** - [`mutable QKV_Type MultiHeadAttentionBlock::V_`](src/Attention.hpp)
    - Cached `mutable` `Tensor` for `V`, used by `Backward`

- ***MultiHeadAttentionBlock::attn_weights_*** - [
  `mutable Scores_Type MultiHeadAttentionBlock::attn_weights_`](src/Attention.hpp)
    - Cached `mutable` `Tensor` for attention matrix, used by `Backward`

- ***MultiHeadAttentionBlock::attended_*** - [
  `mutable Attended_Type MultiHeadAttentionBlock::attended_`](src/Attention.hpp)
    - Cached `mutable` `Tensor` for attended embeddings, used by `Backward`


- ***MultiHeadAttentionBlock::bQ_buf_*** - [
  `mutable std::vector<float> MultiHeadAttentionBlock::bQ_buf_`](src/Attention.hpp)
    - Cached `std::vector<float>` for batched `Q`, used by `BatchedBackward` (not a `Tensor` because
      `Batch` is a function template parameter, not a class parameter)

- ***MultiHeadAttentionBlock::bK_buf_*** - [
  `mutable std::vector<float> MultiHeadAttentionBlock::bK_buf_`](src/Attention.hpp)
    - Cached `std::vector<float>` for batched `K`, used by `BatchedBackward` (not a `Tensor` because
      `Batch` is a function template parameter, not a class parameter)

- ***MultiHeadAttentionBlock::bV_buf_*** - [
  `mutable std::vector<float> MultiHeadAttentionBlock::bV_buf_`](src/Attention.hpp)
    - Cached `std::vector<float>` for batched `V`, used by `BatchedBackward` (not a `Tensor` because
      `Batch` is a function template parameter, not a class parameter)

- ***MultiHeadAttentionBlock::battn_buf_*** - [
  `mutable std::vector<float> MultiHeadAttentionBlock::battn_buf_`](src/Attention.hpp)
    - Cached `std::vector<float>` for batched attention matrix, used by `BatchedBackward` (not a `Tensor` because
      `Batch` is a function template parameter, not a class parameter)

- ***MultiHeadAttentionBlock::battended_buf_*** - [
  `mutable std::vector<float> MultiHeadAttentionBlock::battended_buf_`](src/Attention.hpp)
    - Cached `std::vector<float>` for batched attended embeddings, used by `BatchedBackward` (not a `Tensor` because
      `Batch` is a function template parameter, not a class parameter)

- ***MultiHeadAttentionBlock::bcache_store*** - [
  `template<typename T> static void MultiHeadAttentionBlock::bcache_store(const T &t, std::vector<float> &buf)`](src/Attention.hpp)
    - Setter for batch cache `std::vector<float>`s

- ***MultiHeadAttentionBlock::bcache_load*** - [
  `template<typename T> static T MultiHeadAttentionBlock::bcache_load(const std::vector<float> &buf)`](src/Attention.hpp)
    - Setter for batch cache `std::vector<float>`s

- ***MultiHeadAttentionBlock::all_params()*** - [`auto MultiHeadAttentionBlock::all_params()`](src/Attention.hpp)
    - Returns `std::tuple` of `Param&`

- ***MultiHeadAttentionBlock::all_params() const*** - [
  `auto MultiHeadAttentionBlock::all_params() const`](src/Attention.hpp)
    - Returns `std::tuple` of `const Param&`

- ***MultiHeadAttentionBlock::attn_weights()*** - [
  `const Scores_Type &MultiHeadAttentionBlock::attn_weights() const`](src/Attention.hpp)
    - Getter for attention pattern matrix

- ***MultiHeadAttentionBlock::peek*** - [
  `void MultiHeadAttentionBlock::peek(SnapshotMap &out, const std::string &prefix) const`](src/Attention.hpp)
    - `PeekableBlock` satisfier function; returns `attn_weights_`

- ***MultiHeadAttentionBlock::MultiHeadAttentionBlock()*** - [
  `MultiHeadAttentionBlock::MultiHeadAttentionBlock()`](src/Attention.hpp)
    - Calls `XavierInitMD` on all weight `Tensor`s (`WQ_`, `WK_`, `WV_`, `WO_`)

- ***MultiHeadAttentionBlock::QKV_Contract*** - [
  `static constexpr auto MultiHeadAttentionBlock::QKV_Contract(const InputTensor &X, const W_QKV_Type &wm)`](src/Attention.hpp)
    - Abstraction of the contraction that occurs between `InputTensor x` and `W_QKV_Type w` (for **query**, **key**, and
      **value**)

- ***MultiHeadAttentionBlock::Forward*** - [
  `OutputTensor MultiHeadAttentionBlock::Forward(const InputTensor &X) const`](src/Attention.hpp)
    - Forward pass:
        - Take `InputTensor x` as input
        - Map to **heads**
        - Within each head, map to `Q`, `K`, and `V` representations
        - Compute attention matrix for each head (`softmax(Q . K)`)
        - Contract `V` with `x` according to attention matrix (attend)
        - Map from **heads** to normal embedding dimension, return
    - Excellent comments in the code

- ***MultiHeadAttentionBlock::Backward*** - [
  `InputTensor MultiHeadAttentionBlock::Backward(const OutputTensor &delta_A, const OutputTensor &a, const InputTensor &a_prev)`](src/Attention.hpp)
    - Backpropagates gradients through attended, attention pattern,`Q`, `K`, and
      `V` representations, passing same-shape gradient upstream
    - Extensive commenting below and in the code

- ***MultiHeadAttentionBlock::BatchedQKV_Contract*** - [
  `template<size_t Batch> static auto MultiHeadAttentionBlock::BatchedQKV_Contract(const Tensor<Batch, SeqLen, EmbDims...> &X, const W_QKV_Type &W)`](src/Attention.hpp)
    - Abstraction of the batched contraction that occurs between `PrependBatch<InputTensor> x` and `W_QKV_Type w` (for *
      *query**, **key**, and
      **value**)

- ***MultiHeadAttentionBlock::BatchedDAttended*** - [
  `template<size_t Batch> static auto MultiHeadAttentionBlock::BatchedDAttended(const Tensor<Batch, SeqLen, EmbDims...> &dA, const W_O_Type &WO)`](src/Attention.hpp)
    - Abstraction of the contraction that occurs between `delta_A` coming back and
      `WO_` to produce the attended tokens pre out-projection

- ***MultiHeadAttentionBlock::BatchedForward*** - [
  `template<size_t Batch> Tensor<Batch, SeqLen, EmbDims...> MultiHeadAttentionBlock::BatchedForward(const Tensor<Batch, SeqLen, EmbDims...> &X) const`](src/Attention.hpp)
    - Echoes `Forward`, but incorporates `Batch`
    - Batched forward pass goes from `[Batch, SeqLen, EmbDims...]` -> `[Batch, SeqLen, EmbDims...]`
    - Extensive comments in code

- ***MultiHeadAttentionBlock::BatchedBackward*** - [
  `template<size_t Batch> Tensor<Batch, SeqLen, EmbDims...> MultiHeadAttentionBlock::BatchedBackward(const Tensor<Batch, SeqLen, EmbDims...> &delta_A, const Tensor<Batch, SeqLen, EmbDims...> &a, const Tensor<Batch, SeqLen, EmbDims...> &a_prev)`](src/Attention.hpp)
    - Echoes 'Backward', but incorporates 'Batch'
    - Batched backward pass goes from `[Batch, SeqLen, EmbDims...]` -> `[Batch, SeqLen, EmbDims...]`

- ***MHAttention*** - [`template<size_t Heads, size_t... EmbDims> struct MHAttention`](src/Attention.hpp)
    - `BlockRecipe` for unmasked `MultiHeadAttentionBlock`
    - Takes in `size_t Heads` for head count and `size_t...EmbDims` indicating the dimensionality of the embeddings
    - `InputT` passed to `Resolve` should have its first axis be `SeqLen`
        - `Resolve = MultiHeadAttentionBlock<TensorFirstDim<InputT>::value, Heads, false, EmbDims...>`

- ***MHCausalAttention*** - [`template<size_t Heads, size_t... EmbDims> struct MHCausalAttention`](src/Attention.hpp)
    - `BlockRecipe` for causal (masked) `MultiHeadAttentionBlock`
    - Identical to `MHAttention` but passes `Masked = true` — positions `k > q` are masked to `-inf` before softmax
        - `Resolve = MultiHeadAttentionBlock<TensorFirstDim<InputT>::value, Heads, true, EmbDims...>`

---

### Forward pass

```cpp
OutputTensor Forward(const InputTensor &X) const {
    const float inv_sqrt = 1.f / std::sqrt(static_cast<float>(HeadDim));
    X_cache_ = X;

    // [SeqLen, EmbDims...] x [Heads, HeadDim, EmbDims...] -> [SeqLen, Heads, HeadDim]
    Q_ = QKV_Contract(X, WQ_.value);
    K_ = QKV_Contract(X, WK_.value);
    V_ = QKV_Contract(X, WV_.value);

    // Q: [SeqLen, Heads, HeadDim]
    // K: [SeqLen, Heads, HeadDim]
    // ?(Q, K) -> [Heads, SeqLen, SeqLen]
    // ? :
    //      batch axes: Q/K[1] = 'Heads`
    //      contract axes: Q/K[2] = 'HeadDim'
    auto rawQK = BatchContract<AxisList<1>{}, AxisList<1>{}, AxisList<2>{}, AxisList<2>{}, Mul, Add>(Q_, K_);
    rawQK *= inv_sqrt;
    // rawQK: [Heads, SeqLen_Q, SeqLen_K]
    // softmax needs to apply 'over the keys', so index 2
    attn_weights_ = Softmax<2>(rawQK);

    // attn_weights_: [Heads, SeqLen_Q, SeqLen_K]
    // ?(attn_weights_, V_) -> [Heads, SeqLen, HeadDim]
    // ? :
    //      batch axes: attn_weights_/V_[0] = 'Heads'
    //      contract axes: attn_weights_[2] = 'SeqLen_K', V_[0] = 'SeqLen'
    attended_ = BatchContract<AxisList<0>{}, AxisList<1>{}, AxisList<2>{}, AxisList<0>{}, Mul, Add>(
        attn_weights_, V_);

    // attended_: [Heads, SeqLen, HeadDim]
    // WO_: [EmbDims..., Heads, HeadDim]
    // ?(attended_, WO_) -> [SeqLen, EmbDims...]
    //      contract axes: attended_[0, 2] = ['Heads', 'HeadDim'], WO_[N_emb, N_emb+1] = ['Heads', 'HeadDim']
    return Contract<AxisList<0, 2>{}, AxisList<N_emb, N_emb + 1>{}, Mul, Add>(attended_, WO_.value);
}
```

### Backward pass

```cpp
InputTensor Backward(const OutputTensor &delta_A,
                     const OutputTensor & /*a*/,
                     const InputTensor & /*a_prev*/) {
    const float inv_sqrt = 1.f / std::sqrt(static_cast<float>(HeadDim));

    // delta_A: [SeqLen, EmbDims...]
    // attended_: [Heads, SeqLen, HeadDim]
    // ?(delta_A, attended_) -> [EmbDims..., Heads, HeadDim]
    //      contract delta_A[0] = 'SeqLen' with attended_[1] = 'SeqLen'
    // this collapses the gradient coming back to this Block along SeqLen
    // ie WO_ is punished at each free index for its effect on *all* tokens in the sequence
    // when we strip SeqLen, we have [EmbDims...] x [Heads, HeadDim]
    // the result is a sum of: Outer(delta_A[s, EmbDims...], attended_[Heads, s, HeadDim]) for s in SeqLen
    WO_.grad += Einsum<0, 1>(delta_A, attended_);

    // now need dLoss/d_attended_
    // delta_A: [SeqLen, EmbDims...]
    // WO_: [EmbDims..., Heads, HeadDim]
    // ?(delta_A, WO_) -> [SeqLen, Heads, HeadDim]
    //      ΣΠ contracted over EmbDims... is exactly what we want
    // NOTE: d_att ([SeqLen, Heads, HeadDim]) does not have the same shape as attended_ ([Heads, SeqLen, HeadDim])
    // this is because attended_ itself is an intermediary (no weights) and we can use flexible contraction below to grab the axes we want
    const auto d_attended = ΣΠ<N_emb>(delta_A, WO_.value);

    // now propagate gradients from d_attended to the attention pattern itself
    // d_attended: [SeqLen, Heads, HeadDim]
    // V_: [SeqLen, Heads, HeadDim]
    // ?(d_attended, V_) -> [Heads, SeqLen_Q, SeqLen_K]
    //      batch axes: d_attended[1] = 'Heads', V_[1] = 'Heads'
    //      contract axes: d_attended[2] = 'HeadDim', V_[2] = 'HeadDim'
    // NOTE: d_attended is 'A' in the BatchContract call, meaning its free axes (SeqLen) go first
    // so SeqLen_Q comes from d_attended, SeqLen_K comes from V_.

    // d_attended[s_q, h, d] carries 'how much did attended token s_q need to change?'
    // V_[s_k, h, d] carries 'what did key s_k contribute to values it attended?'
    // so d_attention[h, s_q, s_k] carries 'how much should the WEIGHT on key s_k change for query s_q?'
    //      -> this is exactly the dot product over d (over the whole HeadDim)
    // Why? because s_k's WEIGHT needs to change for s_q exactly according to:
    //      (A) how much the *attended* s_q should change (post s_k)
    //      (B) what s_k passes to s_q (regardless of weight)
    const auto d_attention = BatchContract<AxisList<1>{}, AxisList<1>{}, AxisList<2>{}, AxisList<2>{}, Mul,
        Add>(
        d_attended, V_);

    // V_[s_k, h, d] carries 'what did key s_k contribute to values it attended?'
    // V_ is combined with attn_weights_ to make attended_...
    //      V_ x attn_weights_ = attended_; d_attended_/d_V_ = attn_weights_... so feed grad to attn_weights_:
    // attn_weights_: [Heads, SeqLen_Q, SeqLen_K]
    // d_attended: [SeqLen, Heads, HeadDim]
    // ?(attn_weights_, d_attended) -> [Heads, SeqLen, HeadDim]
    //      (...d_V_ is a temp, so doesn't need V_'s exact shape, just the same info!)
    //      batch axes: attn_weights_[0] = 'Heads', d_attended[1] = 'Heads'
    //      contract axes: attn_weights_[1] = 'SeqLen_Q', d_attended[0] = 'SeqLen'
    const auto d_V = BatchContract<AxisList<0>{}, AxisList<1>{}, AxisList<1>{}, AxisList<0>{}, Mul, Add>(
        attn_weights_, d_attended);

    // take softmax derivative
    // again, axis = 2, over SeqLen_K (which is the axis over which Softmax was computed)
    auto d_scores = SoftmaxPrime<2>(d_attention, attn_weights_);
    d_scores *= inv_sqrt;

    // d_scores: [Heads, SeqLen_Q, SeqLen_K]
    // K_: [SeqLen, Heads, HeadDim]
    // ?(d_scores, K_) -> [Heads, SeqLen, HeadDim] (d_Q is also a temp...when WQ_.grad is updated, shapes will agree)
    //      batch axes: d_scores[0] = 'Heads', K_[1] = 'Heads'
    //      contract axes: d_scores[2] = 'SeqLen_K', K_[0] = 'SeqLen'
    const auto d_Q = BatchContract<AxisList<0>{}, AxisList<1>{}, AxisList<2>{}, AxisList<0>{}, Mul, Add>(
        d_scores, K_);
    // same logic for grad wrt K
    const auto d_K = BatchContract<AxisList<0>{}, AxisList<1>{}, AxisList<1>{}, AxisList<0>{}, Mul, Add>(
        d_scores, Q_);

    // now for WQ_, we have d_Q (coming back) times X (which multiplies with WQ_ to make Q!)
    // d_Q: [Heads, SeqLen, HeadDim]
    // X_cache_: [SeqLen, EmbDims...]
    // ?(d_Q, X_cache_) -> [Heads, HeadDim, EmbDims...] (this must be actual WQ_ shape)
    //      contract: d_Q[1] = 'SeqLen' with X_cache_[0] = 'SeqLen'
    // these weights are duly punished for all tokens!
    WQ_.grad += Einsum<1, 0>(d_Q, X_cache_);
    WK_.grad += Einsum<1, 0>(d_K, X_cache_);
    WV_.grad += Einsum<1, 0>(d_V, X_cache_);

    // now derivatives of all Q, K, V representations of the sequence need to be added
    // to represent what they all depend on: the input tokens

    // d_Q: [Heads, SeqLen, HeadDim]
    // WQ_: [Heads, HeadDim, EmbDims...]
    // ?(d_Q, WQ_) -> [SeqLen, EmbDims...]
    //      contract d_Q[0, 2] = ['Heads', 'HeadDim'] with WQ_[0, 1] = ['Heads', 'HeadDim']
    auto result = Contract<AxisList<0, 2>{}, AxisList<0, 1>{}, Mul, Add>(d_Q, WQ_.value);
    result += Contract<AxisList<0, 2>{}, AxisList<0, 1>{}, Mul, Add>(d_K, WK_.value);
    result += Contract<AxisList<0, 2>{}, AxisList<0, 1>{}, Mul, Add>(d_V, WV_.value);
    return result;
}
```

---

## [MoreNets.hpp](src/MoreNets.hpp) -- Helper Block types

---

### `class IdentityBlock<T>`

- ***IdentityBlock*** - [`template<IsTensor T> class IdentityBlock`](src/MoreNets.hpp)
    - `Block` that performs identity transformation
    - Used by `ResidualBlock`

- ***IdentityBlock::InputTensor*** - [`using IdentityBlock::InputTensor`](src/MoreNets.hpp)
    - Alias for `IsTensor T` template parameter

- ***IdentityBlock::OutputTensor*** - [`using IdentityBlock::OutputTensor`](src/MoreNets.hpp)
    - Alias for `IsTensor T` template parameter

- ***IdentityBlock::all_params*** - [`auto IdentityBlock::all_params()`](src/MoreNets.hpp)
    - Returns emtpy `std::tuple`

- ***IdentityBlock::all_params*** - [`auto IdentityBlock::all_params() const`](src/MoreNets.hpp)
    - Returns emtpy `std::tuple`


- ***IdentityBlock::Forward*** - [`static OutputTensor IdentityBlock::Forward(const InputTensor &x)`](src/MoreNets.hpp)
    - Return `X`

- ***IdentityBlock::Backward*** - [
  `static InputTensor IdentityBlock::Backward(const OutputTensor &delta_A, const OutputTensor &a, const InputTensor &a_prev)`](src/MoreNets.hpp)
    - Return `delta_A`

- ***IdentityBlock::BatchedForward*** - [
  `template<size_t Batch> static auto IdentityBlock::BatchedForward(const PrependBatch<Batch, InputTensor>::type &X) -> PrependBatch<Batch, OutputTensor>::type`](src/MoreNets.hpp)
    - Return `X`

- ***IdentityBlock::BatchedBackward*** - [
  `template<size_t Batch> static auto IdentityBlock::BatchedBackward(const PrependBatch<Batch, OutputTensor>::type &delta_A, const PrependBatch<Batch, OutputTensor>::type &a, const PrependBatch<Batch, InputTensor>::type &a_prev) -> PrependBatch<Batch, InputTensor>::type`](src/MoreNets.hpp)
    - Return `delta_A`

---

### `class ParallelBlock<BlockA, BlockB>`

- ***ParallelBlock*** - [`template<Block BlockA, Block BlockB> class ParallelBlock`](src/MoreNets.hpp)
    - Take two `Block`s who have the same `InputTensor`s and
      `OutputTensor`s and compute their forward and backward passes, effectively, in parallel
    - Forward pass results are summed

- ***ParallelBlock::InputTensor*** - [`using ParallelBlock::InputTensor`](src/MoreNets.hpp)
    - Alias for `BlockA::`/`BlockB::InputTensor`

- ***ParallelBlock::OutputTensor*** - [`using ParallelBlock::OutputTensor`](src/MoreNets.hpp)
    - Alias for `BlockA::`/`BlockB::OutputTensor`

- ***ParallelBlock::a_*** - [`BlockA ParallelBlock::a_`](src/MoreNets.hpp)
    - `Block` member variable for first `Block BlockA`

- ***ParallelBlock::b_*** - [`BlockB ParallelBlock::b_`](src/MoreNets.hpp)
    - `Block` member variable for second `Block BlockB`

- ***ParallelBlock::a_out_*** - [`mutable OutputTensor ParallelBlock::a_out_`](src/MoreNets.hpp)
    - `mutable` cache for `a_`'s intermediate resulting `OutputTensor`, used in backward pass

- ***ParallelBlock::b_out_*** - [`mutable OutputTensor ParallelBlock::b_out_`](src/MoreNets.hpp)
    - `mutable` cache for `b_`'s intermediate resulting `OutputTensor`, used in backward pass

- ***ParallelBlock::a_out_buf_*** - [`mutable std::vector<float> ParallelBlock::a_out_buf_`](src/MoreNets.hpp)
    - `mutable` cache for batched intermediate forward results
    - Uses `std::vector<float>` because `Batch` is not a class parameter

- ***ParallelBlock::b_out_buf_*** - [`mutable std::vector<float> ParallelBlock::b_out_buf_`](src/MoreNets.hpp)
    - `mutable` cache for batched intermediate forward results
    - Uses `std::vector<float>` because `Batch` is not a class parameter

- ***ParallelBlock::block_a*** - [`const BlockA &ParallelBlock::block_a() const`](src/MoreNets.hpp)
    - `const &` getter for `BlockA a_`

- ***ParallelBlock::block_b*** - [`const BlockB &ParallelBlock::block_b() const`](src/MoreNets.hpp)
    - `const &` getter for `BlockB b_`

- ***ParallelBlock::peek*** - [
  `void ParallelBlock::peek(SnapshotMap &out, const std::string &prefix) const`](src/MoreNets.hpp)
    - Forwards peek to `a_` and `b_` if they satisfy `PeekableBlock`; no extra prefix added (transparency layer)

- ***ParallelBlock::all_params*** - [`auto ParallelBlock::all_params()`](src/MoreNets.hpp)
    - `std::tuple_cat` of `a_.all_params()` and `b_.all_params()`

- ***ParallelBlock::all_params*** - [`auto ParallelBlock::all_params() const`](src/MoreNets.hpp)
    - `std::tuple_cat` of `a_.all_params()` and `b_.all_params()`


- ***ParallelBlock::Forward*** - [`OutputTensor ParallelBlock::Forward(const InputTensor &x) const`](src/MoreNets.hpp)
    - Call `Forward` on `a_` and `b_` and return their sum
    - Caches intermediate results

- ***ParallelBlock::Backward*** - [
  `InputTensor ParallelBlock::Backward(const OutputTensor &delta_A, const OutputTensor &a, const InputTensor &a_prev)`](src/MoreNets.hpp)
    - Call `Backward` on `a_out_` and `b_out_` (cached individual results) and pass their sum upstream

- ***ParallelBlock::BatchedForward*** - [
  `template<size_t Batch> auto ParallelBlock::BatchedForward(const PrependBatch<Batch, InputTensor>::type &X) const -> PrependBatch<Batch, OutputTensor>::type`](src/MoreNets.hpp)
    - Call `BatchedForward` on `a_` and `b_` and return their sum
    - Caches intermediate results

- ***ParallelBlock::BatchedBackward*** - [
  `template<size_t Batch> auto ParallelBlock::BatchedBackward(const PrependBatch<Batch, OutputTensor>::type &delta_A, const PrependBatch<Batch, OutputTensor>::type &a, const PrependBatch<Batch, InputTensor>::type &a_prev) -> PrependBatch<Batch, InputTensor>::type`](src/MoreNets.hpp)
    - Call `BatchedBackward` on `a_out_buf_` and `b_out_buf_` (cached individual results) and pass their sum upstream

- ***Parallel*** - [`template<typename RecipeA, typename RecipeB> struct Parallel`](src/MoreNets.hpp)
    - `BlockRecipe` for `ParallelBlock`

- ***Parallel::OutputTensor*** - [`using Parallel::OutputTensor`](src/MoreNets.hpp)
    - Alias for output `Tensor` type

- ***Parallel::Resolve*** - [`template<IsTensor InputT> using Parallel::Resolve`](src/MoreNets.hpp)
    - Given `IsTensor InputT`, define `ParallelBlock` type that should be created

---

### `using ResidualBlock<B>`

- ***ResidualBlock*** - [`template<Block B> using ResidualBlock`](src/MoreNets.hpp)
    - `Block` type alias for residual connections, defined as a `ParallelBlock` with some `Block B` and `IdentityBlock`
    - `ParallelBlock` handles both forward and backward passes!

- ***Residual*** - [`template<typename Recipe> struct Residual`](src/MoreNets.hpp)
    - `Block` type alias for residual connections, defined as a `ParallelBlock` with some `Block B` and `IdentityBlock`
    - `ParallelBlock` handles both forward and backward passes!


- ***Residual::OutputTensor*** - [`using Residual::OutputTensor`](src/MoreNets.hpp)
    - Alias for output `Tensor` type

- ***Residual::Resolve*** - [`template<IsTensor InputT> using Residual::Resolve`](src/MoreNets.hpp)
    - Given `IsTensor T`, define `ResidualBlock` type that should be created

---

### `class TransposeBlock<InnerBlock>`

- ***TransposeBlock*** - [`template<Block InnerBlock> class TransposeBlock`](src/MoreNets.hpp)
    - Wraps a
      `Block InnerBlock` and runs its forward and backward passes on the transposed input, transposing results back out
    - Requires `InputTensor == OutputTensor` (transposing preserves shape)

- ***TransposeBlock::InputTensor*** - [`using TransposeBlock::InputTensor`](src/MoreNets.hpp)
    - Alias for `InnerBlock::InputTensor`

- ***TransposeBlock::OutputTensor*** - [`using TransposeBlock::OutputTensor`](src/MoreNets.hpp)
    - Alias for `InnerBlock::OutputTensor`

- ***TransposeBlock::inner_*** - [`mutable InnerBlock TransposeBlock::inner_`](src/MoreNets.hpp)
    - The wrapped `Block`

- ***TransposeBlock::inner*** - [`const InnerBlock &TransposeBlock::inner() const`](src/MoreNets.hpp)
    - `const &` getter for `inner_`

- ***TransposeBlock::all_params*** - [`auto TransposeBlock::all_params()`](src/MoreNets.hpp)
    - Delegates to `inner_.all_params()`

- ***TransposeBlock::Forward*** - [`OutputTensor TransposeBlock::Forward(const InputTensor &x) const`](src/MoreNets.hpp)
    - `Permute<1,0>(inner_.Forward(Permute<1,0>(x)))`

- ***TransposeBlock::Backward*** - [
  `InputTensor TransposeBlock::Backward(const OutputTensor &delta_A, const OutputTensor &a, const InputTensor &a_prev)`](src/MoreNets.hpp)
    - Transposes all arguments into `inner_`'s perspective, calls `inner_.Backward`, transposes result back

- ***TransposeBlock::BatchedForward*** - [
  `template<size_t Batch> auto TransposeBlock::BatchedForward(const PrependBatch<Batch, InputTensor>::type &X) const -> PrependBatch<Batch, OutputTensor>::type`](src/MoreNets.hpp)
    - `Permute<0,2,1>` on `Tensor<Batch, R, C>` gives `Tensor<Batch, C, R>`, forward through `inner_`, permute back

- ***TransposeBlock::BatchedBackward*** - [
  `template<size_t Batch> auto TransposeBlock::BatchedBackward(const PrependBatch<Batch, OutputTensor>::type &delta_A, const PrependBatch<Batch, OutputTensor>::type &a, const PrependBatch<Batch, InputTensor>::type &a_prev) -> PrependBatch<Batch, InputTensor>::type`](src/MoreNets.hpp)
    - Same as `Backward` with batched `Permute<0,2,1>` on all arguments

- ***Transposed*** - [`template<typename InnerRecipe> struct Transposed`](src/MoreNets.hpp)
    - `BlockRecipe` for `TransposeBlock`

- ***Transposed::OutputTensor*** - [`using Transposed::OutputTensor`](src/MoreNets.hpp)
    - Alias for `InnerRecipe::OutputTensor`

- ***Transposed::Resolve*** - [`template<IsTensor InputT> using Transposed::Resolve`](src/MoreNets.hpp)
    - Given `IsTensor InputT`, define `TransposeBlock` type wrapping the resolved inner block

---

### `class LayerNormBlock<SeqLen, EmbDims...>`

**Layer Normalization**: z-score transformation around learned mean and variance!

- ***LayerNormBlock*** - [`template<size_t SeqLen, size_t... EmbDims> class LayerNormBlock`](src/MoreNets.hpp)
  -
  `Block` for layer normalization: z-score transformation around learned mean and variance, performed independently over each
  `size_t...EmbDims` in `SeqLen`

- ***LayerNormBlock::EmbSize*** - [`static constexpr size_t LayerNormBlock::EmbSize`](src/MoreNets.hpp)
    - Total element count of (Rank-1 mandated) `EmbDims...`

- ***LayerNormBlock::inv_emb*** - [`static constexpr float LayerNormBlock::inv_emb`](src/MoreNets.hpp)
    - Precompute `1.f / EmbSize`

- ***LayerNormBlock::InputTensor*** - [`using LayerNormBlock::InputTensor`](src/MoreNets.hpp)
    - Alias around `Tensor<SeqLen, EmbDims...>`

- ***LayerNormBlock::OutputTensor*** - [`using LayerNormBlock::OutputTensor`](src/MoreNets.hpp)
    - Alias around `Tensor<SeqLen, EmbDims...>`

- ***LayerNormBlock::Scale_Type*** - [`using LayerNormBlock::Scale_Type`](src/MoreNets.hpp)
    - Alias around the type to be scaled (each `Tensor<EmbDims...>`)

- ***LayerNormBlock::Sigma_Type*** - [`using LayerNormBlock::Sigma_Type`](src/MoreNets.hpp)
    - Scalar multiple for each of the `SeqLen` sub`Tensor<EmbDims...>`s

- ***LayerNormBlock::gamma_*** - [`Param<Scale_Type> LayerNormBlock::gamma_`](src/MoreNets.hpp)
    - Learned `Scale_Type` parameter used in Hadamard product to stretch distribution for each token

- ***LayerNormBlock::beta_*** - [`Param<Scale_Type> LayerNormBlock::beta_`](src/MoreNets.hpp)
    - Learned `Scale_Type` parameter used in elementwise sum to translate/shift mean for each token

- ***LayerNormBlock::x_hat_*** - [`mutable InputTensor LayerNormBlock::x_hat_`](src/MoreNets.hpp)
    - `mutable` cache for pre-`gamma_`- and -`beta_`-transformed tokens

- ***LayerNormBlock::inv_sigma_*** - [`mutable Sigma_Type LayerNormBlock::inv_sigma_`](src/MoreNets.hpp)
    - `mutable` cache for precomputed inverse of standard deviation

- ***LayerNormBlock::LayerNormBlock*** - [`LayerNormBlock::LayerNormBlock()`](src/MoreNets.hpp)
    - Default construct, fill `gamma_` with `1.f`

- ***LayerNormBlock::all_params*** - [`auto LayerNormBlock::all_params()`](src/MoreNets.hpp)
    - Return `std::tuple` of `Param&` for `gamma_` and `beta_`

- ***LayerNormBlock::all_params*** - [`auto LayerNormBlock::all_params() const`](src/MoreNets.hpp)
    - Return `std::tuple` of `const Param&` for `gamma_` and `beta_`


- ***LayerNormBlock::Forward*** - [`OutputTensor LayerNormBlock::Forward(const InputTensor &X) const`](src/MoreNets.hpp)
    - Subtract mean from each token, divide by standard deviation, apply learned elementwise `gamma_` and
      `beta_` transformation

- ***LayerNormBlock::Backward*** - [
  `InputTensor LayerNormBlock::Backward(const OutputTensor &delta_A, const OutputTensor &a, const InputTensor &a_prev)`](src/MoreNets.hpp)
    - Sum `delta_A` over all tokens for `beta_.grad`
    - Sum `delta_A * x_hat_` over all tokens for `gamma_.grad` (product rule 101)
    - Reverse `gamma_`- and `beta_`-transformations, reverse z-score, pass gradient back upstream

- ***LayerNormBlock::BatchedForward*** - [
  `template<size_t Batch> auto LayerNormBlock::BatchedForward(const PrependBatch<Batch, InputTensor>::type &X) const -> PrependBatch<Batch, OutputTensor>::type`](src/MoreNets.hpp)
    - Slices each sample out of the batch, calls `Forward`, writes result back
    - Layer norm has no cross-sample dependencies so per-sample dispatch is correct

- ***LayerNormBlock::BatchedBackward*** - [
  `template<size_t Batch> auto LayerNormBlock::BatchedBackward(const PrependBatch<Batch, OutputTensor>::type &delta_A, const PrependBatch<Batch, OutputTensor>::type &a, const PrependBatch<Batch, InputTensor>::type &a_prev) -> PrependBatch<Batch, InputTensor>::type`](src/MoreNets.hpp)
    - Slices each sample out of the batch, calls `Backward`, writes result back
    - `gamma_.grad` and `beta_.grad` accumulate across the loop then are scaled by `inv_batch`

- ***LayerNorm*** - [`template<size_t... EmbDims> struct LayerNorm`](src/MoreNets.hpp)
    - `BlockRecipe` for `LayerNormBlock`

- ***LayerNorm::OutputTensor*** - [`using LayerNorm::OutputTensor`](src/MoreNets.hpp)
    - Placeholder output type for recipe chain resolution

- ***LayerNorm::Resolve*** - [`template<IsTensor InputT> using LayerNorm::Resolve`](src/MoreNets.hpp)
    - Extracts `SeqLen` from `InputT` via `TensorFirstDim` and resolves to `LayerNormBlock<SeqLen, EmbDims...>`

---

## [TransformerBlock.hpp](src/TransformerBlock.hpp) -- Transformer Block (Pre/Post-norm)

```cpp
// sequence of 28 tokens, each embedded in R^28, 7 heads, FFN projects through 64
NetworkBuilder<
    Input<Tensor<28, 28>>,
    Transformer<7, 64>,
    Transformer<7, 64>,
    MapDense<1, Tensor<2>>,   // per-token binary classifier
    SoftmaxBlock<1>
>::type net;

std::mt19937 rng{42};
net.RunEpoch<CEL, 32>(X_train, Y_train, rng, 1e-3f);
```

---

### `class TransformerBlock<InputT, Heads, FFNHidden, PreNorm, Masked>`

Implementation is purely a composition; all detailed work is handled by denizen `Block`s.

- ***TransformerBlock*** - [
  `template<size_t SeqLen, size_t EmbDim, size_t Heads, size_t FFNHidden, bool PreNorm, bool Masked> class TransformerBlock<Tensor<SeqLen, EmbDim>, Heads, FFNHidden, PreNorm, Masked>`](src/TransformerBlock.hpp)
    - `Block` that defines standard **Transformer Block**: Multi-headed Attention + FFN with optional LayerNorm.
    - `bool PreNorm` (default `true`): LayerNorm precedes each sub-layer inside the residual (modern default);
      `false` matches the original 2017 paper — LayerNorm follows each residual addition.
    - `bool Masked` (default `false`): passed through to `MultiHeadAttentionBlock` for causal masking.
    - Pre-norm topology:  `Residual(LN → MHA)`,  `Residual(LN → FFN)`
    - Post-norm topology: `Residual(MHA) → LN`,  `Residual(FFN) → LN`

- ***TransformerBlock::MHABlock_*** - [`using TransformerBlock::MHABlock_`](src/TransformerBlock.hpp)
    - `MultiHeadAttentionBlock<SeqLen, Heads, Masked, EmbDim>` — shared by both norm orderings

- ***TransformerBlock::MHASub_*** - [`using TransformerBlock::MHASub_`](src/TransformerBlock.hpp)
    - Pre-norm MHA sub-sequence: `BlockSequence<LayerNormBlock, MHABlock_>`

- ***TransformerBlock::FFNOnly_*** - [`using TransformerBlock::FFNOnly_`](src/TransformerBlock.hpp)
    - FFN without LayerNorm (post-norm path): `BlockSequence<MapDenseMDBlock, MapDenseMDBlock>`

- ***TransformerBlock::FFNSub_*** - [`using TransformerBlock::FFNSub_`](src/TransformerBlock.hpp)
    - Pre-norm FFN sub-sequence: `BlockSequence<LayerNormBlock, MapDenseMDBlock, MapDenseMDBlock>`

- ***TransformerBlock::Inner_*** - [`using TransformerBlock::Inner_`](src/TransformerBlock.hpp)
    - Full topology selected by `PreNorm`:
        - `true`:  `BlockSequence<Residual(LN+MHA),  Residual(LN+FFN)>`
        - `false`: `BlockSequence<Residual(MHA), LN, Residual(FFN), LN>`

- ***TransformerBlock::inner_*** - [`Inner_ TransformerBlock::inner_`](src/TransformerBlock.hpp)
    - Instance of topology defined in `Inner_`

- ***TransformerBlock::InputTensor*** - [`using TransformerBlock::InputTensor`](src/TransformerBlock.hpp)
    - Alias for `Tensor<SeqLen, EmbDim>`

- ***TransformerBlock::OutputTensor*** - [`using TransformerBlock::OutputTensor`](src/TransformerBlock.hpp)
    - Alias for `Tensor<SeqLen, EmbDim>`

- ***TransformerBlock::peek*** - [
  `void TransformerBlock::peek(SnapshotMap &out, const std::string &prefix) const`](src/TransformerBlock.hpp)
    - Forwards peek to `inner_`, propagating the prefix so nested peekable blocks (e.g.
      `MultiHeadAttentionBlock`) are reachable

- ***TransformerBlock::all_params*** - [`auto TransformerBlock::all_params()`](src/TransformerBlock.hpp)
    - Return `std::tuple` from `inner_.all_params()`

- ***TransformerBlock::all_params*** - [`auto TransformerBlock::all_params() const`](src/TransformerBlock.hpp)
    - Return `std::tuple` from `inner_.all_params()`


- ***TransformerBlock::Forward*** - [
  `OutputTensor TransformerBlock::Forward(const InputTensor &x) const`](src/TransformerBlock.hpp)
    - Forward pass, simply: `inner_.Forward(x)`

- ***TransformerBlock::Backward*** - [
  `InputTensor TransformerBlock::Backward(const OutputTensor &delta_A, const OutputTensor &a, const InputTensor &a_prev)`](src/TransformerBlock.hpp)
    - Backward pass, simply: `inner_.Backward(delta_A, a, a_prev)`

- ***TransformerBlock::BatchedForward*** - [
  `template<size_t Batch> auto TransformerBlock::BatchedForward(const PrependBatch<Batch, InputTensor>::type &X) const -> PrependBatch<Batch, OutputTensor>::type`](src/TransformerBlock.hpp)
    - Batched forward pass, simply: `inner_.template BatchedForward<Batch>(X)`

- ***TransformerBlock::BatchedBackward*** - [
  `template<size_t Batch> auto TransformerBlock::BatchedBackward(const PrependBatch<Batch, OutputTensor>::type &delta_A, const PrependBatch<Batch, OutputTensor>::type &a, const PrependBatch<Batch, InputTensor>::type &a_prev) -> PrependBatch<Batch, InputTensor>::type`](src/TransformerBlock.hpp)
    - Batched backward pass, simply: `inner_.template BatchedBackward<Batch>(delta_A, a, a_prev)`

- ***Transformer*** - [
  `template<size_t Heads, size_t FFNHidden, bool PreNorm, bool Masked> struct Transformer`](src/TransformerBlock.hpp)
    - `BlockRecipe` struct for building a `TransformerBlock`
    - Parameterized by `Heads`, `FFNHidden`, `PreNorm` (default `true`), and `Masked` (default `false`)
    - Remaining dimensions (`SeqLen`, `EmbDim`) are deduced from `InputT` passed to `Resolve`

- ***Transformer::OutputTensor*** - [`using Transformer::OutputTensor`](src/TransformerBlock.hpp)
    - Placeholder for `BlockRecipe` concept check: `Tensor<1, Heads>` guarantees `EmbSize % Heads == 0`

- ***Transformer::Resolve*** - [`template<IsTensor InputT> using Transformer::Resolve`](src/TransformerBlock.hpp)
    - When given `IsTensor InputT`, create full `TransformerBlock<InputT, Heads, FFNHidden, PreNorm, Masked>`

---

## [BranchTrainer.hpp](src/BranchTrainer.hpp) -- Shared-Trunk Multi-Head Training

A variadic utility for training a single shared trunk network with multiple independent head networks branching off at arbitrary layer indices ("taps"). Each head trains on its own loss and its own learning rate; the trunk receives the summed gradient from every active head (plus an optional trunk-output loss). Each network owns its own Adam state.

---

- ***NoLoss*** - [`struct NoLoss`](src/BranchTrainer.hpp)
    - Sentinel type. Pass as `TrunkLoss` or any
      `HeadLosses...` entry to suppress loss computation and gradient contribution for that branch.

- ***is_head_branch_v*** - [`template<typename T> inline constexpr bool is_head_branch_v`](src/BranchTrainer.hpp)
    - Type trait to SFINAE-match only `HeadBranch` objects

- ***IsHeadBranch*** - [`template<typename T> concept IsHeadBranch`](src/BranchTrainer.hpp)
    - `concept` wrapper around `is_head_branch_v`

- ***HeadBranch*** - [
  `template<size_t TapIndex, typename HeadNet, typename HeadLoss = NoLoss> requires IsTrainableNetwork<HeadNet> && (std::is_same_v<HeadLoss, NoLoss> || LossFunction<HeadLoss, typename HeadNet::OutputTensor>) struct HeadBranch`](src/BranchTrainer.hpp)
    - Wrapper pairing a head network with its trunk tap index and loss function

- ***HeadBranch::Tap*** - [`static constexpr size_t HeadBranch::Tap`](src/BranchTrainer.hpp)
    - Index of trunk network activations to take as input to this `HeadBranch`

- ***HeadBranch::Net*** - [`using HeadBranch::Net`](src/BranchTrainer.hpp)
    - Type alias for underlying `TrainableTensorNetwork` type in this `HeadBranch`

- ***HeadBranch::Loss*** - [`using HeadBranch::Loss`](src/BranchTrainer.hpp)
    - Loss function for this head; `NoLoss` suppresses gradient contribution

- ***HeadBranch::net*** - [`HeadNet HeadBranch::net`](src/BranchTrainer.hpp)
    - Member instance of `Net` for this `HeadBranch`

- ***BranchTrainer*** - [
  `template<typename TrunkNet, typename... Heads> requires IsTrainableNetwork<TrunkNet> && (IsHeadBranch<Heads> && ...) class BranchTrainer`](src/BranchTrainer.hpp)
    - Owns one `TrunkNet` and a `std::tuple<Heads...>` of
      `HeadBranch` instances
    - Tap indices must be non-decreasing (enforced by `static_assert`)
    - Each head's
      `InputTensor` must match the trunk activation tensor at its tap index (enforced by `static_assert`)

- ***BranchTrainer::InputTensor*** - [`using BranchTrainer::InputTensor`](src/BranchTrainer.hpp)
    - Alias for `TrunkNet::InputTensor`

- ***BranchTrainer::TrunkOutputTensor*** - [`using BranchTrainer::TrunkOutputTensor`](src/BranchTrainer.hpp)
    - Alias for `TrunkNet::OutputTensor`

- ***BranchTrainer::TrunkActivations*** - [`using BranchTrainer::TrunkActivations`](src/BranchTrainer.hpp)
    - Alias for `TrunkNet::Activations`

- ***BranchTrainer::HeadTapGrads*** - [`using BranchTrainer::HeadTapGrads`](src/BranchTrainer.hpp)
    - Tuple of per-head input-tap gradient tensors

- ***BranchTrainer::HeadOutputs*** - [`using BranchTrainer::HeadOutputs`](src/BranchTrainer.hpp)
    - Tuple of per-head output target tensors

- ***BranchTrainer::trunk_*** - [`TrunkNet BranchTrainer::trunk_`](src/BranchTrainer.hpp)
    - Member instance of `TrunkNet` type

- ***BranchTrainer::heads_*** - [`std::tuple<Heads...> BranchTrainer::heads_`](src/BranchTrainer.hpp)
    - `std::tuple` of `HeadBranch` objects

- ***BranchTrainer::trunk*** - [`TrunkNet &BranchTrainer::trunk()`](src/BranchTrainer.hpp)
    - Getter for `TrunkNet&`

- ***BranchTrainer::trunk*** - [`const TrunkNet &BranchTrainer::trunk() const`](src/BranchTrainer.hpp)
    - Getter for `const TrunkNet&`

- ***BranchTrainer::head*** - [`template<size_t I> auto &BranchTrainer::head()`](src/BranchTrainer.hpp)
    - Getter for `&` to `I`-th `HeadBranch` from `heads_`

- ***BranchTrainer::head*** - [`template<size_t I> const auto &BranchTrainer::head() const`](src/BranchTrainer.hpp)
    - Getter for `const &` to `I`-th `HeadBranch` from `heads_`


- ***BranchTrainer::forward_heads*** - [
  `template<size_t... I> auto BranchTrainer::forward_heads(const TrunkActivations &A, std::index_sequence<I...>) const`](src/BranchTrainer.hpp)
    - Forward pass through all heads, returning `std::tuple` of heads' activation tuples

- ***BranchTrainer::forward_heads_batched*** - [
  `template<size_t Batch, size_t... I> auto BranchTrainer::forward_heads_batched(const TrunkNet::BatchedActivations<Batch> &A, std::index_sequence<I...>) const`](src/BranchTrainer.hpp)
    - Batched forward pass through all heads, returning `std::tuple` of heads' batched activation tuples

- ***BranchTrainer::compute_losses*** - [
  `template<typename TrunkLoss, size_t... Is> requires (std::is_same_v<TrunkLoss, NoLoss> || LossFunction<TrunkLoss, typename TrunkNet::OutputTensor>) float BranchTrainer::compute_losses(const TrunkOutputTensor &trunk_pred, const TrunkOutputTensor &trunk_target, const HeadOutputs &head_preds, const HeadOutputs &head_targets, std::index_sequence<Is...>) const`](src/BranchTrainer.hpp)
    - Accumulates scalar loss contributions from trunk and all heads using already-computed output tensors

- ***BranchTrainer::compute_losses_batched*** - [
  `template<typename TrunkLoss, size_t Batch, size_t... Is> requires (std::is_same_v<TrunkLoss, NoLoss> || LossFunction<TrunkLoss, typename TrunkNet::OutputTensor>) float BranchTrainer::compute_losses_batched(const TrunkNet::BatchedActivations<Batch> &A, const PrependBatch<Batch, TrunkOutputTensor>::type &trunk_target, const std::tuple<typename PrependBatch<Batch, typename Heads::Net::OutputTensor>::type...> &head_targets, std::index_sequence<Is...>) const`](src/BranchTrainer.hpp)
    - Batched: accumulates scalar loss contributions from trunk and all heads across the batch

- ***BranchTrainer::backward_heads*** - [
  `template<size_t I, size_t PrevTap> void BranchTrainer::backward_heads(const TrunkActivations &A, const TrunkOutputTensor &grad, const HeadTapGrads &head_tap_grads)`](src/BranchTrainer.hpp)
    - Pure gradient propagation: chains
      `BackwardRange` through trunk segments, summing in precomputed per-head tap gradients

- ***BranchTrainer::backward_heads_batched*** - [
  `template<size_t Batch, size_t I, size_t PrevTap> void BranchTrainer::backward_heads_batched(const TrunkNet::BatchedActivations<Batch> &A, const PrependBatch<Batch, TrunkOutputTensor>::type &grad, const std::tuple<typename PrependBatch<Batch, typename Heads::Net::InputTensor>::type...> &head_tap_grads)`](src/BranchTrainer.hpp)
    - Batched recursive backward: same structure as `backward_heads`, operating on `Batch`-prepended tensors

- ***BranchTrainer::Fit*** - [
  `template<typename TrunkLoss = NoLoss> float BranchTrainer::Fit(const InputTensor &x, const TrunkOutputTensor &trunk_target, float trunk_lr, const HeadOutputs &head_targets, const std::array<float, NumHeads> &head_lrs)`](src/BranchTrainer.hpp)
    - ######

- ***BranchTrainer::BatchFit*** - [
  `template<size_t Batch, typename TrunkLoss = NoLoss> float BranchTrainer::BatchFit(const PrependBatch<Batch, InputTensor>::type &X, const PrependBatch<Batch, TrunkOutputTensor>::type &trunk_target, float trunk_lr, const std::tuple<typename PrependBatch<Batch, typename Heads::Net::OutputTensor>::type...> &head_targets, const std::array<float, NumHeads> &head_lrs)`](src/BranchTrainer.hpp)
    - ######

---

## [DataIO.hpp](src/DataIO.hpp) -- Data Loading and Batching

Utilities for loading datasets from disk, drawing random mini-batches, and displaying terminal progress bars. Shapes are compile-time parameters: the type
*is* the schema.


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

