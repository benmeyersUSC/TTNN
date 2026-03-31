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
14. [ChainBlock.hpp: Sequential Block Composition](#chainblockhpp--sequential-block-composition)
14. [Snapshot.hpp: Activation Snapshots](#snapshothpp--activation-snapshots)
15. [Dense.hpp: Fully-Connected Layer](#densehpp--fully-connected-layer)
16. [Attention.hpp: Multi-Head Self-Attention](#attentionhpp--multi-head-self-attention)
17. [DataIO.hpp: Data Loading and Batching](#dataiohpp--data-loading-and-batching)

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

All contractions eventually become a `BatchInnerContraction`, which takes in four arrays of axes: `A`'s `Batch` and
`Contract` axes and `B`'s `Batch` and `Contract` axes. Non-batched `Contract` calls become
`Batch=0` calls to the former. `Dot`, `Matmul`, `ΣΠ`, `Outer`, and other variants all have
`Batch` versions, but, again are all wrappers around `BatchInnerContraction`.

- ***AxisList*** - [`template<size_t... Axes> struct AxisList`](src/TensorContract.hpp)
    - Necessary wrapper around variadic `size_t` axis packs used in `Contract` and `BatchContract` to specify specific
      `Batch` and/or `Contract` axis indices, rather than passing the *count* of left (for `Batch`) or right (for
      `Contract`) axes
    - `static_assert`s no duplicate axis indices

- ***BC_Permute*** - [
  `template<size_t Rank, AxisList BatchAxes, AxisList ContractAxes> struct BC_Permute`](src/TensorContract.hpp)
    - Compile-time helper to generate permutation indices for a `Tensor`'s axes in `BatchInnerContract` form:
      `[Batch..., Free..., Contract...]`
    - `static_assert`s that:
        - `Batch` and `Contract` axes are `disjoint`
        - no indices in `Batch` or `Contract` lists are greater than `Rank`
        - `Batch` and `Contract` axes have no duplicates
    - Intended to allow for `Permute`-`Contract` fusion, obviating many calls (in [
      `TrainableTensorNetwork`](src/TrainableTensorNetwork.hpp) blocks, for example) which allocate temporary `Tensor`s
    - `BatchInnerContract` form is part conventional and part performance-informed:
        - `Batch` being left-aligned adopts common convention for `Tensor` shapes in ML
        - `Inner` being right-aligned departs from the original conventional configuration in which `Inner` means
          `A`'s rightmost and `B`'s leftmost axes. The reason for right-alignment of contracted axes is that
          `Tensor`s in `TTTN` are backed by ***row-major***
          `float` arrays. This means that in the backing array, the only sets of values which are stored contiguously are those in the right-most axes. To maximize vectorization optimizations for
          `Reduce ∘ zipWith(Map)` operations, we want the loops over contracted indices to be traversing contiguous memory. Detailed comments on this subject are resident in the code.

- ***BatchedContractionKernel*** — [
  `template<size_t M_Batched, size_t N_Contracted, size_t... A_Dims, size_t... B_Dims> struct BatchedContractionKernel<M_Batched, N_Contracted, Tensor<A_Dims...>, Tensor<B_Dims...> >`](src/TensorContract.hpp)
    - Unified contraction bookkeeping kernel, used compile-time compute convenient shapes and values used in two versions of
      `BatchInnerContract` (the functions through which every contraction operation are routed)

- ***BatchInnerContract*** - [
  `template<size_t M, size_t N, size_t... ADims, size_t... BDims, FloatBinaryOp Map, FloatBinaryOp Reduce> auto BatchInnerContract(const Tensor<ADims...> &A, const Tensor<BDims...> &B, const float init, Map map, Reduce reduce)`](src/TensorContract.hpp)'s core primitive; all contraction routes through here
    - Core primitive: all contractions become ***BatchInnerContract***
    - See ***BatchContractionKernel*** for more details on implementation

- ***BatchInnerContract*** - [
  `template<size_t M, size_t N, typename Map, typename Reduce, size_t... ADims, size_t... BDims> requires FloatBinaryOp<Map> && FloatBinaryOp<Reduce> && std::default_initializable<Map> && std::default_initializable<Reduce> && requires { { Reduce::identity } -> std::convertible_to<float>; } auto BatchInnerContract(const Tensor<ADims...> &A, const Tensor<BDims...> &B)`](src/TensorContract.hpp)
    - Tag-parameter specialization of ***BatchInnerContract***; calls ***BatchInnerContract***

- ***BatchInnerContract*** - [
  `template<size_t M, size_t N, size_t... ADims, size_t... BDims> auto BatchInnerContract(const Tensor<ADims...> &A, const Tensor<BDims...> &B, float /*init*/, Mul, Add)`](src/TensorContract.hpp)
    - Specialized version of generalized ***BatchInnerContract*** for `Map=Mul` and `Reduce=Add` (most common use-case)
    - Uses `Apple Accelerate`'s
      `cblas_sgemm` function to unlock aggressive vectorization optimization for matrix multiplication
    - Extensive commenting in code

- ***BatchΣΠ*** - [
  `template<size_t M, size_t N, size_t... ADims, size_t... BDims> auto BatchΣΠ(const Tensor<ADims...> &A, const Tensor<BDims...> &B)`](src/TensorContract.hpp)
    - Convenience wrapper for sum of product specialization of ***BatchInnerContract***

- ***BatchSigmaPi*** - [
  `template<size_t M, size_t N, size_t... ADims, size_t... BDims> auto BatchSigmaPi(const Tensor<ADims...> &A, const Tensor<BDims...> &B)`](src/TensorContract.hpp)
    - ASCII overload of ***BatchΣΠ***

- ***InnerContract*** - [
  `template<size_t N, size_t... ADims, size_t... BDims, FloatBinaryOp Map, FloatBinaryOp Reduce> auto InnerContract(const Tensor<ADims...> &A, const Tensor<BDims...> &B, float init, Map map, Reduce reduce)`](src/TensorContract.hpp)
    - Convenience wrapper for non-batched calls to generalized ***BatchInnerContract***

- ***InnerContract*** - [
  `template<size_t N, typename Map, typename Reduce, size_t... ADims, size_t... BDims> requires FloatBinaryOp<Map> && FloatBinaryOp<Reduce> && std::default_initializable<Map> && std::default_initializable<Reduce> && requires { { Reduce::identity } -> std::convertible_to<float>; } auto InnerContract(const Tensor<ADims...> &A, const Tensor<BDims...> &B)`](src/TensorContract.hpp)
    - tag-param specialization of ***InnerContract***

- ***ΣΠ*** - [
  `template<size_t N, size_t... ADims, size_t... BDims> auto ΣΠ(const Tensor<ADims...> &A, const Tensor<BDims...> &B)`](src/TensorContract.hpp)
    - Convenience wrapper for non-batched sum of products contraction

- ***SigmaPi*** - [
  `template<size_t N, size_t... ADims, size_t... BDims> auto SigmaPi(const Tensor<ADims...> &A, const Tensor<BDims...> &B)`](src/TensorContract.hpp)
    - ASCII convenience wrapper for ***ΣΠ***

- ***Contract*** - [
  `template<AxisList AAxes, AxisList BAxes, size_t... ADims, size_t... BDims, FloatBinaryOp Map, FloatBinaryOp Reduce> auto Contract(const Tensor<ADims...> &A, const Tensor<BDims...> &B, float init, Map map, Reduce reduce)`](src/TensorContract.hpp)
    - Convenience wrapper for ***BatchContract*** (and
      ***BatchInnerContract***) for non-Batched, arbitrary-axes contractions
    - Second-most general function in [TensorContract.hpp](src/TensorContract.hpp)

- ***Contract*** - [
  `template<AxisList AAxes, AxisList BAxes, typename Map, typename Reduce, size_t... ADims, size_t... BDims> requires FloatBinaryOp<Map> && FloatBinaryOp<Reduce> && std::default_initializable<Map> && std::default_initializable<Reduce> && requires { { Reduce::identity } -> std::convertible_to<float>; } auto Contract(const Tensor<ADims...> &A, const Tensor<BDims...> &B)`](src/TensorContract.hpp)
    - tag-param specialization of ***Contract***

- ***Einsum*** [
  `template<size_t I, size_t J, size_t... ADims, size_t... BDims> auto Einsum(const Tensor<ADims...> &A, const Tensor<BDims...> &B)`](src/TensorContract.hpp)
    - Variant of ***ΣΠ*** for single-axis sum of product contractions (specified by `I` and `J` for `A` and
      `B`, respectively)

- ***BatchEinsum*** - [
  `template<AxisList ABatchAxes, AxisList BBatchAxes, size_t I, size_t J, size_t... ADims, size_t... BDims> auto BatchEinsum(const Tensor<ADims...> &A, const Tensor<BDims...> &B)`](src/TensorContract.hpp)
    - Batch version of ***Einsum***

- ***Dot*** - [`template<size_t N> auto Dot(const Tensor<N> &A, const Tensor<N> &B)`](src/TensorContract.hpp)
    - Convenience wrapper for sum of product full-rank contraction (dot product) of two Rank-1 `Tensor`s

- ***Matmul*** - [
  `template<size_t M, size_t K, size_t N> auto Matmul(const Tensor<M, K> &A, const Tensor<K, N> &B)`](src/TensorContract.hpp)
    - Convenience wrapper for sum of product contraction (matrix multiplication) of two Rank-2 `Tensor`s
    - NOTE: expects Axis 1 of `A` to be contracted with Axis 0 of `B`, per ***Matmul*** convention

- ***Outer*** - [
  `template<size_t... ADims, size_t... BDims> auto Outer(const Tensor<ADims...> &A, const Tensor<BDims...> &B)`](src/TensorContract.hpp)
    - Convenience wrapper for sum of product no-rank contraction (outer product) of two Rank-1 `Tensor`s

- ***BatchContract*** - [
  `template<AxisList ABatchAxes, AxisList BBatchAxes, AxisList AContractAxes, AxisList BContractAxes, size_t... ADims, size_t... BDims, FloatBinaryOp Map, FloatBinaryOp Reduce> auto BatchContract(const Tensor<ADims...> &A, const Tensor<BDims...> &B, float init, Map map, Reduce reduce)`](src/TensorContract.hpp)
    - Most general function in [TensorContract.hpp](src/TensorContract.hpp)
    - Arbitrary `Align` (specified by `Batch` and `Contract` axes), arbitrary `Map` (to zip aligned elements of `A` and
      `B`), arbitrary `Reduce` (to fold down `Map` results along contracted axes)
    - Utilizes right-alignment convention of contracted axes (explained more in [
      ***BC_Permute***](src/TensorContract.hpp) docs and code) and
      ***tiling*** to utilize vectorization of contiguous reads and computations

- ***BatchContract*** - [
  `template<AxisList ABatchAxes, AxisList BBatchAxes, AxisList AContractAxes, AxisList BContractAxes, typename Map, typename Reduce, size_t... ADims, size_t... BDims> requires FloatBinaryOp<Map> && FloatBinaryOp<Reduce> && std::default_initializable<Map> && std::default_initializable<Reduce> && requires { { Reduce::identity } -> std::convertible_to<float>; } auto BatchContract(const Tensor<ADims...> &A, const Tensor<BDims...> &B)`](src/TensorContract.hpp)
    - Tag-param specialization of ***BatchContract***


- ***Collapse*** - [
  `template<size_t... Dims, FloatBinaryOp M, FloatBinaryOp R> float Collapse(const Tensor<Dims...> &A, const Tensor<Dims...> &B, float init, M map, R reduce)`](src/TensorContract.hpp)
    - Specialization for contraction over *all* axes
    - Known as ***Frobenius Inner Product***, it is a generalization of `Dot` or inner product for arbitrarily-shaped
      `Tensor`s

- ***Collapse*** - [
  `template<typename M, typename R, size_t... Dims> requires FloatBinaryOp<M> && FloatBinaryOp<R> && std::default_initializable<M> && std::default_initializable<R> && requires { { R::identity } -> std::convertible_to<float>; } float Collapse(const Tensor<Dims...> &A, const Tensor<Dims...> &B)`](src/TensorContract.hpp)
    - Tag-param specialization of ***Collapse***

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

`TensorIndex` and
`TensorIndexApply` are the gather/scatter primitives. The axis is compile-time — the compiler knows the slice shape and stride layout. Only the index into that axis is runtime.

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
    - `ActivationOp` for ***Rectified Linear Unit*** (***ReLU***)
    - `operator()` -> `[0, infinity)`
    - `prime` -> `1.0f || 0.0f`

- ***Sigmoid*** - [`struct Sigmoid`](src/TTTN_ML.hpp)
    - `ActivationOp` for ***Sigmoid***
    - `operator()` -> `[0, 1.0f]`
    - `prime` -> `(0.0f, 0.25f]`

- ***Tanh*** - [`struct Tanh`](src/TTTN_ML.hpp)
    - `ActivationOp` for ***Hyperbolic Tangent*** (***Tanh***)
    - `operator()` -> `[-1.0f, 1.0f]`
    - `prime` -> `(0.0f, 1.0f]`

- ***Liner*** - [`struct Linear`](src/TTTN_ML.hpp)
    - `ActivationOp` for ***Linear*** (no activation)
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
  `template<size_t Axis, size_t... Dims> Tensor<Dims...> Softmax(const Tensor<Dims...> &x)`](src/TTTN_ML.hpp)
    - Given an `Axis` on which to normalize, perform ***Softmax*** normalization
    - Elegantly calls
      `BroadcastReduceMove<Axis, Div, Add>(BroadcastReduce<Axis, Compose<Exp, Sub>, Max>(x))` to first map to
      `a = e^(x - max)` and then to `b = a / sum(a)`
    - Shape-preserving

- ***SoftmaxPrime*** — [
  `template<size_t Axis, size_t... Dims> Tensor<Dims...> SoftmaxPrime(const Tensor<Dims...> &grad, const Tensor<Dims...> &a)`](src/TTTN_ML.hpp)
    - Computes derivative of `Softmax`
    - Calls (efficient equivalent of) `a * BroadcastMap<Axis, Sub>(grad, BroadcastReduce<Axis, Add, Mul>(a, grad))`
        - Generalization of `a * (g - (g . a))`
    - Shape-preserving

- ***SoftmaxBlock*** - [
  `template<size_t Axis, size_t... Dims> class SoftmaxBlock<Axis, Tensor<Dims...> >`](src/TTTN_ML.hpp)
    - Class representing the concrete block of a  `Softmax` layer in a `TrainableTensorNetwork`, satisfying the
      `ConcreteBlock` `concept`

- ***SoftmaxBlock::Forward*** — [`OutputTensor SoftmaxBlock::Forward(const InputTensor &x) const`](src/TTTN_ML.hpp)
    - Calls `Softmax<Axis>(x)`

- ***SoftmaxBlock::Backward*** — [
  `InputTensor SoftmaxBlock::Backward(const OutputTensor &delta_A, const OutputTensor &a, const InputTensor & /*a_prev*/)`](src/TTTN_ML.hpp)
    - Calls `SoftmaxPrime<Axis>(delta_A, a)`

- ***SoftmaxBlock::BatchedForward*** — [
  `template<size_t Batch> Tensor<Batch, Dims...> SoftmaxBlock::BatchedForward(const Tensor<Batch, Dims...> &X) const`](src/TTTN_ML.hpp)
    - Calls `Softmax<Axis + 1>(X)`
    - NOTE: assumes first axis is `Batch` axis

- ***SoftmaxBlock::BatchedBackward*** — [
  `template<size_t Batch> Tensor<Batch, Dims...> SoftmaxBlock::BatchedBackward(const Tensor<Batch, Dims...> &delta_A, const Tensor<Batch, Dims...> &a, const Tensor<Batch, Dims...> & /*a_prev*/)`](src/TTTN_ML.hpp)
    - Calls `SoftmaxPrime<Axis + 1>(delta_A, a)`
    - NOTE: assumes first axis is `Batch` axis

- ***SoftmaxBlock::all_params*** — [`auto all_params()`](src/TTTN_ML.hpp)
    - Returns `std::tuple<>{}` (no parameters)


- ***SoftmaxLayer*** - [`template<size_t Axis> struct SoftmaxLayer`](src/TTTN_ML.hpp)
    - `Block`-compliant recipe struct to create `ConcreteBlock SoftmaxBlock`
    - Pass in `Axis` of normalization and
      `Tensor` type whose shape will be preserved from input to output will be deduced

---

### Loss Function Concept

- ***LossFunction*** — [`template<typename L, typename TensorT> concept LossFunction`](src/TTTN_ML.hpp)
    - `concept` to define `LossFunction` structs
    - Requires:
        - `Loss(Tensor<Dims...>, Tensor<Dims...>) -> float`
        - `Grad(Tensor<Dims...>, Tensor<Dims...>) -> Tensor<Dims...>`

#### `struct MSE`

- ***MSE*** - [`struct MSE`](src/TTTN_ML.hpp)
    - `LossFunction` struct for ***Mean Squared Error*** (***MSE***)

- ***MSE::Loss*** - [
  `template<size_t... Dims> static float MSE::Loss(const Tensor<Dims...> &pred, const Tensor<Dims...> &target)`](src/TTTN_ML.hpp)
    - Sum of squares of difference between `target` and `pred`
    - Calls `Collapse<Compose<Sq, Sub>, Add>(pred, target) * Inv`

- ***MSE::Grad*** - [
  `template<size_t... Dims> static Tensor<Dims...> MSE::Grad(const Tensor<Dims...> &pred, const Tensor<Dims...> &target)`](src/TTTN_ML.hpp)
    - Derivative of ***MSE*** loss
    -
  `2(pred - target) / Tensor<Dims...>::Size` (standard power rule derivative, scaled by how many elements composed the original sum)

#### `struct BinaryCEL`

- ***BinaryCEL*** - [`struct BinaryCEL`](src/TTTN_ML.hpp)
    - `LossFunction` struct for ***Binary Cross Entropy Loss*** (***BinaryCEL***)
    - Helper for binary cases, but is just a specialization of `struct CEL`

- ***BinaryCEL::Loss*** - [
  `template<size_t... Dims> static float BinaryCEL::Loss(const Tensor<Dims...> &pred, const Tensor<Dims...> &target)`](src/TTTN_ML.hpp)
    - `-log(pred[true])` (negative log of the predicted value for `true` answer, whose target value is `1.0f`)

- ***BinaryCEL::Grad*** - [
  `template<size_t... Dims> static Tensor<Dims...> BinaryCEL::Grad(const Tensor<Dims...> &pred, const Tensor<Dims...> &target)`](src/TTTN_ML.hpp)
    - `(p - t) / (p * (1.f - p) + EPS)`

#### `struct CEL`

- ***CEL*** - [`struct CEL`](src/TTTN_ML.hpp)
    - `LossFunction` struct for ***Cross Entropy Loss*** (***CEL***)

- ***CEL::Loss*** - [
  `template<size_t... Dims> static float CEL::Loss(const Tensor<Dims...> &pred, const Tensor<Dims...> &target)`](src/TTTN_ML.hpp)
    - `-log(pred[true])` (negative log of the predicted value for `true` answer, whose target value is `1.0f`)
    - Elegantly calls `Collapse<Mul, Add>(target, Map<Compose<Log, Clamp<EPS> > >(pred)) * -1.f`

- ***CEL::Grad*** - [
  `template<size_t... Dims> static Tensor<Dims...> CEL::Grad(const Tensor<Dims...> &pred, const Tensor<Dims...> &target)`](src/TTTN_ML.hpp)
    - Elegantly calls `Zip<Compose<Neg, Div> >(target, Map<Clamp<EPS> >(pred))`

#### `BatchAccuracy`

- ***BatchAccuracy*** - [
  `template<size_t Batch, size_t N> float BatchAccuracy(const Tensor<Batch, N> &pred, const Tensor<Batch, N> &labels)`](src/TTTN_ML.hpp)
    - Computes accuracy for `Tensor`s organized in a batched `Tensor`
    - Takes any `Tensor<Batch, Dims...>` and flattens `Dims...` internally
    - NOTE: assumes ***one-hot encoding*** for labels

---

## [NetworkUtil.hpp](src/NetworkUtil.hpp): Concepts, Types, and Utilities

Defines the two block concepts that gate the type system, the chain-resolution machinery used by
`NetworkBuilder`, and the `ActivationsWrap` safety wrapper.

### Concepts

- ***ConcreteBlock*** - [`template<typename T> concept ConcreteBlock`](src/NetworkUtil.hpp)
    - Any block in a `TrainableTensorNetwork` must satisfy `ConcreteBlock`:
        - Defined `InputTensor` and `OutputTensor` types which are `Tensor` objects
        - `OutputTensor Forward(InputTensor)`
        - `InputTensor Backward(OutputTensor, OutputTensor, InputTensor)`
        - `auto all_params()` and `auto all_params() const`
    - `TrainableTensorNetwork` blocks need not belong to a specific hierarchy; just satisfy this `concept`

- ***Block*** - [`template<typename B> concept Block`](src/NetworkUtil.hpp)
    - Declarable recipe to define a `ConcreteBlock` in a `TrainableTensorNetwork` template argument list
    - `Block`s must define an `OutputTensor` type and alias a `ConcreteBlock` as `Resolve`
    - `Block` argument lists passed to `NetworkBuilder` will be resolved into full `ConcreteBlock`s with chained
      `InputTensor` attributes

- ***PeekableBlock*** - [`template<typename T> concept PeekableBlock`](src/NetworkUtil.hpp)
    - Opt-in `concept` for `ConcreteBlock`s to be able to expose their internal activations to an owning
      `TrainableTensorNetwork`
    - Compliant `ConcreteBlock`s must implement `void peek(SnapshotMap& m, const std::string& s)`


- ***BuildChain*** - [`template<typename Prev, Block Last> struct BuildChain<Prev, Last>`](src/NetworkUtil.hpp)
    - Build `std::tuple` of `ConcreteBlock`s from a variadic argument list of `Block`s
    - Base case for recursive `BuildChain`

- ***BuildChain*** - [
  `template<typename Prev, Block Next, Block... Rest> struct BuildChain<Prev, Next, Rest...>`](src/NetworkUtil.hpp)
    - Build `std::tuple` of `ConcreteBlock`s from a variadic argument list of `Block`s
    - Recursive case: `std::tuple_cat` of
        - first `Block`'s `ConcreteBlock` as given by its `Resolve` member
        - next `Block`s' `ConcreteBlock`s
    - Used by `ApplyBuildChain`

- ***Input*** - [`template<size_t... Dims> struct Input`](src/NetworkUtil.hpp)
    - `Block` type which begins and allows a variadic argument list of `Block`s to be processed by `BuildChain` via
      `ApplyBuildChain`
    - Defines `OutputTensor = Tensor<Dims...>` to begin chain

- ***ApplyBuildChain*** - [
  `template<typename In, Block... Rs> struct ApplyBuildChain<In, std::tuple<Rs...> >`](src/NetworkUtil.hpp)
    - Expects an `Block<Input>` first and a trailing variadic list of `Block`s passes them to `BuildChain`
    - Used in `NetworkBuilder` to define `BlockTuple`, a `TrainableTensorNetwork`'s tuple of `ConcreteBlock`s

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
    - Base case: one single `ConcreteBlock` left, whose `InputTensor` and `OutputTensor` are wrapped in a `std::tuple`

- ***TensorTupleBuilder*** - [
  `template<typename First, typename... Rest> struct TensorTupleBuilder<First, Rest...>`](src/NetworkUtil.hpp)
    - Recursively build `std::tuple` of
      `Tensor` objects representing intermediate activations of the network, wrapped by `ActivationsWrap`
    - Recursive case: `std::tuple_cat` of `First` `InputTensor` object and `TensorTupleBuilder<Rest...>`


- ***BatchedTensorTupleBuilder*** - [
  `template<size_t Batch, typename Last> struct BatchedTensorTupleBuilder<Batch, Last>`](src/NetworkUtil.hpp)
    - For `Batched` functions and use-cases, create a `Batched` version of a `std::tuple` of activations by passing
      `PrependBatch<Batch, ...>` on all `Tensor`s that `TensorTupleBuilder` adds raw
    - Base case: one single `ConcreteBlock` left, whose `InputTensor` and `OutputTensor` are wrapped in
      `PrependBatch<Batch, ...>` and then in a `std::tuple`

- ***BatchedTensorTupleBuilder*** - [
  `template<size_t Batch, typename First, typename... Rest> struct BatchedTensorTupleBuilder<Batch, First, Rest...>`](src/NetworkUtil.hpp)
    - For `Batched` functions and use-cases, create a `Batched` version of a `std::tuple` of activations by passing
      `PrependBatch<Batch, ...>` on all `Tensor`s that `TensorTupleBuilder` adds raw
    - Recursive case: `std::tuple_cat` of `First` `PrependBatch<Batch, InputTensor>` object and
      `BatchedTensorTupleBuilder<Rest...>`

---

## [TrainableTensorNetwork.hpp](src/TrainableTensorNetwork.hpp): The Network

The top-level network class and the `NetworkBuilder` factory. Owns all blocks in a
`std::tuple` and one `AdamState` instance. Orchestrates forward/backward passes; drives `ZeroGrad`, `Update`, `Save`,
`Load` on every block via `all_params()` — no block implements these directly.

### `class TrainableTensorNetwork<ConcreteBlock... Blocks>`

- ***TrainableTensorNetwork*** - [
  `template<ConcreteBlock... Blocks> class TrainableTensorNetwork`](src/TrainableTensorNetwork.hpp)
    - Capstone object of the library
    - Wrap `std::tuple` of shape-compliant `ConcreteBlock`s in a network API that enables and/or enforces:
        - `Forward` and `BatchedForward`
        - `Backward` and `BatchedBackward`
        - `Update`, `ZeroGrad`, `TrainStep` and `BatchedTrainStep`, `Fit` and `BatchFit`
        - `snap` (snapshot of activations)
        - `Save` and `Load` serialization

- ***TrainableTensorNetwork::check_connected()*** - [
  `static constexpr bool TrainableTensorNetwork::check_connected()`](src/TrainableTensorNetwork.hpp)
    - Immediate `static_assert` function to ensure that `ConcreteBlock... Blocks` have compliant shapes:
      -
      `std::is_same_v<typename std::tuple_element_t<Is, BlockTuple>::OutputTensor, typename std::tuple_element_t<Is + 1, BlockTuple>::InputTensor> && ...)`

### Type Aliases and Public Members

- ***TrainableTensorNetwork::BlockTuple*** - [`TrainableTensorNetwork::BlockTuple`](src/TrainableTensorNetwork.hpp)
    - `using BlockTuple = std::tuple<Blocks...>`
    - NOTE: not a `std::tuple` of `Blocks...` *objects* but *types*

- ***TrainableTensorNetwork::InputTensor*** - [`TrainableTensorNetwork::InputTensor`](src/TrainableTensorNetwork.hpp)
    - Extract `InputTensor` type from first element of `BlockTuple`

- ***TrainableTensorNetwork::OutputTensor*** - [`TrainableTensorNetwork::OutputTensor`](src/TrainableTensorNetwork.hpp)
    - Extract `OutputTensor` type from last element of `BlockTuple`

- ***TrainableTensorNetwork::InSize*** - [`TrainableTensorNetwork::InSize`](src/TrainableTensorNetwork.hpp)
    - Convenience member for total size of `InputTensor` type

- ***TrainableTensorNetwork::OutSize*** - [`TrainableTensorNetwork::OutSize`](src/TrainableTensorNetwork.hpp)
    - Convenience member for total size of `OutputTensor` type

### Private Members and Functions

- ***TrainableTensorNetwork::mBlocks*** - [`TrainableTensorNetwork::mBlocks`](src/TrainableTensorNetwork.hpp)
    - Default-constructed `BlockTuple` type containing actual `ConcreteBlock` values

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

## [ChainBlock.hpp](src/ChainBlock.hpp): Composed Block Object

## [Snapshot.hpp](src/Snapshot.hpp): Activation Snapshots

Runtime-typed storage for capturing named activation tensors. `SnapshotEntry` holds a shape vector and a flat
`float` copy — erasing the compile-time type so snapshots can be stored in a uniform `SnapshotMap` (
`unordered_map<string, SnapshotEntry>`). Used by visualization and debugging tools.

- ***snap_add*** — [
  `template<size_t... Dims> void snap_add(SnapshotMap& out, const std::string& key, const Tensor<Dims...>& t)`](src/Snapshot.hpp)
    - #########

--- 

## [Dense.hpp](src/Dense.hpp): Fully-Connected Layer

Implements the general multidimensional dense layer (`DenseMDBlock`) and its recipe types. Weights have shape
`Tensor<OutDims..., InDims...>`; forward pass is a generalized matrix-vector product via
`ΣΠ`. Includes Adam optimizer state.


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

