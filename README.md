# TTTN -- Trainable Template Tensor Network

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
  ** templates and the type system, we enforce shape correctness at compile time while enabling aggressive precomputation of traversal structure. Runtime execution becomes a planned walk over constant topologies—fully fused, vectorizable, and allocation-free.

- The dimension-typed `Tensor` unlocks a unified
  `Contraction` abstraction, parameterized by three orthogonal components:
    - `Align`: how two `Tensor`s are brought into correspondence (which elements meet)
    - `Map`: how aligned elements interact (`Map :: (T, T) -> T`)
    - `Reduce`: how mapped results are aggregated along contracted axes

- Under this formulation, a contraction is:
    - `Contraction = Reduce ∘ zipWith(Map) ∘ Align`

- Classical tensor operations such as `Einsum`, `ΣΠ` (generalized *sum of products*), `Matmul`, and
  `Dot` arise as specializations of this pattern. By choosing different `Align`, `Map`, and
  `Reduce` components, one can express and efficiently execute a wide range of computations—from linear algebra to activation functions, loss functions, and full training pipelines—within a single fused kernel.

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
    auto C = InnerContract<1>(                  // Align <1> inner axis
        A, B,
        0.0f,
        [](float a, float b) { return a * b; }, // Map
        std::plus<>{}                           // Reduce
    );
    ```

    - Different contractions on the same type:
        - **Dot product / Frobenius inner product**
          ```cpp
          float sim = Collapse(
              A, B,
              0.0f,
              [](float a, float b) { return a * b; },
              std::plus<>{}
          );
           ```
        - **L1 Distance**
          ```cpp
          float dist = Collapse(
              A, B,
              0.0f,
              [](float a, float b) { return std::abs(a - b); },
              std::plus<>{}
          );
             ```
        - **Max product**
          ```cpp
          auto C = InnerContract<1>(
              A, B,
              -std::numeric_limits<float>::infinity(),
              [](float a, float b) { return a * b; },
              [](float x, float y) { return std::max(x, y); }
          );
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

- **SDL Visualization Plugin** — render
  `Tensor`s and networks below rank 4; watch weight matrices, attention patterns, and activation volumes live during training
- **Interpretability APIs
  ** — comparison frameworks for two same-topology networks (weight similarity, head alignment, per-layer Frobenius cosine similarity across checkpoints or runs)
- **GPU Utilization** — replace parallel CPU dispatch with CUDA/Metal kernels for real throughput on large networks

---

## Table of Contents

1. [Tensor.hpp -- The Foundational Object](#tensorhpp--the-foundational-object)
2. [TensorOps.hpp -- Tensor Operations](#tensoropshpp--tensor-operations)
3. [TTTN_ML.hpp -- ML Primitives](#tttn_mlhpp--ml-primitives)
4. [Dense.hpp -- Fully-Connected Layer](#densehpp--fully-connected-layer)
5. [Attention.hpp -- Multi-Head Self-Attention](#attentionhpp--multi-head-self-attention)
6. [NetworkUtil.hpp -- Concepts, Types, and Utilities](#networkutilhpp--concepts-types-and-utilities)
7. [TrainableTensorNetwork.hpp -- The Network](#trainabletensornetworkhpp--the-network)
8. [DataIO.hpp -- Data Loading and Batching](#dataiohpp--data-loading-and-batching)

---

## [Tensor.hpp](src/Tensor.hpp) -- The Foundational Object

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

- ***SizeTemplateGet*** — [`struct SizeTemplateGet<size_t N, size_t... Ds>`](src/Tensor.hpp)
    - Template-specialization-based recursion grab `N`-th `size_t` from `<size_t...Ds>`
    - Uses functional-style aggregation and pattern-matching to decrement `N` and peel off
      `size_t`s from variadic array until reaching basecase where `N = 0`
    - Used for clean, compile-time syntax in [TensorOps.hpp](src/TensorOps.hpp)

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

---

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

- ***FlatToMulti*** — [`static constexpr std::array<size_t, Rank> FlatToMulti(size_t flat)`](src/Tensor.hpp)
    - Inverse of `MultiToFlat`; map a flat index `[0, Size)` to its `Rank`-term index
    - Pattern matches a `std::index_sequence` parameterized by `<size_t... Rank>` (=
      `[0, ..., Rank]`) and unpacks into an array:
      `[(flat / Strides[0]) % Shape[0], ..., (flat / Strides[Rank]) % Shape[Rank]]`

- ***MultiToFlat*** — [`static constexpr size_t MultiToFlat(const std::array<size_t, Rank>& multi)`](src/Tensor.hpp)
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

- ***operator float*** — [`operator float() const`](src/Tensor.hpp)
    - Implicit scalar conversion — only valid for `Tensor<>` (rank-0, `Rank == 0`). Allows
      `ΣΠ<N>(A, B)` and `InnerContract<N>(...)` results to be used directly as `float` without calling `.flat(0)`.

**Functional transforms:**

- ***map*** — [`template<typename F> Tensor map(F f) const`](src/Tensor.hpp)
    - Use `std::execution::par_unseq` to `std::transform` `Tensor`'s underlying data by `float -> float` map
      `f`, returning a new `Tensor`

- ***zip*** — [`template<typename F> Tensor zip(const Tensor& other, F f) const`](src/Tensor.hpp)
    - Use `std::execution::par_unseq` to `std::transform` two `Tensor`s' underlying data by `float -> float -> float`
      map `f`, returning a new `Tensor`

- ***apply*** — [`template<typename F> void apply(F f)`](src/Tensor.hpp)
    - Use `std::execution::par_unseq` + `std::for_each` to apply `float -> float` map `f` to
      `Tensor`'s underlying data in-place

- ***zip_apply*** — [`template<typename F> void zip_apply(const Tensor& other, F f)`](src/Tensor.hpp)
    - Use `std::execution::par_unseq` to `std::transform` two `Tensor`s' underlying data in-place by
      `float -> float -> float` map `f`

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

### Type Traits

- ***is_tensor*** — [`struct is_tensor<T>`](src/Tensor.hpp)
    - SFINAE type traits for verifying that a type is a `Tensor`
        - Specialize `<size_t...Dims>`: matches into `<Tensor<Dims...>`, inherits from `std::true_type`
        - Specialize <>: inherits from `std::false_type`, backup when former fails


- ***IsTensor*** — [`concept IsTensor<T>`](src/Tensor.hpp)
    - Wrapper `concept` around `is_tensor` type trait, satisfied if `T` is a `Tensor`

---

### Tensor Type Algebra

Shape-only metaprogramming. No data, no runtime — purely compile-time type-level operations on `Tensor` dimension packs. Available to any code that includes `Tensor.hpp`.

- ***TensorConcat*** — [`struct TensorConcat<typename T1, typename T2>`](src/Tensor.hpp)
    - Concatenate the dimension packs of two `Tensor` types
    - `TensorConcat<Tensor<A,B>, Tensor<C>>::type == Tensor<A,B,C>`

- ***ArrayToTensor*** — [`struct ArrayToTensor<typename KeptIdxs, typename Iota>`](src/Tensor.hpp)
    - Convert a compile-time `std::array<size_t, N>` (held in `KeptIdxs::value`) into a `Tensor` type
    - `type = Tensor<arr[Iota]...>` where `Iota` is an `index_sequence` over `[0, N)`

- ***KeptDimsHolder*** — [`struct KeptDimsHolder<size_t Skip, size_t... Dims>`](src/Tensor.hpp)
    - Compute `Dims...` with the axis at position `Skip` removed
    - `value` holds the resulting `std::array<size_t, sizeof...(Dims) - 1>`

- ***RemoveAxis*** — [`struct RemoveAxis<size_t Skip, size_t... Dims>`](src/Tensor.hpp)
    - `Tensor<Dims...>` with axis `Skip` dropped
    - `RemoveAxis<1, A, B, C>::type == Tensor<A, C>`

- ***InsertAxisHolder*** — [`struct InsertAxisHolder<size_t Axis, size_t N, size_t... Dims>`](src/Tensor.hpp)
    - Compute `Dims...` with dimension `N` inserted at position `Axis`
    - `value` holds the resulting `std::array<size_t, sizeof...(Dims) + 1>`

- ***InsertAxis*** — [`struct InsertAxis<size_t Axis, size_t N, size_t... Dims>`](src/Tensor.hpp)
    - `Tensor<Dims...>` with dimension `N` inserted at position `Axis`
    - `InsertAxis<1, 4, A, C>::type == Tensor<A, 4, C>`

- ***SliceDimsHolder*** — [`struct SliceDimsHolder<size_t Start, size_t Len, size_t... Dims>`](src/Tensor.hpp)
    - Extract `Len` contiguous dimensions starting at `Start` from `Dims...`
    - `value` holds the resulting `std::array<size_t, Len>`

- ***TensorSlice*** — [`struct TensorSlice<size_t Start, size_t Len, size_t... Dims>`](src/Tensor.hpp)
    - `Tensor` type formed from dimensions `[Start, Start+Len)` of `Tensor<Dims...>`
    - `TensorSlice<1, 2, A, B, C, D>::type == Tensor<B, C>`

- ***PermutedTensorType*** — [`struct PermutedTensorType<typename T, size_t... Perm>`](src/Tensor.hpp)
    - `Tensor` type with dimensions reordered according to `Perm`
    - `PermutedTensorType<Tensor<4,5,3>, 2,0,1>::type == Tensor<3,4,5>`

### Tensor Demonstration

```cpp
// shapes are types -- different shapes are different types
Tensor<3, 4> mat;                                  // 3x4 matrix
Tensor<2, 3, 4> cube;                              // rank-3 tensor

// compile-time statics
static_assert(decltype(mat)::Rank == 2);
static_assert(decltype(mat)::Size == 12);
static_assert(decltype(mat)::Shape[0] == 3);
static_assert(decltype(mat)::Strides[0] == 4);     // row-major: stride[0] = cols

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
vec.apply([](float& x){ x *= -1.0f; });            // in-place

```

---

## [TensorOps.hpp](src/TensorOps.hpp) -- Tensor Operations

Generalized tensor algebra: contractions, permutations, reductions, slicing, and broadcast operations. All shapes are derived at compile time.

### Concepts

- **`FloatUnaryOp`** — `std::regular_invocable<F, float>` with return type `float`. Used by `Tensor::map`.
- **`FloatBinaryOp`** — `std::regular_invocable<F, float, float>` with return type `float`. Used by `ReduceApply`,
  `BroadcastApply`, `ReduceBroadcast`, `TensorIndexApply`.

---

### Arithmetic Operators

- ***operator+*** — [
  `template<size_t... Dims> Tensor<Dims...> operator+(const Tensor<Dims...>& a, const Tensor<Dims...>& b)`](src/TensorOps.hpp)
    - Element-wise add, uses parallel functional `zip`

- ***operator-*** — [
  `template<size_t... Dims> Tensor<Dims...> operator-(const Tensor<Dims...>& a, const Tensor<Dims...>& b)`](src/TensorOps.hpp)
    - Element-wise subtract, uses parallel functional `zip`

- ***operator+=*** — [
  `template<size_t... Dims> Tensor<Dims...>& operator+=(Tensor<Dims...>& a, const Tensor<Dims...>& b)`](src/TensorOps.hpp)
    - Element-wise add-to, uses parallel functional `zip_apply`

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

- ***Permute*** — [
  `template<size_t... Perm, size_t... Dims> auto Permute(const Tensor<Dims...>& src)`](src/TensorOps.hpp)
    - Parallelized arbitrary permutation of `Tensor`'s indices
    - Returns `PermutedTensorType`
    - Algorithm:
        - `using Source = Tensor<Dims...>`
        - `using Result = PermutedTensorType<Source, Perm>::type`
        - `std::array<size_t, Rank> perm_arr = {Perm...}`
        - `Result dst;`
        - For each (parallelized) individual index in `Result::Size`:
            - `auto dst_idx = Result::FlatToMulti(i)`
                - get `Result` multi-index
            - `std::array<size_t, Rank> src_multi = [perm_arr[Rank]...]`
                - get `Source` multi-index
            - `size_t src_index = Source::MultiToFlat(src_multi)`
                - get `Source` flat index
            - dst.flat(i) = src.flat(src_index)
                - Assign `Source` value at that index to `Result`

- ***MoveToLastPerm*** — [`struct MoveToLastPerm<size_t Src, size_t Rank>`](src/TensorOps.hpp)
    - Compile-time helper to rearrange `Tensor`'s shape such that `Src` is at index
      `[Rank - 1]` and all others are kept in order

- ***MoveToFirstPerm*** — [`struct MoveToFirstPerm<size_t Src, size_t Rank>`](src/TensorOps.hpp)
    - Compile-time helper to rearrange `Tensor`'s shape such that `Src` is at index
      `[0]` and all others are kept in order

- ***PermuteFromHolder*** — [
  `template<typename PermHolder, size_t... I, size_t... Dims> auto PermuteFromHolder(const Tensor<Dims...>& t, std::index_sequence<I...>)`](src/TensorOps.hpp)
    - Unpack a `constexpr` permutation indices array into a proper `Permute`-given `Tensor` type
    - `PermHolder` is an array of permutation indices, typically the result of `MoveToLastPerm` or `MoveToFirstPerm`
    - Call with `PermHolder` as template arg, `Tensor<Dims...>` as first arg,
      `std::make_index_sequence<sizeof...(Dims)>{}` as second arg

- ***Transpose*** — [`template<size_t... Dims> auto Transpose(const Tensor<Dims...>& t)`](src/TensorOps.hpp)
    - Reverse all dimensions of a `Tensor<Dims...>`
    - `Permute<(sizeof...(Dims) - 1 - I)...>(s)`, for `I` in `Dims...`
    - so if `Dims` are `<3, 5, 4>` (`sizeof = 3`) we will have:
        - `Permute<`(3-1) - 0 = `2,` (3-1) - 1 = `1,` (3-1) - 2 = `0>`

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

### Tensor Contraction

Tensor contraction is the unified `Reduce ∘ zipWith(Map) ∘ Align` operation on `Tensor`s. Every named operation —
`ΣΠ`, `Dot`, `Matmul`, `Outer`, `Einsum`, `Collapse` — is a specialization.

- ***ContractionKernel*** — [`template<size_t N, typename TA, typename TB> struct ContractionKernel`](src/TensorOps.hpp)
    - Unified compile-time index kernel. Specialized for `<N, Tensor<ADims...>, Tensor<BDims...>>`.
    - Compile-time `static constexpr`:
        - `RankA`, `RankB` — ranks of `A` and `B`
        - Asserts `N <= RankA && N <= RankB` and last `N` dims of `A` match first `N` dims of `B`
        - `A_Free = TensorSlice<0, RankA-N, ADims...>::type`
        - `B_Free = TensorSlice<N, RankB-N, BDims...>::type`
        - `Contracted = TensorSlice<RankA-N, N, ADims...>::type`
        - `ResultType = TensorConcat<A_Free, B_Free>::type`
        - `struct { std::array<size_t, Contracted::Size> a, b; } offsets` — flat-index offset into `A` and `B` for
          every contracted position; precomputed once per `(N, ADims, BDims)` and shared across all `(Map, Reduce)` variants
        - `b_free_size`, `contracted_size` — compile-time constants used by `InnerContract` to compute per-output base
          offsets as `O(1)` arithmetic (`base_a = (o / b_free_size) * contracted_size`, `base_b = o % b_free_size`)
          rather than a precomputed table — the compiler strength-reduces these to multiply-shift at `-O2`
    - **These pay real dividends for [TrainableTensorNetwork](./src/TrainableTensorNetwork.hpp) training schedules.
      Any weight `Tensor`'s `Dot`s, `Matmul`s, and `Outer`s (*in forward and backward passes*) are saved structs,
      and the runtime computations are parallelized and vectorized, following known, saved paths**

- ***InnerContract*** — [
  `template<size_t N, ..., FloatBinaryOp Map, FloatBinaryOp Reduce> auto InnerContract(const Tensor<ADims...>& A, const Tensor<BDims...>& B, float init, Map map, Reduce reduce)`](src/TensorOps.hpp)
    - N-inner-axis contraction with custom `Map` and `Reduce`
    - Aligns the last `N` axes of `A` with the first `N` axes of `B`
    - Reads precomputed `offsets` from `ContractionKernel`; base offsets computed inline as O(1) arithmetic
    - `result.flat(o) = Reduce_c map(A[A_Free(o), c], B[c, B_Free(o)])`, for `o ∈ [0, ResultType::Size)`

- ***ΣΠ*** / ***SigmaPi*** — [
  `template<size_t N, size_t... ADims, size_t... BDims> auto ΣΠ(const Tensor<ADims...>& A, const Tensor<BDims...>& B)`](src/TensorOps.hpp)
    - `InnerContract<N>` specialized to `map = multiply`, `reduce = plus` — the classical sum-of-products
    - `result.flat(o) = Σ_c A[A_Free(o), c] * B[c, B_Free(o)]`
    - `SigmaPi<N>(A, B)` is an ASCII alias for `ΣΠ<N>(A, B)`

- ***Einsum*** — [
  `template<size_t I, size_t J, size_t... ADims, size_t... BDims> auto Einsum(const Tensor<ADims...>& A, const Tensor<BDims...>& B)`](src/TensorOps.hpp)
    - `ΣΠ`-contracts over single selected indices `I` and `J` from `A` and `B`, respectively
    - Permutes `A` to move axis `I` last, `B` to move axis `J` first, then calls `ΣΠ<1>`

- ***Dot*** — [`template<size_t N> auto Dot(const Tensor<N>& A, const Tensor<N>& B)`](src/TensorOps.hpp)
    - `ΣΠ<1>` on two `Tensor<N>`s — returns `Tensor<>` (rank-0 scalar)

- ***Matmul*** — [
  `template<size_t M, size_t K, size_t N> auto Matmul(const Tensor<M,K>& A, const Tensor<K,N>& B)`](src/TensorOps.hpp)
    - `ΣΠ<1>` on rank-2 tensors — returns `Tensor<M,N>`

- ***Outer*** — [
  `template<size_t... ADims, size_t... BDims> auto Outer(const Tensor<ADims...>& A, const Tensor<BDims...>& B)`](src/TensorOps.hpp)
    - `ΣΠ<0>`: contract nothing — returns `Tensor<ADims..., BDims...>`

- ***Contract*** — [
  `template<AxisList AAxes, AxisList BAxes, ..., FloatBinaryOp Map, FloatBinaryOp Reduce> auto Contract(const Tensor<ADims...>& A, const Tensor<BDims...>& B, float init, Map map, Reduce reduce)`](src/TensorOps.hpp)
    - Grand-generalized contraction over arbitrary axis sets
    - Permutes `A` and `B` to align the selected axes, then delegates to `InnerContract<N>`

- ***Collapse*** — [
  `template<size_t... Dims, FloatBinaryOp M, FloatBinaryOp R> float Collapse(const Tensor<Dims...>& A, const Tensor<Dims...>& B, float init, M m, R r)`](src/TensorOps.hpp)
    - Full-rank same-shape scalar reduction: `Reduce_i map(A[i], B[i])`
    - Implemented as a direct `std::transform_reduce` over flat data — no index tables needed
    - `Collapse(A, B, 0, mul, plus)` == Frobenius inner product; `Collapse(A, B, 0, abs_diff, plus)` == L1 distance

`InnerContract<N>` is the core primitive. `N` controls how many trailing dims of `A` contract with leading dims of `B`.
Every named operation is a specialization — and every result shape is resolved at compile time from the input shapes.

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

float frob = Collapse(W, W, 0.f,                    // full contraction → scalar Frobenius inner product
    [](float a, float b){ return a * b; },
    std::plus<float>{});

// ── what the type system rejects ─────────────────────────────────────────────
// ΣΠ<1>(x, W);                                    // ✗ last dim of Tensor<784> ≠ first dim of Tensor<128,784>
// ΣΠ<1>(W3, W);                                   // ✗ last dim 4 ≠ first dim 3
// Dot(u, Tensor<5>{});                             // ✗ Tensor<3> · Tensor<5> — dimension mismatch
```

---

### Reduction and Broadcast

- ***ReduceKernel*** — [`template<size_t Axis, size_t... Dims> struct ReduceKernel`](src/TensorOps.hpp)
    - Shared kernel for all axis-reduction and broadcast operations
    - Compile-time `static constexpr`:
        - `axis_dim = SizeTemplateGet<Axis, Dims...>::value`
        - `axis_stride = Source::Strides[Axis]`
        - `std::array<size_t, Result::Size> bases` — flat index in `Source` for each output index with axis set to 0
        - `static constexpr size_t project(size_t i)` — flat index in `Result` for source flat index
          `i` (axis contribution stripped); closed-form `i - ((i / axis_stride) % axis_dim) * axis_stride`; no table,
          `axis_stride` compile-time so division compiles to multiply-shift

- ***ReduceApply*** — [
  `template<size_t Axis, FloatBinaryOp ReduceFn, size_t... Dims> typename RemoveAxis<Axis, Dims...>::type ReduceApply(const Tensor<Dims...>& src, float init, ReduceFn rfn)`](src/TensorOps.hpp)
    - Generalized axis reduction: collapses `Axis` by folding elements with `rfn(acc, val)` starting from `init`
    - `ReduceSum`  == `ReduceApply<Axis>(src, 0.f,  std::plus<float>{})`
    - `ReduceMax`  == `ReduceApply<Axis>(src, -inf, [](float a, float b){ return std::max(a,b); })`

- ***ReduceSum*** — [
  `template<size_t Axis, size_t... Dims> typename RemoveAxis<Axis, Dims...>::type ReduceSum(const Tensor<Dims...>& src)`](src/TensorOps.hpp)
    - Reduce an axis with `Tensor` addition — `ReduceSum<P>(Tensor<P,Q>) -> Tensor<Q>`
    - Routes through `ReduceApply<Axis>(src, 0.f, std::plus<float>{})`

- ***ReduceMean*** — [
  `template<size_t Axis, size_t... Dims> typename RemoveAxis<Axis, Dims...>::type ReduceMean(const Tensor<Dims...>& src)`](src/TensorOps.hpp)
    - Reduce an axis with `Tensor` averaging — `ReduceMean<P>(Tensor<P,Q>) -> Tensor<Q>`

- ***ReduceMax*** — [
  `template<size_t Axis, size_t... Dims> typename RemoveAxis<Axis, Dims...>::type ReduceMax(const Tensor<Dims...>& src)`](src/TensorOps.hpp)
    - Reduce an axis with `Tensor` maxing — `ReduceMax<P>(Tensor<P,Q>) -> Tensor<Q>`
    - Routes through `ReduceApply<Axis>(src, -inf, std::max)`

- ***Expand*** — [
  `template<size_t Axis, size_t N, size_t... Dims> InsertAxis<Axis, N, Dims...>::type Expand(const Tensor<Dims...>& src)`](src/TensorOps.hpp)
    - Dual of `ReduceSum`/`ReduceMax`: broadcasts a reduced tensor back up by repeating it `N` times along `Axis`
    - `Expand<0, 5>(Tensor<3>)` → `Tensor<5, 3>` — 5 copies stacked along axis 0
    - Uses `ReduceKernel::project` to map each output element to its source element

- ***BroadcastApply*** — [
  `template<size_t Axis, FloatBinaryOp F, size_t... Dims> Tensor<Dims...> BroadcastApply(const Tensor<Dims...>& A, const typename RemoveAxis<Axis, Dims...>::type& b, F f)`](src/TensorOps.hpp)
    - Apply binary `f(a_elem, b_elem) -> float` element-wise between `A` and `b` broadcast along `Axis`
    - For each element `i` of `A`: `result[i] = f(A[i], b[project(i)])`
    - `BroadcastAdd(A, b)` == `BroadcastApply<Axis>(A, b, std::plus<float>{})`

- ***ReduceBroadcast*** — [
  `template<size_t Axis, FloatBinaryOp ReduceFn, FloatBinaryOp ApplyFn, size_t... Dims> Tensor<Dims...> ReduceBroadcast(const Tensor<Dims...>& src, float init, ReduceFn rfn, ApplyFn afn)`](src/TensorOps.hpp)
    - Compose `ReduceApply` + `BroadcastApply` in one call: reduce along `Axis` with
      `rfn`, then broadcast the result back with `afn`
    - `ReduceBroadcast<Axis>(src, init, rfn, afn)` ==
      `BroadcastApply<Axis>(src, ReduceApply<Axis>(src, init, rfn), afn)`
    - Powers `Softmax`: two calls — `(max, exp(a-m))` then `(sum, e/s)`

- ***BroadcastAdd*** — [
  `template<size_t Axis, size_t... Dims> Tensor<Dims...> BroadcastAdd(const Tensor<Dims...>& A, const typename RemoveAxis<Axis, Dims...>::type& b)`](src/TensorOps.hpp)
    - Add a reduced tensor to all `Axis` slices of `A` — convenience wrapper over
      `BroadcastApply<Axis>(A, b, std::plus<float>{})`

Every operation below takes
`Axis` as a compile-time template argument. The compiler resolves the output shape, the stride arithmetic, and the projection function at compile time — the runtime loop is a flat parallel sweep with zero shape logic.

```cpp
Tensor<32, 10> logits;                              // batch of 32, 10 classes

// ── built-in reductions ──────────────────────────────────────────────────────
auto col_sum  = ReduceSum<0>(logits);               // sum over batch    → Tensor<10>
auto row_max  = ReduceMax<1>(logits);               // max per sample    → Tensor<32>
auto col_mean = ReduceMean<0>(logits);              // mean over batch   → Tensor<10>
static_assert(std::is_same_v<decltype(col_sum),  Tensor<10>>);
static_assert(std::is_same_v<decltype(row_max),  Tensor<32>>);
static_assert(std::is_same_v<decltype(col_mean), Tensor<10>>);

// ── ReduceApply: bring your own fold ─────────────────────────────────────────
// any FloatBinaryOp works — the axis, shape, and stride are all compile-time
auto row_min = ReduceApply<1>(logits,
    std::numeric_limits<float>::infinity(),
    [](float a, float b) { return std::min(a, b); });
static_assert(std::is_same_v<decltype(row_min), Tensor<32>>);   // min per sample

// ── Expand: the dual of reduction ────────────────────────────────────────────
// insert a new axis and repeat values along it
Tensor<10> bias;
auto stacked = Expand<0, 32>(bias);                 // Tensor<10> → Tensor<32, 10>
static_assert(std::is_same_v<decltype(stacked), Tensor<32, 10>>);

// works at any rank
Tensor<4, 5> slice;
auto vol = Expand<1, 3>(slice);                     // Tensor<4,5> → Tensor<4,3,5>
static_assert(std::is_same_v<decltype(vol), Tensor<4, 3, 5>>);

// ── BroadcastApply: element-wise op between full tensor and reduced tensor ───
auto biased = BroadcastApply<0>(logits, bias,
    std::plus<float>{});                            // add bias to every row
static_assert(std::is_same_v<decltype(biased), Tensor<32, 10>>);

// BroadcastAdd is the convenience wrapper:
auto biased2 = BroadcastAdd<0>(logits, bias);       // same thing
static_assert(std::is_same_v<decltype(biased2), Tensor<32, 10>>);

// but BroadcastApply takes ANY binary op:
auto scaled = BroadcastApply<0>(logits, row_max,
    std::divides<float>{});                         // divide each row by its max
static_assert(std::is_same_v<decltype(scaled), Tensor<32, 10>>);

// ── ReduceBroadcast: reduce then broadcast in one call ───────────────────────
// the most powerful primitive — composes ReduceApply + BroadcastApply.
// two lambdas: one to reduce along the axis, one to apply the result back.

// subtract each row's max from every element (numerically stable pre-softmax):
auto centered = ReduceBroadcast<1>(logits,
    -std::numeric_limits<float>::infinity(),
    [](float a, float b) { return std::max(a, b); },   // reduce: find max
    [](float a, float m) { return a - m; });            // apply:  subtract it
static_assert(std::is_same_v<decltype(centered), Tensor<32, 10>>);

// Softmax itself is just two ReduceBroadcast calls:
//   1. (max, exp(x - max))   — stable exponentials
//   2. (sum, e / sum)        — normalize to probabilities
// both fully parallel, both with compile-time-known shapes and strides.

// ── higher-rank example: 3D tensor ───────────────────────────────────────────
Tensor<8, 16, 64> activations;                      // batch × seq × embed

auto per_token_norm = ReduceApply<2>(activations,   // reduce embed dim
    0.f, std::plus<float>{});                       // → Tensor<8, 16>
static_assert(std::is_same_v<decltype(per_token_norm), Tensor<8, 16>>);

auto per_batch_max = ReduceMax<0>(activations);     // reduce batch dim
static_assert(std::is_same_v<decltype(per_batch_max), Tensor<16, 64>>);

// Expand it right back:
auto restored = Expand<0, 8>(per_batch_max);        // → Tensor<8, 16, 64>
static_assert(std::is_same_v<decltype(restored), Tensor<8, 16, 64>>);
```

---

### Indexed Slice Access

- ***TensorIndex*** — [
  `template<size_t Axis, size_t... Dims> typename RemoveAxis<Axis, Dims...>::type TensorIndex(const Tensor<Dims...>& src, size_t idx)`](src/TensorOps.hpp)
    - Extract the `idx`-th `RemoveAxis<Axis, Dims...>::type` sub-`Tensor` from `Tensor<Dims...> src` on the `Axis` axis
    - Essentially fills new `Tensor` with values from `src` by looping through dimensions in `Rank`, but passing
      `idx` for `Axis` dimension on all values

- ***TensorIndexApply*** — [
  `template<size_t Axis, FloatBinaryOp F, size_t... Dims> void TensorIndexApply(Tensor<Dims...>& dst, size_t idx, const typename RemoveAxis<Axis, Dims...>::type& src, F f)`](src/TensorOps.hpp)
    - Apply binary `f(existing, incoming) -> float` to each element of the `idx`-th slice of `dst` along
      `Axis` using the corresponding element of `src`

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

### Parallel Loop Helper

- ***ParForEach*** — [`template<std::invocable<size_t> F> void ParForEach(size_t n, F f)`](src/TensorOps.hpp)
    - Helper to parallel-execute `std::for_each` on a `std::views::iota(size_t{0}, n)`, calling `f` (something
      `std::invocable` on `size_t`) on each index

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

// softmax output (two ReduceBroadcast calls — stable, parallel, one line each):
auto exps  = ReduceBroadcast<0>(z2,                 // axis 0 is the only axis on Tensor<10>
    -std::numeric_limits<float>::infinity(),
    [](float a, float b) { return std::max(a, b); },
    [](float a, float m) { return std::exp(a - m); });
auto probs = ReduceBroadcast<0>(exps,
    0.f, std::plus<float>{}, std::divides<float>{});

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

## [TTTN_ML.hpp](src/TTTN_ML.hpp) -- ML Primitives

Activation functions, their derivatives, loss functions, and the `SoftmaxBlock` layer. Depends on `TensorOps.hpp`.

**Constants:**

- ***EPS*** — [`static constexpr float EPS`](src/TTTN_ML.hpp)
  -

### Activation Function Enum

- ***ActivationFunction*** — [`enum class ActivationFunction`](src/TTTN_ML.hpp) -- `Linear`, `Sigmoid`, `ReLU`,
  `Softmax`, `Tanh`
    -

---

### Free Functions

- ***CrossEntropyLoss*** — [
  `template<size_t N> float CrossEntropyLoss(const Tensor<N>& output, const Tensor<N>& target)`](src/TTTN_ML.hpp)
    -

- ***XavierInitMD*** — [
  `template<size_t... Dims> void XavierInitMD(Tensor<Dims...>& W, const size_t fan_in, const size_t fan_out)`](src/TTTN_ML.hpp)
    -

- ***Activate*** — [
  `template<size_t N> Tensor<N> Activate(const Tensor<N>& z, ActivationFunction act)`](src/TTTN_ML.hpp)
    -

- ***ActivatePrime*** — [
  `template<size_t N> Tensor<N> ActivatePrime(const Tensor<N>& grad, const Tensor<N>& a, ActivationFunction act)`](src/TTTN_ML.hpp)
    -

- ***BatchedActivate*** — [
  `template<size_t Batch, size_t N> Tensor<Batch, N> BatchedActivate(const Tensor<Batch, N>& Z, ActivationFunction act)`](src/TTTN_ML.hpp)
    -

- ***BatchedActivatePrime*** — [
  `template<size_t Batch, size_t N> Tensor<Batch, N> BatchedActivatePrime(const Tensor<Batch, N>& grad, const Tensor<Batch, N>& a, ActivationFunction act)`](src/TTTN_ML.hpp)
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

A shape-preserving, parameter-free block that applies axis-generalized softmax. `ParamCount == 0`; `Update`, `Save`,
`Load` are all no-ops.

- ***Forward*** — [`OutputTensor Forward(const InputTensor& x) const`](src/TTTN_ML.hpp)
  -

- ***ZeroGrad*** — [`void ZeroGrad()`](src/TTTN_ML.hpp)
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

- ***Update*** — [`static void Update(float, float, float, float, float, float)`](src/TTTN_ML.hpp) *(no-op)*
  -

- ***Save*** — [`static void Save(std::ofstream&) const`](src/TTTN_ML.hpp) *(no-op)*
  -

- ***Load*** — [`static void Load(std::ifstream&)`](src/TTTN_ML.hpp) *(no-op)*
  -

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
    -
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
      `ReduceSum<1>(pred ⊙ labels)` (probability assigned to the true class) vs
      `ReduceMax<1>(pred)` (highest predicted probability) — no explicit argmax loop required.

---

## [Dense.hpp](src/Dense.hpp) -- Fully-Connected Layer

Implements the general multidimensional dense layer (`DenseMDBlock`) and its recipe types. Weights have shape
`Tensor<OutDims..., InDims...>`; forward pass is a generalized matrix-vector product via
`ΣΠ`. Includes Adam optimizer state.

### Multi-Dimensional Activation Helpers

- ***ActivateMD*** — [
  `template<size_t... Dims> Tensor<Dims...> ActivateMD(const Tensor<Dims...>& z, ActivationFunction act)`](src/Dense.hpp)
    -

- ***ActivatePrimeMD*** — [
  `template<size_t... Dims> Tensor<Dims...> ActivatePrimeMD(const Tensor<Dims...>& grad, const Tensor<Dims...>& a, ActivationFunction act)`](src/Dense.hpp)
    -

- ***WTBlockSwapPerm*** — [`struct WTBlockSwapPerm<size_t N_out, size_t N_in>`](src/Dense.hpp)
  -

---

### `class DenseMDBlock<typename InT, typename OutT, ActivationFunction Act_>`

The concrete fully-connected block. `W = Tensor<OutDims..., InDims...>`, `b = Tensor<OutDims...>`.

- ***DenseMDBlock*** — [`DenseMDBlock()`](src/Dense.hpp)
  -

- ***Forward*** — [`OutputTensor Forward(const InputTensor& x) const`](src/Dense.hpp)
  -

- ***ZeroGrad*** — [`void ZeroGrad()`](src/Dense.hpp)
  -

- ***Backward*** — [
  `InputTensor Backward(const OutputTensor& delta_A, const OutputTensor& a, const InputTensor& a_prev)`](src/Dense.hpp)
    -

- ***BatchedForward*** — [
  `template<size_t Batch> Tensor<Batch, OutDims...> BatchedForward(const Tensor<Batch, InDims...>& X) const`](src/Dense.hpp)
    -

- ***BatchedBackward*** — [
  `template<size_t Batch> Tensor<Batch, InDims...> BatchedBackward(const Tensor<Batch, OutDims...>& delta_A, const Tensor<Batch, OutDims...>& a, const Tensor<Batch, InDims...>& a_prev)`](src/Dense.hpp)
    -

- ***Update*** — [
  `void Update(float adamBeta1, float adamBeta2, float lr, float mCorr, float vCorr, float eps)`](src/Dense.hpp)
    -

- ***Save*** — [`void Save(std::ofstream& f) const`](src/Dense.hpp)
  -

- ***Load*** — [`void Load(std::ifstream& f)`](src/Dense.hpp)
  -

---

### `struct DenseMD<typename OutT, ActivationFunction Act_>` *(Block recipe)*

- ***Resolve*** — [`template<typename InputT> using Resolve = DenseMDBlock<InputT, OutT, Act_>`](src/Dense.hpp)
  -

### `template<size_t N, ActivationFunction Act_> using Dense`

- ***Dense*** — [`using Dense = DenseMD<Tensor<N>, Act_>`](src/Dense.hpp)
  -

---

## [Attention.hpp](src/Attention.hpp) -- Multi-Head Self-Attention

Implements scaled dot-product multi-head self-attention over sequences of arbitrary-rank token embeddings. Forward-pass cache is stored as
`mutable` members. All four weight matrices (`W_Q`, `W_K`, `W_V`, `W_O`) are updated with Adam.

### `class MultiHeadAttentionBlock<size_t SeqLen, size_t Heads, size_t... EmbDims>`

`InputTensor = OutputTensor = Tensor<SeqLen, EmbDims...>`. Constraint: `EmbSize % Heads == 0`.

- ***MultiHeadAttentionBlock*** — [`MultiHeadAttentionBlock()`](src/Attention.hpp)
  -

- ***Forward*** — [`OutputTensor Forward(const InputTensor& X) const`](src/Attention.hpp)
  -

- ***ZeroGrad*** — [`void ZeroGrad()`](src/Attention.hpp)
  -

- ***Backward*** — [
  `InputTensor Backward(const OutputTensor& delta_A, const OutputTensor& a, const InputTensor& a_prev)`](src/Attention.hpp)
    -

- ***BatchedForward*** — [
  `template<size_t Batch> Tensor<Batch, SeqLen, EmbDims...> BatchedForward(const Tensor<Batch, SeqLen, EmbDims...>& X)`](src/Attention.hpp)
    -

- ***BatchedBackward*** — [
  `template<size_t Batch> Tensor<Batch, SeqLen, EmbDims...> BatchedBackward(const Tensor<Batch, SeqLen, EmbDims...>& delta_A, const Tensor<Batch, SeqLen, EmbDims...>& a, const Tensor<Batch, SeqLen, EmbDims...>& a_prev)`](src/Attention.hpp)
    -

- ***Update*** — [
  `void Update(float adamBeta1, float adamBeta2, float lr, float mCorr, float vCorr, float eps)`](src/Attention.hpp)
    -

- ***Save*** — [`void Save(std::ofstream& f) const`](src/Attention.hpp)
  -

- ***Load*** — [`void Load(std::ifstream& f)`](src/Attention.hpp)
  -

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

## [NetworkUtil.hpp](src/NetworkUtil.hpp) -- Concepts, Types, and Utilities

Defines the two block concepts that gate the type system, the chain-resolution machinery used by
`NetworkBuilder`, and the `ActivationsWrap` safety wrapper.

### Concepts

- ***ConcreteBlock*** — [`concept ConcreteBlock<T>`](src/NetworkUtil.hpp)
  -

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

## [TrainableTensorNetwork.hpp](src/TrainableTensorNetwork.hpp) -- The Network

The top-level network class and the `NetworkBuilder` factory. Owns all blocks in a
`std::tuple`, orchestrates forward and backward passes, and manages Adam optimizer state (bias-correction counters).

### `class TrainableTensorNetwork<ConcreteBlock... Blocks>`

**Constants:**

- ***ADAM_BETA_1*** — [`static constexpr float ADAM_BETA_1`](src/TrainableTensorNetwork.hpp)
  -
- ***ADAM_BETA_2*** — [`static constexpr float ADAM_BETA_2`](src/TrainableTensorNetwork.hpp)
  -

**Type aliases and constants:**

- ***InputTensor*** — [`using InputTensor`](src/TrainableTensorNetwork.hpp) -- tensor type of the first block's input
  -
- ***OutputTensor*** — [`using OutputTensor`](src/TrainableTensorNetwork.hpp) -- tensor type of the last block's output
  -
- ***InSize*** — [`static constexpr size_t InSize`](src/TrainableTensorNetwork.hpp)
  -
- ***OutSize*** — [`static constexpr size_t OutSize`](src/TrainableTensorNetwork.hpp)
  -
- ***TotalParamCount*** — [`static constexpr size_t TotalParamCount`](src/TrainableTensorNetwork.hpp)
  -
- ***Activations*** — [`using Activations`](src/TrainableTensorNetwork.hpp)
  -
- ***BatchedActivations*** — [`template<size_t Batch> using BatchedActivations`](src/TrainableTensorNetwork.hpp)
  -

**Single-sample interface:**

- ***ForwardAll*** — [`Activations ForwardAll(const InputTensor& x) const`](src/TrainableTensorNetwork.hpp)
    - Run forward pass through entire network, returning `Activations` tuple of `Tensor`s from each layer

- ***Forward*** — [`OutputTensor Forward(const InputTensor& x) const`](src/TrainableTensorNetwork.hpp)
    - Run forward pass through entire network, returning a `Tensor` of type: `OutputTensor`, the final activation

- ***BackwardAll*** — [
  `void BackwardAll(const Activations& A, const OutputTensor& grad)`](src/TrainableTensorNetwork.hpp)
    -

- ***Update*** — [`void Update(float lr)`](src/TrainableTensorNetwork.hpp)
  -

- ***ZeroGrad*** — [`void ZeroGrad()`](src/TrainableTensorNetwork.hpp)
  -

- ***TrainStep*** — [
  `void TrainStep(const InputTensor& x, const OutputTensor& grad, float lr)`](src/TrainableTensorNetwork.hpp)
    -

- ***Fit*** — [
  `template<typename Loss> float Fit(const InputTensor& x, const OutputTensor& target, float lr)`](src/TrainableTensorNetwork.hpp)
    -

**Batched interface:**

- ***BatchedForwardAll*** — [
  `template<size_t Batch> BatchedActivations<Batch> BatchedForwardAll(const typename PrependBatch<Batch, InputTensor>::type& X) const`](src/TrainableTensorNetwork.hpp)
    - Inference a batch and get a `Tensor` of type: `BatchedActivations<Batch>`

- ***BatchedForward*** — [
  `template <size_t Batch> PrependBatch<Batch, OutputTensor>::type BatchedForward(const typename PrependBatch<Batch, InputTensor>::type& X)`](src/TrainableTensorNetwork.hpp)
    - Inference the model with a batch dimension, getting in return a `Tensor` of type:
      `PrependBatch<Batch, OutputTensor>::type`


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

Utilities for loading datasets from disk, drawing random mini-batches, and displaying terminal progress bars. Shapes are compile-time parameters -- the type
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
