# TTTN — Trainable Template Tensor Network

A header-only C++20 neural network library built around a fully type-safe, arbitrary-rank `Tensor` type. Network
topology, tensor shapes, and layer connectivity are all encoded at compile time.

Used in my [AlphaToe](https://github.com/benmeyersUSC/AlphaToe) library.

**Namespace:** `TTTN`
**Umbrella include:** `#include "src/TTTN.hpp"`

---

## Table of Contents

1. [Tensor.hpp — The Foundational Object](#tensorhpp--the-foundational-object)
2. [TensorOps.hpp — Tensor Operations](#tensoropshpp--tensor-operations)
3. [TTTN_ML.hpp — ML Primitives](#tttn_mlhpp--ml-primitives)
4. [Dense.hpp — Fully-Connected Layer](#densehpp--fully-connected-layer)
5. [Attention.hpp — Multi-Head Self-Attention](#attentionhpp--multi-head-self-attention)
6. [NetworkUtil.hpp — Concepts, Types, and Utilities](#networkutilhpp--concepts-types-and-utilities)
7. [TrainableTensorNetwork.hpp — The Network](#trainabletensornetworkhpp--the-network)
8. [DataIO.hpp — Data Loading and Batching](#dataiohpp--data-loading-and-batching)

---

## [Tensor.hpp](src/Tensor.hpp) — The Foundational Object

The impetus and core data structure of the library is the variadic-dimension-templated `Tensor<size_t...Dims>`.
In `TTTN`, ***shapes*** are ***types*** and ***types*** denote ***shapes***. At compile-time, a `Tensor` is declared
and its `std::tuple<size_t>` of ***dimensions*** (shape), ***rank***, and ***size*** are all known. `Tensor`s' data are
all stored in flat heap arrays; to index these efficiently and in accordance with rank, we also compute a ***strides***
vector (row-major) for the type at compile-time which compose in a dot-product with the N-rank index provided to locate
the requested index in the flat array.

### Compile-Time Shape Metaprogramming

- **[`struct TensorDimsProduct<size_t... Ds>`](src/Tensor.hpp)**
    - Template-specialization-based recursion to collapse variadic template `<size_t...Ds>` into single `size_t`, stored
      statically as `TensorDimsProduct<size_t...Ds>::value`
    - Used to compute `Tensor::Shape`, used in [`struct ComputeStrides<size_t... Ds>`](src/Tensor.hpp) to compute
      `Tensor::Strides`

- **[`struct SizeTemplateGet<size_t N, size_t... Ds>`](src/Tensor.hpp)**
    - Template-specialization-based recursion grab `N`-th `size_t` from `<size_t...Ds>`
    - Uses functional-style aggregation and pattern-matching to decrement `N` and peel off `size_t`s from variadic array
      until reaching basecase where `N = 0`
    - Used for clean, compile-time syntax in [TensorOps.hpp](src/TensorOps.hpp)

- **[`struct ComputeStrides<size_t... Ds>`](src/Tensor.hpp)**
    - Template-specialization-based recursion to compute `Tensor::Strides` array
        - The `Tensor::Strides` array is vital to mapping from indices into `Tensor::Shape` to flat indices for the
          backing array
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

- **[`static constexpr size_t Rank`](src/Tensor.hpp)**
    - Number of dimensions
- **[`static constexpr size_t Size`](src/Tensor.hpp)**
    - Product of all dimensions = total distinct values in `Tensor`
- **[`static constexpr std::array<size_t, Rank> Shape`](src/Tensor.hpp)**
    - `<size_t... Dims>` captured into an array
- **[`static constexpr std::array<size_t, Rank> Strides`](src/Tensor.hpp)**
    - Uses `ComputeStrides` to create array
    - The `Tensor::Strides` array is vital to mapping from indices into `Tensor::Shape` to flat indices for the
      backing array
    - In general, for a `Tensor` with `Tensor::Shape = [A, B, ..., N]`, its
      `Tensor::Strides = [A * B * ... * N, B * ... * N, ..., 1]`

**Static methods:**

- **[`static constexpr std::array<size_t, Rank> FlatToMulti(size_t flat)`](src/Tensor.hpp)**
    - Inverse of `MultiToFlat`; map a flat index `[0, Size)` to its `Rank`-term index
    - Pattern matches a `std::index_sequence` parameterized by `<size_t... Rank>` (= `[0, ..., Rank]`) and unpacks
      into an array: `[(flat / Strides[0]) % Shape[0], ..., (flat / Strides[Rank]) % Shape[Rank]]`

- **[`static constexpr size_t MultiToFlat(const std::array<size_t, Rank>& multi)`](src/Tensor.hpp)**
    - Inverse of `FlatToMulti`; map a `Rank`-term index to its flat index `[0, Size)`
    - Pattern matches a `std::index_sequence` parameterized by `<size_t... Rank>` (= `[0, ..., Rank]`) and `+`-folds
      terms `multi[0] * Strides[0] + ... + multi[Rank] * Strides[Rank]`
    - Dot product of given `multi` index and `Strides`

**Constructors / Rule of Five:**

- **[`Tensor()`](src/Tensor.hpp)**
    - Default constructor
    - Initialize `float[Size]` on the heap

- **[`Tensor(std::initializer_list<float> init)`](src/Tensor.hpp)**
    - Initializer list constructor
    - Fill the first `Size` elements of `std::initializer_list<float> init` to flat indices of backing array

- **[`~Tensor() = default`](src/Tensor.hpp)**
    - Default destructor
    - RAII: destructs `std::unique_ptr` to `float[Size]` on heap

- **[`Tensor(const Tensor& other)`](src/Tensor.hpp)**
    - Deep copy constructor
    - Allocate new `float[Size]` on heap and `std::memcpy` from `other.data()`

- **[`Tensor& operator=(const Tensor& other)`](src/Tensor.hpp)**
    - Deep copy assignment operator
    - `std::memcpy` from `other.data()`

- **[`Tensor(Tensor&&) noexcept = default`](src/Tensor.hpp)**
    - Default move constructor
    - `std::unique_ptr` to data handles this already

- **[`Tensor& operator=(Tensor&&) noexcept = default`](src/Tensor.hpp)**
    - Default move assigment operator
    - `std::unique_ptr` to data handles this already

**Data access:**

- **[`void fill(const float v) const`](src/Tensor.hpp)**
    - Fill a `Tensor`'s underlying array with some `float v`

- **[`float* data()`](src/Tensor.hpp)**
    - Returns pointer to `Tensor`'s underlying array

- **[`const float* data() const`](src/Tensor.hpp)**
    - Returns `const` pointer to `Tensor`'s underlying array

- **[`float& flat(size_t idx)`](src/Tensor.hpp)**
    - Returns `float&` reference to item at `idx` in underlying array

- **[`float flat(size_t idx) const`](src/Tensor.hpp)**
    - Returns `const float&` reference to item at `idx` in underlying array

**Functional transforms:**

- **[`template<typename F> Tensor map(F f) const`](src/Tensor.hpp)**
    - Use `std::execution::par_unseq` to `std::transform` `Tensor`'s underlying data by `float -> float` map `f`,
      returning a new `Tensor`

- **[`template<typename F> Tensor zip(const Tensor& other, F f) const`](src/Tensor.hpp)**
    - Use `std::execution::par_unseq` to `std::transform` two `Tensor`s' underlying data by `float -> float -> float`
      map `f`, returning a new `Tensor`

- **[`template<typename F> void apply(F f)`](src/Tensor.hpp)**
    - Use `std::execution::par_unseq` + `std::for_each` to apply `float -> float` map `f` to `Tensor`'s underlying data
      in-place

- **[`template<typename F> void zip_apply(const Tensor& other, F f)`](src/Tensor.hpp)**
    - Use `std::execution::par_unseq` to `std::transform` two `Tensor`s' underlying data in-place by
      `float -> float -> float` map `f`

**Indexing operators:**

- **[`float& operator()(Indices... idxs)`](src/Tensor.hpp)**
    - Variadic multi-index access, returns reference
    - Uses compile-time-templated `MultiToFlat` for efficient access

- **[`float operator()(Indices... idxs) const`](src/Tensor.hpp)**
    - Variadic multi-index access, returns copy
    - Uses compile-time-templated `MultiToFlat` for efficient access

- **[`float& operator()(const std::array<size_t, Rank>& multi)`](src/Tensor.hpp)**
    - Array-based multi-index access, returns reference
    - Uses compile-time-templated `MultiToFlat` for efficient access

- **[`float operator()(const std::array<size_t, Rank>& multi) const`](src/Tensor.hpp)**
    - Array-based multi-index access, returns copy
    - Uses compile-time-templated `MultiToFlat` for efficient access

**Serialization:**

- **[`void Save(std::ofstream& f) const`](src/Tensor.hpp)**
    - Writes entirety of flat backing array (`float[Size]`) to binary file

- **[`void Load(std::ifstream& f)`](src/Tensor.hpp)**
    - Reads binary file into flat backing array (`float[Size]`)

---

### Type Traits

- **[`struct is_tensor<T>`](src/Tensor.hpp)**
    - SFINAE type traits for verifying that a type is a `Tensor`
        - Specialize `<size_t...Dims>`: matches into `<Tensor<Dims...>`, inherits from `std::true_type`
        - Specialize <>: inherits from `std::false_type`, backup when former fails


- **[`concept IsTensor<T>`](src/Tensor.hpp)**
    - Wrapper `concept` around `is_tensor` type trait, satisfied if `T` is a `Tensor`

### Tensor Demonstration

```cpp
// shapes are types — different shapes are different types
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

## [TensorOps.hpp](src/TensorOps.hpp) — Tensor Operations

Generalized tensor algebra: contractions, permutations, reductions, slicing, and broadcast operations. All shapes are
derived at compile time.

### Shape Manipulation Helpers

- **[`struct TensorConcat<typename T1, typename T2>`](src/TensorOps.hpp)**
    - Templated utility struct for concatenating the dimensions of two `Tensor` objects
        - Templated for two types, `<typename T1, typename T2>`
        - Specialized for two `Tensor`s, `<size_t... Ds1, size_t... Ds2>` and pattern-matched to
          `<Tensor<Ds1...>, Tensor<Ds2...>>`, creating `type = Tensor<Ds1..., Ds2...>`

- **[`struct KeptDimsHolder<size_t Skip, size_t... Dims>`](src/TensorOps.hpp)**
    - Given a pack `Dims...` and an axis to `Skip`, produce `Tensor<remaining dims...>`
    - `static constexpr value` holds the new array of `sizeof...(Dims) - 1` dimensions

- **[`struct ArrayToTensor<typename KeptIdxs, typename Iota>`](src/TensorOps.hpp)**
    - Unpack `KeptDimsHolder::value` into new `Tensor` type defined by those kept dimensions
    - Beautiful syntax: `type = Tensor<arr[Iota]...>`, where `arr = KeptDimsHolder::value` and `Iota...` represents the
      `[0, arr.size())` indices

- **[`struct RemoveAxis<size_t Skip, size_t... Dims>`](src/TensorOps.hpp)**
    - Compact operator to make new `Tensor` type by removing `Skip` dimension from given `Tensor`
    - `type = ArrayToTensor<KeptDimsHolder<Skip, Dims...>, std::make_index_sequence<sizeof...(Dims) - 1>`

- **[`struct SliceDimsHolder<size_t Start, size_t Len, size_t... Dims>`](src/TensorOps.hpp)**
    - Helper struct to hold `std::array<size_t, Len> value` representing contiguous dimensions `[Start, Start+Len)` of a
      `Tensor<Dims...>`
    - Will not compile if `Start + Len > Rank`

- **[`struct TensorSlice<size_t Start, size_t Len, size_t... Dims>`](src/TensorOps.hpp)**
    - Using `ArrayToTensor` and `SliceDimsHolder`, create new `Tensor` object out of a set of dimensions,
      `[Start, Start+Len)`, from original `Tensor<Dims...>`

---

### Arithmetic Operators

- **[
  `template<size_t... Dims> Tensor<Dims...> operator+(const Tensor<Dims...>& a, const Tensor<Dims...>& b)`](src/TensorOps.hpp)
  **
    - Element-wise add, uses parallel functional `zip`

- **[
  `template<size_t... Dims> Tensor<Dims...> operator-(const Tensor<Dims...>& a, const Tensor<Dims...>& b)`](src/TensorOps.hpp)
  **
    - Element-wise subtract, uses parallel functional `zip`

- **[
  `template<size_t... Dims> Tensor<Dims...>& operator+=(Tensor<Dims...>& a, const Tensor<Dims...>& b)`](src/TensorOps.hpp)
  **
    - Element-wise add-to, uses parallel functional `zip_apply`

- **[`template<size_t... Dims> Tensor<Dims...> operator*(const Tensor<Dims...>& a, float s)`](src/TensorOps.hpp)**
    - Scalar multiply, uses parallel functional `map`

- **[`template<size_t... Dims> Tensor<Dims...> operator*(float s, const Tensor<Dims...>& a)`](src/TensorOps.hpp)**
    - Scalar multiply, uses parallel functional `map`

- **[
  `template<size_t... Dims> Tensor<Dims...> operator*(const Tensor<Dims...>& a, const Tensor<Dims...>& b)`](src/TensorOps.hpp)
  **
    - Hadamard (element-wise) product, uses parallel functional `zip`

---

### Permutation

- **[`struct PermutedTensorType<typename T, size_t... Perm>`](src/TensorOps.hpp)**
    - Type for a permuted `Tensor`
    - Arbitrary reorganization of a `Tensor`'s `Shape`
        - Templated to `<typename T, size_t... Perm>`
        - Specialized by `<size_t... Dims, size_t... Perm>` and matched to `<Tensor<Dims...>, Perm...>`
            - Unpack and reassign `Shape`'s indices according to `Perm`:
              `Tensor<Tensor<Dims...>::Shape[Perm]...>`
    - Example: `PermutedTensorType<Tensor<4, 5, 3>, 2, 0, 1>::type = Tensor<3, 4, 5>`

- **[`template<size_t... Perm, size_t... Dims> auto Permute(const Tensor<Dims...>& src)`](src/TensorOps.hpp)**
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

- **[`struct MoveToLastPerm<size_t Src, size_t Rank>`](src/TensorOps.hpp)**
    - Compile-time helper to rearrange `Tensor`'s shape such that `Src` is at index `[Rank - 1]` and all others are kept
      in order

- **[`struct MoveToFirstPerm<size_t Src, size_t Rank>`](src/TensorOps.hpp)**
    - Compile-time helper to rearrange `Tensor`'s shape such that `Src` is at index `[0]` and all others are kept
      in order

- **[
  `template<typename PermHolder, size_t... I, size_t... Dims> auto PermuteFromHolder(const Tensor<Dims...>& t, std::index_sequence<I...>)`](src/TensorOps.hpp)
  **
    - Unpack a `constexpr` permutation indices array into a proper `Permute`-given `Tensor` type
    - `PermHolder` is an array of permutation indices, typically the result of `MoveToLastPerm` or `MoveToFirstPerm`
    - Call with `PermHolder` as template arg, `Tensor<Dims...>` as first arg,
      `std::make_index_sequence<sizeof...(Dims)>{}` as second arg

- **[`template<size_t... Dims> auto Transpose(const Tensor<Dims...>& t)`](src/TensorOps.hpp)**
    - Reverse all dimensions of a `Tensor<Dims...>`
    - `Permute<(sizeof...(Dims) - 1 - I)...>(s)`, for `I` in `Dims...`
    - so if `Dims` are `<3, 5, 4>` (`sizeof = 3`) we will have:
        - `Permute<`(3-1) - 0 = `2,` (3-1) - 1 = `1,` (3-1) - 2 = `0>`

---

### Tensor Contraction : ΣΠ

Tensor contraction is the generalized ***Sum of Products*** operation on `Tensor`s. `Dot`, `Matmul`, `Outer`, and
`Einsum` are all special cases of this powerful operation.

- **[`template<size_t N, typename TA, typename TB> struct SigmaPiKernel`](src/TensorOps.hpp)**
    - Struct templated on `<size_t N, size_t... ADims, size_t... BDims>`, matched to
      `<N, Tensor<ADims...>, Tensor<BDims...>>`
    - Compile-time `static constexpr`:
        - `RankA = sizeof...(ADims)`
        - `RankB = sizeof...(BDims)`
        - Asserts that `N <= RankA && N <= RankB`
        - Asserts that last `N` dimensions of `ADims...` are each equal in size to the first `N` dimensions of
          `BDims...`
        - Free dimensions of `A` and `B`
        - `A_Free = TensorSlice<0, RankA - N, ADims...>::type`
            - `B_Free = TensorSlice<N, RankB - N, BDims...>::type`
        - Types for contracted and resulting `Tensor`s
            - `Contracted = TensorSlice<RankA - N, N, ADims...>::type`
            - `ResultType = TensorConcat<A_Free, B_Free>::type`
        - Index tables
            - `struct { std::array<size_t, Contracted::Size> a, b; } offsets;`
                - Precomputes flat-index contribution of every contracted position for A and B
            - `struct { std::array<size_t, ResultType::Size> a, b; } bases;`
                - Precomputes free-dimension base offset in A and B for every output index
        - **These pay real dividends for [TrainableTensorNetwork](./src/TrainableTensorNetwork.hpp) training schedules.
          Any weight `Tensor`'s `Dot`s, `Matmul`s, and `Outer`s (*in forward and backward passes*) are saved structs,
          and
          the runtime computations are parallelized and vectorized, following known, saved paths**

- **[
  `template<size_t N, size_t... ADims, size_t... BDims> auto ΣΠ(const Tensor<ADims...>& A, const Tensor<BDims...>& B)`](src/TensorOps.hpp)
  **
    - For every ***product*** in Union(Free Axes of A, Free Axes of B), ***Sum every Product*** of contracted axes.
        - We obtain the free axes of `A` and `B` and combine them into a `Tensor` as such:
          `ResultType = TensorConcat<A_Free, B_Free>::type`, where `A_Free = TensorSlice<0, RankA - N, ADims...>::type`
          and `B_Free = TensorSlice<N, RankB - N, BDims...>::type`
    - `result.flat(o) = Σ_c A[A_Free(o), c] * B[c, B_Free(o)]`, for `o ∈ [0, ResultType::Size)` — where `A_Free(o)` and
      `B_Free(o)` are the first `RankA−N` and last `RankB−N` components of output multi-index `o`, and `c` ranges over
      `Contracted`


- **[
  `template<size_t N, size_t... ADims, size_t... BDims> auto ΣΠ(const Tensor<ADims...>& A, const Tensor<BDims...>& B)`](src/TensorOps.hpp)
  **
    - Wrapper function around `SigmaPiKernel::Compute`
    - returns `SigmaPiKernel<N, Tensor<ADims...>, Tensor<BDims...> >::compute(A, B)`


- **[
  `template<size_t N, size_t... ADims, size_t... BDims> auto SigmaPi(const Tensor<ADims...>& A, const Tensor<BDims...>& B)`](src/TensorOps.hpp)
  **
    - English wrapper around greek-aliased function
    - returns `ΣΠ<N>(A, B)`

- **[
  `template<size_t I, size_t J, size_t... ADims, size_t... BDims> auto Einsum(const Tensor<ADims...>& A, const Tensor<BDims...>& B)`](src/TensorOps.hpp)
  **
    - `ΣΠ`-contracts over single selected indices, `I` and `J`, from `Tensor<ADims...>` and `Tensor<BDims...>`,
      respectively
    - Calls `ΣΠ<1>` on `PermuteFromHolder<MoveToLastPerm<I, sizeof...(ADims)>...>` and
      `PermuteFromHolder<MoveToFirstPerm<J, sizeof...(BDims)>...>`

- **[`template<size_t N> auto Dot(const Tensor<N>& a, const Tensor<N>& b)`](src/TensorOps.hpp)**
    - `ΣΠ`-contracts over `1` (the only) inner dimension of two `Tensor<N>`s, `a` and `b`, returning a `Tensor<>` with
      `Rank = 0`

- **[
  `template<size_t M, size_t K, size_t N> auto Matmul(const Tensor<M,K>& A, const Tensor<K,N>& B)`](src/TensorOps.hpp)
  **
    - `ΣΠ`-contracts over `1` inner dimension of two `Tensor<_,K>`s, `A` and `A`, returning a `Tensor<M,N>` with
      `Rank = 2`

- **[
  `template<size_t... ADims, size_t... BDims> auto Outer(const Tensor<ADims...>& a, const Tensor<BDims...>& b)`](src/TensorOps.hpp)
  **
    - `ΣΠ`-contracts over `0` inner dimension of `Tensor<ADims...> a` and `Tensor<BDims...> b`, returning a
      `Tensor<ADims..., BDims...>` with
      `Rank = sizeof...(ADims) + sizeof...(BDims)`

---

### Reduction and Broadcast

- **[
  `template<size_t Axis, size_t... Dims> auto ReduceSum(const Tensor<Dims...>& src) -> typename RemoveAxis<Axis, Dims...>::type`](src/TensorOps.hpp)
  **
    -

- **[
  `template<size_t Axis, size_t... Dims> Tensor<Dims...> BroadcastAdd(const Tensor<Dims...>& A, const typename RemoveAxis<Axis, Dims...>::type& b)`](src/TensorOps.hpp)
  **
    -

- **[
  `template<size_t Axis, size_t... Dims> auto ReduceMean(const Tensor<Dims...>& src) -> typename RemoveAxis<Axis, Dims...>::type`](src/TensorOps.hpp)
  **
    -

- **[
  `template<size_t Axis, size_t... Dims> auto ReduceMax(const Tensor<Dims...>& src) -> typename RemoveAxis<Axis, Dims...>::type`](src/TensorOps.hpp)
  **
    -

---

### Indexed Slice Access

- **[
  `template<size_t Axis, size_t... Dims> auto TensorIndex(const Tensor<Dims...>& src, size_t idx) -> typename RemoveAxis<Axis, Dims...>::type`](src/TensorOps.hpp)
  **
    -

- **[
  `template<size_t Axis, size_t... Dims> void TensorIndexAdd(Tensor<Dims...>& dst, size_t idx, const typename RemoveAxis<Axis, Dims...>::type& src)`](src/TensorOps.hpp)
  **
    -

---

### Softmax (Axis-Generalized)

- **[`template<size_t Axis, size_t... Dims> Tensor<Dims...> Softmax(const Tensor<Dims...>& x)`](src/TensorOps.hpp)**
  -

- **[
  `template<size_t Axis, size_t... Dims> Tensor<Dims...> SoftmaxPrime(const Tensor<Dims...>& grad, const Tensor<Dims...>& a)`](src/TensorOps.hpp)
  **
    -

### Parallel Loop Helper

- **[`template<typename F> void ParForEach(size_t n, F f)`](src/TensorOps.hpp)**
  -

---

## [TTTN_ML.hpp](src/TTTN_ML.hpp) — ML Primitives

Activation functions, their derivatives, loss functions, and the `SoftmaxBlock` layer. Depends on `TensorOps.hpp`.

**Constants:**

- **[`static constexpr float EPS`](src/TTTN_ML.hpp)**
  -

### Activation Function Enum

- **[`enum class ActivationFunction`](src/TTTN_ML.hpp)** — `Linear`, `Sigmoid`, `ReLU`, `Softmax`, `Tanh`
  -

---

### Free Functions

- **[`template<size_t N> float CrossEntropyLoss(const Tensor<N>& output, const Tensor<N>& target)`](src/TTTN_ML.hpp)**
  -

- **[`template<size_t... Dims> void XavierInitMD(Tensor<Dims...>& W, size_t fan_in, size_t fan_out)`](src/TTTN_ML.hpp)**
  -

- **[`template<size_t N> Tensor<N> Activate(const Tensor<N>& z, ActivationFunction act)`](src/TTTN_ML.hpp)**
  -

- **[
  `template<size_t N> Tensor<N> ActivatePrime(const Tensor<N>& grad, const Tensor<N>& a, ActivationFunction act)`](src/TTTN_ML.hpp)
  **
    -

- **[
  `template<size_t Batch, size_t N> Tensor<Batch, N> BatchedActivate(const Tensor<Batch, N>& Z, ActivationFunction act)`](src/TTTN_ML.hpp)
  **
    -

- **[
  `template<size_t Batch, size_t N> Tensor<Batch, N> BatchedActivatePrime(const Tensor<Batch, N>& grad, const Tensor<Batch, N>& a, ActivationFunction act)`](src/TTTN_ML.hpp)
  **
    -

---

### `class SoftmaxBlock<size_t Axis, typename TensorT>`

A shape-preserving, parameter-free block that applies axis-generalized softmax. `ParamCount == 0`; `Update`, `Save`,
`Load` are all no-ops.

- **[`OutputTensor Forward(const InputTensor& x) const`](src/TTTN_ML.hpp)**
  -

- **[`void ZeroGrad()`](src/TTTN_ML.hpp)**
  -

- **[
  `InputTensor Backward(const OutputTensor& delta_A, const OutputTensor& a, const InputTensor& a_prev)`](src/TTTN_ML.hpp)
  **
    -

- **[
  `template<size_t Batch> Tensor<Batch, Dims...> BatchedForward(const Tensor<Batch, Dims...>& X) const`](src/TTTN_ML.hpp)
  **
    -

- **[
  `template<size_t Batch> Tensor<Batch, Dims...> BatchedBackward(const Tensor<Batch, Dims...>& delta_A, const Tensor<Batch, Dims...>& a, const Tensor<Batch, Dims...>& a_prev)`](src/TTTN_ML.hpp)
  **
    -

- **[`void Update(float, float, float, float, float, float)`](src/TTTN_ML.hpp)** *(no-op)*
  -

- **[`void Save(std::ofstream&) const`](src/TTTN_ML.hpp)** *(no-op)*
  -

- **[`void Load(std::ifstream&)`](src/TTTN_ML.hpp)** *(no-op)*
  -

---

### `struct SoftmaxLayer<size_t Axis>` *(Block recipe)*

- **[`template<typename InputT> using Resolve = SoftmaxBlock<Axis, InputT>`](src/TTTN_ML.hpp)**
  -

---

### Loss Functions

- **[`concept LossFunction<typename L, typename TensorT>`](src/TTTN_ML.hpp)**
  -

#### `struct MSE`

- **[
  `template<size_t... Dims> static float Loss(const Tensor<Dims...>& pred, const Tensor<Dims...>& target)`](src/TTTN_ML.hpp)
  **
    -
- **[
  `template<size_t... Dims> static Tensor<Dims...> Grad(const Tensor<Dims...>& pred, const Tensor<Dims...>& target)`](src/TTTN_ML.hpp)
  **
    -

#### `struct BinaryCEL`

- **[
  `template<size_t... Dims> static float Loss(const Tensor<Dims...>& pred, const Tensor<Dims...>& target)`](src/TTTN_ML.hpp)
  **
    -
- **[
  `template<size_t... Dims> static Tensor<Dims...> Grad(const Tensor<Dims...>& pred, const Tensor<Dims...>& target)`](src/TTTN_ML.hpp)
  **
    -

#### `struct CEL`

- **[
  `template<size_t... Dims> static float Loss(const Tensor<Dims...>& pred, const Tensor<Dims...>& target)`](src/TTTN_ML.hpp)
  **
    -
- **[
  `template<size_t... Dims> static Tensor<Dims...> Grad(const Tensor<Dims...>& pred, const Tensor<Dims...>& target)`](src/TTTN_ML.hpp)
  **
    -

---

## [Dense.hpp](src/Dense.hpp) — Fully-Connected Layer

Implements the general multi-dimensional dense layer (`DenseMDBlock`) and its recipe types. Weights have shape
`Tensor<OutDims..., InDims...>`; forward pass is a generalized matrix-vector product via `ΣΠ`. Includes Adam optimizer
state.

### Multi-Dimensional Activation Helpers

- **[
  `template<size_t... Dims> Tensor<Dims...> ActivateMD(const Tensor<Dims...>& z, ActivationFunction act)`](src/Dense.hpp)
  **
    -

- **[
  `template<size_t... Dims> Tensor<Dims...> ActivatePrimeMD(const Tensor<Dims...>& grad, const Tensor<Dims...>& a, ActivationFunction act)`](src/Dense.hpp)
  **
    -

- **[`struct WTBlockSwapPerm<size_t N_out, size_t N_in>`](src/Dense.hpp)**
  -

---

### `class DenseMDBlock<typename InT, typename OutT, ActivationFunction Act_>`

The concrete fully-connected block. `W = Tensor<OutDims..., InDims...>`, `b = Tensor<OutDims...>`.

- **[`DenseMDBlock()`](src/Dense.hpp)**
  -

- **[`OutputTensor Forward(const InputTensor& x) const`](src/Dense.hpp)**
  -

- **[`void ZeroGrad()`](src/Dense.hpp)**
  -

- **[
  `InputTensor Backward(const OutputTensor& delta_A, const OutputTensor& a, const InputTensor& a_prev)`](src/Dense.hpp)
  **
    -

- **[
  `template<size_t Batch> Tensor<Batch, OutDims...> BatchedForward(const Tensor<Batch, InDims...>& X) const`](src/Dense.hpp)
  **
    -

- **[
  `template<size_t Batch> Tensor<Batch, InDims...> BatchedBackward(const Tensor<Batch, OutDims...>& delta_A, const Tensor<Batch, OutDims...>& a, const Tensor<Batch, InDims...>& a_prev)`](src/Dense.hpp)
  **
    -

- **[`void Update(float adamBeta1, float adamBeta2, float lr, float mCorr, float vCorr, float eps)`](src/Dense.hpp)**
  -

- **[`void Save(std::ofstream& f) const`](src/Dense.hpp)**
  -

- **[`void Load(std::ifstream& f)`](src/Dense.hpp)**
  -

---

### `struct DenseMD<typename OutT, ActivationFunction Act_>` *(Block recipe)*

- **[`template<typename InputT> using Resolve = DenseMDBlock<InputT, OutT, Act_>`](src/Dense.hpp)**
  -

### `template<size_t N, ActivationFunction Act_> using Dense`

- **[`using Dense = DenseMD<Tensor<N>, Act_>`](src/Dense.hpp)**
  -

---

## [Attention.hpp](src/Attention.hpp) — Multi-Head Self-Attention

Implements scaled dot-product multi-head self-attention over sequences of arbitrary-rank token embeddings. Forward-pass
cache is stored as `mutable` members. All four weight matrices (`W_Q`, `W_K`, `W_V`, `W_O`) are updated with Adam.

### `class MultiHeadAttentionBlock<size_t SeqLen, size_t Heads, size_t... EmbDims>`

`InputTensor = OutputTensor = Tensor<SeqLen, EmbDims...>`. Constraint: `EmbSize % Heads == 0`.

- **[`MultiHeadAttentionBlock()`](src/Attention.hpp)**
  -

- **[`OutputTensor Forward(const InputTensor& X) const`](src/Attention.hpp)**
  -

- **[`void ZeroGrad()`](src/Attention.hpp)**
  -

- **[
  `InputTensor Backward(const OutputTensor& delta_A, const OutputTensor& a, const InputTensor& a_prev)`](src/Attention.hpp)
  **
    -

- **[
  `template<size_t Batch> Tensor<Batch, SeqLen, EmbDims...> BatchedForward(const Tensor<Batch, SeqLen, EmbDims...>& X)`](src/Attention.hpp)
  **
    -

- **[
  `template<size_t Batch> Tensor<Batch, SeqLen, EmbDims...> BatchedBackward(const Tensor<Batch, SeqLen, EmbDims...>& delta_A, const Tensor<Batch, SeqLen, EmbDims...>& a, const Tensor<Batch, SeqLen, EmbDims...>& a_prev)`](src/Attention.hpp)
  **
    -

- **[`void Update(float adamBeta1, float adamBeta2, float lr, float mCorr, float vCorr, float eps)`](src/Attention.hpp)
  **
    -

- **[`void Save(std::ofstream& f) const`](src/Attention.hpp)**
  -

- **[`void Load(std::ifstream& f)`](src/Attention.hpp)**
  -

---

### `struct TensorFirstDim<typename T>`

- **[`static constexpr size_t value`](src/Attention.hpp)**
  -

---

### `struct MHAttention<size_t Heads, size_t... EmbDims>` *(Block recipe)*

- **[
  `template<typename InputT> using Resolve = MultiHeadAttentionBlock<TensorFirstDim<InputT>::value, Heads, EmbDims...>`](src/Attention.hpp)
  **
    -

---

## [NetworkUtil.hpp](src/NetworkUtil.hpp) — Concepts, Types, and Utilities

Defines the two block concepts that gate the type system, the chain-resolution machinery used by `NetworkBuilder`, and
the `ActivationsWrap` safety wrapper.

### Concepts

- **[`concept ConcreteBlock<T>`](src/NetworkUtil.hpp)**
  -

- **[`concept Block<B>`](src/NetworkUtil.hpp)**
  -

---

### Chain Resolution

- **[`struct BuildChain<typename In, Block... Recipes>`](src/NetworkUtil.hpp)**
  -

- **[`struct Input<size_t... Dims>`](src/NetworkUtil.hpp)**
  -

- **[`struct PrependBatch<size_t Batch, typename T>`](src/NetworkUtil.hpp)**
  -

---

### `class ActivationsWrap<typename TupleT>`

Thin owning wrapper around an activations tuple. Deletes the rvalue `get()` overload at compile time to prevent dangling
references.

- **[`explicit ActivationsWrap(TupleT t)`](src/NetworkUtil.hpp)**
  -

- **[`template<size_t N> auto get() const& -> const std::tuple_element_t<N, TupleT>&`](src/NetworkUtil.hpp)**
  -

- **[`template<size_t N> auto get() & -> std::tuple_element_t<N, TupleT>&`](src/NetworkUtil.hpp)**
  -

- **[`template<size_t N> auto get() && -> std::tuple_element_t<N, TupleT>&& = delete`](src/NetworkUtil.hpp)**
  -

- **[`const TupleT& tuple() const`](src/NetworkUtil.hpp)**
  -

---

## [TrainableTensorNetwork.hpp](src/TrainableTensorNetwork.hpp) — The Network

The top-level network class and the `NetworkBuilder` factory. Owns all blocks in a `std::tuple`, orchestrates forward
and backward passes, and manages Adam optimizer state (bias-correction counters).

### `class TrainableTensorNetwork<ConcreteBlock... Blocks>`

**Constants:**

- **[`static constexpr float ADAM_BETA_1`](src/TrainableTensorNetwork.hpp)**
  -
- **[`static constexpr float ADAM_BETA_2`](src/TrainableTensorNetwork.hpp)**
  -

**Type aliases and constants:**

- **[`using InputTensor`](src/TrainableTensorNetwork.hpp)** — tensor type of the first block's input
  -
- **[`using OutputTensor`](src/TrainableTensorNetwork.hpp)** — tensor type of the last block's output
  -
- **[`static constexpr size_t InSize`](src/TrainableTensorNetwork.hpp)**
  -
- **[`static constexpr size_t OutSize`](src/TrainableTensorNetwork.hpp)**
  -
- **[`static constexpr size_t TotalParamCount`](src/TrainableTensorNetwork.hpp)**
  -
- **[`using Activations`](src/TrainableTensorNetwork.hpp)**
  -
- **[`template<size_t Batch> using BatchedActivations`](src/TrainableTensorNetwork.hpp)**
  -

**Single-sample interface:**

- **[`Activations ForwardAll(const InputTensor& x) const`](src/TrainableTensorNetwork.hpp)**
    - Run forward pass through entire network, returning `Activations` tuple of `Tensor`s from each layer

- **[`OutputTensor Forward(const InputTensor& x) const`](src/TrainableTensorNetwork.hpp)**
    - Run forward pass through entire network, returning a `Tensor` of type: `OutputTensor`, the final activation

- **[`void BackwardAll(const Activations& A, const OutputTensor& grad)`](src/TrainableTensorNetwork.hpp)**
  -

- **[`void Update(float lr)`](src/TrainableTensorNetwork.hpp)**
  -

- **[`void ZeroGrad()`](src/TrainableTensorNetwork.hpp)**
  -

- **[`void TrainStep(const InputTensor& x, const OutputTensor& grad, float lr)`](src/TrainableTensorNetwork.hpp)**
  -

- **[
  `template<typename Loss> float Fit(const InputTensor& x, const OutputTensor& target, float lr)`](src/TrainableTensorNetwork.hpp)
  **
    -

**Batched interface:**

- **[
  `template<size_t Batch> BatchedActivations<Batch> BatchedForwardAll(const typename PrependBatch<Batch, InputTensor>::type& X) const`](src/TrainableTensorNetwork.hpp)
  **
    - Inference a batch and get a `Tensor` of type: `BatchedActivations<Batch>`

- **[
  `template <size_t Batch> PrependBatch<Batch, OutputTensor>::type BatchedForward(const typename PrependBatch<Batch, InputTensor>::type& X)`](src/TrainableTensorNetwork.hpp)
  **
    - Inference the model with a batch dimension, getting in return a `Tensor` of type:
      `PrependBatch<Batch, OutputTensor>::type`


- **[
  `template<size_t Batch> void BatchedBackwardAll(const BatchedActivations<Batch>& A, const typename PrependBatch<Batch, OutputTensor>::type& grad)`](src/TrainableTensorNetwork.hpp)
  **
    -

- **[
  `template<size_t Batch> void BatchTrainStep(const typename PrependBatch<Batch, InputTensor>::type& X, const typename PrependBatch<Batch, OutputTensor>::type& grad, float lr)`](src/TrainableTensorNetwork.hpp)
  **
    -

- **[
  `template<typename Loss, size_t Batch> float BatchFit(const typename PrependBatch<Batch, InputTensor>::type& X, const typename PrependBatch<Batch, OutputTensor>::type& Y, float lr)`](src/TrainableTensorNetwork.hpp)
  **
    -

**Serialization:**

- **[`void Save(const std::string& path) const`](src/TrainableTensorNetwork.hpp)**
  -

- **[`void Load(const std::string& path)`](src/TrainableTensorNetwork.hpp)**
  -

---

### Free Functions

- **[
  `template<typename Loss, size_t Batch, ConcreteBlock... Blocks, size_t N, size_t... DataDims, typename PrepFn> float RunEpoch(TrainableTensorNetwork<Blocks...>& net, const Tensor<N, DataDims...>& dataset, std::mt19937& rng, float lr, PrepFn prep)`](src/TrainableTensorNetwork.hpp)
  **
    -

---

### `struct NetworkBuilder<typename In, Block... Recipes>`

- **[`using type`](src/TrainableTensorNetwork.hpp)** — the fully resolved `TrainableTensorNetwork<...>` type
  -

---

### `struct CombineNetworks<typename NetA, typename NetB>`

Type-level composition of two networks into one. Concatenates the block lists of `NetA` and `NetB`
into a single `TrainableTensorNetwork`. A compile-time `static_assert` enforces that
`NetA::OutputTensor == NetB::InputTensor`. No shared weight state — the result is an independent
network whose parameter count equals `NetA::TotalParamCount + NetB::TotalParamCount`. All three
types (`NetA`, `NetB`, and the combined type) can be instantiated and trained independently.

```cpp
using Encoder     = NetworkBuilder<Input<784>, Dense<128, ReLU>, Dense<32>>::type;
using Decoder     = NetworkBuilder<Input<32>,  Dense<128, ReLU>, Dense<784>>::type;
using Autoencoder = CombineNetworks<Encoder, Decoder>::type;

Encoder     enc;   // train for representations
Decoder     dec;   // train for generation
Autoencoder ae;    // train end-to-end — all blocks update together
```

- **[`using type`](src/TrainableTensorNetwork.hpp)** — the combined `TrainableTensorNetwork<BlocksA..., BlocksB...>`
    - Result of splicing the block lists of `NetA` and `NetB`; a complete network supporting all single-sample and
      batched interfaces

---

## [DataIO.hpp](src/DataIO.hpp) — Data Loading and Batching

Utilities for loading datasets from disk, drawing random mini-batches, and displaying terminal progress bars. Shapes are
compile-time parameters — the type *is* the schema.

### `class ProgressBar`

Lightweight terminal progress bar. Construct with a total step count and optional label; call `tick()` each step.

- **[`explicit ProgressBar(size_t total, std::string label = "")`](src/DataIO.hpp)**
  -

- **[`void tick(const std::string& suffix = "", size_t n = 1)`](src/DataIO.hpp)**
    - Advances by `n` steps and redraws. `suffix` is printed to the right of the bar (e.g. `"loss=0.312"`).

- **[`void set_label(const std::string& label)`](src/DataIO.hpp)**
  -

- **[`void reset()`](src/DataIO.hpp)**
  -

---

- **[
  `template<size_t Rows, size_t Cols> Tensor<Rows, Cols> LoadCSV(const std::string& path, bool skip_header = false)`](src/DataIO.hpp)
  **
    - Parses a CSV into a `Tensor<Rows, Cols>`. On first call shows a progress bar, then writes a binary cache at
      `<path>.<Rows>x<Cols>.bin`; subsequent calls load that file directly (pure binary read, no CSV parsing). Delete
      the `.bin` file if the underlying CSV changes.

- **[
  `template<size_t Batch, size_t N, size_t... RestDims> Tensor<Batch, RestDims...> RandomBatch(const Tensor<N, RestDims...>& ds, std::mt19937& rng)`](src/DataIO.hpp)
  **
    -
