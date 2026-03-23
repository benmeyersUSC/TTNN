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

The primary data structure. `Dims...` encodes the full shape at compile time.

**Static members:**

- **[`static constexpr size_t Rank`](src/Tensor.hpp)**
  -
- **[`static constexpr size_t Size`](src/Tensor.hpp)**
  -
- **[`static constexpr std::array<size_t, Rank> Shape`](src/Tensor.hpp)**
  -
- **[`static constexpr std::array<size_t, Rank> Strides`](src/Tensor.hpp)**
  -

**Static methods:**

- **[`static auto FlatToMulti(size_t flat) -> std::array<size_t, Rank>`](src/Tensor.hpp)**
    - .

- **[`static size_t MultiToFlat(const std::array<size_t, Rank>& multi)`](src/Tensor.hpp)**
    - .

**Constructors / Rule of Five:**

- **[`Tensor()`](src/Tensor.hpp)**
  -

- **[`Tensor(std::initializer_list<float> init)`](src/Tensor.hpp)**
  -

- **[`~Tensor() = default`](src/Tensor.hpp)**
  -

- **[`Tensor(const Tensor& other)`](src/Tensor.hpp)**
  -

- **[`Tensor& operator=(const Tensor& other)`](src/Tensor.hpp)**
  -

- **[`Tensor(Tensor&&) noexcept = default`](src/Tensor.hpp)**
  -

- **[`Tensor& operator=(Tensor&&) noexcept = default`](src/Tensor.hpp)**
  -

**Data access:**

- **[`void fill(float v)`](src/Tensor.hpp)**
  -

- **[`float* data()`](src/Tensor.hpp)**
  -

- **[`const float* data() const`](src/Tensor.hpp)**
  -

- **[`float& flat(size_t idx)`](src/Tensor.hpp)**
  -

- **[`float flat(size_t idx) const`](src/Tensor.hpp)**
  -

**Functional transforms:**

- **[`template<typename F> Tensor map(F f) const`](src/Tensor.hpp)**
  -

- **[`template<typename F> Tensor zip(const Tensor& other, F f) const`](src/Tensor.hpp)**
  -

- **[`template<typename F> void apply(F f)`](src/Tensor.hpp)**
  -

- **[`template<typename F> void zip_apply(const Tensor& other, F f)`](src/Tensor.hpp)**
  -

**Indexing operators:**

- **[`float& operator()(Indices... idxs)`](src/Tensor.hpp)**
  -

- **[`float operator()(Indices... idxs) const`](src/Tensor.hpp)**
  -

- **[`float& operator()(const std::array<size_t, Rank>& multi)`](src/Tensor.hpp)**
  -

- **[`float operator()(const std::array<size_t, Rank>& multi) const`](src/Tensor.hpp)**
  -

**Serialization:**

- **[`void Save(std::ofstream& f) const`](src/Tensor.hpp)**
  -

- **[`void Load(std::ifstream& f)`](src/Tensor.hpp)**
  -

---

### Type Traits

- **[`struct is_tensor<T>`](src/Tensor.hpp)**
  -

- **[`concept IsTensor<T>`](src/Tensor.hpp)**
  -

---

## [TensorOps.hpp](src/TensorOps.hpp) — Tensor Operations

Generalized tensor algebra: contractions, permutations, reductions, slicing, and broadcast operations. All shapes are
derived at compile time.

**Constants:**

- **[`static constexpr float EPS`](src/TensorOps.hpp)**
  -
- **[`static constexpr float ADAM_BETA_1`](src/TensorOps.hpp)**
  -
- **[`static constexpr float ADAM_BETA_2`](src/TensorOps.hpp)**
  -

### Shape Manipulation Helpers

- **[`struct TensorConcat<typename T1, typename T2>`](src/TensorOps.hpp)**
  -

- **[`struct ArrayToTensor<typename KeptIdxs, typename Iota>`](src/TensorOps.hpp)**
  -

- **[`struct KeptDimsHolder<size_t Skip, size_t... Dims>`](src/TensorOps.hpp)**
  -

- **[`struct RemoveAxis<size_t Skip, size_t... Dims>`](src/TensorOps.hpp)**
  -

- **[`struct SliceDimsHolder<size_t Start, size_t Len, size_t... Dims>`](src/TensorOps.hpp)**
  -

- **[`struct TensorSlice<size_t Start, size_t Len, size_t... Dims>`](src/TensorOps.hpp)**
  -

---

### Arithmetic Operators

- **[
  `template<size_t... Dims> Tensor<Dims...> operator+(const Tensor<Dims...>& a, const Tensor<Dims...>& b)`](src/TensorOps.hpp)
  **
    -

- **[
  `template<size_t... Dims> Tensor<Dims...> operator-(const Tensor<Dims...>& a, const Tensor<Dims...>& b)`](src/TensorOps.hpp)
  **
    -

- **[
  `template<size_t... Dims> Tensor<Dims...>& operator+=(Tensor<Dims...>& a, const Tensor<Dims...>& b)`](src/TensorOps.hpp)
  **
    -

- **[`template<size_t... Dims> Tensor<Dims...> operator*(const Tensor<Dims...>& a, float s)`](src/TensorOps.hpp)**
  -

- **[`template<size_t... Dims> Tensor<Dims...> operator*(float s, const Tensor<Dims...>& a)`](src/TensorOps.hpp)**
  -

- **[`template<size_t... Dims> Tensor<Dims...> operator*(const Tensor<Dims...>& a, const Tensor<Dims...>& b)`](src/TensorOps.hpp)**
  -

---

### Permutation

- **[`struct PermutedTensorType<typename T, size_t... Perm>`](src/TensorOps.hpp)**
  -

- **[`template<size_t... Perm, size_t... Dims> auto Permute(const Tensor<Dims...>& src)`](src/TensorOps.hpp)**
  -

- **[`struct MoveToLastPerm<size_t Src, size_t P>`](src/TensorOps.hpp)**
  -

- **[`struct MoveToFirstPerm<size_t Src, size_t P>`](src/TensorOps.hpp)**
  -

- **[
  `template<typename PermHolder, size_t... I, size_t... Dims> auto PermuteFromHolder(const Tensor<Dims...>& t, std::index_sequence<I...>)`](src/TensorOps.hpp)
  **
    -

- **[`template<size_t... Dims> auto Transpose(const Tensor<Dims...>& t)`](src/TensorOps.hpp)**
  -

---

### Tensor Contraction

- **[
  `template<size_t N, size_t... ADims, size_t... BDims> auto ΣΠ(const Tensor<ADims...>& A, const Tensor<BDims...>& B)`](src/TensorOps.hpp)
  **
    -

- **[
  `template<size_t N, size_t... ADims, size_t... BDims> auto SigmaPi(const Tensor<ADims...>& A, const Tensor<BDims...>& B)`](src/TensorOps.hpp)
  **
    -

- **[
  `template<size_t I, size_t J, size_t... ADims, size_t... BDims> auto Einsum(const Tensor<ADims...>& A, const Tensor<BDims...>& B)`](src/TensorOps.hpp)
  **
    -

- **[
  `template<size_t... ADims, size_t... BDims> auto Einsum(const Tensor<ADims...>& A, const Tensor<BDims...>& B)`](src/TensorOps.hpp)
  ** *(outer product overload — no axis arguments)*
    -

- **[`template<size_t N> auto Dot(const Tensor<N>& a, const Tensor<N>& b)`](src/TensorOps.hpp)**
    -

- **[`template<size_t M, size_t K, size_t N> auto Matmul(const Tensor<M,K>& A, const Tensor<K,N>& B)`](src/TensorOps.hpp)**
    -

- **[`template<size_t... ADims, size_t... BDims> auto Outer(const Tensor<ADims...>& a, const Tensor<BDims...>& b)`](src/TensorOps.hpp)**
    -

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

---

## [TTTN_ML.hpp](src/TTTN_ML.hpp) — ML Primitives

Activation functions, their derivatives, loss functions, and the `SoftmaxBlock` layer. Depends on `TensorOps.hpp`.

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
    - Inference the model with a batch dimension, getting in return a `Tensor` of type: `PrependBatch<Batch, OutputTensor>::type`
      

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

### `struct NetworkBuilder<typename In, Block... Recipes>`

- **[`using type`](src/TrainableTensorNetwork.hpp)** — the fully resolved `TrainableTensorNetwork<...>` type
  -

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
