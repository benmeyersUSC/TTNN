#pragma once
#include <array>
#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <type_traits>

#if defined(__APPLE__)
#  define PSTLD_HEADER_ONLY
#  define PSTLD_HACK_INTO_STD
#  include "pstld.h"
#else
#  include <algorithm>
#  include <execution>
#endif

namespace TTTN {
    // TENSOR DIMENSIONS PRODUCT
    // templatized dimension list --> product (for Tensor underlying array size)
    // @doc: struct TensorDimsProduct<size_t... Ds>
    /**
     * Template-specialization-based recursion to collapse variadic template `<size_t...Ds>` into single `size_t`, stored statically as `TensorDimsProduct<size_t...Ds>::value`
     * Used to compute `Tensor::Shape`, used in [`struct ComputeStrides<size_t... Ds>`](src/Tensor.hpp) to compute `Tensor::Strides`
     */
    template<size_t... Ds>
    struct TensorDimsProduct;

    // base case
    template<>
    struct TensorDimsProduct<> {
        static constexpr size_t value = 1;
    };

    // recursive case
    template<size_t First, size_t... Rest>
    struct TensorDimsProduct<First, Rest...> {
        static constexpr size_t value = First * TensorDimsProduct<Rest...>::value;
    };


    // GET DIMENSION FROM TENSOR DIMENSIONS LIST
    // @doc: struct SizeTemplateGet<size_t N, size_t... Ds>
    /**
     * Template-specialization-based recursion grab `N`-th `size_t` from `<size_t...Ds>`
     * Uses functional-style aggregation and pattern-matching to decrement `N` and peel off `size_t`s from variadic array until reaching basecase where `N = 0`
     * Used for clean, compile-time syntax in [TensorOps.hpp](src/TensorOps.hpp)
     */
    template<size_t N, size_t... Ds>
    struct SizeTemplateGet;

    // base case: first element
    template<size_t First, size_t... Rest>
    struct SizeTemplateGet<0, First, Rest...> {
        static constexpr size_t value = First;
    };

    // recursive case
    template<size_t N, size_t First, size_t... Rest>
    struct SizeTemplateGet<N, First, Rest...> {
        // peel off, ditch first, until N = 0
        static constexpr size_t value = SizeTemplateGet<N - 1, Rest...>::value;
    };


    // STRIDES: for N-Rank indexing in flat vector, we need stride vector
    // stride[0] = TensorDimsProduct<A, B, ..., N>, stride[-2] = N, stride[-1] = 1
     // @doc: struct ComputeStrides<size_t... Ds>
     /**
      * Template-specialization-based recursion to compute `Tensor::Strides` array
      *   - The `Tensor::Strides` array is vital to mapping from indices into `Tensor::Shape` to flat indices for the backing array
      *   - In general, for a `Tensor` with `Tensor::Shape = [A, B, ..., N]`, its `Tensor::Strides = [A * B * ... * N, B * ... * N, ..., 1]`
      * Specialize for `<>` and `<size_t D>`
      *   - `value = ` `[]` and `[1]`, respectively
      * Specialize for `<size_t First, size_t Second, size_t... Rest>`
      *   - recursively compute `tail = ComputeStrides<Second, Rest...>::value`
      *   - `value[0] = TensorDimsProduct<Second, Rest...>::value`
      *   - `value[i] = tail[i + 1]`
      */
    template<size_t... Ds>
    struct ComputeStrides;

    // base case: no dim
    template<>
    struct ComputeStrides<> {
        static constexpr std::array<size_t, 0> value = {};
    };

    // base case: one dim
    template<size_t D>
    struct ComputeStrides<D> {
        static constexpr std::array<size_t, 1> value = {1};
    };

    // recursive case: stride[0] is product of remaining dims, then recurse
    template<size_t First, size_t Second, size_t... Rest>
    struct ComputeStrides<First, Second, Rest...> {
        static constexpr auto tail = ComputeStrides<Second, Rest...>::value;
        static constexpr size_t N = 1 + tail.size();

        static constexpr std::array<size_t, N> compute() {
            std::array<size_t, N> result{};
            // stride[0] = Second * Rest... = product of everything after first
            result[0] = TensorDimsProduct<Second, Rest...>::value;
            for (size_t i = 0; i < tail.size(); i++) {
                result[i + 1] = tail[i];
            }
            return result;
        }

        static constexpr auto value = compute();
    };

    //<f=3,s=2,r=>
    //  <2,>--tail: [1]
    //      <>--tail: []
    //      N = 1; result = [1] (from product base case)
    //      value = [1]
    //  N = 2; result = [2,1]
    //  value = [2,1]


    // TENSOR
    template<size_t... Dims>
    class Tensor {
    public:
        // rank = number of dimensions
        static constexpr size_t Rank = sizeof...(Dims);
        // size = total number of values (size of array)
        static constexpr size_t Size = TensorDimsProduct<Dims...>::value;
        // shape = array of dims
        static constexpr std::array<size_t, Rank> Shape = {Dims...};
        // strides = used for seamless indexing
        static constexpr std::array<size_t, Rank> Strides = ComputeStrides<Dims...>::value;

        // FlatToMulti: decompose a flat index into a multi-index
        // EXAMPLE: Tensor<3, 4, 5>
        //      Rank = 3, Size = 60, range = 0-59
        //      Strides = [20, 5, 1]
        //
        // suppose: flat = 33
        //      multi = [_, _, _]; flat = 33
        //          multi[0] = 33 / Strides[0] = 33 / 20 = 1
        //          flat = 33 % 20 = 13
        //      multi = [1, _, _]; flat = 13
        //          multi[1] = 13 / Strides[1] = 13 / 5 = 2
        //          flat = 13 % 5 = 3
        //      multi = [1, 2, _]; flat = 3
        //          multi[2] = 3 / Strides[2] = 3 / 1 = 3
        //          flat = 3 % 1 = 0
        //      multi = [1, 2, 3]
        //
        // suppose: flat = 58 (2nd to last item, we should get [2, 3, 3])
        //      multi[0] = 58 / 20 = 2,  flat = 58 % 20 = 18
        //      multi[1] = 18 / 5  = 3,  flat = 18 % 5  = 3
        //      multi[2] =  3 / 1  = 3,  flat =  3 % 1  = 0
        //      multi = [2, 3, 3]
        // @doc: static auto FlatToMulti(size_t flat) -> std::array<size_t, Rank>
        /** . */
        static auto FlatToMulti(size_t flat) -> std::array<size_t, Rank> {
            assert(flat < Size && "flat index out of bounds");
            std::array<size_t, Rank> multi{};
            for (size_t d = 0; d < Rank; d++) {
                multi[d] = flat / Strides[d];
                flat %= Strides[d];
            }
            return multi;
        }

        // MultiToFlat: dot a multi-index against Strides to get a flat index
        // just a dot product between multi index and strides!
        // EXAMPLE: Tensor<3, 4, 5>
        //      Rank = 3, Size = 60, Strides = [20, 5, 1]
        //
        // suppose multi = [2, 3, 3] (second to last item, should be 58)
        //      flat = 20*2 + 5*3 + 1*3 = 40 + 15 + 3 = 58
        // @doc: static size_t MultiToFlat(const std::array<size_t, Rank>& multi)
        /** . */
        static size_t MultiToFlat(const std::array<size_t, Rank> &multi) {
            size_t flat = 0;
            for (size_t d = 0; d < Rank; d++) {
                flat += multi[d] * Strides[d];
            }
            assert(flat < Size && "computed flat index out of bounds");
            return flat;
        }

    private:
        std::unique_ptr<float[]> data_;

    public:
        // Default: heap-allocate and zero-initialize.
        Tensor() : data_(std::make_unique<float[]>(Size)) {
        }

        // Initializer list: heap-allocate, fill from list (remaining elements stay zero).
        Tensor(std::initializer_list<float> init) : data_(std::make_unique<float[]>(Size)) {
            size_t i = 0;
            for (auto v: init)
                if (i < Size) data_[i++] = v;
        }

        // Rule of Five ─────────────────────────────────────────────────────────

        // Destructor: unique_ptr<float[]> calls delete[] automatically.
        ~Tensor() = default;

        // Copy ctor: deep copy — allocate a new buffer and memcpy the data.
        Tensor(const Tensor &other) : data_(std::make_unique<float[]>(Size)) {
            for (size_t i = 0; i < Size; ++i) data_[i] = other.data_[i];
        }

        // Copy assignment: destination already owns a live buffer, just overwrite it.
        Tensor &operator=(const Tensor &other) {
            if (this != &other)
                for (size_t i = 0; i < Size; ++i) data_[i] = other.data_[i];
            return *this;
        }

        // Move ctor / move assignment: unique_ptr transfers ownership for free.
        Tensor(Tensor &&) noexcept = default;

        Tensor &operator=(Tensor &&) noexcept = default;

        // ──────────────────────────────────────────────────────────────────────

        // fill with a value
        void fill(float v) { for (size_t i = 0; i < Size; ++i) data_[i] = v; }

        // get underlying array
        float *data() { return data_.get(); }
        [[nodiscard]] const float *data() const { return data_.get(); }

        // flat indexing
        float &flat(size_t idx) { return data_[idx]; }
        [[nodiscard]] float flat(size_t idx) const { return data_[idx]; }

        // map: apply f(float) -> float element-wise, return new tensor
        template<typename F>
        Tensor map(F f) const {
            Tensor out;
            std::transform(std::execution::par_unseq,
                           data_.get(), data_.get() + Size, out.data_.get(), f);
            return out;
        }

        // zip: apply f(float, float) -> float element-wise with another tensor, return new tensor
        template<typename F>
        Tensor zip(const Tensor &other, F f) const {
            Tensor out;
            std::transform(std::execution::par_unseq,
                           data_.get(), data_.get() + Size, other.data_.get(), out.data_.get(), f);
            return out;
        }

        // apply: mutate each element in-place with f(float&)
        template<typename F>
        void apply(F f) {
            std::for_each(std::execution::par_unseq, data_.get(), data_.get() + Size, f);
        }

        // zip_apply: mutate each element in-place using corresponding element from other with f(float&, float)
        template<typename F>
        void zip_apply(const Tensor &other, F f) {
            std::transform(std::execution::par_unseq,
                           data_.get(), data_.get() + Size, other.data_.get(), data_.get(),
                           [&f](float a, float b) -> float { f(a, b); return a; });
        }

#define ACCESS_IMPL {                                                                       \
        static_assert(sizeof...(idxs) == Rank,"Number of indices must match tensor rank");  \
        const size_t idx_arr[] = {static_cast<size_t>(idxs)...};                            \
        size_t flat_index = 0;                                                              \
        for (size_t i = 0; i < Rank; i++)                                                   \
            flat_index += idx_arr[i] * Strides[i];                                          \
        return data_[flat_index];                                                           \
    }

        template<typename... Indices>
        float &operator()(Indices... idxs) ACCESS_IMPL

        template<typename... Indices>
        float operator()(Indices... idxs) const ACCESS_IMPL
#undef ACCESS_IMPL

        // array multi-index overload — runtime values, no compile-time bounds check
        float &operator()(const std::array<size_t, Rank> &multi) {
            return data_[MultiToFlat(multi)];
        }

        float operator()(const std::array<size_t, Rank> &multi) const {
            return data_[MultiToFlat(multi)];
        }

        void Save(std::ofstream &f) const {
            f.write(reinterpret_cast<const char *>(data_.get()), Size * sizeof(float));
        }

        void Load(std::ifstream &f) {
            f.read(reinterpret_cast<char *>(data_.get()), Size * sizeof(float));
        }
    };

    // IS_TENSOR type trait -> concept
    // allows us to make sure Block parameters are Tensors
    // dummy SFINAE backup
    template<typename T>
    struct is_tensor : std::false_type {
    };

    // substitution success: able to pattern match the T in is_tensor<T> to a Tensor<any dims at all> (ie any Tensor)
    template<size_t... Dims>
    struct is_tensor<Tensor<Dims...> > : std::true_type {
    };

    // concept to wrap it
    template<typename T>
    concept IsTensor = is_tensor<T>::value;
}
