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
    // @doc: struct TensorDimsProduct<size_t... Ds>
    /**
     * Template-specialization-based recursion to collapse variadic template `<size_t...Ds>` into single `size_t`, stored statically as `TensorDimsProduct<size_t...Ds>::value`
     * Used to compute `Tensor::Shape`, used in [`struct ComputeStrides<size_t... Ds>`](src/Tensor.hpp) to compute `Tensor::Strides`
     */
    template<size_t... Ds>
    struct TensorDimsProduct;

    template<>
    struct TensorDimsProduct<> {
        static constexpr size_t value = 1;
    };

    template<size_t First, size_t... Rest>
    struct TensorDimsProduct<First, Rest...> {
        static constexpr size_t value = First * TensorDimsProduct<Rest...>::value;
    };


    // @doc: struct SizeTemplateGet<size_t N, size_t... Ds>
    /**
     * Template-specialization-based recursion grab `N`-th `size_t` from `<size_t...Ds>`
     * Uses functional-style aggregation and pattern-matching to decrement `N` and peel off `size_t`s from variadic array until reaching basecase where `N = 0`
     * Used for clean, compile-time syntax in [TensorOps.hpp](src/TensorOps.hpp)
     */
    template<size_t N, size_t... Ds>
    struct SizeTemplateGet;

    template<size_t First, size_t... Rest>
    struct SizeTemplateGet<0, First, Rest...> {
        static constexpr size_t value = First;
    };

    template<size_t N, size_t First, size_t... Rest>
    struct SizeTemplateGet<N, First, Rest...> {
        // peel off, ditch first, until N = 0
        static constexpr size_t value = SizeTemplateGet<N - 1, Rest...>::value;
    };


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

    template<>
    struct ComputeStrides<> {
        static constexpr std::array<size_t, 0> value = {};
    };

    template<size_t D>
    struct ComputeStrides<D> {
        static constexpr std::array<size_t, 1> value = {1};
    };

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

    template<size_t... Dims>
    class Tensor {
    public:
        // @doc: static constexpr size_t Rank
        /** Number of dimensions */
        static constexpr size_t Rank = sizeof...(Dims);
        // @doc: static constexpr size_t Size
        /** Product of all dimensions = total distinct values in `Tensor` */
        static constexpr size_t Size = TensorDimsProduct<Dims...>::value;
        // @doc: static constexpr std::array<size_t, Rank> Shape
        /** `<size_t... Dims>` captured into an array */
        static constexpr std::array<size_t, Rank> Shape = {Dims...};
        // @doc: static constexpr std::array<size_t, Rank> Strides
        /**
         * Uses `ComputeStrides` to create array
         * The `Tensor::Strides` array is vital to mapping from indices into `Tensor::Shape` to flat indices for the backing array
         * In general, for a `Tensor` with `Tensor::Shape = [A, B, ..., N]`, its `Tensor::Strides = [A * B * ... * N, B * ... * N, ..., 1]`
         */
        static constexpr std::array<size_t, Rank> Strides = ComputeStrides<Dims...>::value;

        // @doc: static constexpr std::array<size_t, Rank> FlatToMulti(size_t flat)
        /**
         * Inverse of `MultiToFlat`; map a flat index `[0, Size)` to its `Rank`-term index
         * Pattern matches a `std::index_sequence` parameterized by `<size_t... Rank>` (= `[0, ..., Rank]`) and unpacks into an array: `[(flat / Strides[0]) % Shape[0], ..., (flat / Strides[Rank]) % Shape[Rank]]`
         */
        static constexpr std::array<size_t, Rank> FlatToMulti(size_t flat) {
            return [&]<size_t... Is>(std::index_sequence<Is...>) {
                return std::array<size_t, Rank>{(flat / Strides[Is]) % Shape[Is]...};
            }(std::make_index_sequence<Rank>{});
        }


        // @doc: static constexpr size_t MultiToFlat(const std::array<size_t, Rank>& multi)
        /**
         * Inverse of `FlatToMulti`; map a `Rank`-term index to its flat index `[0, Size)`
         * Pattern matches a `std::index_sequence` parameterized by `<size_t... Rank>` (= `[0, ..., Rank]`) and `+`-folds terms `multi[0] * Strides[0] + ... + multi[Rank] * Strides[Rank]`
         * Dot product of given `multi` index and `Strides`
         */
        static constexpr size_t MultiToFlat(const std::array<size_t, Rank> &multi) {
            return [&]<size_t... Is>(std::index_sequence<Is...>) {
                return (... + (multi[Is] * Strides[Is]));
            }(std::make_index_sequence<Rank>{});
        }

    private:
        std::unique_ptr<float[]> data_;

    public:
        // Default: heap-allocate and zero-initialize.
        // @doc: Tensor()
        /**
         * Default constructor
         * Initialize `float[Size]` on the heap
         */
        Tensor() : data_(std::make_unique<float[]>(Size)) {
        }

        // Initializer list: heap-allocate, fill from list (remaining elements stay zero).
        // @doc: Tensor(std::initializer_list<float> init)
        /**
         * Initializer list constructor
         * Fill the first `Size` elements of `std::initializer_list<float> init` to flat indices of backing array
         */
        Tensor(std::initializer_list<float> init) : data_(std::make_unique<float[]>(Size)) {
            size_t i = 0;
            for (const auto v: init) {
                if (i < Size) {
                    data_[i++] = v;
                }
            }
        }

        // @doc: ~Tensor() = default
        /**
         * Default destructor
         * RAII: destructs `std::unique_ptr` to `float[Size]` on heap
         */
        ~Tensor() = default;

        // @doc: Tensor(const Tensor& other)
        /**
         * Deep copy constructor
         * Allocate new `float[Size]` on heap and `std::memcpy` from `other.data()`
         */
        Tensor(const Tensor &other) : data_(std::make_unique<float[]>(Size)) {
            std::memcpy(data_.get(), other.data_.get(), sizeof(float) * Size);
        }

        // @doc: Tensor& operator=(const Tensor& other)
        /**
         * Deep copy assignment operator
         * `std::memcpy` from `other.data()`
         */
        Tensor &operator=(const Tensor &other) {
            if (this != &other) {
                std::memcpy(data_.get(), other.data_.get(), sizeof(float) * Size);
            }
            return *this;
        }

        // @doc: Tensor(Tensor&&) noexcept = default
        /**
         * Default move constructor
         * `std::unique_ptr` to data handles this already
         */
        Tensor(Tensor &&) noexcept = default;

        // @doc: Tensor &operator=(Tensor&&) noexcept = default
        /**
         * Default move assigment operator
         * `std::unique_ptr` to data handles this already
         */
        Tensor &operator=(Tensor &&) noexcept = default;

        // @doc: void fill(const float v) const
        /** Fill a `Tensor`'s underlying array with some `float v` */
        void fill(const float v) const { std::fill_n(data_.get(), Size, v); }

        // @doc: const float* data() const
        /** Returns `const` pointer to `Tensor`'s underlying array */
        float *data() { return data_.get(); }
        [[nodiscard]] const float *data() const { return data_.get(); }

        // @doc: float flat(size_t idx) const
        // @doc: float& flat(size_t idx)
        /** Returns `float&` reference to item at `idx` in underlying array */
        float &flat(size_t idx) { return data_[idx]; }
        [[nodiscard]] float flat(size_t idx) const { return data_[idx]; }

        // @doc: operator float() const
        /** Implicit scalar conversion — only valid for `Tensor<>` (rank-0, `Rank == 0`). Allows `Contract(A, B)` results to be used directly as `float` without calling `.flat(0)`. */
        operator float() const requires (Rank == 0) { return data_[0]; }

        // @doc: template<typename F> Tensor map(F f) const
        /** Use `std::execution::par_unseq` to `std::transform` `Tensor`'s underlying data by `float -> float` map `f`, returning a new `Tensor` */
        template<typename F>
        Tensor map(F f) const {
            Tensor out;
            std::transform(std::execution::par_unseq,
                           data_.get(), data_.get() + Size, out.data_.get(), f);
            return out;
        }

        // @doc: template<typename F> Tensor zip(const Tensor& other, F f) const
        /** Use `std::execution::par_unseq` to `std::transform` two `Tensor`s' underlying data by `float -> float -> float` map `f`, returning a new `Tensor` */
        template<typename F>
        Tensor zip(const Tensor &other, F f) const {
            Tensor out;
            std::transform(std::execution::par_unseq,
                           data_.get(), data_.get() + Size, other.data_.get(), out.data_.get(), f);
            return out;
        }

        // @doc: template<typename F> void apply(F f)
        /** Use `std::execution::par_unseq` + `std::for_each` to apply `float -> float` map `f` to `Tensor`'s underlying data in-place */
        template<typename F>
        void apply(F f) {
            std::for_each(std::execution::par_unseq, data_.get(), data_.get() + Size, f);
        }

        // @doc: template<typename F> void zip_apply(const Tensor& other, F f)
        /** Use `std::execution::par_unseq` to `std::transform` two `Tensor`s' underlying data in-place by `float -> float -> float` map `f` */
        template<typename F>
        void zip_apply(const Tensor &other, F f) {
            std::transform(std::execution::par_unseq,
                           data_.get(), data_.get() + Size, other.data_.get(), data_.get(),
                           [&f](float a, float b) -> float {
                               f(a, b);
                               return a;
                           });
        }

#define ACCESS_IMPL {                                                                       \
                static_assert(sizeof...(idxs) == Rank, "Number of indices must match tensor rank");  \
                const size_t idx_arr[] = {static_cast<size_t>(idxs)...};                            \
                size_t flat_index = 0;                                                              \
                for (size_t i = 0; i < Rank; i++)                                                   \
                    flat_index += idx_arr[i] * Strides[i];                                          \
                return data_[flat_index];                                                           \
            }

        // @doc: float& operator()(Indices... idxs)
        /**
         * Variadic multi-index access, returns reference
         * Uses compile-time-templated `MultiToFlat` for efficient access
         */
        template<typename... Indices>
        float &operator()(Indices... idxs) ACCESS_IMPL

        // @doc: float operator()(Indices... idxs) const
        /**
         * Variadic multi-index access, returns copy
         * Uses compile-time-templated `MultiToFlat` for efficient access
         */
        template<typename... Indices>
        float operator()(Indices... idxs) const ACCESS_IMPL
#undef ACCESS_IMPL

        // @doc: float operator()(const std::array<size_t, Rank>& multi) const
        /**
         * Array-based multi-index access, returns copy
         * Uses compile-time-templated `MultiToFlat` for efficient access
         */
        // @doc: float& operator()(const std::array<size_t, Rank>& multi)
        /**
         * Array-based multi-index access, returns reference
         * Uses compile-time-templated `MultiToFlat` for efficient access
         */
        float &operator()(const std::array<size_t, Rank> &multi) {
            return data_[MultiToFlat(multi)];
        }

        float operator()(const std::array<size_t, Rank> &multi) const {
            return data_[MultiToFlat(multi)];
        }

        // @doc: void Save(std::ofstream& f) const
        /** Writes entirety of flat backing array (`float[Size]`) to binary file */
        void Save(std::ofstream &f) const {
            f.write(reinterpret_cast<const char *>(data_.get()), Size * sizeof(float));
        }

        // @doc: void Load(std::ifstream& f)
        /** Reads binary file into flat backing array (`float[Size]`) */
        void Load(std::ifstream &f) {
            f.read(reinterpret_cast<char *>(data_.get()), Size * sizeof(float));
        }
    };

    // @doc: struct is_tensor<T>
    /**
     * SFINAE type traits for verifying that a type is a `Tensor`
     *   - Specialize `<size_t...Dims>`: matches into `<Tensor<Dims...>`, inherits from `std::true_type`
     *   - Specialize <>: inherits from `std::false_type`, backup when former fails
     */
    template<typename T>
    struct is_tensor : std::false_type {
    };

    template<size_t... Dims>
    struct is_tensor<Tensor<Dims...> > : std::true_type {
    };

    // @doc: concept IsTensor<T>
    /** Wrapper `concept` around `is_tensor` type trait, satisfied if `T` is a `Tensor` */
    template<typename T>
    concept IsTensor = is_tensor<T>::value;
}
