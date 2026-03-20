#pragma once
#include <array>
#include <fstream>
#include <iostream>
#include <numeric>
#include <type_traits>

namespace TTTN {
    // TENSOR DIMENSIONS PRODUCT
    // templatized dimension list --> product (for Tensor underlying array size)
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

    private:
        std::array<float, Size> data_{};

    public:
        Tensor() = default;

        // construct from initializer list
        Tensor(std::initializer_list<float> init) {
            size_t i = 0;
            for (auto v: init) {
                if (i < Size) {
                    data_[i++] = v;
                }
            }
        }

        // fill with a value
        void fill(float v) { data_.fill(v); }

        // get underlying array
        float *data() { return data_.data(); }
        [[nodiscard]] const float *data() const { return data_.data(); }

        // flat indexing
        float &flat(size_t idx) { return data_[idx]; }
        [[nodiscard]] float flat(size_t idx) const { return data_[idx]; }

        // map: apply f(float) -> float element-wise, return new tensor
        template<typename F>
        Tensor map(F f) const {
            Tensor out;
            for (size_t i = 0; i < Size; ++i) {
                out.data_[i] = f(data_[i]);
            }
            return out;
        }

        // zip: apply f(float, float) -> float element-wise with another tensor, return new tensor
        template<typename F>
        Tensor zip(const Tensor &other, F f) const {
            Tensor out;
            for (size_t i = 0; i < Size; ++i) {
                out.data_[i] = f(data_[i], other.data_[i]);
            }
            return out;
        }

        // apply: mutate each element in-place with f(float&)
        template<typename F>
        void apply(F f) {
            for (size_t i = 0; i < Size; ++i) {
                f(data_[i]);
            }
        }

        // zip_apply: mutate each element in-place using corresponding element from other with f(float&, float)
        template<typename F>
        void zip_apply(const Tensor &other, F f) {
            for (size_t i = 0; i < Size; ++i) {
                f(data_[i], other.data_[i]);
            }
        }

#define ACCESS_IMPL {                                                                       \
        static_assert(sizeof...(idxs) == Rank,"Number of indices must match tensor rank");  \
        const size_t idx_arr[] = {static_cast<size_t>(idxs)...};                            \
        /* flat index is simply a dot product between requested indices and strides ! */    \
        size_t flat_index = 0;                                                              \
        for (size_t i = 0; i < Rank; i++) {                                                 \
            flat_index += idx_arr[i] * Strides[i];                                          \
        }                                                                                   \
        return data_[flat_index];                                                           \
    }
        // proper dimensional indexing
        template<typename... Indices>
        float &operator()(Indices... idxs) ACCESS_IMPL

        template<typename... Indices>
        float operator()(Indices... idxs) const ACCESS_IMPL
#undef ACCESS_IMPL

        // these functions are extremely easy because the Tensor type itself (prereq to calling the function)
        // already has all the metadata
        void Save(std::ofstream &f) const {
            f.write(reinterpret_cast<char *>(data_.data()), Size * sizeof(float));
        }

        void Load(std::ifstream &f) {
            f.read(reinterpret_cast<char *>(data_.data()), Size * sizeof(float));
        }
    };
}
