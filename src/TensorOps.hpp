#pragma once
#include <random>
#include <concepts>
#include <numeric>
#include <ranges>
#include "Tensor.hpp"

namespace TTTN {
    static constexpr float EPS         = 1e-8f;
    static constexpr float ADAM_BETA_1  = 0.9f;
    static constexpr float ADAM_BETA_2  = 0.999f;
    // EINSUM: generalized tensor contraction
    // contract index I from Tensor A and index J from Tensor B
    // Example:
    //      - multiplication of two Rank-0 Tensors (two floats) : no contraction, no dimensions, just multiply
    //      - dot product of two Rank-1 Tensors: contract non-1 dimension of each (user must specify...
    //          for two row vectors it would be einsum<1,1>)
    //      - outer product of two Rank-1 Tensors: contract the 1-dimensions of each (for two row vectors it would be einsum<0,0>)
    //      - matmul two Rank-2 Tensors: typically einsum<1,0>


    //  Helpers:

    // ConcatTensors: concatenate two tensor shapes into one
    template<typename T1, typename T2>
    struct TensorConcat;

    // since generic takes two types, we separate the two variadic dimension lists and match them (only!) to two tensors of
    // those sizes.
    template<size_t... Ds1, size_t... Ds2>
    struct TensorConcat<Tensor<Ds1...>, Tensor<Ds2...> > {
        using type = Tensor<Ds1..., Ds2...>;
    };

    // thin wrapper to present as new Tensor type
    // so when we need to remove axis and suture, we'll get the type this way


    // now need beautiful helper: constexpr array --> Tensor
    template<typename KeptIdxs, typename Iota>
    struct ArrayToTensor;

    template<typename KeptIdxs, size_t... Iota>
    struct ArrayToTensor<KeptIdxs, std::index_sequence<Iota...> > {
        static constexpr auto arr = KeptIdxs::value;
        using type = Tensor<arr[Iota]...>;
    };


    // Build a Tensor type from selected indices of a Shape array
    // Given a pack Dims... and an axis to skip, produce Tensor<remaining dims...>
    // wrap kept dimensions array in a type so it can be passed to the above as template!
    template<size_t Skip, size_t... Dims>
    struct KeptDimsHolder {
        static constexpr auto value = [] {
            constexpr std::array<size_t, sizeof...(Dims)> all = {Dims...};
            std::array<size_t, sizeof...(Dims) - 1> result{};
            size_t out = 0;
            for (size_t i = 0; i < sizeof...(Dims); ++i) {
                if (i != Skip) {
                    result[out++] = all[i];
                }
            }
            return result;
        }();
    };


    // Finally; RemoveAxis. this returns the new Tensor type
    template<size_t Skip, size_t... Dims>
    struct RemoveAxis {
        using type = typename ArrayToTensor<
            // kept indices (actual values)
            KeptDimsHolder<Skip, Dims...>,
            // iota of the above to pattern-match and grab them into new Tensor type!
            std::make_index_sequence<sizeof...(Dims) - 1>
        >::type;
    };


    // now ready for Einsum

    template<size_t I, size_t J, typename TA, typename TB>
    struct EinsumResultType;

    template<size_t I, size_t J, size_t... ADims, size_t... BDims>
    struct EinsumResultType<I, J, Tensor<ADims...>, Tensor<BDims...> > {
        static_assert(
            SizeTemplateGet<I, ADims...>::value == SizeTemplateGet<J, BDims...>::value,
            "axis I from A and axis J from B must be same size!"
        );

        using A_Reduced = typename RemoveAxis<I, ADims...>::type;
        using B_Reduced = typename RemoveAxis<J, BDims...>::type;
        using type = typename TensorConcat<A_Reduced, B_Reduced>::type;
    };


    // EINSUM FUNCTION
    // Einsum<I, J>(A, B) contracts axis I from A with axis J from B
    //
    // we iterate over free indices in EinsumResultType, and sum over contracted indices

    template<size_t I, size_t J, size_t... ADims, size_t... BDims>
    auto Einsum(const Tensor<ADims...> &A, const Tensor<BDims...> &B) {
        using Result = typename EinsumResultType<I, J, Tensor<ADims...>, Tensor<BDims...> >::type;

        constexpr size_t A_Rank = Tensor<ADims...>::Rank;
        constexpr size_t B_Rank = Tensor<BDims...>::Rank;
        constexpr size_t ContractDimSize = SizeTemplateGet<I, ADims...>::value;

        constexpr auto A_Strides = Tensor<ADims...>::Strides;
        constexpr auto B_Strides = Tensor<BDims...>::Strides;

        Result C;
        C.fill(0.0f);

        constexpr size_t C_Rank = Result::Rank;
        constexpr auto C_Strides = Result::Strides;

        // loop over raw C
        //      to do this, we must decompose a C-array index into a multi C index
        //      map the first (A_Rank - 1) components to A's free indices
        //      map the next/last (B_Rank - 1) components to B's free indices
        //      loop over contracted dimension

#define GET_FREE_AXES(idx, tensor_rank) [] {        \
    std::array<size_t, tensor_rank - 1> result{};   \
    size_t out = 0;                                 \
    for (size_t i = 0; i < tensor_rank; i++) {      \
        if (i != idx) {                             \
            result[out++] = i;                      \
        }                                           \
    }                                               \
    return result;                                  \
}();

        constexpr auto a_free_axes = GET_FREE_AXES(I, A_Rank)
        constexpr auto b_free_axes = GET_FREE_AXES(J, B_Rank)
#undef GET_FREE_AXES


        // main loop
        for (size_t c_flat = 0; c_flat < Result::Size; c_flat++) {
            // decompose c_flat into multi-index
            std::array<size_t, C_Rank> c_multi_index{};
            // starting at the flat index
            size_t temp = c_flat;
            // for each dimension (whose strides bumped us to c_flat!)
            for (size_t dimension = 0; dimension < C_Rank; dimension++) {
                // reverse the multiplication it contributed, leaving the strided
                c_multi_index[dimension] = temp / C_Strides[dimension];
                temp %= C_Strides[dimension];
            }
            /*
             * Example: Tensor<3,3> x;
             * x(2, 1) = 3 * 2 + 1 = 7
             * ---the array is 0-8 inclusive, [2][1] is second to last item!
             *
             * Example: Tensor<3,3,3> x;
             * x(0, 1, 2) = 9 * 0 + 3 * 1 + 2 = 5
             * ---harder to picture array, but this is first row, second column, third matrix.
             * x(2, 2, 1) = 9 * 2 + 3 * 2 + 1 = 25
             * ---array is 0-26 inclusive, [2][2][1] is second to last item!
             */

            // build A's and B's base flat indices (which will get us to the right zone to iterate over ContractDim)
            // a_base brings us to A[contracted_index][0]
            size_t a_base = 0;
            // for each free axis in A
            for (size_t freeA = 0; freeA < A_Rank - 1; freeA++) {
                // get that axis from C's multi-index and adjust with proper A stride
                // a_free_axes[] maps from free-space to A_Rank space
                a_base += c_multi_index[freeA] * A_Strides[a_free_axes[freeA]];
            }
            size_t b_base = 0;
            for (size_t freeB = 0; freeB < B_Rank - 1; freeB++) {
                // need to adjust for dimensions already taken by A
                b_base += c_multi_index[A_Rank - 1 + freeB] * B_Strides[b_free_axes[freeB]];
            }

            // CONTRACTION
            float sum = 0.0f;
            for (size_t k = 0; k < ContractDimSize; k++) {
                // go to base starting point, add dimension-adjusted stride for k-th item in that array
                const float a_val = A.flat(a_base + k * A_Strides[I]);
                const float b_val = B.flat(b_base + k * B_Strides[J]);
                sum += a_val * b_val;
            }

            C.flat(c_flat) = sum;
        }
        return C;
    }

    // outer product (no contraction) specialization
    template<size_t... ADims, size_t... BDims>
    auto Einsum(const Tensor<ADims...> &A, const Tensor<BDims...> &B) {
        Tensor<ADims..., BDims...> C;
        // for each value of A
        for (size_t i = 0; i < Tensor<ADims...>::Size; i++) {
            // for each value of B
            for (size_t j = 0; j < Tensor<BDims...>::Size; j++) {
                C.flat(i * Tensor<BDims...>::Size + j) = A.flat(i) * B.flat(j);
                /*
                 * Example: Tensor<3, 1> A; Tensor<1, 2> B;
                 * [1][1] --> 1 * 2 + 1 = 3
                 *
                 * essentially, every new dimension of A (in the array) is separated by a full subarray 'from B', scaled by A[i]
                 */
            }
        }
        return C;
    }


    template<size_t... Dims>
    Tensor<Dims...> operator+(const Tensor<Dims...> &a, const Tensor<Dims...> &b) {
        return a.zip(b, [](float x, float y) { return x + y; });
    }

    template<size_t... Dims>
    Tensor<Dims...> operator-(const Tensor<Dims...> &a, const Tensor<Dims...> &b) {
        return a.zip(b, [](float x, float y) { return x - y; });
    }

    template<size_t... Dims>
    Tensor<Dims...> &operator+=(Tensor<Dims...> &a, const Tensor<Dims...> &b) {
        a.zip_apply(b, [](float &x, float y) { x += y; });
        return a;
    }

    template<size_t... Dims>
    Tensor<Dims...> operator*(const Tensor<Dims...> &a, float s) {
        return a.map([s](float x) { return x * s; });
    }

    template<size_t... Dims>
    Tensor<Dims...> operator*(float s, const Tensor<Dims...> &a) { return a * s; }


    // CROSS ENTROPY LOSS
    template<size_t N>
    float CrossEntropyLoss(const Tensor<N> &output, const Tensor<N> &target) {
        auto indices = std::views::iota(size_t{0}, N);

        return std::accumulate(indices.begin(), indices.end(), 0.0f,
                               [&target, &output](float current_loss, size_t i) {
                                   return current_loss - target.flat(i) * std::log(std::max(output.flat(i), EPS));
                               }
        );
    }

    // xavier init for controlled init variance
    template<size_t In, size_t Out>
    void XavierInit(Tensor<Out, In> &W) {
        static std::mt19937 rng{std::random_device{}()};
        const float limit = std::sqrt(6.f / static_cast<float>(In + Out));
        std::uniform_real_distribution<float> dist{-limit, limit};
        W.apply([&dist](float &x) { x = dist(rng); });
    }


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
};
