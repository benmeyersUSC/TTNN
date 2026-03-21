#pragma once
#include <ranges>
#include "Tensor.hpp"


// EINSUM: generalized tensor contraction
// contract index I from Tensor A and index J from Tensor B
// Example:
//      - multiplication of two Rank-0 Tensors (two floats) : no contraction, no dimensions, just multiply
//      - dot product of two Rank-1 Tensors: contract non-1 dimension of each (user must specify...
//          for two row vectors it would be einsum<1,1>)
//      - outer product of two Rank-1 Tensors: contract the 1-dimensions of each (for two row vectors it would be einsum<0,0>)
//      - matmul two Rank-2 Tensors: typically einsum<1,0>

namespace TTTN {
    static constexpr float EPS = 1e-8f;
    static constexpr float ADAM_BETA_1 = 0.9f;
    static constexpr float ADAM_BETA_2 = 0.999f;

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
        using type = ArrayToTensor<
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

        using A_Reduced = RemoveAxis<I, ADims...>::type;
        using B_Reduced = RemoveAxis<J, BDims...>::type;
        using type = TensorConcat<A_Reduced, B_Reduced>::type;
    };


    // EINSUM FUNCTION
    // Einsum<I, J>(A, B) contracts axis I from A with axis J from B
    //
    // we iterate over free indices in EinsumResultType, and sum over contracted indices

    template<size_t I, size_t J, size_t... ADims, size_t... BDims>
    auto Einsum(const Tensor<ADims...> &A, const Tensor<BDims...> &B) {
        using Result = EinsumResultType<I, J, Tensor<ADims...>, Tensor<BDims...> >::type;

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
std::array<size_t, tensor_rank - 1> result{};       \
size_t out = 0;                                     \
for (size_t i = 0; i < tensor_rank; i++) {          \
if (i != idx) {                                     \
result[out++] = i;                                  \
}                                                   \
}                                                   \
return result;                                      \
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
        return a.map([s](const float x) { return x * s; });
    }

    template<size_t... Dims>
    Tensor<Dims...> operator*(float s, const Tensor<Dims...> &a) { return a * s; }


    template<typename T, size_t... Perm>
    struct PermutedTensorType;

    template<size_t... Dims, size_t... Perm>
    struct PermutedTensorType<Tensor<Dims...>, Perm...> {
        static_assert(sizeof...(Dims) == sizeof...(Perm), "Permutation specification must be same size as Tensor dims");
        static constexpr std::array<size_t, sizeof...(Dims)> shape = {Dims...};
        using type = Tensor<shape[Perm]...>;
    };

    template<size_t... Perm, size_t... Dims>
    auto Permute(const Tensor<Dims...> &src) {
        using Result = PermutedTensorType<Tensor<Dims...>, Perm...>::type;
        static constexpr size_t Rank = sizeof...(Dims);
        // [multi-index] . [strides] = flat index
        static constexpr auto src_strides = Tensor<Dims...>::Strides;
        static constexpr auto dst_strides = Result::Strides;
        // perm_arr[d], for some dimension index d, is its new position
        // if og dimensions are <3, 4, 5> and we Permute<1, 2, 0>, then:
        //      d = 0: Dims[0] = 3, perm_arr[0] = 1...result will be <_, 3, _>
        //      d = 1: Dims[1] = 4, perm_arr[1] = 2...result will be <_, 3, 4>
        //      d = 2: Dims[2] = 5, perm_arr[2] = 0...result will be <5, 3, 4>
        // the d-th dimension in src is the perm_arr[d]-th dimension in dst
        static constexpr std::array<size_t, Rank> perm_arr = {Perm...};


        // continuing example src = Tensor<3, 4, 5>, Rank = 3, Size = 60
        // src_strides = [20, 5, 1]...Tensor[2][3][3] = 20*2 + 5*3 + 3 = 58 = 2nd to last item!
        // for Permute<1, 2, 0>, dst_strides = [12, 4, 1]
        Result dst;
        for (size_t i = 0; i < Result::Size; i++) {
            // decompose into dst multi-index
            std::array<size_t, Rank> dst_idx{}; // [_, _, _]
            size_t temp = i;
            for (size_t d = 0; d < Rank; d++) {
                // standard algo: each dimension needs to factor out its own stride-mult from the flat idx
                dst_idx[d] = temp / dst_strides[d];
                // then update temp to factor out that dimension's stride-mult
                temp %= dst_strides[d];
            }
            // say i = 33
            // temp = 33
            //      dst_index = [33 / 12 = 2, _, _]
            //      temp = 33 % 12 = 9
            // temp = 9
            //      dst_index = [2, 9 / 4 = 2, _]
            //      temp = 9 % 4 = 1
            // temp = 1
            //      dst_index = [2, 2, 1 / 1 = 1]
            //      temp = 1 % 1 = 0

            // for i = 33
            // dst_index = [2, 2, 1]
            // verify that: dst_index . dst_strides = 33
            // [2, 2, 1] . [12, 4, 1] = 2*12 + 2*4 + 1 = 24 + 8 + 1 = 33


            // map dst multi-index to src flat index
            size_t src_index = 0;
            for (size_t d = 0; d < Rank; d++) {
                // incr by dst-idx at this dimension times appropriate source stride-mult
                src_index += dst_idx[d] * src_strides[perm_arr[d]];
            }
            // i = 33 continued...

            // src_index = 0
            //      src_index = 0 + dst_idx[0] * src_strides[perm_arr[d]]
            //                = 2 * src_strides[1]
            //                = 2 * 5
            // src_index = 10
            //      src_index = 10 + 2 * src_strides[2]
            //                = 10 + 2 * 1
            //                = 10 + 2
            // src_index = 12
            //      src_index = 12 + 1 * src_strides[0]
            //                = 12 + 1 * 20
            // src_index = 32
            //
            // what is the src multi-index for src.flat(32)?
            // [_, _, _] . [20, 5, 1] = _*20 + _*5 + _*1 = 32
            // [1, _, _] . [20, 5, 1] = 1*20 + _*5 + _*1 = 32
            //                        = _*5 + _*1 = 12
            // [1, 2, _] . [20, 5, 1] = 1*20 + 2*5 + _*1 = 32
            //                        = _*1 = 2
            // [1, 2, 2] . [20, 5, 1] = 1*20 + 2*5 + 2*1 = 32
            // src-idx for i = 33 is [1, 2, 2]
            //
            // recall that we permuted <1, 2, 0>
            // we know that for i = 33, dst_index is [2, 2, 1], which must be [src-idx[1], src-idx[2], src-idx[0]]...
            // [src-idx[1], src-idx[2], src-idx[0]] = [2, 2, 1]


            dst.flat(i) = src.flat(src_index);
        }
        return dst;
    }

    // Reverse all dimensions by calling Permute<NumDims-1, NumDims-2, ..., 0>
    template<size_t... Dims>
    auto Transpose(const Tensor<Dims...> &t) {
        auto f = []<size_t... Is>(const Tensor<Is...> &s, std::index_sequence<Is...>) {
            // this iterates over Is...

            // so if Dims are <3, 5, 4> (sizeof = 3) we will have:
            //      Permute<(3-1) - 0 = 2, 2 - 1 = 1, 0>
            return Permute<(sizeof...(Dims) - 1 - Is)...>(s);
        };
        return f(t, std::make_index_sequence<sizeof...(Dims)>{});
    }
};
