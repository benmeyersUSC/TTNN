#pragma once
#include <cassert>
#include <limits>
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
    // ParForEach: run f(i) for i in [0, n) in parallel.
    // Thin wrapper around std::for_each(par_unseq, iota) — the canonical parallel loop pattern.
    // @doc: template<std::invocable<size_t> F> void ParForEach(size_t n, F f)
    /** Helper to parallel-execute `std::for_each` on a `std::views::iota(size_t{0}, n)`, calling `f` (something `std::invocable` on `size_t`) on each index */
    template<std::invocable<size_t> F>
    void ParForEach(size_t n, F f) {
        auto range = std::views::iota(size_t{0}, n);
        std::for_each(std::execution::par_unseq, range.begin(), range.end(), f);
    }

    //  Helpers:

    // ConcatTensors: concatenate two tensor shapes into one
    // @doc: struct TensorConcat<typename T1, typename T2>
    /**
     * Templated utility struct for concatenating the dimensions of two `Tensor` objects
     *   - Templated for two types, `<typename T1, typename T2>`
     *   - Specialized for two `Tensor`s, `<size_t... Ds1, size_t... Ds2>` and pattern-matched to `<Tensor<Ds1...>, Tensor<Ds2...>>`, creating `type = Tensor<Ds1..., Ds2...>`
     */
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
    // @doc: struct ArrayToTensor<typename KeptIdxs, typename Iota>
    /**
     * Unpack `KeptDimsHolder::value` into new `Tensor` type defined by those kept dimensions
     * Beautiful syntax: `type = Tensor<arr[Iota]...>`, where `arr = KeptDimsHolder::value` and `Iota...` represents the `[0, arr.size())` indices
     */
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
    // @doc: struct KeptDimsHolder<size_t Skip, size_t... Dims>
    /**
     * Given a pack `Dims...` and an axis to `Skip`, produce `Tensor<remaining dims...>`
     * `static constexpr value` holds the new array of `sizeof...(Dims) - 1` dimensions
     */
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
    // @doc: struct RemoveAxis<size_t Skip, size_t... Dims>
    /**
     * Compact operator to make new `Tensor` type by removing `Skip` dimension from given `Tensor`
     * `type = ArrayToTensor<KeptDimsHolder<Skip, Dims...>, std::make_index_sequence<sizeof...(Dims) - 1>`
     */
    template<size_t Skip, size_t... Dims>
    struct RemoveAxis {
        using type = ArrayToTensor<
            // kept indices (actual values)
            KeptDimsHolder<Skip, Dims...>,
            // iota of the above to pattern-match and grab them into new Tensor type!
            std::make_index_sequence<sizeof...(Dims) - 1>
        >::type;
    };


    // @doc: template<size_t... Dims> Tensor<Dims...> operator+(const Tensor<Dims...>& a, const Tensor<Dims...>& b)
    /** Element-wise add, uses parallel functional `zip` */
    template<size_t... Dims>
    Tensor<Dims...> operator+(const Tensor<Dims...> &a, const Tensor<Dims...> &b) {
        return a.zip(b, [](float x, float y) { return x + y; });
    }

    // @doc: template<size_t... Dims> Tensor<Dims...> operator-(const Tensor<Dims...>& a, const Tensor<Dims...>& b)
    /** Element-wise subtract, uses parallel functional `zip` */
    template<size_t... Dims>
    Tensor<Dims...> operator-(const Tensor<Dims...> &a, const Tensor<Dims...> &b) {
        return a.zip(b, [](float x, float y) { return x - y; });
    }

    template<size_t... Dims>
    Tensor<Dims...> &operator+=(Tensor<Dims...> &a, const Tensor<Dims...> &b) {
        a.zip_apply(b, [](float &x, float y) { x += y; });
        return a;
    }

    // @doc: template<size_t... Dims> Tensor<Dims...> operator*(const Tensor<Dims...>& a, float s)
    /** Scalar multiply, uses parallel functional `map` */
    template<size_t... Dims>
    Tensor<Dims...> operator*(const Tensor<Dims...> &a, float s) {
        return a.map([s](const float x) { return x * s; });
    }

    // @doc: template<size_t... Dims> Tensor<Dims...> operator*(float s, const Tensor<Dims...>& a)
    /** Scalar multiply, uses parallel functional `map` */
    template<size_t... Dims>
    Tensor<Dims...> operator*(float s, const Tensor<Dims...> &a) { return a * s; }

    // Hadamard (element-wise) product of two same-shape tensors
    // @doc: template<size_t... Dims> Tensor<Dims...> operator*(const Tensor<Dims...>& a, const Tensor<Dims...>& b)
    /** Hadamard (element-wise) product, uses parallel functional `zip` */
    template<size_t... Dims>
    Tensor<Dims...> operator*(const Tensor<Dims...> &a, const Tensor<Dims...> &b) {
        return a.zip(b, [](const float x, const float y) { return x * y; });
    }


    // @doc: struct PermutedTensorType<typename T, size_t... Perm>
    /**
     * Type for a permuted `Tensor`
     * Arbitrary reorganization of a `Tensor`'s `Shape`
     *   - Templated to `<typename T, size_t... Perm>`
     *   - Specialized by `<size_t... Dims, size_t... Perm>` and matched to `<Tensor<Dims...>, Perm...>`
     *   - Unpack and reassign `Shape`'s indices according to `Perm`: `Tensor<Tensor<Dims...>::Shape[Perm]...>`
     * Example: `PermutedTensorType<Tensor<4, 5, 3>, 2, 0, 1>::type = Tensor<3, 4, 5>`
     */
    template<typename T, size_t... Perm>
    struct PermutedTensorType;

    template<size_t... Dims, size_t... Perm>
    struct PermutedTensorType<Tensor<Dims...>, Perm...> {
        static_assert(sizeof...(Dims) == sizeof...(Perm), "Permutation specification must be same size as Tensor dims");
        static constexpr std::array<size_t, sizeof...(Dims)> shape = {Dims...};
        using type = Tensor<shape[Perm]...>;
    };

    // @doc: template<size_t... Perm, size_t... Dims> auto Permute(const Tensor<Dims...>& src)
    /**
     * Parallelized arbitrary permutation of `Tensor`'s indices
     * Returns `PermutedTensorType`
     * Algorithm:
     *   - `using Source = Tensor<Dims...>`
     *   - `using Result = PermutedTensorType<Source, Perm>::type`
     *   - `std::array<size_t, Rank> perm_arr = {Perm...}`
     *   - `Result dst;`
     *   - For each (parallelized) individual index in `Result::Size`:
     *   - `auto dst_idx = Result::FlatToMulti(i)`
     *   - get `Result` multi-index
     *   - `std::array<size_t, Rank> src_multi = [perm_arr[Rank]...]`
     *   - get `Source` multi-index
     *   - `size_t src_index = Source::MultiToFlat(src_multi)`
     *   - get `Source` flat index
     *   - dst.flat(i) = src.flat(src_index)
     *   - Assign `Source` value at that index to `Result`
     */
    template<size_t... Perm, size_t... Dims>
    auto Permute(const Tensor<Dims...> &src) {
        using Source = Tensor<Dims...>;
        using Result = PermutedTensorType<Source, Perm...>::type;
        static constexpr size_t Rank = sizeof...(Dims);
        // [multi-index] . [strides] = flat index
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
        ParForEach(Result::Size, [&](size_t i) {
            // decompose into dst multi-index
            auto dst_idx = Result::FlatToMulti(i);
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
            // invert the permutation: dst_idx[d] belongs at position perm_arr[d] in src space
            auto src_multi = [&]<size_t... Is>(std::index_sequence<Is...>) {
                std::array<size_t, Rank> m{};
                ((m[perm_arr[Is]] = dst_idx[Is]), ...);
                return m;
            }(std::make_index_sequence<Rank>{});
            size_t src_index = Source::MultiToFlat(src_multi);
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
        });
        return dst;
    }


    // TENSOR SLICE
    // Select a contiguous range [Start, Start+Len) of dims, producing a new Tensor type.
    // Peer to RemoveAxis — same ArrayToTensor machinery, different selection predicate.
    // @doc: struct SliceDimsHolder<size_t Start, size_t Len, size_t... Dims>
    /**
     * Helper struct to hold `std::array<size_t, Len> value` representing contiguous dimensions `[Start, Start+Len)` of a `Tensor<Dims...>`
     * Will not compile if `Start + Len > Rank`
     */
    template<size_t Start, size_t Len, size_t... Dims>
    struct SliceDimsHolder {
        static constexpr auto value = [] {
            constexpr std::array<size_t, sizeof...(Dims)> all = {Dims...};
            std::array<size_t, Len> result{};
            for (size_t i = 0; i < Len; ++i) {
                result[i] = all[Start + i];
            }
            return result;
        }();
    };

    // @doc: struct TensorSlice<size_t Start, size_t Len, size_t... Dims>
    /** Using `ArrayToTensor` and `SliceDimsHolder`, create new `Tensor` object out of a set of dimensions, `[Start, Start+Len)`, from original `Tensor<Dims...>` */
    template<size_t Start, size_t Len, size_t... Dims>
    struct TensorSlice {
        using type = ArrayToTensor<
            SliceDimsHolder<Start, Len, Dims...>,
            std::make_index_sequence<Len>
        >::type;
    };


    // SIGMA PI (Σ Π)
    // SigmaPi<N>(A, B): contract the last N axes of A with the first N axes of B.
    // Name: Σ over contracted indices of Π(A element, B element) — the sum-of-products that
    // is the kernel of every tensor contraction, generalised to any rank and N axes at once.
    // The contracted axes must have matching sizes.
    // Result shape: Tensor< (A's first RankA-N dims)..., (B's last RankB-N dims)... >
    //
    // Natural Dense convention: W = Tensor<OutDims..., InDims...>, x = Tensor<InDims...>
    //      SigmaPi<sizeof...(InDims)>(W, x)  -->  Tensor<OutDims...>
    //
    // Special cases:
    //      N=0           : outer product
    //      N=RankA=RankB : full contraction (scalar result, Tensor<>)

    // SigmaPiKernel: struct templated on (N, ADims..., BDims...) that holds all four
    // compile-time index tables as static constexpr members.  One instantiation per
    // unique shape combination — tables are evaluated once (before main) and reused
    // for every ΣΠ call on those shapes across all epochs.
    //
    // a_offsets[c] / b_offsets[c]:  A's and B's flat-index contribution from contracted index c.
    // a_bases[out_i] / b_bases[out_i]: base flat offset in A and B from the free dimensions.
    //
    // Hot inner loop then becomes two array lookups + multiply-accumulate — no FlatToMulti,
    // no multi-index construction at runtime.
    template<size_t N, typename TA, typename TB>
    struct SigmaPiKernel;

    // @doc: template<size_t N, typename TA, typename TB> struct SigmaPiKernel
    /**
     * Struct templated on `<size_t N, size_t... ADims, size_t... BDims>`, matched to `<N, Tensor<ADims...>, Tensor<BDims...>>`
     * Compile-time `static constexpr`:
     *   - `RankA = sizeof...(ADims)`
     *   - `RankB = sizeof...(BDims)`
     *   - Asserts that `N <= RankA && N <= RankB`
     *   - Asserts that last `N` dimensions of `ADims...` are each equal in size to the first `N` dimensions of `BDims...`
     *   - Free dimensions of `A` and `B`
     *   - `A_Free = TensorSlice<0, RankA - N, ADims...>::type`
     *   - `B_Free = TensorSlice<N, RankB - N, BDims...>::type`
     *   - Types for contracted and resulting `Tensor`s
     *   - `Contracted = TensorSlice<RankA - N, N, ADims...>::type`
     *   - `ResultType = TensorConcat<A_Free, B_Free>::type`
     *   - Index tables
     *   - `struct { std::array<size_t, Contracted::Size> a, b; } offsets;`
     *   - Precomputes flat-index contribution of every contracted position for A and B
     *   - `struct { std::array<size_t, ResultType::Size> a, b; } bases;`
     *   - Precomputes free-dimension base offset in A and B for every output index
     *   - **These pay real dividends for [TrainableTensorNetwork](./src/TrainableTensorNetwork.hpp) training schedules. Any weight `Tensor`'s `Dot`s, `Matmul`s, and `Outer`s (*in forward and backward passes*) are saved structs, and the runtime computations are parallelized and vectorized, following known, saved paths**
     */
    template<size_t N, size_t... ADims, size_t... BDims>
    struct SigmaPiKernel<N, Tensor<ADims...>, Tensor<BDims...> > {
        static constexpr size_t RankA = sizeof...(ADims);
        static constexpr size_t RankB = sizeof...(BDims);
        static_assert(N <= RankA && N <= RankB);
        static_assert([] {
            constexpr std::array<size_t, RankA> a = {ADims...};
            constexpr std::array<size_t, RankB> b = {BDims...};
            for (size_t i = 0; i < N; ++i)
                if (a[RankA - N + i] != b[i]) return false;
            return true;
        }(), "last N dims of A must match first N dims of B");

        using A_Free = TensorSlice<0, RankA - N, ADims...>::type;
        using B_Free = TensorSlice<N, RankB - N, BDims...>::type;
        using Contracted = TensorSlice<RankA - N, N, ADims...>::type;
        using ResultType = TensorConcat<A_Free, B_Free>::type;

        static constexpr size_t a_free_rank = A_Free::Rank;
        static constexpr size_t b_free_rank = B_Free::Rank;

        // contracted-index → flat offsets in A and B
        static constexpr auto offsets = [] {
            constexpr auto sa = ComputeStrides<ADims...>::value;
            constexpr auto sb = ComputeStrides<BDims...>::value;
            struct {
                std::array<size_t, Contracted::Size> a{}, b{};
            } t;
            for (size_t c = 0; c < Contracted::Size; ++c) {
                const auto cm = Contracted::FlatToMulti(c);
                size_t ao = 0, bo = 0;
                for (size_t i = 0; i < N; ++i) {
                    ao += cm[i] * sa[a_free_rank + i];
                    bo += cm[i] * sb[i];
                }
                t.a[c] = ao;
                t.b[c] = bo;
            }
            return t;
        }();

        // output-index → base flat offsets in A and B from free dims
        static constexpr auto bases = [] {
            constexpr auto sa = ComputeStrides<ADims...>::value;
            constexpr auto sb = ComputeStrides<BDims...>::value;
            struct {
                std::array<size_t, ResultType::Size> a{}, b{};
            } t;
            for (size_t o = 0; o < ResultType::Size; ++o) {
                const auto om = ResultType::FlatToMulti(o);
                size_t ab = 0, bb = 0;
                for (size_t i = 0; i < A_Free::Rank; ++i) ab += om[i] * sa[i];
                for (size_t i = 0; i < B_Free::Rank; ++i) bb += om[A_Free::Rank + i] * sb[N + i];
                t.a[o] = ab;
                t.b[o] = bb;
            }
            return t;
        }();

        static ResultType compute(const Tensor<ADims...> &A, const Tensor<BDims...> &B) {
            ResultType result;
            ParForEach(ResultType::Size, [&](size_t o) {
                result.flat(o) = std::transform_reduce(
                    std::execution::unseq,
                    offsets.a.begin(), offsets.a.end(),
                    offsets.b.begin(),
                    0.0f,
                    std::plus<>{},
                    [&](size_t oa, size_t ob) {
                        return A.flat(bases.a[o] + oa) * B.flat(bases.b[o] + ob);
                    });
            });
            return result;
        }
    };

    // @doc: template<size_t N, size_t... ADims, size_t... BDims> auto ΣΠ(const Tensor<ADims...>& A, const Tensor<BDims...>& B)
    /**
     * Wrapper function around `SigmaPiKernel::Compute`
     * returns `SigmaPiKernel<N, Tensor<ADims...>, Tensor<BDims...> >::compute(A, B)`
     */
    template<size_t N, size_t... ADims, size_t... BDims>
    auto ΣΠ(const Tensor<ADims...> &A, const Tensor<BDims...> &B) {
        return SigmaPiKernel<N, Tensor<ADims...>, Tensor<BDims...> >::compute(A, B);
    }

    // @doc: template<size_t N, size_t... ADims, size_t... BDims> auto SigmaPi(const Tensor<ADims...>& A, const Tensor<BDims...>& B)
    /**
     * English wrapper around greek-aliased function
     * returns `ΣΠ<N>(A, B)`
     */
    template<size_t N, size_t... ADims, size_t... BDims>
    auto SigmaPi(const Tensor<ADims...> &A, const Tensor<BDims...> &B) {
        return ΣΠ<N>(A, B);
    }


    // Permutation helpers for Einsum: move one axis to last or first position,
    // preserving the relative order of all other axes.
    // @doc: struct MoveToLastPerm<size_t Src, size_t Rank>
    /** Compile-time helper to rearrange `Tensor`'s shape such that `Src` is at index `[Rank - 1]` and all others are kept in order */
    template<size_t Src, size_t Rank>
    struct MoveToLastPerm {
        static constexpr auto value = [] {
            std::array<size_t, Rank> p{};
            size_t out = 0;
            for (size_t i = 0; i < Rank; ++i) {
                if (i != Src) {
                    p[out++] = i;
                }
            }
            p[Rank - 1] = Src;
            return p;
        }();
    };

    // @doc: struct MoveToFirstPerm<size_t Src, size_t Rank>
    /** Compile-time helper to rearrange `Tensor`'s shape such that `Src` is at index `[0]` and all others are kept in order */
    template<size_t Src, size_t Rank>
    struct MoveToFirstPerm {
        static constexpr auto value = [] {
            std::array<size_t, Rank> p{};
            p[0] = Src;
            size_t out = 1;
            for (size_t i = 0; i < Rank; ++i) {
                if (i != Src) {
                    p[out++] = i;
                }
            }
            return p;
        }();
    };

    // unpack a constexpr perm array into Permute<perm[I]...> via index_sequence
    // @doc: template<typename PermHolder, size_t... I, size_t... Dims> auto PermuteFromHolder(const Tensor<Dims...>& t, std::index_sequence<I...>)
    /**
     * Unpack a `constexpr` permutation indices array into a proper `Permute`-given `Tensor` type
     * `PermHolder` is an array of permutation indices, typically the result of `MoveToLastPerm` or `MoveToFirstPerm`
     * Call with `PermHolder` as template arg, `Tensor<Dims...>` as first arg, `std::make_index_sequence<sizeof...(Dims)>{}` as second arg
     */
    template<typename PermHolder, size_t... I, size_t... Dims>
    auto PermuteFromHolder(const Tensor<Dims...> &t, std::index_sequence<I...>) {
        return Permute<PermHolder::value[I]...>(t);
    }


    // EINSUM
    // Einsum<I, J>(A, B): contract axis I of A with axis J of B.
    // Derived from primitives: move I to last in A, J to first in B, then ΣΠ<1>.
    // @doc: template<size_t I, size_t J, size_t... ADims, size_t... BDims> auto Einsum(const Tensor<ADims...>& A, const Tensor<BDims...>& B)
    /**
     * `ΣΠ`-contracts over single selected indices, `I` and `J`, from `Tensor<ADims...>` and `Tensor<BDims...>`, respectively
     * Calls `ΣΠ<1>` on `PermuteFromHolder<MoveToLastPerm<I, sizeof...(ADims)>...>` and `PermuteFromHolder<MoveToFirstPerm<J, sizeof...(BDims)>...>`
     */
    template<size_t I, size_t J, size_t... ADims, size_t... BDims>
    auto Einsum(const Tensor<ADims...> &A, const Tensor<BDims...> &B) {
        static_assert(
            SizeTemplateGet<I, ADims...>::value == SizeTemplateGet<J, BDims...>::value,
            "axis I of A and axis J of B must have the same size");
        constexpr size_t RankA = sizeof...(ADims);
        constexpr size_t RankB = sizeof...(BDims);
        return ΣΠ<1>(
            PermuteFromHolder<MoveToLastPerm<I, RankA> >(A, std::make_index_sequence<RankA>{}),
            PermuteFromHolder<MoveToFirstPerm<J, RankB> >(B, std::make_index_sequence<RankB>{})
        );
    }


    // Dot: full contraction of two rank-1 tensors → Tensor<> (scalar)
    // @doc: template<size_t N> auto Dot(const Tensor<N>& a, const Tensor<N>& b)
    // @doc: template<size_t N> auto Dot(const Tensor<N>& A, const Tensor<N>& B)
    /** `ΣΠ`-contracts over `1` (the only) inner dimension of two `Tensor<N>`s, `A` and `B`, returning a `Tensor<>` with `Rank = 0` */
    template<size_t N>
    auto Dot(const Tensor<N> &A, const Tensor<N> &B) {
        return ΣΠ<1>(A, B);
    }

    // Matmul: standard 2D matrix multiplication Tensor<M,K> × Tensor<K,N> → Tensor<M,N>
    // @doc: template<size_t M, size_t K, size_t N> auto Matmul(const Tensor<M,K>& A, const Tensor<K,N>& B)
    /** `ΣΠ`-contracts over `1` inner dimension of two `Tensor<_,K>`s, `A` and `A`, returning a `Tensor<M,N>` with `Rank = 2` */
    template<size_t M, size_t K, size_t N>
    auto Matmul(const Tensor<M, K> &A, const Tensor<K, N> &B) {
        return ΣΠ<1>(A, B);
    }

    // Outer: named wrapper for the outer product (no contracted axes)
    // @doc: template<size_t... ADims, size_t... BDims> auto Outer(const Tensor<ADims...>& a, const Tensor<BDims...>& b)
    // @doc: template<size_t... ADims, size_t... BDims> auto Outer(const Tensor<ADims...>& A, const Tensor<BDims...>& B)
    /** `ΣΠ`-contracts over `0` inner dimension of `Tensor<ADims...> A` and `Tensor<BDims...> B`, returning a `Tensor<ADims..., BDims...>` with `Rank = sizeof...(ADims) + sizeof...(BDims)` */
    template<size_t... ADims, size_t... BDims>
    auto Outer(const Tensor<ADims...> &A, const Tensor<BDims...> &B) {
        return ΣΠ<0>(A, B);
    }

    // @doc: template<size_t... Dims> auto Contract(const Tensor<Dims...> &A, const Tensor<Dims...> &B)
    /**
     * `ΣΠ`-contracts *every* dimension of the congruent `A` and `B` `Tensor<Dims...>`s
     * `(⊕ ∘ ⊙)(A, B)` — Hadamard product (⊙) of `A` and `B`, then flat accumulation (⊕) over the result
     * Returns `Tensor<>` (scalar)
     */
    template<size_t... Dims>
    auto Contract(const Tensor<Dims...> &A, const Tensor<Dims...> &B) {
        return ΣΠ<sizeof...(Dims)>(A, B);
    }


    // Reverse all dimensions by calling Permute<NumDims-1, NumDims-2, ..., 0>
    // @doc: template<size_t... Dims> auto Transpose(const Tensor<Dims...>& t)
    /**
     * Reverse all dimensions of a `Tensor<Dims...>`
     * `Permute<(sizeof...(Dims) - 1 - I)...>(s)`, for `I` in `Dims...`
     * so if `Dims` are `<3, 5, 4>` (`sizeof = 3`) we will have:
     *   - `Permute<`(3-1) - 0 = `2,` (3-1) - 1 = `1,` (3-1) - 2 = `0>`
     */
    template<size_t... Dims>
    auto Transpose(const Tensor<Dims...> &t) {
        auto f = []<size_t... I>(const Tensor<I...> &s, std::index_sequence<I...>) {
            // so if Dims are <3, 5, 4> (sizeof = 3) we will have:
            //      Permute<(3-1) - 0 = 2, 2 - 1 = 1, 0>
            return Permute<(sizeof...(Dims) - 1 - I)...>(s);
        };
        return f(t, std::make_index_sequence<sizeof...(Dims)>{});
    }

    // REDUCE KERNEL
    // Precomputes per-output base flat indices and the compile-time axis stride,
    // shared across ReduceSum, ReduceMax, and BroadcastAdd.
    //
    // bases[out_i]  = flat index in Source for output out_i with axis dim = 0
    //                 inner loop: src.flat(bases[out_i] + k * axis_stride), k in [0, axis_dim)
    // project[i]    = flat index in Result for source flat index i (axis contribution stripped)
    //                 used by BroadcastAdd: b.flat(project[i])
    //
        // @doc: template<size_t Axis, size_t... Dims> struct ReduceKernel
        /**
         * Struct templated on `<size_t Axis, size_t... Dims>`, shared across `ReduceSum`, `ReduceMax`, and `BroadcastAdd`
         * Compile-time `static constexpr`:
         *   - `axis_dim = SizeTemplateGet<Axis, Dims...>::value`
         *   - `axis_stride = Source::Strides[Axis]`
         *   - `std::array<size_t, Result::Size> bases`
         *   - flat index in `Source` for each output index with axis set to 0
         *   - inner loop: `src.flat(bases[out_i] + k * axis_stride)`
         *   - `static constexpr size_t project(size_t i)`
         *   - flat index in `Result` for source flat index `i` (axis contribution stripped); closed-form `i - ((i / axis_stride) % axis_dim) * axis_stride`
         *   - no table, `axis_stride` compile-time so division compiles to multiply-shift
         */
    template<size_t Axis, size_t... Dims>
    struct ReduceKernel {
        using Source = Tensor<Dims...>;
        using Result = typename RemoveAxis<Axis, Dims...>::type;
        static constexpr size_t axis_dim = SizeTemplateGet<Axis, Dims...>::value;
        static constexpr size_t axis_stride = Source::Strides[Axis];

        static constexpr auto bases = [] {
            std::array<size_t, Result::Size> t{};
            for (size_t out_i = 0; out_i < Result::Size; ++out_i) {
                const auto dm = Result::FlatToMulti(out_i);
                std::array<size_t, Source::Rank> sm{};
                size_t d = 0;
                for (size_t sd = 0; sd < Source::Rank; ++sd)
                    sm[sd] = (sd == Axis) ? 0 : dm[d++];
                t[out_i] = Source::MultiToFlat(sm);
            }
            return t;
        }();

        // project(i): flat index in Result for source flat index i (axis contribution stripped)
        // closed-form — no table, axis_stride is compile-time so division → multiply-shift
        static constexpr size_t project(size_t i) {
            return i - ((i / axis_stride) % axis_dim) * axis_stride;
        }
    };

    // REDUCE SUM
    // Sum over an axis, collapsing it.
    // Ex: ReduceSum<0>(Tensor<Batch, In>) --> Tensor<In> where each 'column' is summed
    // (ReduceSum<Axis>(A) is mathematically Einsum<Axis,0>(A, ones) — contraction with a ones vector.)
    // @doc: template<size_t Axis, size_t... Dims> typename RemoveAxis<Axis, Dims...>::type ReduceSum(const Tensor<Dims...>& src)
    /**
     * Reduce an axis with `Tensor` addition
     * `ReduceSum<P>(Tensor<P,Q>) -> Tensor<Q>` where all `Q` final values are the `sum` of `P` collapsed values
     * `ReduceSum<Axis>(A)` is mathematically equivalent to `Einsum<Axis, 0>(A, ones)`, where `ones = Tensor<SizeTemplateGet<Axis, Dims...>::value>`, i.e., `ones` is a `Rank-1` tensor whose size is the same as the `Axis` dimension in `src`
     */
    template<size_t Axis, size_t... Dims>
    typename RemoveAxis<Axis, Dims...>::type ReduceSum(const Tensor<Dims...> &src) {
        using K = ReduceKernel<Axis, Dims...>;
        typename K::Result dst;
        auto k_range = std::views::iota(size_t{0}, K::axis_dim);
        ParForEach(K::Result::Size, [&](size_t out_i) {
            dst.flat(out_i) = std::transform_reduce(
                std::execution::unseq,
                k_range.begin(), k_range.end(),
                0.0f, std::plus<>{},
                [&](size_t k) {
                    return src.flat(K::bases[out_i] + k * K::axis_stride);
                });
        });
        return dst;
    }

    // BROADCAST ADD
    // Add a lower-rank tensor to every slice of a higher-rank tensor along Axis.
    // BroadcastAdd<0>(Tensor<Batch,Out>, Tensor<Out>) --> Tensor<Batch,Out>
    // @doc: template<size_t Axis, size_t... Dims> Tensor<Dims...> BroadcastAdd(const Tensor<Dims...>& A, const typename RemoveAxis<Axis, Dims...>::type& b)
    /** Add a `RemoveAxis<Axis, Dims...>::type` (`Tensor<sizeof...(Dims) - 1>`, where the removed dimension is `Axis`) to all `Axis` slices of a `Tensor<Dims...>` */
    template<size_t Axis, size_t... Dims>
    Tensor<Dims...> BroadcastAdd(const Tensor<Dims...> &A, const typename RemoveAxis<Axis, Dims...>::type &b) {
        using K = ReduceKernel<Axis, Dims...>;
        Tensor<Dims...> result;
        ParForEach(Tensor<Dims...>::Size, [&](size_t i) {
            result.flat(i) = A.flat(i) + b.flat(K::project(i));
        });
        return result;
    }

    // REDUCE MEAN
    // Mean over an axis — ReduceSum then divide by the axis size.
    // @doc: template<size_t Axis, size_t... Dims> typename RemoveAxis<Axis, Dims...>::type ReduceMean(const Tensor<Dims...>& src)
    /**
     * Reduce an axis with `Tensor` averaging
     * `ReduceSum<P>(Tensor<P,Q>) -> Tensor<Q>` where all `Q` final values are the `average` of `P` collapsed values
     */
    template<size_t Axis, size_t... Dims>
    typename RemoveAxis<Axis, Dims...>::type ReduceMean(const Tensor<Dims...> &src) {
        constexpr float inv = 1.f / static_cast<float>(SizeTemplateGet<Axis, Dims...>::value);
        return ReduceSum<Axis>(src) * inv;
    }

    // REDUCE MAX
    // Max over an axis, collapsing it.
    // @doc: template<size_t Axis, size_t... Dims> typename RemoveAxis<Axis, Dims...>::type ReduceMax(const Tensor<Dims...>& src)
    /**
     * Reduce an axis with `Tensor` maxing
     * `ReduceSum<P>(Tensor<P,Q>) -> Tensor<Q>` where all `Q` final values are the `max` of `P` collapsed values
     */
    template<size_t Axis, size_t... Dims>
    typename RemoveAxis<Axis, Dims...>::type ReduceMax(const Tensor<Dims...> &src) {
        using K = ReduceKernel<Axis, Dims...>;
        typename K::Result dst;
        auto k_range = std::views::iota(size_t{0}, K::axis_dim);
        ParForEach(K::Result::Size, [&](size_t out_i) {
            dst.flat(out_i) = std::transform_reduce(
                std::execution::unseq,
                k_range.begin(), k_range.end(),
                -std::numeric_limits<float>::infinity(),
                [](float a, float b) { return std::max(a, b); },
                [&](size_t k) { return src.flat(K::bases[out_i] + k * K::axis_stride); });
        });
        return dst;
    }

    // TENSOR INDEX
    // Extract the rank-(R-1) subtensor at position idx along Axis, peeling that dimension off.
    // TensorIndex<0>(Tensor<SeqLen, EmbDims...>, s) --> Tensor<EmbDims...>
    //
    // Inverse of TensorIndexAdd: iterates over the result, inserts idx at Axis to address the source.
    // @doc: template<size_t Axis, size_t... Dims> typename RemoveAxis<Axis, Dims...>::type TensorIndex(const Tensor<Dims...>& src, size_t idx)
    /**
     * Extract the `idx`-th `RemoveAxis<Axis, Dims...>::type` sub-`Tensor` from `Tensor<Dims...> src` on the `Axis` axis
     * Essentially fills new `Tensor` with values from `src` by looping through dimensions in `Rank`, but passing `idx` for `Axis` dimension on all values
     */
    template<size_t Axis, size_t... Dims>
    typename RemoveAxis<Axis, Dims...>::type TensorIndex(const Tensor<Dims...> &src, size_t idx) {
        using Source = Tensor<Dims...>;
        // we're decrementing rank
        using Result = typename RemoveAxis<Axis, Dims...>::type;

        Result dst;
        // for each value we're returning
        for (size_t i = 0; i < Result::Size; ++i) {
            // where are we going in result in terms of dims?
            auto dst_multi = Result::FlatToMulti(i);

            // rebuild source multi-index by inserting idx at Axis
            std::array<size_t, Source::Rank> src_multi{};
            size_t dst_d = 0;
            for (size_t d = 0; d < Source::Rank; ++d) {
                src_multi[d] = d == Axis ? idx : dst_multi[dst_d++];
            }
            dst.flat(i) = src.flat(Source::MultiToFlat(src_multi));
        }
        return dst;
    }

    // TENSOR INDEX ADD
    // Accumulate a rank-(R-1) tensor into one slice of a rank-R tensor along Axis at position idx.
    // Used in backward passes to scatter per-token gradients back into the full sequence gradient.
    // TensorIndexAdd<0>(dX, s, dx_s): dX[s, ...] += dx_s[...]
    // @doc: template<size_t Axis, size_t... Dims> void TensorIndexAdd(Tensor<Dims...>& dst, size_t idx, const typename RemoveAxis<Axis, Dims...>::type& src)
    /** Accumulate (`+=`) a `RemoveAxis<Axis, Dims...>::type` to the `idx`-th sub-`Tensor` of `Tensor<Dims...> dst` on the `Axis` axis */
    template<size_t Axis, size_t... Dims>
    void TensorIndexAdd(Tensor<Dims...> &dst, size_t idx,
                        const typename RemoveAxis<Axis, Dims...>::type &src) {
        using Dest = Tensor<Dims...>;
        using Slice = typename RemoveAxis<Axis, Dims...>::type;

        for (size_t i = 0; i < Slice::Size; ++i) {
            auto src_multi = Slice::FlatToMulti(i);
            // rebuild destination multi-index by inserting idx at Axis
            std::array<size_t, Dest::Rank> dst_multi{};
            size_t src_d = 0;
            for (size_t d = 0; d < Dest::Rank; ++d) {
                dst_multi[d] = d == Axis ? idx : src_multi[src_d++];
            }
            dst.flat(Dest::MultiToFlat(dst_multi)) += src.flat(i);
        }
    }

};
