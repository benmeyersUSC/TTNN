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
    static constexpr float EPS = 1e-8f;
    static constexpr float ADAM_BETA_1 = 0.9f;
    static constexpr float ADAM_BETA_2 = 0.999f;

    // ParForEach: run f(i) for i in [0, n) in parallel.
    // Thin wrapper around std::for_each(par_unseq, iota) — the canonical parallel loop pattern.
    template<typename F>
    void ParForEach(size_t n, F f) {
        auto range = std::views::iota(size_t{0}, n);
        std::for_each(std::execution::par_unseq, range.begin(), range.end(), f);
    }

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

    // Hadamard (element-wise) product of two same-shape tensors
    template<size_t... Dims>
    Tensor<Dims...> operator*(const Tensor<Dims...> &a, const Tensor<Dims...> &b) {
        return a.zip(b, [](float x, float y) { return x * y; });
    }


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
        for (size_t i = 0; i < Result::Size; i++) {
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
            std::array<size_t, Rank> src_multi{};
            for (size_t d = 0; d < Rank; d++)
                src_multi[perm_arr[d]] = dst_idx[d];
            size_t src_index = Source::MultiToFlat(src_multi);
            assert(src_index < Source::Size && "permuted src index out of bounds");
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


    // TENSOR SLICE
    // Select a contiguous range [Start, Start+Len) of dims, producing a new Tensor type.
    // Peer to RemoveAxis — same ArrayToTensor machinery, different selection predicate.
    template<size_t Start, size_t Len, size_t... Dims>
    struct SliceDimsHolder {
        static constexpr auto value = [] {
            constexpr std::array<size_t, sizeof...(Dims)> all = {Dims...};
            std::array<size_t, Len> result{};
            for (size_t i = 0; i < Len; ++i)
                result[i] = all[Start + i];
            return result;
        }();
    };

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

    template<size_t N, size_t... ADims, size_t... BDims>
    auto ΣΠ(const Tensor<ADims...> &A, const Tensor<BDims...> &B) {
        constexpr size_t RankA = sizeof...(ADims);
        constexpr size_t RankB = sizeof...(BDims);
        static_assert(N <= RankA, "N exceeds rank of A");
        static_assert(N <= RankB, "N exceeds rank of B");

        // validate that last N dims of A match first N dims of B
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

        constexpr size_t a_free_rank = A_Free::Rank;
        constexpr size_t b_free_rank = B_Free::Rank;

        ResultType result;

        // Each out_i owns result.flat(out_i) exclusively — no write conflicts.
        // Accumulate into a local float to keep the hot value in a register.
        ParForEach(ResultType::Size, [&](size_t out_i) {
                auto out_multi = ResultType::FlatToMulti(out_i);

                // split output multi-index into A's free part and B's free part
                std::array<size_t, RankA - N> a_free{};
                std::array<size_t, RankB - N> b_free{};
                for (size_t i = 0; i < a_free_rank; ++i) a_free[i] = out_multi[i];
                for (size_t i = 0; i < b_free_rank; ++i) b_free[i] = out_multi[a_free_rank + i];

                // Σ over every contracted index combination
                float acc = 0.f;
                for (size_t c = 0; c < Contracted::Size; ++c) {
                    auto c_multi = Contracted::FlatToMulti(c);

                    // A's full multi-index: [a_free..., c...]
                    std::array<size_t, RankA> a_multi{};
                    for (size_t i = 0; i < a_free_rank; ++i) a_multi[i] = a_free[i];
                    for (size_t i = 0; i < N; ++i) a_multi[a_free_rank + i] = c_multi[i];

                    // B's full multi-index: [c..., b_free...]
                    std::array<size_t, RankB> b_multi{};
                    for (size_t i = 0; i < N; ++i) b_multi[i] = c_multi[i];
                    for (size_t i = 0; i < b_free_rank; ++i) b_multi[N + i] = b_free[i];

                    acc += A(a_multi) * B(b_multi); // Π
                }
                result.flat(out_i) = acc;
            });
        return result;
    }

    template<size_t N, size_t... ADims, size_t... BDims>
    auto SigmaPi(const Tensor<ADims...> &A, const Tensor<BDims...> &B) {
        return ΣΠ<N>(A, B);
    }


    // Permutation helpers for Einsum: move one axis to last or first position,
    // preserving the relative order of all other axes.
    template<size_t Src, size_t P>
    struct MoveToLastPerm {
        static constexpr auto value = [] {
            std::array<size_t, P> p{};
            size_t out = 0;
            for (size_t i = 0; i < P; ++i)
                if (i != Src) p[out++] = i;
            p[P - 1] = Src;
            return p;
        }();
    };

    template<size_t Src, size_t P>
    struct MoveToFirstPerm {
        static constexpr auto value = [] {
            std::array<size_t, P> p{};
            p[0] = Src;
            size_t out = 1;
            for (size_t i = 0; i < P; ++i)
                if (i != Src) p[out++] = i;
            return p;
        }();
    };

    // unpack a constexpr perm array into Permute<perm[I]...> via index_sequence
    template<typename PermHolder, size_t... I, size_t... Dims>
    auto PermuteFromHolder(const Tensor<Dims...> &t, std::index_sequence<I...>) {
        return Permute<PermHolder::value[I]...>(t);
    }


    // EINSUM
    // Einsum<I, J>(A, B): contract axis I of A with axis J of B.
    // Derived from primitives: move I to last in A, J to first in B, then ΣΠ<1>.
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

    // outer product: no axes contracted — SigmaPi<0>
    template<size_t... ADims, size_t... BDims>
    auto Einsum(const Tensor<ADims...> &A, const Tensor<BDims...> &B) {
        return ΣΠ<0>(A, B);
    }

    // Dot: full contraction of two rank-1 tensors → Tensor<> (scalar)
    template<size_t N>
    auto Dot(const Tensor<N> &a, const Tensor<N> &b) {
        return ΣΠ<1>(a, b);
    }

    // Matmul: standard 2D matrix multiplication Tensor<M,K> × Tensor<K,N> → Tensor<M,N>
    template<size_t M, size_t K, size_t N>
    auto Matmul(const Tensor<M, K> &A, const Tensor<K, N> &B) {
        return ΣΠ<1>(A, B);
    }

    // Outer: named wrapper for the outer product (no contracted axes)
    template<size_t... ADims, size_t... BDims>
    auto Outer(const Tensor<ADims...> &a, const Tensor<BDims...> &b) {
        return ΣΠ<0>(a, b);
    }


    // Reverse all dimensions by calling Permute<NumDims-1, NumDims-2, ..., 0>
    template<size_t... Dims>
    auto Transpose(const Tensor<Dims...> &t) {
        auto f = []<size_t... I>(const Tensor<I...> &s, std::index_sequence<I...>) {
            // so if Dims are <3, 5, 4> (sizeof = 3) we will have:
            //      Permute<(3-1) - 0 = 2, 2 - 1 = 1, 0>
            return Permute<(sizeof...(Dims) - 1 - I)...>(s);
        };
        return f(t, std::make_index_sequence<sizeof...(Dims)>{});
    }

    // REDUCE SUM
    // Sum over an axis, collapsing it.
    // Ex: ReduceSum<0>(Tensor<Batch, In>) --> Tensor<In> where each 'column' is summed
    //
    // Parallelized by iterating over *output* elements (no write contention),
    // accumulating the axis dimension sequentially inside each job — same pattern as ΣΠ.
    // (ReduceSum<Axis>(A) is mathematically Einsum<Axis,0>(A, ones) — contraction with a ones vector.)
    template<size_t Axis, size_t... Dims>
    auto ReduceSum(const Tensor<Dims...> &src) -> typename RemoveAxis<Axis, Dims...>::type {
        using Source = Tensor<Dims...>;
        using Result = typename RemoveAxis<Axis, Dims...>::type;
        static constexpr size_t axis_dim = SizeTemplateGet<Axis, Dims...>::value;

        Result dst;
        ParForEach(Result::Size, [&](size_t out_i) {
            auto dst_multi = Result::FlatToMulti(out_i);
            std::array<size_t, Source::Rank> src_multi{};
            size_t d = 0;
            for (size_t sd = 0; sd < Source::Rank; ++sd)
                src_multi[sd] = (sd == Axis) ? 0 : dst_multi[d++];
            float acc = 0.f;
            for (size_t k = 0; k < axis_dim; ++k) {
                src_multi[Axis] = k;
                acc += src.flat(Source::MultiToFlat(src_multi));
            }
            dst.flat(out_i) = acc;
        });
        return dst;
    }

    // BROADCAST ADD
    // Add a lower-rank tensor to every slice of a higher-rank tensor along Axis.
    // BroadcastAdd<0>(Tensor<Batch,Out>, Tensor<Out>) --> Tensor<Batch,Out>
    // Each output element is independent — parallelized directly.
    template<size_t Axis, size_t... Dims>
    Tensor<Dims...> BroadcastAdd(const Tensor<Dims...> &A, const typename RemoveAxis<Axis, Dims...>::type &b) {
        using Slice = typename RemoveAxis<Axis, Dims...>::type;
        using Result = Tensor<Dims...>;

        Result result;
        ParForEach(Result::Size, [&](size_t i) {
            auto idx = Result::FlatToMulti(i);
            std::array<size_t, Slice::Rank> b_multi{};
            size_t b_dim_idx = 0;
            for (size_t d = 0; d < Result::Rank; d++)
                if (d != Axis) b_multi[b_dim_idx++] = idx[d];
            result.flat(i) = A.flat(i) + b.flat(Slice::MultiToFlat(b_multi));
        });
        return result;
    }

    // REDUCE MEAN
    // Mean over an axis — ReduceSum then divide by the axis size.
    template<size_t Axis, size_t... Dims>
    auto ReduceMean(const Tensor<Dims...> &src) -> typename RemoveAxis<Axis, Dims...>::type {
        constexpr float inv = 1.f / static_cast<float>(SizeTemplateGet<Axis, Dims...>::value);
        return ReduceSum<Axis>(src) * inv;
    }

    // REDUCE MAX
    // Max over an axis, collapsing it — same parallel pattern as ReduceSum.
    template<size_t Axis, size_t... Dims>
    auto ReduceMax(const Tensor<Dims...> &src) -> typename RemoveAxis<Axis, Dims...>::type {
        using Source = Tensor<Dims...>;
        using Result = typename RemoveAxis<Axis, Dims...>::type;
        static constexpr size_t axis_dim = SizeTemplateGet<Axis, Dims...>::value;

        Result dst;
        ParForEach(Result::Size, [&](size_t out_i) {
            auto dst_multi = Result::FlatToMulti(out_i);
            std::array<size_t, Source::Rank> src_multi{};
            size_t d = 0;
            for (size_t sd = 0; sd < Source::Rank; ++sd)
                src_multi[sd] = (sd == Axis) ? 0 : dst_multi[d++];
            float best = -std::numeric_limits<float>::infinity();
            for (size_t k = 0; k < axis_dim; ++k) {
                src_multi[Axis] = k;
                best = std::max(best, src.flat(Source::MultiToFlat(src_multi)));
            }
            dst.flat(out_i) = best;
        });
        return dst;
    }

    // TENSOR INDEX
    // Extract the rank-(R-1) subtensor at position idx along Axis, peeling that dimension off.
    // TensorIndex<0>(Tensor<SeqLen, EmbDims...>, s) --> Tensor<EmbDims...>
    //
    // Inverse of TensorIndexAdd: iterates over the result, inserts idx at Axis to address the source.
    template<size_t Axis, size_t... Dims>
    auto TensorIndex(const Tensor<Dims...> &src, size_t idx)
        -> typename RemoveAxis<Axis, Dims...>::type {
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
            for (size_t d = 0; d < Source::Rank; ++d)
                src_multi[d] = (d == Axis) ? idx : dst_multi[dst_d++];
            dst.flat(i) = src.flat(Source::MultiToFlat(src_multi));
        }
        return dst;
    }

    // TENSOR INDEX ADD
    // Accumulate a rank-(R-1) tensor into one slice of a rank-R tensor along Axis at position idx.
    // Used in backward passes to scatter per-token gradients back into the full sequence gradient.
    // TensorIndexAdd<0>(dX, s, dx_s): dX[s, ...] += dx_s[...]
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
            for (size_t d = 0; d < Dest::Rank; ++d)
                dst_multi[d] = (d == Axis) ? idx : src_multi[src_d++];
            dst.flat(Dest::MultiToFlat(dst_multi)) += src.flat(i);
        }
    }

    // SOFTMAX
    // Normalizes over the Axis dimension: for each (outer, inner) pair the Axis-slice is
    // exp-normalized to sum to 1.  Max-subtraction ensures numerical stability.
    //
    // Flat-index layout:
    //   axis_dim   = Dims[Axis]
    //   inner_size = stride[Axis] = product of all dims after Axis
    //   outer_size = Size / (axis_dim * inner_size)
    //   element k in pool (outer, inner): outer*(axis_dim*inner_size) + k*inner_size + inner
    template<size_t Axis, size_t... Dims>
    Tensor<Dims...> Softmax(const Tensor<Dims...> &x) {
        using T = Tensor<Dims...>;
        static constexpr size_t axis_dim = SizeTemplateGet<Axis, Dims...>::value;
        static constexpr size_t inner_size = ComputeStrides<Dims...>::value[Axis];
        static constexpr size_t outer_size = T::Size / (axis_dim * inner_size);

        T result;
        for (size_t outer = 0; outer < outer_size; ++outer) {
            for (size_t inner = 0; inner < inner_size; ++inner) {
                const size_t base = outer * (axis_dim * inner_size) + inner;
                float maxV = x.flat(base);
                for (size_t k = 1; k < axis_dim; ++k)
                    maxV = std::max(maxV, x.flat(base + k * inner_size));
                float sum = 0.f;
                for (size_t k = 0; k < axis_dim; ++k) {
                    result.flat(base + k * inner_size) = std::exp(x.flat(base + k * inner_size) - maxV);
                    sum += result.flat(base + k * inner_size);
                }
                for (size_t k = 0; k < axis_dim; ++k)
                    result.flat(base + k * inner_size) /= sum;
            }
        }
        return result;
    }

    // SOFTMAX BACKWARD
    // Vector-Jacobian product of softmax: δx_i = a_i * (δy_i - dot(δy, a))
    // One dot product per pool, then pointwise scale — O(axis_dim) per pool, no Jacobian matrix.
    template<size_t Axis, size_t... Dims>
    Tensor<Dims...> SoftmaxPrime(const Tensor<Dims...> &grad, const Tensor<Dims...> &a) {
        using T = Tensor<Dims...>;
        static constexpr size_t axis_dim = SizeTemplateGet<Axis, Dims...>::value;
        static constexpr size_t inner_size = ComputeStrides<Dims...>::value[Axis];
        static constexpr size_t outer_size = T::Size / (axis_dim * inner_size);

        T result;
        for (size_t outer = 0; outer < outer_size; ++outer) {
            for (size_t inner = 0; inner < inner_size; ++inner) {
                const size_t base = outer * (axis_dim * inner_size) + inner;
                float dot = 0.f;
                for (size_t k = 0; k < axis_dim; ++k)
                    dot += a.flat(base + k * inner_size) * grad.flat(base + k * inner_size);
                for (size_t k = 0; k < axis_dim; ++k)
                    result.flat(base + k * inner_size) =
                            a.flat(base + k * inner_size) * (grad.flat(base + k * inner_size) - dot);
            }
        }
        return result;
    }
};
