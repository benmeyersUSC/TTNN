#pragma once
#include "TensorShapeOps.hpp"


namespace TTTN {
    struct Add {
        constexpr float operator()(float a, float b) const { return a + b; }
        static constexpr float identity = 0.f;
    };

    struct Mul {
        constexpr float operator()(float a, float b) const { return a * b; }
        static constexpr float identity = 1.f;
    };

    struct Max {
        constexpr float operator()(float a, float b) const { return std::max(a, b); }
        static constexpr float identity = -std::numeric_limits<float>::infinity();
    };

    struct Sub {
        constexpr float operator()(float a, float b) const { return a - b; }
    };

    struct Div {
        constexpr float operator()(float a, float b) const { return a / b; }
    };

    struct AbsDiff {
        constexpr float operator()(float a, float b) const { return std::abs(a - b); }
    };

    struct SqAdd {
        constexpr float operator()(float acc, float x) const { return acc + x * x; }
        static constexpr float identity = 0.f;
    };

    template<size_t D>
    struct SubMean {
        constexpr float operator()(float x, float sum) const { return x - sum / static_cast<float>(D); }
    };

    struct Log {
        constexpr float operator()(float x) const { return std::log(x); }
    };

    struct Exp {
        constexpr float operator()(float x) const { return std::exp(x); }
    };

    struct Neg {
        constexpr float operator()(float x) const { return -x; }
    };

    struct Sq {
        constexpr float operator()(float x) const { return x * x; }
    };

    struct Abs {
        constexpr float operator()(float x) const { return std::abs(x); }
    };

    struct OneMinus {
        constexpr float operator()(float x) const { return 1.f - x; }
    };

    template<float Lo, float Hi = std::numeric_limits<float>::infinity()>
    struct Clamp {
        constexpr float operator()(float x) const { return std::min(std::max(x, Lo), Hi); }
    };

    template<float T>
    struct Step {
        constexpr float operator()(float x) const { return x < T ? 1.f : 0.f; }
    };

    // @doc: template<typename F, typename G> struct Compose
    /** Compose a `FloatUnaryOp` with either another `FloatUnaryOp` or a `FloatBinaryOp`, creating a new operation that can be passed as a template tag */
    template<typename F, typename G>
    struct Compose {
        constexpr float operator()(float x) const
            requires FloatUnaryOp<F> && FloatUnaryOp<G> { return F{}(G{}(x)); }

        constexpr float operator()(float a, float b) const
            requires FloatUnaryOp<F> && FloatBinaryOp<G> { return F{}(G{}(a, b)); }
    };


    // @doc: template<typename Op, size_t... Dims> requires FloatUnaryOp<Op> && std::default_initializable<Op> Tensor<Dims...> Map(const Tensor<Dims...> &src)
    /** Apply a `FloatUnaryOp` to every element of `Tensor<Dims...>& src` and return a copy */
    template<typename Op, size_t... Dims> requires FloatUnaryOp<Op> && std::default_initializable<Op>
    Tensor<Dims...> Map(const Tensor<Dims...> &src) {
        return src.map(Op{});
    }

    // @doc: template<typename Op, size_t... Dims> requires FloatUnaryOp<Op> && std::default_initializable<Op> void Apply(Tensor<Dims...> &&src)
    /** Apply a `FloatUnaryOp` inplace */
    template<typename Op, size_t... Dims> requires FloatUnaryOp<Op> && std::default_initializable<Op>
    void Apply(Tensor<Dims...> &&src) {
        src.apply(Op{});
    }

    // @doc: template<typename Op, size_t... Dims> requires FloatUnaryOp<Op> && std::default_initializable<Op> Tensor<Dims...> MapMove(Tensor<Dims...> &&src)
    /**
     * `Apply` a `FloatUnaryOp` inplace on `src`, return moved version
     * Memory efficient way to call `Apply` on a `Tensor` that is part of a composition or nested call
     */
    template<typename Op, size_t... Dims> requires FloatUnaryOp<Op> && std::default_initializable<Op>
    Tensor<Dims...> MapMove(Tensor<Dims...> &&src) {
        src.apply(Op{});
        return std::move(src);
    }


    // @doc: template<typename Op, size_t... Dims> requires FloatBinaryOp<Op> && std::default_initializable<Op> Tensor<Dims...> Zip(const Tensor<Dims...> &A, const Tensor<Dims...> &B)
    /** Create copy of `A`, call `FloatBinaryOp` taking in elements from `A` and `B`, return new copy */
    template<typename Op, size_t... Dims> requires FloatBinaryOp<Op> && std::default_initializable<Op>
    Tensor<Dims...> Zip(const Tensor<Dims...> &A, const Tensor<Dims...> &B) {
        Tensor<Dims...> result(A);
        result.zip_apply(B, Op{});
        return result;
    }

    // @doc: template<typename Op, size_t... Dims> requires FloatBinaryOp<Op> && std::default_initializable<Op> Tensor<Dims...> ZipMove(Tensor<Dims...> &&A, const Tensor<Dims...> &B)
    /** Call `zip_apply` inplace on `A`, taking in `B` values, return moved-from version */
    template<typename Op, size_t... Dims> requires FloatBinaryOp<Op> && std::default_initializable<Op>
    Tensor<Dims...> ZipMove(Tensor<Dims...> &&A, const Tensor<Dims...> &B) {
        A.zip_apply(B, Op{});
        return std::move(A);
    }


    // @doc: template<size_t Axis, FloatBinaryOp F, size_t... Dims> void TensorIndexApply(Tensor<Dims...> &dst, size_t idx, const typename RemoveAxis<Axis, Dims...>::type &src, F f)
    /** Using indexing conventions described in `TensorIndex`, apply a `FloatBinaryOp` on a sub-`Tensor`, combining `src` elements with those of `dst`, a sub-`Tensor` of the same type as the slice */
    template<size_t Axis, FloatBinaryOp F, size_t... Dims>
    void TensorIndexApply(Tensor<Dims...> &dst, size_t idx, const typename RemoveAxis<Axis, Dims...>::type &src, F f) {
        using Dest = Tensor<Dims...>;
        using Slice = RemoveAxis<Axis, Dims...>::type;
        ParForEach(Slice::Size, [&](const size_t i) {
            auto src_multi = Slice::FlatToMulti(i);
            std::array<size_t, Dest::Rank> dst_multi{};
            size_t src_d = 0;
            for (size_t d = 0; d < Dest::Rank; ++d) {
                dst_multi[d] = d == Axis ? idx : src_multi[src_d++];
            }
            const size_t flat = Dest::MultiToFlat(dst_multi);
            dst.flat(flat) = f(dst.flat(flat), src.flat(i));
        });
    }


    // @doc: template<size_t... Perm, size_t... Dims> Tensor<SizeTemplateGet<Perm, Dims...>::value...> Permute(const Tensor<Dims...> &src)
    /**
     * Given a `size_t...Perm`, the same length as `Dims...`, representing a new ordering of `Tensor<Dims...> src`'s axes, perform a permutation and return a copy
     * `Permute<1, 2, 0>(Tensor<8, 4, 7>)` returns a `Tensor<4, 7, 8>`
     */
    template<size_t... Perm, size_t... Dims>
    Tensor<SizeTemplateGet<Perm, Dims...>::value...> Permute(const Tensor<Dims...> &src) {
        static_assert(sizeof...(Perm) == sizeof...(Dims), "Permutation length must match rank");
        using Source = Tensor<Dims...>;
        using Result = Tensor<SizeTemplateGet<Perm, Dims...>::value...>;
        constexpr std::array<size_t, sizeof...(Dims)> perm = {Perm...};
        Result dst;
        ParForEach(Result::Size, [&](size_t i) {
            auto dst_multi = Result::FlatToMulti(i);
            std::array<size_t, sizeof...(Dims)> src_multi{};
            for (size_t j = 0; j < sizeof...(Dims); ++j)
                src_multi[perm[j]] = dst_multi[j];
            dst.flat(i) = src.flat(Source::MultiToFlat(src_multi));
        });
        return dst;
    }


    // @doc: template<size_t... Dims> auto Transpose(const Tensor<Dims...> &src)
    /**
     * Call `Permute` on `Tensor<Dims...>` with permutation indices: `<Tensor<Dims...>::Rank - 1, ..., 0>` (reverse all axes of `Tensor<Dims...>`)
     * `Transpose(Tensor<8, 4, 7>)` returns a `Tensor<7, 4, 8>`
     */
    template<size_t... Dims>
    auto Transpose(const Tensor<Dims...> &src) {
        return []<size_t... I>(const auto &s, std::index_sequence<I...>) {
            return Permute<(sizeof...(Dims) - 1 - I)...>(s);
        }(src, std::make_index_sequence<sizeof...(Dims)>{});
    }

    // @doc: template<auto Perm, size_t... I, size_t... Dims> auto PermuteFromArray(const Tensor<Dims...> &t, std::index_sequence<I...>)
    /**
     * Takes a `std::array<size_t, Rank>` representing requested permutation ordering and unpacks them with a `std::index_sequence` into a call to `Permute`
     * Returns a `Tensor` of new permuted shape
     */
    template<auto Perm, size_t... I, size_t... Dims>
    auto PermuteFromArray(const Tensor<Dims...> &t, std::index_sequence<I...>) {
        return Permute<Perm[I]...>(t);
    }

    // @doc: template<auto Perm, size_t... I, size_t... Dims> decltype(auto) ConditionalPermute(const Tensor<Dims...> &t, std::index_sequence<I...>)
    /**
     * Zero-cost identity check: if `Perm` is the identity permutation (all `Perm[i] == i`), returns `t` as a `const&` with no allocation
     * Otherwise falls back to `Permute`, returning a permuted copy
     * Used internally by `Contract` and `BatchContract` to skip the permutation step when incoming tensors are already in minor-aligned layout
     */
    template<auto Perm, size_t... I, size_t... Dims>
    decltype(auto) ConditionalPermute(const Tensor<Dims...> &t, std::index_sequence<I...>) {
        // if requested permutation axes are all already in place, return const &
        if constexpr (((Perm[I] == I) && ...))
            return t;
            // otherwise, call Permute and return copy
        else
            return Permute<Perm[I]...>(t);
    }


    // @doc: template<size_t... NewDims, size_t... OldDims> Tensor<NewDims...> Reshape(const Tensor<OldDims...> &src)
    /**
     * Reinterpret `Tensor<OldDims...>` as `Tensor<NewDims...>` — total size must match
     * Same flat data, new shape; copies via `std::copy`
     */
    template<size_t... NewDims, size_t... OldDims>
    Tensor<NewDims...> Reshape(const Tensor<OldDims...> &src) {
        static_assert(Tensor<NewDims...>::Size == Tensor<OldDims...>::Size, "Reshape: total size must match");
        Tensor<NewDims...> result;
        std::copy(src.data(), src.data() + src.Size, result.data());
        return result;
    }

    // @doc: template<size_t... Dims> Tensor<Tensor<Dims...>::Size> Flatten(const Tensor<Dims...> &src)
    /**
     * Collapse `Tensor<Dims...>` to rank-1 `Tensor<Size>`
     * Convenience wrapper around `Reshape<Size>`
     */
    template<size_t... Dims>
    Tensor<Tensor<Dims...>::Size> Flatten(const Tensor<Dims...> &src) {
        return Reshape<Tensor<Dims...>::Size>(src);
    }


    // @doc: template<size_t Axis, size_t Index, size_t... Dims> RemoveAxis<Axis, Dims...>::type TensorIndex(const Tensor<Dims...> &src)
    /**
     * Get a `Tensor` slice from `Tensor<Dims...> src` by specifying a `size_t Axis` and a `size_t Index` *from* that axis
     * If we have a `Tensor<3, 2, 4>` and call `TensorIndex<0, 1>()`, we will get the `1-th` (second) `Tensor<2, 4>` that lives on the first axis
     */
    template<size_t Axis, size_t Index, size_t... Dims>
    RemoveAxis<Axis, Dims...>::type TensorIndex(const Tensor<Dims...> &src) {
        static_assert(Index < Tensor<Dims...>::Shape[Axis], "Index out of range for specified Axis!");
        using Source = Tensor<Dims...>;
        using Result = RemoveAxis<Axis, Dims...>::type;

        Result dst;
        ParForEach(Result::Size, [&](const size_t i) {
            auto dst_multi = Result::FlatToMulti(i);
            std::array<size_t, Source::Rank> src_multi{};
            size_t dst_d = 0;
            for (size_t d = 0; d < Source::Rank; ++d) {
                src_multi[d] = d == Axis ? Index : dst_multi[dst_d++];
            }
            dst.flat(i) = src.flat(Source::MultiToFlat(src_multi));
        });
        return dst;
    }


    // @doc: template<size_t Axis, size_t... Dims> RemoveAxis<Axis, Dims...>::type TensorGet(const Tensor<Dims...> &src, size_t idx)
    /**
     * Runtime-index read: extract the slice at position `idx` along `Axis`, returning a `Tensor` with that axis removed
     * Runtime counterpart to compile-time `TensorIndex`
     */
    template<size_t Axis, size_t... Dims>
    RemoveAxis<Axis, Dims...>::type TensorGet(const Tensor<Dims...> &src, size_t idx) {
        using Source = Tensor<Dims...>;
        using Result = RemoveAxis<Axis, Dims...>::type;
        Result dst;
        ParForEach(Result::Size, [&](const size_t i) {
            auto dst_multi = Result::FlatToMulti(i);
            std::array<size_t, Source::Rank> src_multi{};
            size_t dst_d = 0;
            for (size_t d = 0; d < Source::Rank; ++d)
                src_multi[d] = d == Axis ? idx : dst_multi[dst_d++];
            dst.flat(i) = src.flat(Source::MultiToFlat(src_multi));
        });
        return dst;
    }

    // @doc: template<size_t Axis, size_t... Dims> void TensorSet(Tensor<Dims...> &dst, size_t idx, const RemoveAxis<Axis, Dims...>::type &src)
    /**
     * Runtime-index write: assign `src` into the slice at position `idx` along `Axis` of `dst`
     * Plain-assignment counterpart to `TensorIndexApply` (no binary op needed)
     */
    template<size_t Axis, size_t... Dims>
    void TensorSet(Tensor<Dims...> &dst, size_t idx, const typename RemoveAxis<Axis, Dims...>::type &src) {
        using Dest = Tensor<Dims...>;
        using Slice = RemoveAxis<Axis, Dims...>::type;
        ParForEach(Slice::Size, [&](const size_t i) {
            auto src_multi = Slice::FlatToMulti(i);
            std::array<size_t, Dest::Rank> dst_multi{};
            size_t src_d = 0;
            for (size_t d = 0; d < Dest::Rank; ++d)
                dst_multi[d] = d == Axis ? idx : src_multi[src_d++];
            dst.flat(Dest::MultiToFlat(dst_multi)) = src.flat(i);
        });
    }


    // @doc: template<size_t Src, size_t Rank> struct MoveToLastPerm
    /**
     * Create member `std::array<size_t, Rank> value`, representing `Rank` dimensions, permuted such that `src` is the *last* index
     * (Pass to `PermuteFromArray`)
     */
    template<size_t Src, size_t Rank>
    struct MoveToLastPerm {
        static constexpr auto value = [] {
            std::array<size_t, Rank> p{};
            size_t j = 0;
            for (size_t i = 0; i < Rank; ++i)
                if (i != Src) p[j++] = i;
            p[Rank - 1] = Src;
            return p;
        }();
    };

    // @doc: template<size_t Src, size_t Rank> struct MoveToFirstPerm
    /**
     * Create member `std::array<size_t, Rank> value`, representing `Rank` dimensions, permuted such that `src` is the *first* index
     * (Pass to `PermuteFromArray`)
     */
    template<size_t Src, size_t Rank>
    struct MoveToFirstPerm {
        static constexpr auto value = [] {
            std::array<size_t, Rank> p{};
            p[0] = Src;
            size_t j = 1;
            for (size_t i = 0; i < Rank; ++i)
                if (i != Src) p[j++] = i;
            return p;
        }();
    };


    // @doc: template<size_t N, size_t... Dims, typename Fn> auto BatchMap(const Tensor<N, Dims...> &src, Fn fn)
    /** Using a map `Fn` from `Tensor<Dims...>` to some other `Tensor` shape, map `Tensor<N, Dims...> -> PrependBatch<N, OutSlice>::type`, where `OutSlice` is the return `Tensor` type from `Fn` */
    template<size_t N, size_t... Dims, typename Fn>
    auto BatchMap(const Tensor<N, Dims...> &src, Fn fn) {
        // what is the type/shape of the function's return
        using OutSlice = std::invoke_result_t<Fn, Tensor<Dims...> >;
        // make sure that's a Tensor
        static_assert(IsTensor<OutSlice>, "BatchMap: fn must return a Tensor");
        // result is N many of these
        using ResultT = PrependBatch<N, OutSlice>::type;

        // get size of input and output tensors for mem copying
        constexpr size_t InSliceSize = Tensor<Dims...>::Size;
        constexpr size_t OutSliceSize = OutSlice::Size;

        // default construct result type
        ResultT result;
        // for each (batch)
        for (size_t b = 0; b < N; ++b) {
            Tensor<Dims...> slice;
            // fill a sub result tensor with source
            std::copy(src.data() + b * InSliceSize, src.data() + (b + 1) * InSliceSize, slice.data());
            // get result
            const auto out = fn(slice);
            // fill b-th result subtensor with result of fn
            std::copy(out.data(), out.data() + OutSliceSize, result.data() + b * OutSliceSize);
        }
        return result;
    }


    // @doc: template<size_t N, size_t... Dims, typename Fn> auto BatchZip(const Tensor<N, Dims...> &A, const Tensor<N, Dims...> &B, Fn fn)
    /** Using a map `Fn` from `(Tensor<Dims...>, Tensor<Dims...>)` to some other `Tensor` shape, map `(Tensor<N, Dims...>, Tensor<N, Dims...>) -> PrependBatch<N, OutSlice>::type`, where `OutSlice` is the return `Tensor` type from `Fn` */
    template<size_t N, size_t... Dims, typename Fn>
    auto BatchZip(const Tensor<N, Dims...> &A, const Tensor<N, Dims...> &B, Fn fn) {
        // same internal logic as BatchMap
        using OutSlice = std::invoke_result_t<Fn, Tensor<Dims...>, Tensor<Dims...> >;
        static_assert(IsTensor<OutSlice>, "BatchZip: fn must return a Tensor");
        using ResultT = PrependBatch<N, OutSlice>::type;
        constexpr size_t InSliceSize = Tensor<Dims...>::Size;
        constexpr size_t OutSliceSize = OutSlice::Size;
        ResultT result;
        for (size_t b = 0; b < N; ++b) {
            Tensor<Dims...> a_slice, b_slice;
            std::copy(A.data() + b * InSliceSize, A.data() + (b + 1) * InSliceSize, a_slice.data());
            std::copy(B.data() + b * InSliceSize, B.data() + (b + 1) * InSliceSize, b_slice.data());
            const auto out = fn(a_slice, b_slice);
            std::copy(out.data(), out.data() + OutSliceSize, result.data() + b * OutSliceSize);
        }
        return result;
    }


    // @doc: template<size_t SeqAxis = 0, size_t... Dims> void AddPositionalEncoding(Tensor<Dims...> &X)
    /**
     * Add sinusoidal positional encoding in-place to `X` along `SeqAxis` (default 0)
     * For each position `pos` along `SeqAxis` and embedding index `j` (row-major over the remaining axes):
     * `freq = 1 / 10000^(2*(j/2) / EmbSize)`
     * `PE[pos][j] = sin(pos * freq)` if `j` even, `cos(pos * freq)` if `j` odd
     * PE tensor depends only on template parameters — computed once as a `static` local and added via `zip_apply`
     */
    template<size_t SeqAxis = 0, size_t... Dims>
    void AddPositionalEncoding(Tensor<Dims...> &X) {
        using T = Tensor<Dims...>;
        static_assert(T::Rank >= 1, "AddPositionalEncoding requires Rank >= 1");
        static_assert(SeqAxis < T::Rank, "SeqAxis out of range");
        constexpr size_t SeqLen = T::Shape[SeqAxis];
        constexpr size_t EmbSize = T::Size / SeqLen;
        static const T pe = [] {
            T p;
            for (size_t flat = 0; flat < T::Size; ++flat) {
                const auto multi = T::FlatToMulti(flat);
                const size_t pos = multi[SeqAxis];
                size_t emb_idx = 0, stride = 1;
                for (size_t d = T::Rank; d-- > 0;) {
                    if (d == SeqAxis) continue;
                    emb_idx += multi[d] * stride;
                    stride *= T::Shape[d];
                }
                const float freq = 1.f / std::pow(
                                       10000.f,
                                       static_cast<float>(2 * (static_cast<double>(emb_idx) / 2.0)) / static_cast<float>
                                       (EmbSize));
                p.flat(flat) = (emb_idx % 2 == 0)
                                   ? std::sin(static_cast<float>(pos) * freq)
                                   : std::cos(static_cast<float>(pos) * freq);
            }
            return p;
        }();
        X.zip_apply(pe, Add{});
    }


    // @doc: template<size_t Axis, size_t... ADims, size_t... BDims> auto ConcatAxis(const Tensor<ADims...> &a, const Tensor<BDims...> &b)
    /** Apply a `FloatUnaryOp` inplace */
    template<size_t Axis, size_t... ADims, size_t... BDims>
    auto ConcatAxis(const Tensor<ADims...> &a, const Tensor<BDims...> &b) {
        constexpr size_t Rank = sizeof...(ADims);
        static_assert(sizeof...(BDims) == Rank, "ConcatAxis: rank must match");
        static_assert(Axis < Rank, "ConcatAxis: Axis out of range");
        constexpr std::array<size_t, Rank> ad = {ADims...};
        constexpr std::array<size_t, Rank> bd = {BDims...};
        static_assert([] {
            constexpr std::array<size_t, Rank> a = {ADims...}, b = {BDims...};
            for (size_t i = 0; i < Rank; ++i)
                if (i != Axis && a[i] != b[i]) return false;
            return true;
        }(), "ConcatAxis: all dims except Axis must match");

        using ResultT = ReplaceAxisDims<Axis, ad[Axis] + bd[Axis], Tensor<ADims...>>::type;
        ResultT out;

        constexpr size_t trailing = [] {
            constexpr std::array<size_t, Rank> d = {ADims...};
            size_t p = 1;
            for (size_t i = Axis + 1; i < Rank; ++i) p *= d[i];
            return p;
        }();
        constexpr size_t a_slab = ad[Axis] * trailing;
        constexpr size_t b_slab = bd[Axis] * trailing;
        constexpr size_t out_slab = a_slab + b_slab;
        constexpr size_t leading = Tensor<ADims...>::Size / a_slab;

        for (size_t i = 0; i < leading; ++i) {
            std::copy_n(a.data() + i * a_slab, a_slab, out.data() + i * out_slab);
            std::copy_n(b.data() + i * b_slab, b_slab, out.data() + i * out_slab + a_slab);
        }
        return out;
    }

    // @doc: template<size_t Axis, size_t SplitN, size_t... Dims> auto SplitAxis(const Tensor<Dims...> &src)
    /**
     * Collapse `Tensor<Dims...>` to rank-1 `Tensor<Size>`
     * Convenience wrapper around `Reshape<Size>`
     */
    template<size_t Axis, size_t SplitN, size_t... Dims>
    auto SplitAxis(const Tensor<Dims...> &src) {
        constexpr size_t Rank = sizeof...(Dims);
        static_assert(Axis < Rank, "SplitAxis: Axis out of range");
        constexpr std::array<size_t, Rank> dims = {Dims...};
        static_assert(SplitN <= dims[Axis], "SplitAxis: SplitN exceeds axis dim");

        using A = ReplaceAxisDims<Axis, SplitN, Tensor<Dims...>>::type;
        using B = ReplaceAxisDims<Axis, dims[Axis] - SplitN, Tensor<Dims...>>::type;
        A a;
        B b;

        constexpr size_t trailing = [] {
            constexpr std::array<size_t, Rank> d = {Dims...};
            size_t p = 1;
            for (size_t i = Axis + 1; i < Rank; ++i) p *= d[i];
            return p;
        }();
        constexpr size_t a_slab = SplitN * trailing;
        constexpr size_t b_slab = (dims[Axis] - SplitN) * trailing;
        constexpr size_t src_slab = a_slab + b_slab;
        constexpr size_t leading = Tensor<Dims...>::Size / src_slab;

        for (size_t i = 0; i < leading; ++i) {
            std::copy_n(src.data() + i * src_slab, a_slab, a.data() + i * a_slab);
            std::copy_n(src.data() + i * src_slab + a_slab, b_slab, b.data() + i * b_slab);
        }
        return std::pair{a, b};
    }
}
