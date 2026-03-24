#pragma once
#include <cassert>
#include <concepts>
#include <limits>
#include <ranges>
#include "Tensor.hpp"


namespace TTTN {
    template<typename F>
    concept FloatUnaryOp = std::regular_invocable<F, float> &&
                           std::same_as<std::invoke_result_t<F, float>, float>;

    template<typename F>
    concept FloatBinaryOp = std::regular_invocable<F, float, float> &&
                            std::same_as<std::invoke_result_t<F, float, float>, float>;


    // @doc: template<std::invocable<size_t> F> void ParForEach(size_t n, F f)
    /** Helper to parallel-execute `std::for_each` on a `std::views::iota(size_t{0}, n)`, calling `f` (something `std::invocable` on `size_t`) on each index */
    template<std::invocable<size_t> F>
    void ParForEach(size_t n, F f) {
        auto range = std::views::iota(size_t{0}, n);
        std::for_each(std::execution::par_unseq, range.begin(), range.end(), f);
    }

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

    // @doc: template<size_t... Dims> Tensor<Dims...>& operator+=(Tensor<Dims...>& a, const Tensor<Dims...>& b)
    /** Element-wise add-to, uses parallel functional `zip_apply` */
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

    // @doc: template<size_t... Dims> Tensor<Dims...> operator*(const Tensor<Dims...>& a, const Tensor<Dims...>& b)
    /** Hadamard (element-wise) product, uses parallel functional `zip` */
    template<size_t... Dims>
    Tensor<Dims...> operator*(const Tensor<Dims...> &a, const Tensor<Dims...> &b) {
        return a.zip(b, [](const float x, const float y) { return x * y; });
    }

    // @doc: template<size_t... Dims> Tensor<Dims...> operator/(const Tensor<Dims...>& a, const Tensor<Dims...>& b)
    /** Element-wise) division, uses parallel functional `zip` */
    template<size_t... Dims>
    Tensor<Dims...> operator/(const Tensor<Dims...> &a, const Tensor<Dims...> &b) {
        return a.zip(b, [](const float x, const float y) { return x / y; });
    }


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
        static constexpr std::array<size_t, Rank> perm_arr = {Perm...};

        Result dst;
        ParForEach(Result::Size, [&](size_t i) {
            auto dst_idx = Result::FlatToMulti(i);
            auto src_multi = [&]<size_t... Is>(std::index_sequence<Is...>) {
                std::array<size_t, Rank> m{};
                ((m[perm_arr[Is]] = dst_idx[Is]), ...);
                return m;
            }(std::make_index_sequence<Rank>{});
            dst.flat(i) = src.flat(Source::MultiToFlat(src_multi));
        });
        return dst;
    }


    // =========================================================================
    // Contraction
    // =========================================================================
    //
    // Contraction = Reduce ∘ zipWith(Map) ∘ Align
    //
    //   ContractionKernel  — unified compile-time index tables + compute
    //   InnerContract<N>   — N-inner-axis contraction, custom Map and Reduce
    //   ΣΠ<N> / SigmaPi<N> — InnerContract specialized to multiply + sum
    //   Contract<A,B>      — grand generalized: arbitrary axes, permute → InnerContract
    //   Collapse           — full-rank same-shape → scalar, custom Map and Reduce
    //
    // =========================================================================


    // @doc: template<size_t N, typename TA, typename TB> struct ContractionKernel
    /**
     * Unified compile-time index kernel. Specialized for `<N, Tensor<ADims...>, Tensor<BDims...>>`.
     * Compile-time `static constexpr`:
     *   - `RankA`, `RankB` — ranks of `A` and `B`
     *   - Asserts `N <= RankA && N <= RankB` and last `N` dims of `A` match first `N` dims of `B`
     *   - `A_Free = TensorSlice<0, RankA-N, ADims...>::type`
     *   - `B_Free = TensorSlice<N, RankB-N, BDims...>::type`
     *   - `Contracted = TensorSlice<RankA-N, N, ADims...>::type`
     *   - `ResultType = TensorConcat<A_Free, B_Free>::type`
     *   - `struct { std::array<size_t, Contracted::Size> a, b; } offsets` — flat-index offset into `A` and `B` for every contracted position; precomputed once per `(N, ADims, BDims)` and shared across all `(Map, Reduce)` variants
     *   - `b_free_size`, `contracted_size` — compile-time constants used by `InnerContract` to compute per-output base offsets as `O(1)` arithmetic (`base_a = (o / b_free_size) * contracted_size`, `base_b = o % b_free_size`) rather than a precomputed table — the compiler strength-reduces these to multiply-shift at `-O2`
     * **These pay real dividends for [TrainableTensorNetwork](./src/TrainableTensorNetwork.hpp) training schedules. Any weight `Tensor`'s `Dot`s, `Matmul`s, and `Outer`s (*in forward and backward passes*) are saved structs, and the runtime computations are parallelized and vectorized, following known, saved paths**
     */
    // @doc: struct ContractionKernel<size_t N, typename TA, typename TB>
    /**
     * Unified contraction kernel: compile-time index tables parameterized purely on tensor shapes
     *
     * `Contraction = Reduce ∘ zipWith(Map) ∘ Align`
     *
     * Aligns the last `N` axes of `TA` with the first `N` axes of `TB`.
     * Index tables are independent of `Map` and `Reduce` — one kernel instance serves every
     * operation variant for a given pair of shapes.
     * `InnerContract` reads these tables and supplies the operation types.
     *
     * Precomputes two compile-time tables:
     *   - `offsets.{a,b}` — flat-index contribution of each contracted position for A and B
     *   - `bases.{a,b}`   — free-dimension base offset in A and B for each output index
     *
     * `bases` is computed without intermediate multi-index arrays (running-remainder style)
     * to stay within the compiler's constexpr evaluation step budget for large tensors.
     *
     * These tables pay real dividends for training schedules: all forward and backward
     * contractions follow known, precomputed paths — fully parallelizable and vectorizable.
     */
    template<size_t N, typename TA, typename TB>
    struct ContractionKernel;

    template<size_t N, size_t... ADims, size_t... BDims>
    struct ContractionKernel<N, Tensor<ADims...>, Tensor<BDims...>> {
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

        using A_Free     = TensorSlice<0,        RankA - N, ADims...>::type;
        using B_Free     = TensorSlice<N,        RankB - N, BDims...>::type;
        using Contracted = TensorSlice<RankA - N, N,        ADims...>::type;
        using ResultType = TensorConcat<A_Free, B_Free>::type;

        static constexpr size_t a_free_rank = A_Free::Rank;
        static constexpr size_t b_free_rank = B_Free::Rank;

        // contracted-index → flat offsets in A and B
        static constexpr auto offsets = [] {
            constexpr auto sa = ComputeStrides<ADims...>::value;
            constexpr auto sb = ComputeStrides<BDims...>::value;
            struct { std::array<size_t, Contracted::Size> a{}, b{}; } t;
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
        //
        // ResultType = TensorConcat<A_Free, B_Free>, so output index o factors cleanly:
        //   base_a(o) = (o / B_Free::Size) * Contracted::Size
        //   base_b(o) = o % B_Free::Size
        //
        // These are O(1) to compute at runtime, so no precomputed table is needed.
        // InnerContract computes them inline per output element.
        static constexpr size_t b_free_size    = B_Free::Size;
        static constexpr size_t contracted_size = Contracted::Size;
    };


    // @doc: template<size_t N, ..., FloatBinaryOp Map, FloatBinaryOp Reduce> auto InnerContract(const Tensor<ADims...>& A, const Tensor<BDims...>& B, float init, Map map, Reduce reduce)
    /**
     * N-inner-axis contraction with custom `Map` and `Reduce`
     * Aligns the last `N` axes of `A` with the first `N` axes of `B`
     * Reads precomputed `offsets` from `ContractionKernel`; base offsets computed inline as O(1) arithmetic
     * `result.flat(o) = Reduce_c map(A[A_Free(o), c], B[c, B_Free(o)])`, for `o ∈ [0, ResultType::Size)`
     */
    // @doc: template<size_t N, size_t... ADims, size_t... BDims, FloatBinaryOp Map, FloatBinaryOp Reduce> auto InnerContract(const Tensor<ADims...>& A, const Tensor<BDims...>& B, float init, Map map, Reduce reduce)
    /**
     * N-inner-axis contraction with custom `Map` and `Reduce`
     * Aligns the last `N` axes of `A` with the first `N` axes of `B`
     * Reads precomputed index tables from `ContractionKernel` — the kernel is instantiated
     * once per (N, shape-of-A, shape-of-B) and shared across all (Map, Reduce) variants
     * All named contractions (`ΣΠ`, `Dot`, `Matmul`, `Outer`, `Einsum`) route through here
     * `ΣΠ<N>(A, B)` == `InnerContract<N>(A, B, 0.f, multiply, plus)`
     */
    template<size_t N, size_t... ADims, size_t... BDims, FloatBinaryOp Map, FloatBinaryOp Reduce>
    auto InnerContract(const Tensor<ADims...> &A,
                       const Tensor<BDims...> &B,
                       float init,
                       Map map,
                       Reduce reduce) {
        using K = ContractionKernel<N, Tensor<ADims...>, Tensor<BDims...>>;
        using ResultType = typename K::ResultType;
        ResultType result;
        ParForEach(ResultType::Size, [&](size_t o) {
            const size_t base_a = (o / K::b_free_size) * K::contracted_size;
            const size_t base_b =  o % K::b_free_size;
            result.flat(o) = std::transform_reduce(
                std::execution::unseq,
                K::offsets.a.begin(), K::offsets.a.end(),
                K::offsets.b.begin(),
                init,
                reduce,
                [&](size_t oa, size_t ob) {
                    return map(A.flat(base_a + oa), B.flat(base_b + ob));
                });
        });
        return result;
    }


    // @doc: template<size_t N, size_t... ADims, size_t... BDims> auto ΣΠ(const Tensor<ADims...>& A, const Tensor<BDims...>& B)
    /**
     * `InnerContract<N>` specialized to `map = multiply`, `reduce = plus` — the classical sum-of-products
     * `result.flat(o) = Σ_c A[A_Free(o), c] * B[c, B_Free(o)]`
     * `SigmaPi<N>(A, B)` is an ASCII alias for `ΣΠ<N>(A, B)`
     */
    template<size_t N, size_t... ADims, size_t... BDims>
    auto ΣΠ(const Tensor<ADims...> &A, const Tensor<BDims...> &B) {
        return InnerContract<N>(
            A, B,
            0.0f,
            [](const float a, const float b) { return a * b; },
            std::plus<>{}
        );
    }

    // @doc: template<size_t N, size_t... ADims, size_t... BDims> auto SigmaPi(const Tensor<ADims...>& A, const Tensor<BDims...>& B)
    /** ASCII alias for `ΣΠ<N>(A, B)` */
    template<size_t N, size_t... ADims, size_t... BDims>
    auto SigmaPi(const Tensor<ADims...> &A, const Tensor<BDims...> &B) {
        return ΣΠ<N>(A, B);
    }


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


    // @doc: template<size_t I, size_t J, size_t... ADims, size_t... BDims> auto Einsum(const Tensor<ADims...>& A, const Tensor<BDims...>& B)
    /**
     * `ΣΠ`-contracts over single selected indices `I` and `J` from `A` and `B`, respectively
     * Permutes `A` to move axis `I` last, `B` to move axis `J` first, then calls `ΣΠ<1>`
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


    // @doc: template<size_t N> auto Dot(const Tensor<N>& A, const Tensor<N>& B)
    /** `ΣΠ<1>` on two `Tensor<N>`s — returns `Tensor<>` (rank-0 scalar) */
    template<size_t N>
    auto Dot(const Tensor<N> &A, const Tensor<N> &B) {
        return ΣΠ<1>(A, B);
    }

    // @doc: template<size_t M, size_t K, size_t N> auto Matmul(const Tensor<M,K>& A, const Tensor<K,N>& B)
    /** `ΣΠ<1>` on rank-2 tensors — returns `Tensor<M,N>` */
    template<size_t M, size_t K, size_t N>
    auto Matmul(const Tensor<M, K> &A, const Tensor<K, N> &B) {
        return ΣΠ<1>(A, B);
    }

    // @doc: template<size_t... ADims, size_t... BDims> auto Outer(const Tensor<ADims...>& A, const Tensor<BDims...>& B)
    /** `ΣΠ<0>`: contract nothing — returns `Tensor<ADims..., BDims...>` */
    template<size_t... ADims, size_t... BDims>
    auto Outer(const Tensor<ADims...> &A, const Tensor<BDims...> &B) {
        return ΣΠ<0>(A, B);
    }


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
            return Permute<(sizeof...(Dims) - 1 - I)...>(s);
        };
        return f(t, std::make_index_sequence<sizeof...(Dims)>{});
    }

        // @doc: template<size_t Axis, size_t... Dims> struct ReduceKernel
        /**
         * Shared kernel for all axis-reduction and broadcast operations
         * Compile-time `static constexpr`:
         *   - `axis_dim = SizeTemplateGet<Axis, Dims...>::value`
         *   - `axis_stride = Source::Strides[Axis]`
         *   - `std::array<size_t, Result::Size> bases` — flat index in `Source` for each output index with axis set to 0
         *   - `static constexpr size_t project(size_t i)` — flat index in `Result` for source flat index `i` (axis contribution stripped); closed-form `i - ((i / axis_stride) % axis_dim) * axis_stride`; no table, `axis_stride` compile-time so division compiles to multiply-shift
         */
    template<size_t Axis, size_t... Dims>
    struct ReduceKernel {
        using Source = Tensor<Dims...>;
        using Result = typename RemoveAxis<Axis, Dims...>::type;
        static constexpr size_t axis_dim    = SizeTemplateGet<Axis, Dims...>::value;
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

        static constexpr size_t project(size_t i) {
            const size_t after  = i % axis_stride;
            const size_t d_axis = (i / axis_stride) % axis_dim;
            const size_t before = i - after - d_axis * axis_stride;
            return before / axis_dim + after;
        }
    };

    // @doc: template<size_t Axis, FloatBinaryOp ReduceFn, size_t... Dims> typename RemoveAxis<Axis, Dims...>::type ReduceApply(const Tensor<Dims...>& src, float init, ReduceFn rfn)
    /**
     * Generalized axis reduction: collapses `Axis` by folding elements with `rfn(acc, val)` starting from `init`
     * `ReduceSum`  == `ReduceApply<Axis>(src, 0.f,  std::plus<float>{})`
     * `ReduceMax`  == `ReduceApply<Axis>(src, -inf, [](float a, float b){ return std::max(a,b); })`
     */
    template<size_t Axis, FloatBinaryOp ReduceFn, size_t... Dims>
    typename RemoveAxis<Axis, Dims...>::type ReduceApply(const Tensor<Dims...> &src, float init, ReduceFn rfn) {
        using K = ReduceKernel<Axis, Dims...>;
        auto k_range = std::views::iota(size_t{0}, K::axis_dim);
        typename K::Result dst;
        ParForEach(K::Result::Size, [&](size_t out_i) {
            dst.flat(out_i) = std::transform_reduce(
                std::execution::unseq,
                k_range.begin(), k_range.end(),
                init, rfn,
                [&](size_t k) { return src.flat(K::bases[out_i] + k * K::axis_stride); });
        });
        return dst;
    }

    // @doc: template<size_t Axis, size_t... Dims> typename RemoveAxis<Axis, Dims...>::type ReduceSum(const Tensor<Dims...>& src)
    /**
     * Reduce an axis with `Tensor` addition — `ReduceSum<P>(Tensor<P,Q>) -> Tensor<Q>`
     * Routes through `ReduceApply<Axis>(src, 0.f, std::plus<float>{})`
     */
    template<size_t Axis, size_t... Dims>
    typename RemoveAxis<Axis, Dims...>::type ReduceSum(const Tensor<Dims...> &src) {
        return ReduceApply<Axis>(src, 0.f, std::plus<float>{});
    }

    // @doc: template<size_t Axis, size_t N, size_t... Dims> InsertAxis<Axis, N, Dims...>::type Expand(const Tensor<Dims...>& src)
    /**
     * Dual of `ReduceSum`/`ReduceMax`: broadcasts a reduced tensor back up by repeating it `N` times along `Axis`
     * `Expand<0, 5>(Tensor<3>)` → `Tensor<5, 3>` — 5 copies stacked along axis 0
     * Uses `ReduceKernel::project` to map each output element to its source element
     */
    template<size_t Axis, size_t N, size_t... Dims>
    typename InsertAxis<Axis, N, Dims...>::type Expand(const Tensor<Dims...> &src) {
        using Full = typename InsertAxis<Axis, N, Dims...>::type;
        return [&]<size_t... FullDims>(std::type_identity<Tensor<FullDims...> >) {
            using K = ReduceKernel<Axis, FullDims...>;
            Tensor<FullDims...> result;
            ParForEach(Full::Size, [&](size_t i) {
                result.flat(i) = src.flat(K::project(i));
            });
            return result;
        }(std::type_identity<Full>{});
    }

    // @doc: template<size_t Axis, FloatBinaryOp F, size_t... Dims> Tensor<Dims...> BroadcastApply(const Tensor<Dims...>& A, const typename RemoveAxis<Axis, Dims...>::type& b, F f)
    /**
     * Apply binary `f(a_elem, b_elem) -> float` element-wise between `A` and `b` broadcast along `Axis`
     * For each element `i` of `A`: `result[i] = f(A[i], b[project(i)])`
     * `BroadcastAdd(A, b)` == `BroadcastApply<Axis>(A, b, std::plus<float>{})`
     */
    template<size_t Axis, FloatBinaryOp F, size_t... Dims>
    Tensor<Dims...> BroadcastApply(const Tensor<Dims...> &A, const typename RemoveAxis<Axis, Dims...>::type &b, F f) {
        using K = ReduceKernel<Axis, Dims...>;
        Tensor<Dims...> result;
        ParForEach(Tensor<Dims...>::Size, [&](size_t i) {
            result.flat(i) = f(A.flat(i), b.flat(K::project(i)));
        });
        return result;
    }

    // @doc: template<size_t Axis, size_t... Dims> Tensor<Dims...> BroadcastAdd(const Tensor<Dims...>& A, const typename RemoveAxis<Axis, Dims...>::type& b)
    /** Add a reduced tensor to all `Axis` slices of `A` — convenience wrapper over `BroadcastApply<Axis>(A, b, std::plus<float>{})` */
    template<size_t Axis, size_t... Dims>
    Tensor<Dims...> BroadcastAdd(const Tensor<Dims...> &A, const typename RemoveAxis<Axis, Dims...>::type &b) {
        return BroadcastApply<Axis>(A, b, std::plus<float>{});
    }

    // @doc: template<size_t Axis, FloatBinaryOp ReduceFn, FloatBinaryOp ApplyFn, size_t... Dims> Tensor<Dims...> ReduceBroadcast(const Tensor<Dims...>& src, float init, ReduceFn rfn, ApplyFn afn)
    /**
     * Compose `ReduceApply` + `BroadcastApply` in one call: reduce along `Axis` with `rfn`, then broadcast the result back with `afn`
     * `ReduceBroadcast<Axis>(src, init, rfn, afn)` == `BroadcastApply<Axis>(src, ReduceApply<Axis>(src, init, rfn), afn)`
     * Powers `Softmax`: two calls — `(max, exp(a-m))` then `(sum, e/s)`
     */
    template<size_t Axis, FloatBinaryOp ReduceFn, FloatBinaryOp ApplyFn, size_t... Dims>
    Tensor<Dims...> ReduceBroadcast(const Tensor<Dims...> &src, float init, ReduceFn rfn, ApplyFn afn) {
        return BroadcastApply<Axis>(src, ReduceApply<Axis>(src, init, rfn), afn);
    }

    // @doc: template<size_t Axis, size_t... Dims> typename RemoveAxis<Axis, Dims...>::type ReduceMean(const Tensor<Dims...>& src)
    /** Reduce an axis with `Tensor` averaging — `ReduceMean<P>(Tensor<P,Q>) -> Tensor<Q>` */
    template<size_t Axis, size_t... Dims>
    typename RemoveAxis<Axis, Dims...>::type ReduceMean(const Tensor<Dims...> &src) {
        constexpr float inv = 1.f / static_cast<float>(SizeTemplateGet<Axis, Dims...>::value);
        return ReduceSum<Axis>(src) * inv;
    }

    // @doc: template<size_t Axis, size_t... Dims> typename RemoveAxis<Axis, Dims...>::type ReduceMax(const Tensor<Dims...>& src)
    /**
     * Reduce an axis with `Tensor` maxing — `ReduceMax<P>(Tensor<P,Q>) -> Tensor<Q>`
     * Routes through `ReduceApply<Axis>(src, -inf, std::max)`
     */
    template<size_t Axis, size_t... Dims>
    typename RemoveAxis<Axis, Dims...>::type ReduceMax(const Tensor<Dims...> &src) {
        return ReduceApply<Axis>(src, -std::numeric_limits<float>::infinity(),
                                 [](float a, float b) { return std::max(a, b); });
    }

    // @doc: template<size_t Axis, size_t... Dims> typename RemoveAxis<Axis, Dims...>::type TensorIndex(const Tensor<Dims...>& src, size_t idx)
    /**
     * Extract the `idx`-th `RemoveAxis<Axis, Dims...>::type` sub-`Tensor` from `Tensor<Dims...> src` on the `Axis` axis
     * Essentially fills new `Tensor` with values from `src` by looping through dimensions in `Rank`, but passing `idx` for `Axis` dimension on all values
     */
    template<size_t Axis, size_t... Dims>
    typename RemoveAxis<Axis, Dims...>::type TensorIndex(const Tensor<Dims...> &src, size_t idx) {
        using Source = Tensor<Dims...>;
        using Result = typename RemoveAxis<Axis, Dims...>::type;

        Result dst;
        for (size_t i = 0; i < Result::Size; ++i) {
            auto dst_multi = Result::FlatToMulti(i);
            std::array<size_t, Source::Rank> src_multi{};
            size_t dst_d = 0;
            for (size_t d = 0; d < Source::Rank; ++d) {
                src_multi[d] = d == Axis ? idx : dst_multi[dst_d++];
            }
            dst.flat(i) = src.flat(Source::MultiToFlat(src_multi));
        }
        return dst;
    }

    // @doc: template<size_t Axis, FloatBinaryOp F, size_t... Dims> void TensorIndexApply(Tensor<Dims...>& dst, size_t idx, const typename RemoveAxis<Axis, Dims...>::type& src, F f)
    /** Apply binary `f(existing, incoming) -> float` to each element of the `idx`-th slice of `dst` along `Axis` using the corresponding element of `src` */
    template<size_t Axis, FloatBinaryOp F, size_t... Dims>
    void TensorIndexApply(Tensor<Dims...> &dst, size_t idx,
                          const typename RemoveAxis<Axis, Dims...>::type &src, F f) {
        using Dest  = Tensor<Dims...>;
        using Slice = typename RemoveAxis<Axis, Dims...>::type;
        for (size_t i = 0; i < Slice::Size; ++i) {
            auto src_multi = Slice::FlatToMulti(i);
            std::array<size_t, Dest::Rank> dst_multi{};
            size_t src_d = 0;
            for (size_t d = 0; d < Dest::Rank; ++d)
                dst_multi[d] = d == Axis ? idx : src_multi[src_d++];
            const size_t flat = Dest::MultiToFlat(dst_multi);
            dst.flat(flat) = f(dst.flat(flat), src.flat(i));
        }
    }


    // =========================================================================
    // Generalized Contract and Collapse
    // =========================================================================


    // @doc: struct MoveAxesToEnd<size_t Rank, size_t... Axes>
    /** Compile-time permutation that moves the listed `Axes` to the end, free axes first in original order */
    template<size_t Rank, size_t... Axes>
    struct MoveAxesToEnd {
        static constexpr auto value = [] {
            std::array<size_t, Rank> p{};
            bool is_contract[Rank] = {false};
            ((is_contract[Axes] = true), ...);
            size_t out = 0;
            for (size_t i = 0; i < Rank; ++i)
                if (!is_contract[i])
                    p[out++] = i;
            ((p[out++] = Axes), ...);
            return p;
        }();
    };

    // @doc: struct MoveAxesToFront<size_t Rank, size_t... Axes>
    /** Compile-time permutation that moves the listed `Axes` to the front, free axes last in original order */
    template<size_t Rank, size_t... Axes>
    struct MoveAxesToFront {
        static constexpr auto value = [] {
            std::array<size_t, Rank> p{};
            bool is_contract[Rank] = {false};
            ((is_contract[Axes] = true), ...);
            size_t out = 0;
            for (size_t i = 0; i < Rank; ++i)
                if (is_contract[i])
                    p[out++] = i;
            for (size_t i = 0; i < Rank; ++i)
                if (!is_contract[i])
                    p[out++] = i;
            return p;
        }();
    };

    // @doc: template<std::size_t N> struct AxisList
    /**
     * Compile-time list of `N` axis indices, passed as a non-type template parameter to `Contract`
     * Usage: `Contract<AxisList<2>{{1, 3}}, AxisList<2>{{0, 2}}>(A, B, init, map, reduce)`
     */
    template<std::size_t N>
    struct AxisList {
        std::size_t data[N]{};

        consteval explicit AxisList(const std::size_t (&input)[N]) {
            for (std::size_t i = 0; i < N; ++i) data[i] = input[i];
        }
    };

    template<size_t... Dims, const auto &PermArray, std::size_t... Is>
    auto ApplyPerm(const Tensor<Dims...> &tensor, std::index_sequence<Is...>) {
        return Permute<PermArray[Is]...>(tensor);
    }


    // @doc: template<size_t... Dims> auto Contract(const Tensor<Dims...> &A, const Tensor<Dims...> &B)
    /**
     * `ΣΠ`-contracts *every* dimension of the congruent `A` and `B` `Tensor<Dims...>`s
     * `(⊕ ∘ ⊙)(A, B)` -- Hadamard product (⊙) of `A` and `B`, then flat accumulation (⊕) over the result
     * Returns `Tensor<>` (scalar)
     */
    // @doc: template<AxisList AAxes, AxisList BAxes, ..., FloatBinaryOp Map, FloatBinaryOp Reduce> auto Contract(const Tensor<ADims...>& A, const Tensor<BDims...>& B, float init, Map map, Reduce reduce)
    /**
     * Grand-generalized contraction over arbitrary axis sets
     * Permutes `A` and `B` to align the selected axes, then delegates to `InnerContract<N>`
     */
    // @doc: template<AxisList AAxes, AxisList BAxes, size_t... ADims, size_t... BDims, FloatBinaryOp Map, FloatBinaryOp Reduce> auto Contract(const Tensor<ADims...>& A, const Tensor<BDims...>& B, float init, Map map, Reduce reduce)
    /**
     * Grand generalized contraction over arbitrary named axes with custom `Map` and `Reduce`
     * `Contraction = Reduce ∘ zipWith(Map) ∘ Align`
     *
     * `AAxes` and `BAxes` name the axes to contract — they are permuted into inner position,
     * then `InnerContract<N>` is called with the given `Map` and `Reduce`
     *
     * Specializations:
     *   - `ΣΠ<N>(A, B)` == `Contract<last-N-of-A, first-N-of-B>(A, B, 0, mul, plus)`
     *   - `Collapse(A, B, init, m, r)` == `Contract<all, all>(A, B, init, m, r)` for same-shape A, B
     */
    template<AxisList AAxes, AxisList BAxes,
             size_t... ADims, size_t... BDims,
             FloatBinaryOp Map, FloatBinaryOp Reduce>
    auto Contract(const Tensor<ADims...> &A,
                  const Tensor<BDims...> &B,
                  float init,
                  Map map,
                  Reduce reduce) {
        static_assert(sizeof(AAxes.data) == sizeof(BAxes.data),
                      "Contract: axis counts for A and B must match");
        constexpr size_t N    = sizeof(AAxes.data) / sizeof(size_t);
        constexpr size_t ARank = sizeof...(ADims);
        constexpr size_t BRank = sizeof...(BDims);

        using PermA = decltype([]<size_t... Is>(std::index_sequence<Is...>) {
            return MoveAxesToEnd<ARank, AAxes.data[Is]...>{};
        }(std::make_index_sequence<N>{}));

        using PermB = decltype([]<size_t... Is>(std::index_sequence<Is...>) {
            return MoveAxesToFront<BRank, BAxes.data[Is]...>{};
        }(std::make_index_sequence<N>{}));

        auto permutedA = ApplyPerm<PermA::value>(A, std::make_index_sequence<ARank>{});
        auto permutedB = ApplyPerm<PermB::value>(B, std::make_index_sequence<BRank>{});

        return InnerContract<N>(permutedA, permutedB, init, map, reduce);
    }


    // @doc: template<size_t... Dims, FloatBinaryOp M, FloatBinaryOp R> float Collapse(const Tensor<Dims...>& A, const Tensor<Dims...>& B, float init, M m, R r)
    /**
     * Full-rank same-shape scalar reduction: `Reduce_i map(A[i], B[i])`
     * Implemented as a direct `std::transform_reduce` over flat data — no index tables needed
     * `Collapse(A, B, 0, mul, plus)` == Frobenius inner product; `Collapse(A, B, 0, abs_diff, plus)` == L1 distance
     */
    template<size_t... Dims, FloatBinaryOp M, FloatBinaryOp R>
    float Collapse(const Tensor<Dims...> &A,
                   const Tensor<Dims...> &B,
                   float init,
                   M m,
                   R r) {
        return std::transform_reduce(
            std::execution::unseq,
            A.data(), A.data() + Tensor<Dims...>::Size,
            B.data(),
            init, r, m
        );
    }

}
