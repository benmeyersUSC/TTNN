#pragma once
#include "TensorOps.hpp"

namespace TTTN {

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
     *   - `b_free_size`, `contracted_size` — compile-time constants used by `InnerContract` to compute per-output base offsets as `O(1)` arithmetic ( `base_a = (o / b_free_size) * contracted_size`, `base_b = o % b_free_size`) rather than a precomputed table — the compiler strength-reduces these to multiply-shift at `-O2`
     * **These pay real dividends for [TrainableTensorNetwork](./src/TrainableTensorNetwork.hpp) training schedules. Any weight `Tensor`'s `Dot`s, `Matmul`s, and `Outer`s (*in forward and backward passes*) are saved structs, and the runtime computations are parallelized and vectorized, following known, saved paths**
     */
    template<size_t N, typename TA, typename TB>
    struct ContractionKernel;

    template<size_t N, size_t... ADims, size_t... BDims>
    struct ContractionKernel<N, Tensor<ADims...>, Tensor<BDims...> > {
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
        //
        // ResultType = TensorConcat<A_Free, B_Free>, so output index o factors cleanly:
        //   base_a(o) = (o / B_Free::Size) * Contracted::Size
        //   base_b(o) = o % B_Free::Size
        //
        // These are O(1) to compute at runtime, so no precomputed table is needed.
        // InnerContract computes them inline per output element.
        static constexpr size_t b_free_size = B_Free::Size;
        static constexpr size_t contracted_size = Contracted::Size;
    };


    // @doc: template<size_t N, size_t... ADims, size_t... BDims, FloatBinaryOp Map, FloatBinaryOp Reduce> auto InnerContract(const Tensor<ADims...>& A, const Tensor<BDims...>& B, float init, Map map, Reduce reduce)
    /**
     * N-inner-axis contraction with custom `Map` and `Reduce` (lambda form)
     * Aligns the last `N` axes of `A` with the first `N` axes of `B`
     * `result.flat(o) = Reduce_c map(A[A_Free(o), c], B[c, B_Free(o)])`, for `o ∈ [0, ResultType::Size)`
     * **Tag-param overload**: `InnerContract<N, Map, Reduce>(A, B)` — `Reduce::identity` used as init; requires monoid `Reduce`
     */
    template<size_t N, size_t... ADims, size_t... BDims, FloatBinaryOp Map, FloatBinaryOp Reduce>
    auto InnerContract(const Tensor<ADims...> &A,
                       const Tensor<BDims...> &B,
                       float init,
                       Map map,
                       Reduce reduce) {
        using K = ContractionKernel<N, Tensor<ADims...>, Tensor<BDims...> >;
        using ResultType = typename K::ResultType;
        ResultType result;
        ParForEach(ResultType::Size, [&](size_t o) {
            const size_t base_a = (o / K::b_free_size) * K::contracted_size;
            const size_t base_b = o % K::b_free_size;
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


    // Tag-param overload: InnerContract<N, Map, Reduce>(A, B) — uses Reduce::identity as init
    template<size_t N, typename Map, typename Reduce, size_t... ADims, size_t... BDims>
        requires FloatBinaryOp<Map> && FloatBinaryOp<Reduce> &&
                 std::default_initializable<Map> && std::default_initializable<Reduce> &&
                 requires { { Reduce::identity } -> std::convertible_to<float>; }
    auto InnerContract(const Tensor<ADims...> &A, const Tensor<BDims...> &B) {
        return InnerContract<N>(A, B, Reduce::identity, Map{}, Reduce{});
    }


    // @doc: template<size_t N, size_t... ADims, size_t... BDims> auto ΣΠ(const Tensor<ADims...>& A, const Tensor<BDims...>& B)
    /**
     * `InnerContract<N, Mul, Add>` — classical sum-of-products
     * `result.flat(o) = Σ_c A[A_Free(o), c] * B[c, B_Free(o)]`
     * `SigmaPi<N>(A, B)` is an ASCII alias for `ΣΠ<N>(A, B)`
     */
    template<size_t N, size_t... ADims, size_t... BDims>
    auto ΣΠ(const Tensor<ADims...> &A, const Tensor<BDims...> &B) {
        return InnerContract<N, Mul, Add>(A, B);
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


    // @doc: template<AxisList AAxes, AxisList BAxes, ..., FloatBinaryOp Map, FloatBinaryOp Reduce> auto Contract(const Tensor<ADims...>& A, const Tensor<BDims...>& B, float init, Map map, Reduce reduce)
    /**
     * Grand-generalized contraction over arbitrary axis sets (lambda form)
     * Permutes `A` and `B` to align the selected axes, then delegates to `InnerContract<N>`
     * **Tag-param overload**: `Contract<AAxes, BAxes, Map, Reduce>(A, B)` — `Reduce::identity` used as init
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
        constexpr size_t N = sizeof(AAxes.data) / sizeof(size_t);
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

    // Tag-param overload: Contract<AAxes, BAxes, Map, Reduce>(A, B) — uses Reduce::identity as init
    template<AxisList AAxes, AxisList BAxes, typename Map, typename Reduce,
        size_t... ADims, size_t... BDims>
        requires FloatBinaryOp<Map> && FloatBinaryOp<Reduce> &&
                 std::default_initializable<Map> && std::default_initializable<Reduce> &&
                 requires { { Reduce::identity } -> std::convertible_to<float>; }
    auto Contract(const Tensor<ADims...> &A, const Tensor<BDims...> &B) {
        return Contract<AAxes, BAxes>(A, B, Reduce::identity, Map{}, Reduce{});
    }


    // @doc: template<size_t... Dims, FloatBinaryOp M, FloatBinaryOp R> float Collapse(const Tensor<Dims...>& A, const Tensor<Dims...>& B, float init, M m, R r)
    /**
     * Full-rank same-shape scalar reduction: `Reduce_i map(A[i], B[i])`
     * Direct `std::transform_reduce` over flat data — no index tables needed
     * **Tag-param overload**: `Collapse<M, R>(A, B)` — `R::identity` used as init
     * `Collapse<Mul, Add>(A, B)` == Frobenius inner product; `Collapse<AbsDiff, Add>(A, B)` == L1 distance
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

    // Tag-param overload: Collapse<M, R>(A, B) — uses R::identity as init
    template<typename M, typename R, size_t... Dims>
        requires FloatBinaryOp<M> && FloatBinaryOp<R> &&
                 std::default_initializable<M> && std::default_initializable<R> &&
                 requires { { R::identity } -> std::convertible_to<float>; }
    float Collapse(const Tensor<Dims...> &A, const Tensor<Dims...> &B) {
        return Collapse(A, B, R::identity, M{}, R{});
    }
}
