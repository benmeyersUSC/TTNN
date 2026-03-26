#pragma once
#include "TensorOps.hpp"

namespace TTTN {
    // =========================================================================
    // Batched Contraction
    // =========================================================================
    //
    // BatchInnerContract<M, N>(A, B):
    //   M leading axes of A and B are "mapped" — must match, stay aligned,
    //   appear once in the output (diagonal, not crossed).
    //   Last N axes of A contract with the first N axes of B (after the M mapped axes).
    //
    //   A = Tensor<MapDims..., A_FreeDims..., ContractedDims...>
    //        |--- M ---|    |--- free_a ---|   |---- N ----|
    //
    //   B = Tensor<MapDims..., ContractedDims..., B_FreeDims...>
    //        |--- M ---|    |---- N ----|     |--- free_b ---|
    //
    //   Result = Tensor<MapDims..., A_FreeDims..., B_FreeDims...>
    //
    //   result[map, af, bf] = Σ_c  map( A[map, af, c], B[map, c, bf] )
    //
    //   BatchΣΠ<M, N> / BatchSigmaPi<M, N> — specialized to multiply + sum
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
    // @doc: template<size_t M, size_t N, typename TA, typename TB> struct BatchedContractionKernel
    /**
     * Compile-time index kernel for batched contraction.
     *
     * M leading axes are "mapped" (shared between A and B, preserved in output).
     * Last N axes of A contract with the first N post-map axes of B.
     *
     * Static constexpr:
     *   - `Mapped    = TensorSlice<0, M, ADims...>::type`
     *   - `A_Free    = TensorSlice<M, RankA-M-N, ADims...>::type`
     *   - `B_Free    = TensorSlice<M+N, RankB-M-N, BDims...>::type`
     *   - `Contracted = TensorSlice<RankA-N, N, ADims...>::type`
     *   - `ResultType = TensorConcat<Mapped, TensorConcat<A_Free, B_Free>>`
     *   - `Offsets.a`, `Offsets.b` — contracted-index → flat offset in A and B
     *   - `Map_Size`, `A_Free_Size`, `B_Free_Size`, `Contracted_Size`
     *   - `A_Inner_Size = A_Free_Size * Contracted_Size`
     *   - `B_Inner_Size = Contracted_Size * B_Free_Size`
     */
    template<size_t M, size_t N, typename TA, typename TB>
    struct BatchedContractionKernel;

    template<size_t M_Mapped, size_t N_Contracted, size_t... A_Dims, size_t... B_Dims>
    struct BatchedContractionKernel<M_Mapped, N_Contracted, Tensor<A_Dims...>, Tensor<B_Dims...> > {
        static constexpr size_t Rank_A = sizeof...(A_Dims);
        static constexpr size_t Rank_B = sizeof...(B_Dims);

        static_assert(M_Mapped + N_Contracted <= Rank_A, "M + N must not exceed rank of A");
        static_assert(M_Mapped + N_Contracted <= Rank_B, "M + N must not exceed rank of B");

        static_assert([] {
            constexpr std::array<size_t, Rank_A> a = {A_Dims...};
            constexpr std::array<size_t, Rank_B> b = {B_Dims...};
            for (size_t i = 0; i < M_Mapped; ++i) {
                if (a[i] != b[i]) {
                    return false;
                }
            }
            return true;
        }(), "THE FIRST M AXES OF A AND B MUST BE SAME SIZE");
        static_assert([] {
            constexpr std::array<size_t, Rank_A> a = {A_Dims...};
            constexpr std::array<size_t, Rank_B> b = {B_Dims...};
            for (size_t i = 0; i < N_Contracted; ++i)
                if (a[Rank_A - N_Contracted + i] != b[M_Mapped + i]) return false;
            return true;
        }(), "AXES [M+RANK_A-N, RANK_A) OF A MUST HAVE SAME SHAPE AS AXES [M+N, RANK_B) OF B");

        // INPUT SHAPES
        //  A: [ MapDims(M) | A_FreeDims(RankA-M-N)  | ContractedDims(N)     ]
        //  B: [ MapDims(M) | ContractedDims(N)      | B_FreeDims(RankB-M-N) ]

        // FIRST M AXES WILL BE MAPPED AXES (SAME IN A AND B)
        using Mapped = TensorSlice<0, M_Mapped, A_Dims...>::type;
        // CONTRACTED AXES ARE N LAST AXES OF A_DIMS
        using Contracted = TensorSlice<Rank_A - N_Contracted, N_Contracted, A_Dims...>::type;

        // A_FREE AXES START AT M, TAKE NEXT (RANK_A - M - N) AXES, FROM A_DIMS
        using A_Free = TensorSlice<M_Mapped, Rank_A - M_Mapped - N_Contracted, A_Dims...>::type;
        // B_FREE AXES START AT (M+N), TAKE NEXT (RANK_B - M - N) AXES, FROM B_DIMS
        using B_Free = TensorSlice<M_Mapped + N_Contracted, Rank_B - M_Mapped - N_Contracted, B_Dims...>::type;

        // RESULT SHAPE:
        // [ MapDims(M) | A_Free | B_Free ]
        using AB_Free = TensorConcat<A_Free, B_Free>::type;
        using ResultType = TensorConcat<Mapped, AB_Free>::type;

        // SUBTENSOR SIZES
        static constexpr size_t Map_Size = Mapped::Size;
        static constexpr size_t A_Free_Size = A_Free::Size;
        static constexpr size_t B_Free_Size = B_Free::Size;
        static constexpr size_t Contracted_Size = Contracted::Size;
        static constexpr size_t AB_Free_Size = A_Free_Size * B_Free_Size;

        // INNER SIZE: SIZE PER MAP DIM
        // ONE MAP DIM CONTAINS A FULL PRODUCT OF A_FREE AND A_CONTRACTED
        static constexpr size_t A_Inner_Size = A_Free_Size * Contracted_Size;
        // ONE MAP DIM CONTAINS A FULL PRODUCT OF B_CONTRACTED AND B_FREE
        static constexpr size_t B_Inner_Size = Contracted_Size * B_Free_Size;

        static constexpr size_t A_Free_Rank = A_Free::Rank;
        static constexpr size_t B_Free_Rank = B_Free::Rank;

        // OFFSET TABLES:
        /* Algorithm:
            * For each value in Contracted SubTensor underlying array
                * Contracted_Multi_Index = Map to Cartesian Product of Contracted dimensions
                * A_Offset = (Contracted_Multi_Index[I] * A_Contracted[I] + ...); I...N_Contracted
                * B_Offset = (Contracted_Multi_Index[I] * B_Contracted[I] + ...); I...N_Contracted
         */
        static constexpr auto Offsets = [] {
            struct {
                std::array<size_t, Contracted::Size> a{}, b{};
            } t;
            for (size_t c = 0; c < Contracted::Size; ++c) {
                const auto Contracted_Multi_Index = Contracted::FlatToMulti(c);
                size_t A_Offset = 0, B_Offset = 0;
                for (size_t i = 0; i < N_Contracted; ++i) {
                    A_Offset += Contracted_Multi_Index[i] * Tensor<A_Dims...>::Strides[M_Mapped + A_Free_Rank + i];
                    B_Offset += Contracted_Multi_Index[i] * Tensor<B_Dims...>::Strides[M_Mapped + i];
                }
                t.a[c] = A_Offset;
                t.b[c] = B_Offset;
            }
            return t;
        }();
    };


    // @doc: template<size_t M, size_t N, ...> auto BatchInnerContract(A, B, init, map, reduce)
    /**
     * Batched N-inner-axis contraction with M mapped (batch) leading axes.
     *
     * The first M axes of A and B must match and stay aligned in the output
     * (diagonal — not crossed as in a free product).
     *
     * `result[map, af, bf] = Reduce_c map( A[map, af, c], B[map, c, bf] )`
     *
     * Tag-param overload: `BatchInnerContract<M, N, Map, Reduce>(A, B)`
     */
    template<size_t M, size_t N, size_t... ADims, size_t... BDims,
        FloatBinaryOp Map, FloatBinaryOp Reduce>
    auto BatchInnerContract(const Tensor<ADims...> &A,
                            const Tensor<BDims...> &B,
                            float init,
                            Map map,
                            Reduce reduce) {
        using K = BatchedContractionKernel<M, N, Tensor<ADims...>, Tensor<BDims...> >;
        using ResultType = K::ResultType;
        ResultType result;

        // FOR ALL VALUES OF OUTPUT
        ParForEach(ResultType::Size, [&](size_t o) {
            // DECOMPOSE FLAT OUTPUT INDEX TO SUBTENSOR FLAT INDICES

            // Each value of the Mapped SubTensor has a full AB_Free SubTensor
            // Since we never even look into Mapped, we can abstractly think about it as a Tensor<Mapped::Size>
            const size_t map_flat = o / K::AB_Free_Size;
            // How many values remain after pulling out map_flat * AB_Free_Size values?
            const size_t ab_rem = o % K::AB_Free_Size;
            // Again, A_Free is a SubTensor which we can imagine as a Tensor<A_Free::Size>
            // Each value in A_Free has a full B_Free SubTensor
            const size_t a_free_flat = ab_rem / K::B_Free_Size;
            // However many values remain after factoring out Mapped and A_Free belong to B_Free
            const size_t b_free_flat = ab_rem % K::B_Free_Size;

            // Every dim of Mapped, in A, contains A_Free_Size * A_Contracted_Size values
            //       add to that A_Free_Flat copies of A_Contracted_Size values
            const size_t base_a = map_flat * K::A_Inner_Size + a_free_flat * K::Contracted_Size;
            // Every dim of Mapped, in B, contains B_Free_Size * B_Contracted_Size values
            //       add to that the remaining B_Free_Flat values
            const size_t base_b = map_flat * K::B_Inner_Size + b_free_flat;


            // (MAP(A_i, B_i) REDUCE ...), for i...Contracted::Size

            result.flat(o) = std::transform_reduce(
                std::execution::unseq,
                K::Offsets.a.begin(), K::Offsets.a.end(),
                K::Offsets.b.begin(),
                init,
                reduce,
                [&](const size_t oa, const size_t ob) {
                    return map(A.flat(base_a + oa), B.flat(base_b + ob));
                });
        });
        return result;
    }

    // Tag-param overload: BatchInnerContract<M, N, Map, Reduce>(A, B)
    template<size_t M, size_t N, typename Map, typename Reduce,
        size_t... ADims, size_t... BDims>
        requires FloatBinaryOp<Map> && FloatBinaryOp<Reduce> &&
                 std::default_initializable<Map> && std::default_initializable<Reduce> &&
                 requires { { Reduce::identity } -> std::convertible_to<float>; }
    auto BatchInnerContract(const Tensor<ADims...> &A, const Tensor<BDims...> &B) {
        return BatchInnerContract<M, N>(A, B, Reduce::identity, Map{}, Reduce{});
    }


    // @doc: template<size_t M, size_t N, ...> auto BatchΣΠ(A, B)
    /**
     * `BatchInnerContract<M, N, Mul, Add>` — batched sum-of-products.
     *
     * `result[map, af, bf] = Σ_c A[map, af, c] * B[map, c, bf]`
     *
     * M=1, N=1 on rank-3 tensors = batched matrix multiply.
     */
    template<size_t M, size_t N, size_t... ADims, size_t... BDims>
    auto BatchΣΠ(const Tensor<ADims...> &A, const Tensor<BDims...> &B) {
        return BatchInnerContract<M, N, Mul, Add>(A, B);
    }

    // @doc: template<size_t M, size_t N, ...> auto BatchSigmaPi(A, B)
    /** ASCII alias for `BatchΣΠ<M, N>(A, B)` */
    template<size_t M, size_t N, size_t... ADims, size_t... BDims>
    auto BatchSigmaPi(const Tensor<ADims...> &A, const Tensor<BDims...> &B) {
        return BatchΣΠ<M, N>(A, B);
    }

    template<size_t N, size_t... ADims, size_t... BDims,
        FloatBinaryOp Map, FloatBinaryOp Reduce>
    // @doc: template<size_t N, size_t... ADims, size_t... BDims, FloatBinaryOp Map, FloatBinaryOp Reduce> auto InnerContract(const Tensor<ADims...>& A, const Tensor<BDims...>& B, float init, Map map, Reduce reduce)
    /**
     * N-inner-axis contraction with custom `Map` and `Reduce` (lambda form)
     * Aligns the last `N` axes of `A` with the first `N` axes of `B`
     * `result.flat(o) = Reduce_c map(A[A_Free(o), c], B[c, B_Free(o)])`, for `o ∈ [0, ResultType::Size)`
     * **Tag-param overload**: `InnerContract<N, Map, Reduce>(A, B)` — `Reduce::identity` used as init; requires monoid `Reduce`
     */
    auto InnerContract(const Tensor<ADims...> &A,
                       const Tensor<BDims...> &B,
                       float init, Map map, Reduce reduce) {
        return BatchInnerContract<0, N>(A, B, init, map, reduce);
    }

    // Tag-param overload
    template<size_t N, typename Map, typename Reduce,
        size_t... ADims, size_t... BDims>
        requires FloatBinaryOp<Map> && FloatBinaryOp<Reduce> &&
                 std::default_initializable<Map> && std::default_initializable<Reduce> &&
                 requires { { Reduce::identity } -> std::convertible_to<float>; }
    auto InnerContract(const Tensor<ADims...> &A, const Tensor<BDims...> &B) {
        return BatchInnerContract<0, N, Map, Reduce>(A, B);
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
