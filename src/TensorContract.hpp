#pragma once
#include <Accelerate/Accelerate.h>
#include "TensorOps.hpp"

namespace TTTN {
    template<size_t... Axes>
    struct AxisList {
        static constexpr std::array<size_t, sizeof...(Axes)> data = {Axes...};

        // --- helper: check for duplicates ---
        static constexpr bool has_duplicates() {
            for (size_t i = 0; i < data.size(); ++i)
                for (size_t j = i + 1; j < data.size(); ++j)
                    if (data[i] == data[j]) return true;
            return false;
        }
    };

    template<size_t Rank, AxisList BatchAxes, AxisList ContractAxes, bool LeftContract>
    struct BC_Permute {
    private:
        static constexpr bool valid_indices(const auto &arr) {
            for (size_t i = 0; i < arr.size(); ++i)
                if (arr[i] >= Rank) return false;
            return true;
        }

    public:
        static_assert(valid_indices(BatchAxes.data), "BatchAxes index >= Rank");
        static_assert(valid_indices(ContractAxes.data), "ContractAxes index >= Rank");
        static_assert(!BatchAxes.has_duplicates(), "BatchAxes contains duplicate indices");
        static_assert(!ContractAxes.has_duplicates(), "ContractAxes contains duplicate indices");

        static constexpr bool disjoint() {
            for (size_t i = 0; i < BatchAxes.data.size(); ++i)
                for (size_t j = 0; j < ContractAxes.data.size(); ++j)
                    if (BatchAxes.data[i] == ContractAxes.data[j]) return false;
            return true;
        }

        static_assert(disjoint(), "BatchAxes and ContractAxes must be disjoint");

        static constexpr auto value = [] {
            constexpr size_t BatchSize = BatchAxes.data.size();
            constexpr size_t ContractSize = ContractAxes.data.size();
            static_assert(Rank >= BatchSize + ContractSize, "Batch size + Contract size is greater than Rank");

            std::array<size_t, Rank> p{};

            bool is_batch[Rank] = {};
            bool is_contract[Rank] = {};

            for (size_t i = 0; i < BatchSize; ++i) {
                is_batch[BatchAxes.data[i]] = true;
            }

            for (size_t i = 0; i < ContractSize; ++i) {
                is_contract[ContractAxes.data[i]] = true;
            }

            // loop through batch, assign at indices
            // loop through contract, assign at indices


            size_t batch_i = 0;
            size_t contract_i, free_i;
            if constexpr (LeftContract) {
                contract_i = BatchSize;
                free_i = BatchSize + ContractSize;
            } else {
                free_i = BatchSize;
                contract_i = BatchSize + (Rank - BatchSize - ContractSize);
            }

            for (size_t i = 0; i < Rank; ++i) {
                if (is_batch[i]) {
                    p[batch_i++] = i;
                } else if (is_contract[i]) {
                    p[contract_i++] = i;
                } else {
                    p[free_i++] = i;
                }
            }

            return p;
        }();
    };

    // Permutation that moves axis Src to the last position, shifting others left.
    // e.g. MoveToLastPerm<0, 3>::value = {1, 2, 0}
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

    // Permutation that moves axis Src to position 0, shifting others right.
    // e.g. MoveToFirstPerm<2, 3>::value = {2, 0, 1}
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

    // Type-based: PermuteFromHolder<SomeType>(t, idx_seq)
    // SomeType must have a static constexpr .value array member
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

    // Value-based: PermuteFromArray<some_constexpr_array>(t, idx_seq)
    // Uses C++20 auto NTTP for passing constexpr arrays directly
    template<auto Perm, size_t... I, size_t... Dims>
    auto PermuteFromArray(const Tensor<Dims...> &t, std::index_sequence<I...>) {
        return Permute<Perm[I]...>(t);
    }


    ; // =========================================================================
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

    template<size_t M, size_t N, typename TA, typename TB>
    struct BatchedContractionKernel;

    // @doc: template<AxisList AAxes, AxisList BAxes, ..., FloatBinaryOp Map, FloatBinaryOp Reduce> auto Contract(const Tensor<ADims...>& A, const Tensor<BDims...>& B, float init, Map map, Reduce reduce)
    /**
     * Grand-generalized contraction over arbitrary axis sets (lambda form)
     * Permutes `A` and `B` to align the selected axes, then delegates to `InnerContract<N>`
     * **Tag-param overload**: `Contract<AAxes, BAxes, Map, Reduce>(A, B)` — `Reduce::identity` used as init
     */
    template<size_t M_Batched, size_t N_Contracted, size_t... A_Dims, size_t... B_Dims>
    struct BatchedContractionKernel<M_Batched, N_Contracted, Tensor<A_Dims...>, Tensor<B_Dims...> > {
        static constexpr size_t Rank_A = sizeof...(A_Dims);
        static constexpr size_t Rank_B = sizeof...(B_Dims);

        static_assert(M_Batched + N_Contracted <= Rank_A, "M + N must not exceed rank of A");
        static_assert(M_Batched + N_Contracted <= Rank_B, "M + N must not exceed rank of B");

        static_assert([] {
            constexpr std::array<size_t, Rank_A> a = {A_Dims...};
            constexpr std::array<size_t, Rank_B> b = {B_Dims...};
            for (size_t i = 0; i < M_Batched; ++i) {
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
                if (a[Rank_A - N_Contracted + i] != b[M_Batched + i]) return false;
            return true;
        }(), "AXES [M+RANK_A-N, RANK_A) OF A MUST HAVE SAME SHAPE AS AXES [M+N, RANK_B) OF B");

        // INPUT SHAPES
        //  A: [ MapDims(M) | A_FreeDims(RankA-M-N)  | ContractedDims(N)     ]
        //  B: [ MapDims(M) | ContractedDims(N)      | B_FreeDims(RankB-M-N) ]

        // FIRST M AXES WILL BE MAPPED AXES (SAME IN A AND B)
        using Batched = TensorSlice<0, M_Batched, A_Dims...>::type;
        // CONTRACTED AXES ARE N LAST AXES OF A_DIMS
        using Contracted = TensorSlice<Rank_A - N_Contracted, N_Contracted, A_Dims...>::type;

        // A_FREE AXES START AT M, TAKE NEXT (RANK_A - M - N) AXES, FROM A_DIMS
        using A_Free = TensorSlice<M_Batched, Rank_A - M_Batched - N_Contracted, A_Dims...>::type;
        // B_FREE AXES START AT (M+N), TAKE NEXT (RANK_B - M - N) AXES, FROM B_DIMS
        using B_Free = TensorSlice<M_Batched + N_Contracted, Rank_B - M_Batched - N_Contracted, B_Dims...>::type;

        // RESULT SHAPE:
        // [ MapDims(M) | A_Free | B_Free ]
        using AB_Free = TensorConcat<A_Free, B_Free>::type;
        using ResultType = TensorConcat<Batched, AB_Free>::type;

        // SUBTENSOR SIZES
        static constexpr size_t Batch_Size = Batched::Size;
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
                * A_Offset = (Contracted_Multi_Index[I] * A_Contracted_Strides[I] + ...); I...N_Contracted
                * B_Offset = (Contracted_Multi_Index[I] * B_Contracted_Strides[I] + ...); I...N_Contracted
         */
        static constexpr auto Offsets = [] {
            struct {
                std::array<size_t, Contracted::Size> a{}, b{};
            } t;
            for (size_t c = 0; c < Contracted::Size; ++c) {
                const auto Contracted_Multi_Index = Contracted::flat_to_multi(c);
                size_t A_Offset = 0, B_Offset = 0;
                for (size_t i = 0; i < N_Contracted; ++i) {
                    A_Offset += Contracted_Multi_Index[i] * Tensor<A_Dims...>::Strides[M_Batched + A_Free_Rank + i];
                    B_Offset += Contracted_Multi_Index[i] * Tensor<B_Dims...>::Strides[M_Batched + i];
                }
                t.a[c] = A_Offset;
                t.b[c] = B_Offset;
            }
            return t;
        }();
    };


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

    // BLAS overload: BatchInnerContract<M, N>(A, B, 0, Mul{}, Add{})
    // Dispatches to cblas_sgemm (Apple AMX) instead of the scalar transform_reduce loop.
    // Selected by overload resolution for Mul+Add — the dominant case in training.
    template<size_t M, size_t N, size_t... ADims, size_t... BDims>
    auto BatchInnerContract(const Tensor<ADims...> &A,
                            const Tensor<BDims...> &B,
                            float /*init*/, Mul, Add) {
        using K = BatchedContractionKernel<M, N, Tensor<ADims...>, Tensor<BDims...> >;
        typename K::ResultType result;
        const float *a_ptr = A.data();
        const float *b_ptr = B.data();
        float *c_ptr = result.data();
        // for each batch
        for (size_t m = 0; m < K::Batch_Size; ++m) {
            cblas_sgemm(
                // we have row major flat vectors, we do not want either Tensor transposed
                CblasRowMajor, CblasNoTrans, CblasNoTrans,
                // Free sizes (we ignore batch because we're in one) and contracted size
                static_cast<int>(K::A_Free_Size),
                static_cast<int>(K::B_Free_Size),
                static_cast<int>(K::Contracted_Size),
                1.f, // (A x B) is scaled by 1.f
                // [Data Pointer, Loading Dimensions (num cols for a row-major matrix)]
                // Data pointer:
                //      start of A.data() + batch_num * A_Inner_Size
                //      each increment of the batch (m) brings us to the front of a whole new copy of an unbatched A
                // Loading dimensions:
                //      we are working with Tensors, so to imagine this simply as a matmul
                //      we say that each row of A[m..., free..., contracted...] contains K::Contracted_Size 'columns'
                //      the row index is which free element we're on, the col is which contracted element
                a_ptr + m * K::A_Inner_Size, static_cast<int>(K::Contracted_Size),
                // data pointer for B is also straightforward
                // B loading dimensions:
                //      for B[m..., contracted..., free...], this inner (m-th) virtual matrix is contracted x free
                //      so the row indexes the contracted elements, the column indexes the free elements
                b_ptr + m * K::B_Inner_Size, static_cast<int>(K::B_Free_Size),
                0.f, // C is scaled by 0.f to start
                // data pointer: each increment of batch (m) means a whole copy of AB_Free in the result
                // C loading dimensions:
                //      for C[m..., A_free..., B_free...], the m-th virtual matrix is A_free x B_free
                //      row indexes A_free elements, column indexes B_free elements
                c_ptr + m * K::AB_Free_Size, static_cast<int>(K::B_Free_Size));
        }
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


    template<size_t M, size_t N, size_t... ADims, size_t... BDims>
    auto BatchΣΠ(const Tensor<ADims...> &A, const Tensor<BDims...> &B) {
        return BatchInnerContract<M, N, Mul, Add>(A, B);
    }

    template<size_t M, size_t N, size_t... ADims, size_t... BDims>
    auto BatchSigmaPi(const Tensor<ADims...> &A, const Tensor<BDims...> &B) {
        return BatchΣΠ<M, N>(A, B);
    }

    // @doc: template<size_t N, size_t... ADims, size_t... BDims, FloatBinaryOp Map, FloatBinaryOp Reduce> auto InnerContract(const Tensor<ADims...>& A, const Tensor<BDims...>& B, float init, Map map, Reduce reduce)
    /**
     * N-inner-axis contraction with custom `Map` and `Reduce` (lambda form)
     * Aligns the last `N` axes of `A` with the first `N` axes of `B`
     * `result.flat(o) = Reduce_c map(A[A_Free(o), c], B[c, B_Free(o)])`, for `o ∈ [0, ResultType::Size)`
     * **Tag-param overload**: `InnerContract<N, Map, Reduce>(A, B)` — `Reduce::identity` used as init; requires monoid `Reduce`
     */
    template<size_t N, size_t... ADims, size_t... BDims,
        FloatBinaryOp Map, FloatBinaryOp Reduce>
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


    template<size_t N, size_t... ADims, size_t... BDims>
    auto ΣΠ(const Tensor<ADims...> &A, const Tensor<BDims...> &B) {
        return InnerContract<N, Mul, Add>(A, B);
    }

    // @doc: template<size_t N, size_t... ADims, size_t... BDims> auto ΣΠ(const Tensor<ADims...>& A, const Tensor<BDims...>& B)
    /**
     * `InnerContract<N, Mul, Add>` — classical sum-of-products
     * `result.flat(o) = Σ_c A[A_Free(o), c] * B[c, B_Free(o)]`
     * `SigmaPi<N>(A, B)` is an ASCII alias for `ΣΠ<N>(A, B)`
     */
    template<size_t N, size_t... ADims, size_t... BDims>
    auto SigmaPi(const Tensor<ADims...> &A, const Tensor<BDims...> &B) {
        return ΣΠ<N>(A, B);
    }


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

        using PermA = BC_Permute<ARank, AxisList<>{}, AAxes, false>;
        using PermB = BC_Permute<BRank, AxisList<>{}, BAxes, true>;

        auto permutedA = PermuteFromArray<PermA::value>(A, std::make_index_sequence<ARank>{});
        auto permutedB = PermuteFromArray<PermB::value>(B, std::make_index_sequence<BRank>{});

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
        // constexpr size_t RankA = sizeof...(ADims);
        // constexpr size_t RankB = sizeof...(BDims);
        return Contract<
            AxisList<I>{},
            AxisList<J>{},
            Mul,
            Add
        >(A, B);
    }


    // @doc: template<size_t N> auto Dot(const Tensor<N>& A, const Tensor<N>& B)
    /** `ΣΠ<1>` on two `Tensor<N>`s — returns `Tensor<>` (rank-0 scalar) */
    template<size_t N>
    auto Dot(const Tensor<N> &A, const Tensor<N> &B) {
        return Contract<
            AxisList<0>{},
            AxisList<0>{},
            Mul,
            Add
        >(A, B);
    }

    // @doc: template<size_t M, size_t K, size_t N> auto Matmul(const Tensor<M,K>& A, const Tensor<K,N>& B)
    /** `ΣΠ<1>` on rank-2 tensors — returns `Tensor<M,N>` */
    template<size_t M, size_t K, size_t N>
    auto Matmul(const Tensor<M, K> &A, const Tensor<K, N> &B) {
        return Contract<
            AxisList<1>{}, // K in A
            AxisList<0>{}, // K in B
            Mul,
            Add
        >(A, B);
    }

    // @doc: template<size_t... ADims, size_t... BDims> auto Outer(const Tensor<ADims...>& A, const Tensor<BDims...>& B)
    /** `ΣΠ<0>`: contract nothing — returns `Tensor<ADims..., BDims...>` */
    template<size_t... ADims, size_t... BDims>
    auto Outer(const Tensor<ADims...> &A, const Tensor<BDims...> &B) {
        return Contract<
            AxisList<>{}, // nothing
            AxisList<>{},
            Mul,
            Add
        >(A, B);
    }


    template<AxisList ABatchAxes, AxisList BBatchAxes,
        AxisList AContractAxes, AxisList BContractAxes,
        size_t... ADims, size_t... BDims,
        FloatBinaryOp Map, FloatBinaryOp Reduce>
    auto BatchContract(const Tensor<ADims...> &A,
                       const Tensor<BDims...> &B,
                       float init,
                       Map map,
                       Reduce reduce) {
        constexpr size_t BatchSize = sizeof(ABatchAxes.data) / sizeof(size_t);
        constexpr size_t InnerSize = sizeof(AContractAxes.data) / sizeof(size_t);

        static_assert(BatchSize == sizeof(BBatchAxes.data) / sizeof(size_t));
        static_assert(InnerSize == sizeof(BContractAxes.data) / sizeof(size_t));

        constexpr size_t A_Rank = sizeof...(ADims);
        constexpr size_t B_Rank = sizeof...(BDims);

        static_assert([] {
            for (size_t i = 0; i < BatchSize; ++i)
                for (size_t j = 0; j < InnerSize; ++j)
                    if (ABatchAxes.data[i] == AContractAxes.data[j])
                        return false;
            return true;
        }(), "A batch axes cannot overlap with contract axes");

        static_assert([] {
            for (size_t i = 0; i < BatchSize; ++i)
                for (size_t j = 0; j < InnerSize; ++j)
                    if (BBatchAxes.data[i] == BContractAxes.data[j])
                        return false;
            return true;
        }(), "B batch axes cannot overlap with contract axes");

        static_assert([] {
            constexpr std::array<size_t, A_Rank> a = {ADims...};
            constexpr std::array<size_t, B_Rank> b = {BDims...};
            for (size_t i = 0; i < BatchSize; ++i)
                if (a[ABatchAxes.data[i]] != b[BBatchAxes.data[i]])
                    return false;
            return true;
        }(), "Batch axes must match in size");

        static_assert([] {
            constexpr std::array<size_t, A_Rank> a = {ADims...};
            constexpr std::array<size_t, B_Rank> b = {BDims...};
            for (size_t i = 0; i < InnerSize; ++i)
                if (a[AContractAxes.data[i]] != b[BContractAxes.data[i]])
                    return false;
            return true;
        }(), "Contract axes must match in size");

        using PermA = BC_Permute<A_Rank, ABatchAxes, AContractAxes, false>;
        using PermB = BC_Permute<B_Rank, BBatchAxes, BContractAxes, true>;

        auto permA = PermuteFromArray<PermA::value>(A, std::make_index_sequence<A_Rank>{});
        auto permB = PermuteFromArray<PermB::value>(B, std::make_index_sequence<B_Rank>{});

        return BatchInnerContract<BatchSize, InnerSize>(
            permA, permB, init, map, reduce
        );
    }

    // Tag-param overload: BatchContract<ABatch, BBatch, AContract, BContract, Map, Reduce>(A, B)
    template<AxisList ABatchAxes, AxisList BBatchAxes,
        AxisList AContractAxes, AxisList BContractAxes,
        typename Map, typename Reduce,
        size_t... ADims, size_t... BDims>
        requires FloatBinaryOp<Map> && FloatBinaryOp<Reduce> &&
                 std::default_initializable<Map> && std::default_initializable<Reduce> &&
                 requires { { Reduce::identity } -> std::convertible_to<float>; }
    auto BatchContract(const Tensor<ADims...> &A, const Tensor<BDims...> &B) {
        return BatchContract<ABatchAxes, BBatchAxes, AContractAxes, BContractAxes>(
            A, B, Reduce::identity, Map{}, Reduce{});
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
                   M map,
                   R reduce) {
        return std::transform_reduce(
            std::execution::unseq,
            A.data(), A.data() + Tensor<Dims...>::Size,
            B.data(),
            init, reduce, map
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
