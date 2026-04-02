#pragma once
#include <Accelerate/Accelerate.h>
#include "TensorOps.hpp"

namespace TTTN {
    // @doc: template<size_t... Axes> struct AxisList
    /**
     * Necessary wrapper around variadic `size_t` axis packs used in `Contract` and `BatchContract` to specify specific `Batch` and/or `Contract` axis indices, rather than passing the *count* of left (for `Batch`) or right (for `Contract`) axes
     * `static_assert`s no duplicate axis indices
     */
    template<size_t... Axes>
    struct AxisList {
        static constexpr std::array<size_t, sizeof...(Axes)> data = {Axes...};

        static constexpr bool has_duplicates() {
            for (size_t i = 0; i < data.size(); ++i)
                for (size_t j = i + 1; j < data.size(); ++j)
                    if (data[i] == data[j]) return true;
            return false;
        }
    };

    // @doc: template<size_t Rank, AxisList BatchAxes, AxisList ContractAxes> struct BC_Permute
    /**
     * Compile-time helper to generate permutation indices for a `Tensor`'s axes in `BatchInnerContract` form: `[Batch..., Free..., Contract...]`
     * `static_assert`s that:
     * `Batch` and `Contract` axes are `disjoint`
     * no indices in `Batch` or `Contract` lists are greater than `Rank`
     * `Batch` and `Contract` axes have no duplicates
     */
    template<size_t Rank, AxisList BatchAxes, AxisList ContractAxes>
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


            size_t batch_i = 0;
            size_t free_i = BatchSize;
            size_t contract_i = BatchSize + (Rank - BatchSize - ContractSize);

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


    template<size_t M, size_t N, typename TA, typename TB>
    struct BatchedContractionKernel;

    // @doc: template<size_t M_Batched, size_t N_Contracted, size_t... A_Dims, size_t... B_Dims> struct BatchedContractionKernel<M_Batched, N_Contracted, Tensor<A_Dims...>, Tensor<B_Dims...> >
    /** Unified contraction bookkeeping kernel, used compile-time compute convenient shapes and values used in two versions of `BatchInnerContract` (the functions through which every contraction operation are routed) */
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
                if (a[Rank_A - N_Contracted + i] != b[Rank_B - N_Contracted + i]) return false;
            return true;
        }(), "LAST N AXES OF A MUST HAVE SAME SHAPE AS LAST N AXES OF B");

        // INPUT SHAPES
        //  A: [ MapDims(M) | A_FreeDims(RankA-M-N)  | ContractedDims(N) ]
        //  B: [ MapDims(M) | B_FreeDims(RankB-M-N)  | ContractedDims(N) ]

        // FIRST M AXES WILL BE MAPPED AXES (SAME IN A AND B)
        using Batched = TensorSlice<0, M_Batched, A_Dims...>::type;
        // CONTRACTED AXES ARE N LAST AXES OF A_DIMS (OR B_DIMS)
        using Contracted = TensorSlice<Rank_A - N_Contracted, N_Contracted, A_Dims...>::type;

        // A_FREE AXES START AT M, TAKE NEXT (RANK_A - M - N) AXES, FROM A_DIMS
        using A_Free = TensorSlice<M_Batched, Rank_A - M_Batched - N_Contracted, A_Dims...>::type;
        // // B_FREE AXES START AT (M+N), TAKE NEXT (RANK_B - M - N) AXES, FROM B_DIMS
        using B_Free = TensorSlice<M_Batched, Rank_B - M_Batched - N_Contracted, B_Dims...>::type;

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
        // // ONE MAP DIM CONTAINS A FULL PRODUCT OF B_CONTRACTED AND B_FREE
        static constexpr size_t B_Inner_Size = B_Free_Size * Contracted_Size;

        static constexpr size_t A_Free_Rank = A_Free::Rank;
        static constexpr size_t B_Free_Rank = B_Free::Rank;
    };


    // @doc: template<size_t M, size_t N, size_t... ADims, size_t... BDims, FloatBinaryOp Map, FloatBinaryOp Reduce> auto BatchInnerContract(const Tensor<ADims...> &A, const Tensor<BDims...> &B, const float init, Map map, Reduce reduce)
    /**
     * Core primitive: all contractions become `BatchInnerContract`
     * See `BatchContractionKernel` for more details on implementation
     */
    template<size_t M, size_t N, size_t... ADims, size_t... BDims, FloatBinaryOp Map, FloatBinaryOp Reduce>
    auto BatchInnerContract(const Tensor<ADims...> &A, const Tensor<BDims...> &B, const float init, Map map,
                            Reduce reduce) {
        using K = BatchedContractionKernel<M, N, Tensor<ADims...>, Tensor<BDims...> >;
        using ResultType = K::ResultType;
        ResultType result;

        auto A_ptr = A.data();
        auto B_ptr = B.data();

        constexpr size_t TILE = 8;
        const size_t num_full_tiles = ResultType::Size / TILE;
        const size_t leftover = ResultType::Size % TILE;

        // process full tiles
        ParForEach(num_full_tiles, [&](const size_t t) {
            // which actual value in result are we (starting) at
            const size_t base = t * TILE;

            // values we're computing (accumulating)
            std::array<float, TILE> acc{};
            acc.fill(init);

            // Result space is Batch x AB_Free
            // if batches are rows, they're non archimedean, and free rows are just contiguous regions
            // then we can take the step to see one big vector:
            //      batches are contiguous regions within the vector...within each
            //          we have contiguous regions, each is a 'free row'
            //              in each free row contiguous array, we have a span which ride the contracted axis

            // so we need to add a factor for the batches * Inner, for the free positions * contracted size......

            for (size_t tile_idx = 0; tile_idx < TILE; ++tile_idx) {
                // offset: base of the tile (8-aligned) + lane (0-7 inclusive)
                const size_t o = base + tile_idx;

                // calculate batch by treating batches as independent rows in the resulting matrix, where each row
                // has a full AB_Free_Size worth of elements
                // NOTE: this single batch value is used to index both A and B
                //      THIS is what gives us batching: A and B are always being indexed at the same batch!
                const size_t batch = o / K::AB_Free_Size;
                const size_t rem = o % K::AB_Free_Size;

                // this result index, o, corresponds to an element [freeA, freeB]
                // it is freeA * Free_B_Size + freeB
                // so:
                const size_t free_a = rem / K::B_Free_Size; // how many full free B_Free tensors
                const size_t free_b = rem % K::B_Free_Size;

                auto a_ptr_tile = &A_ptr[batch * K::A_Inner_Size + free_a * K::Contracted_Size];
                auto b_ptr_tile = &B_ptr[batch * K::B_Inner_Size + free_b * K::Contracted_Size];
                for (size_t c = 0; c < K::Contracted_Size; ++c) {
                    acc[tile_idx] = reduce(acc[tile_idx], map(a_ptr_tile[c], b_ptr_tile[c]));
                }
                result[o] = acc[tile_idx];
            }
        });

        // process leftover elements
        if (leftover > 0) {
            const size_t base = num_full_tiles * TILE;
            std::array<float, TILE> acc{};
            acc.fill(init);
            for (size_t i = 0; i < leftover; ++i) {
                const size_t o = base + i;

                const size_t batch = o / K::AB_Free_Size;
                const size_t rem = o % K::AB_Free_Size;

                const size_t free_a = rem / K::B_Free_Size;
                const size_t free_b = rem % K::B_Free_Size;

                auto a_ptr_tile = &A_ptr[batch * K::A_Inner_Size + free_a * K::Contracted_Size];
                auto b_ptr_tile = &B_ptr[batch * K::B_Inner_Size + free_b * K::Contracted_Size];
                for (size_t c = 0; c < K::Contracted_Size; ++c) {
                    acc[i] = reduce(acc[i], map(a_ptr_tile[c], b_ptr_tile[c]));
                }
                result[o] = acc[i];
            }
        }

        return result;
    }


    // @doc: template<size_t M, size_t N, size_t... ADims, size_t... BDims> auto BatchInnerContract(const Tensor<ADims...> &A, const Tensor<BDims...> &B, float /*init*/, Mul, Add)
    /**
     * Specialized version of generalized `BatchInnerContract` for `Map=Mul` and `Reduce=Add` (most common use-case)
     * Uses `Apple Accelerate`'s `cblas_sgemm` function to unlock aggressive vectorization optimization for matrix multiplication
     * Extensive commenting in code
     */

    template<size_t M, size_t N, size_t... ADims, size_t... BDims>
    auto BatchInnerContract(const Tensor<ADims...> &A, const Tensor<BDims...> &B, float /*init*/, Mul, Add) {
        using K = BatchedContractionKernel<M, N, Tensor<ADims...>, Tensor<BDims...> >;
        typename K::ResultType result;
        const float *a_ptr = A.data();
        const float *b_ptr = B.data();
        float *c_ptr = result.data();
        // for each batch
        for (size_t m = 0; m < K::Batch_Size; ++m) {
            cblas_sgemm(
                // A is [free, contracted] row-major — no transpose; lda = Contracted_Size (cols per row)
                // B is [free, contracted] row-major — CblasTrans so sgemm reads it as [contracted, free]
                CblasRowMajor, CblasNoTrans, CblasTrans,
                // M = A_Free, N = B_Free, K = Contracted
                static_cast<int>(K::A_Free_Size),
                static_cast<int>(K::B_Free_Size),
                static_cast<int>(K::Contracted_Size),
                1.f,
                // A: [batch, free, contracted] — lda = Contracted_Size (cols per row)
                a_ptr + m * K::A_Inner_Size, static_cast<int>(K::Contracted_Size),
                // B: [batch, free, contracted] — ldb = Contracted_Size (cols per row, before transpose)
                b_ptr + m * K::B_Inner_Size, static_cast<int>(K::Contracted_Size),
                0.f, // C is scaled by 0.f to start
                // data pointer: each increment of batch (m) means a whole copy of AB_Free in the result
                // C loading dimensions:
                //      for C[m..., A_free..., B_free...], the m-th virtual matrix is A_free x B_free
                //      row indexes A_free elements, column indexes B_free elements
                c_ptr + m * K::AB_Free_Size, static_cast<int>(K::B_Free_Size));
        }
        return result;
    }


    // @doc: template<size_t M, size_t N, typename Map, typename Reduce, size_t... ADims, size_t... BDims> requires FloatBinaryOp<Map> && FloatBinaryOp<Reduce> && std::default_initializable<Map> && std::default_initializable<Reduce> && requires { { Reduce::identity } -> std::convertible_to<float>; } auto BatchInnerContract(const Tensor<ADims...> &A, const Tensor<BDims...> &B)
    /** Tag-parameter specialization of `BatchInnerContract`; calls `BatchInnerContract` */
    template<size_t M, size_t N, typename Map, typename Reduce, size_t... ADims, size_t... BDims> requires
        FloatBinaryOp<Map> && FloatBinaryOp<Reduce> && std::default_initializable<Map> && std::default_initializable<
            Reduce> && requires { { Reduce::identity } -> std::convertible_to<float>; }
    auto BatchInnerContract(const Tensor<ADims...> &A, const Tensor<BDims...> &B) {
        return BatchInnerContract<M, N>(A, B, Reduce::identity, Map{}, Reduce{});
    }


    // @doc: template<size_t M, size_t N, size_t... ADims, size_t... BDims> auto BatchΣΠ(const Tensor<ADims...> &A, const Tensor<BDims...> &B)
    /** Convenience wrapper for sum of product specialization of `BatchInnerContract` */
    template<size_t M, size_t N, size_t... ADims, size_t... BDims>
    auto BatchΣΠ(const Tensor<ADims...> &A, const Tensor<BDims...> &B) {
        return BatchInnerContract<M, N, Mul, Add>(A, B);
    }


    // @doc: template<size_t M, size_t N, size_t... ADims, size_t... BDims> auto BatchSigmaPi(const Tensor<ADims...> &A, const Tensor<BDims...> &B)
    /** ASCII overload of `BatchΣΠ` */
    template<size_t M, size_t N, size_t... ADims, size_t... BDims>
    auto BatchSigmaPi(const Tensor<ADims...> &A, const Tensor<BDims...> &B) {
        return BatchΣΠ<M, N>(A, B);
    }

    // @doc: template<size_t N, size_t... ADims, size_t... BDims, FloatBinaryOp Map, FloatBinaryOp Reduce> auto InnerContract(const Tensor<ADims...> &A, const Tensor<BDims...> &B, float init, Map map, Reduce reduce)
    /** Convenience wrapper for non-batched calls to generalized `BatchInnerContract` */
    template<size_t N, size_t... ADims, size_t... BDims, FloatBinaryOp Map, FloatBinaryOp Reduce>
    auto InnerContract(const Tensor<ADims...> &A, const Tensor<BDims...> &B, float init, Map map, Reduce reduce) {
        return BatchInnerContract<0, N>(A, B, init, map, reduce);
    }

    // @doc: template<size_t N, typename Map, typename Reduce, size_t... ADims, size_t... BDims> requires FloatBinaryOp<Map> && FloatBinaryOp<Reduce> && std::default_initializable<Map> && std::default_initializable<Reduce> && requires { { Reduce::identity } -> std::convertible_to<float>; } auto InnerContract(const Tensor<ADims...> &A, const Tensor<BDims...> &B)
    /** tag-param specialization of `InnerContract` */
    template<size_t N, typename Map, typename Reduce, size_t... ADims, size_t... BDims> requires
        FloatBinaryOp<Map> && FloatBinaryOp<Reduce> && std::default_initializable<Map> && std::default_initializable<
            Reduce> && requires { { Reduce::identity } -> std::convertible_to<float>; }
    auto InnerContract(const Tensor<ADims...> &A, const Tensor<BDims...> &B) {
        return BatchInnerContract<0, N, Map, Reduce>(A, B);
    }

    // @doc: template<size_t N, size_t... ADims, size_t... BDims> auto ΣΠ(const Tensor<ADims...> &A, const Tensor<BDims...> &B)
    /** Convenience wrapper for non-batched sum of products contraction */
    template<size_t N, size_t... ADims, size_t... BDims>
    auto ΣΠ(const Tensor<ADims...> &A, const Tensor<BDims...> &B) {
        return InnerContract<N, Mul, Add>(A, B);
    }

    // @doc: template<size_t N, size_t... ADims, size_t... BDims> auto SigmaPi(const Tensor<ADims...> &A, const Tensor<BDims...> &B)
    /** ASCII convenience wrapper for `ΣΠ` */
    template<size_t N, size_t... ADims, size_t... BDims>
    auto SigmaPi(const Tensor<ADims...> &A, const Tensor<BDims...> &B) {
        return ΣΠ<N>(A, B);
    }


    // @doc: template<AxisList AAxes, AxisList BAxes, size_t... ADims, size_t... BDims, FloatBinaryOp Map, FloatBinaryOp Reduce> auto Contract(const Tensor<ADims...> &A, const Tensor<BDims...> &B, float init, Map map, Reduce reduce)
    /**
     * Convenience wrapper for `BatchContract` (and `BatchInnerContract`) for non-Batched, arbitrary-axes contractions
     * Second-most general function in [TensorContract.hpp](src/TensorContract.hpp)
     */
    template<AxisList AAxes, AxisList BAxes, size_t... ADims, size_t... BDims, FloatBinaryOp Map, FloatBinaryOp Reduce>
    auto Contract(const Tensor<ADims...> &A, const Tensor<BDims...> &B, float init, Map map, Reduce reduce) {
        static_assert(sizeof(AAxes.data) == sizeof(BAxes.data),
                      "Contract: axis counts for A and B must match");
        constexpr size_t N = sizeof(AAxes.data) / sizeof(size_t);
        constexpr size_t ARank = sizeof...(ADims);
        constexpr size_t BRank = sizeof...(BDims);

        using PermA = BC_Permute<ARank, AxisList<>{}, AAxes>;
        using PermB = BC_Permute<BRank, AxisList<>{}, BAxes>;

        auto permutedA = PermuteFromArray<PermA::value>(A, std::make_index_sequence<ARank>{});
        auto permutedB = PermuteFromArray<PermB::value>(B, std::make_index_sequence<BRank>{});

        return InnerContract<N>(permutedA, permutedB, init, map, reduce);
    }

    // @doc: template<AxisList AAxes, AxisList BAxes, typename Map, typename Reduce, size_t... ADims, size_t... BDims> requires FloatBinaryOp<Map> && FloatBinaryOp<Reduce> && std::default_initializable<Map> && std::default_initializable<Reduce> && requires { { Reduce::identity } -> std::convertible_to<float>; } auto Contract(const Tensor<ADims...> &A, const Tensor<BDims...> &B)
    /** tag-param specialization of `Contract` */
    template<AxisList AAxes, AxisList BAxes, typename Map, typename Reduce, size_t... ADims, size_t... BDims> requires
        FloatBinaryOp<Map> && FloatBinaryOp<Reduce> && std::default_initializable<Map> && std::default_initializable<
            Reduce> && requires { { Reduce::identity } -> std::convertible_to<float>; }
    auto Contract(const Tensor<ADims...> &A, const Tensor<BDims...> &B) {
        return Contract<AAxes, BAxes>(A, B, Reduce::identity, Map{}, Reduce{});
    }

    // @doc: template<size_t I, size_t J, size_t... ADims, size_t... BDims> auto Einsum(const Tensor<ADims...> &A, const Tensor<BDims...> &B)
    /** Variant of `ΣΠ` for single-axis sum of product contractions (specified by `I` and `J` for `A` and `B`, respectively) */
    template<size_t I, size_t J, size_t... ADims, size_t... BDims>
    auto Einsum(const Tensor<ADims...> &A, const Tensor<BDims...> &B) {
        static_assert(
            SizeTemplateGet<I, ADims...>::value == SizeTemplateGet<J, BDims...>::value,
            "axis I of A and axis J of B must have the same size");
        return Contract<
            AxisList<I>{},
            AxisList<J>{},
            Mul,
            Add
        >(A, B);
    }

    // @doc: template<AxisList ABatchAxes, AxisList BBatchAxes, size_t I, size_t J, size_t... ADims, size_t... BDims> auto BatchEinsum(const Tensor<ADims...> &A, const Tensor<BDims...> &B)
    /** Batch version of `Einsum` */
    template<AxisList ABatchAxes, AxisList BBatchAxes, size_t I, size_t J, size_t... ADims, size_t... BDims>
    auto BatchEinsum(const Tensor<ADims...> &A, const Tensor<BDims...> &B) {
        static_assert(
            SizeTemplateGet<I, ADims...>::value == SizeTemplateGet<J, BDims...>::value,
            "axis I of A and axis J of B must have the same size");
        return BatchContract<ABatchAxes, BBatchAxes,
            AxisList<I>{},
            AxisList<J>{},
            Mul,
            Add
        >(A, B);
    }

    // @doc: template<size_t N> auto Dot(const Tensor<N> &A, const Tensor<N> &B)
    /** Convenience wrapper for sum of product full-rank contraction (dot product) of two Rank-1 `Tensor`s */
    template<size_t N>
    auto Dot(const Tensor<N> &A, const Tensor<N> &B) {
        return Contract<
            AxisList<0>{},
            AxisList<0>{},
            Mul,
            Add
        >(A, B);
    }

    // @doc: template<size_t M, size_t K, size_t N> auto Matmul(const Tensor<M, K> &A, const Tensor<K, N> &B)
    /**
     * Convenience wrapper for sum of product contraction (matrix multiplication) of two Rank-2 `Tensor`s
     * NOTE: expects Axis 1 of `A` to be contracted with Axis 0 of `B`, per `Matmul` convention
     */
    template<size_t M, size_t K, size_t N>
    auto Matmul(const Tensor<M, K> &A, const Tensor<K, N> &B) {
        return Contract<
            AxisList<1>{}, // K in A
            AxisList<0>{}, // K in B
            Mul,
            Add
        >(A, B);
    }


    // @doc: template<size_t... ADims, size_t... BDims> auto Outer(const Tensor<ADims...> &A, const Tensor<BDims...> &B)
    /** Convenience wrapper for sum of product no-rank contraction (outer product) of two Rank-1 `Tensor`s */
    template<size_t... ADims, size_t... BDims>
    auto Outer(const Tensor<ADims...> &A, const Tensor<BDims...> &B) {
        return Contract<
            AxisList<>{}, // nothing
            AxisList<>{},
            Mul,
            Add
        >(A, B);
    }


    // @doc: template<AxisList ABatchAxes, AxisList BBatchAxes, AxisList AContractAxes, AxisList BContractAxes, size_t... ADims, size_t... BDims, FloatBinaryOp Map, FloatBinaryOp Reduce> auto BatchContract(const Tensor<ADims...> &A, const Tensor<BDims...> &B, float init, Map map, Reduce reduce)
    /**
     * Most general function in [TensorContract.hpp](src/TensorContract.hpp)
     * Arbitrary `Align` (specified by `Batch` and `Contract` axes), arbitrary `Map` (to zip aligned elements of `A` and `B`), arbitrary `Reduce` (to fold down `Map` results along contracted axes)
     */
    template<AxisList ABatchAxes, AxisList BBatchAxes, AxisList AContractAxes, AxisList BContractAxes, size_t... ADims,
        size_t... BDims, FloatBinaryOp Map, FloatBinaryOp Reduce>
    auto BatchContract(const Tensor<ADims...> &A, const Tensor<BDims...> &B, float init, Map map, Reduce reduce) {
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

        using PermA = BC_Permute<A_Rank, ABatchAxes, AContractAxes>;
        using PermB = BC_Permute<B_Rank, BBatchAxes, BContractAxes>;

        auto permA = PermuteFromArray<PermA::value>(A, std::make_index_sequence<A_Rank>{});
        auto permB = PermuteFromArray<PermB::value>(B, std::make_index_sequence<B_Rank>{});

        return BatchInnerContract<BatchSize, InnerSize>(
            permA, permB, init, map, reduce
        );
    }

    // @doc: template<AxisList ABatchAxes, AxisList BBatchAxes, AxisList AContractAxes, AxisList BContractAxes, typename Map, typename Reduce, size_t... ADims, size_t... BDims> requires FloatBinaryOp<Map> && FloatBinaryOp<Reduce> && std::default_initializable<Map> && std::default_initializable<Reduce> && requires { { Reduce::identity } -> std::convertible_to<float>; } auto BatchContract(const Tensor<ADims...> &A, const Tensor<BDims...> &B)
    /** Tag-param specialization of `BatchContract` */
    template<AxisList ABatchAxes, AxisList BBatchAxes, AxisList AContractAxes, AxisList BContractAxes, typename Map,
        typename Reduce, size_t... ADims, size_t... BDims> requires
        FloatBinaryOp<Map> && FloatBinaryOp<Reduce> && std::default_initializable<Map> && std::default_initializable<
            Reduce> && requires { { Reduce::identity } -> std::convertible_to<float>; }
    auto BatchContract(const Tensor<ADims...> &A, const Tensor<BDims...> &B) {
        return BatchContract<ABatchAxes, BBatchAxes, AContractAxes, BContractAxes>(
            A, B, Reduce::identity, Map{}, Reduce{});
    }


    // @doc: template<size_t... Dims, FloatBinaryOp M, FloatBinaryOp R> float Collapse(const Tensor<Dims...> &A, const Tensor<Dims...> &B, float init, M map, R reduce)
    /**
     * Specialization for contraction over *all* axes
     * Known as ***Frobenius Inner Product***, it is a generalization of `Dot` or inner product for arbitrarily-shaped `Tensor`s
     */
    template<size_t... Dims, FloatBinaryOp M, FloatBinaryOp R>
    float Collapse(const Tensor<Dims...> &A, const Tensor<Dims...> &B, float init, M map, R reduce) {
        return std::transform_reduce(
            std::execution::unseq,
            A.data(), A.data() + Tensor<Dims...>::Size,
            B.data(),
            init, reduce, map
        );
    }

    // @doc: template<typename M, typename R, size_t... Dims> requires FloatBinaryOp<M> && FloatBinaryOp<R> && std::default_initializable<M> && std::default_initializable<R> && requires { { R::identity } -> std::convertible_to<float>; } float Collapse(const Tensor<Dims...> &A, const Tensor<Dims...> &B)
    /** Tag-param specialization of `Collapse` */
    template<typename M, typename R, size_t... Dims> requires
        FloatBinaryOp<M> && FloatBinaryOp<R> && std::default_initializable<M> && std::default_initializable<R> &&
        requires { { R::identity } -> std::convertible_to<float>; }
    float Collapse(const Tensor<Dims...> &A, const Tensor<Dims...> &B) {
        return Collapse(A, B, R::identity, M{}, R{});
    }
}

