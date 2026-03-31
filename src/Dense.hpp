#pragma once
#include "TensorContract.hpp"
#include "TensorReduce.hpp"
#include "TTTN_ML.hpp"
#include "NetworkUtil.hpp"

namespace TTTN {
    // DENSE BLOCK
    // Weights, bias, activation.
    // Implements the Block interface for a fully-connected layer with activation.
    // InputTensor and OutputTensor can be any rank — this is the fully general version.
    //
    // W = Tensor<OutDims..., InDims...>   linear map from In-space to Out-space
    // b = Tensor<OutDims...>              bias in Out-space
    //
    // Forward:  z = ΣΠ<N_in>(W, x) + b        contracts W's last N_in axes with x (generalised matvec)
    //           a = z.map(Act{})
    //
    // Backward: delta_z  = delta_A ⊙ Act::prime(a)           peel off activation derivative → dL/dZ
    //           dW      += ΣΠ<0>(delta_z, a_prev)             outer product → Tensor<OutDims...,InDims...>
    //           dB      += delta_z
    //           dL/dx    = ΣΠ<N_out>(W_T, delta_z)            generalises W^T · delta_z; W_T = Tensor<InDims...,OutDims...>
    //
    // caller must ZeroGrad() before the first Backward in each training step

    template<typename InT, typename OutT, ActivationOp Act_ = Linear>
    class DenseMDBlock;

    template<size_t... InDims, size_t... OutDims, ActivationOp Act_>
    class DenseMDBlock<Tensor<InDims...>, Tensor<OutDims...>, Act_> {
    public:
        using InputTensor = Tensor<InDims...>;
        using OutputTensor = Tensor<OutDims...>;
        using Act = Act_;

    private:
        static constexpr size_t N_in = sizeof...(InDims);
        static constexpr size_t N_out = sizeof...(OutDims);
        using W_Type = Tensor<OutDims..., InDims...>;

        Param<W_Type> W_;
        Param<OutputTensor> b_;

    public:
        // @doc: auto all_params()
        /** Returns `std::tie(W_, b_)`; TTN drives `ZeroGrad`, `Update`, `Save`, `Load` from this */
        auto all_params() { return std::tie(W_, b_); }
        auto all_params() const { return std::tie(W_, b_); }

        // @doc: DenseMDBlock()
        /** Xavier-initializes `W` */
        DenseMDBlock() { XavierInitMD(W_.value, InputTensor::Size, OutputTensor::Size); }

        // @doc: OutputTensor Forward(const InputTensor& x) const
        /** ######### */
        OutputTensor Forward(const InputTensor &x) const {
            return Map<Act>(ΣΠ<N_in>(W_.value, x) + b_.value);
        }

        // @doc: InputTensor Backward(const OutputTensor& delta_A, const OutputTensor& a, const InputTensor& a_prev)
        /** ######### */
        InputTensor Backward(const OutputTensor &delta_A, const OutputTensor &a,
                             const InputTensor &a_prev) {
            const auto delta_z = delta_A.zip(a, [](float g, float ai) { return g * Act::prime(ai); });

            W_.grad += ΣΠ<0>(delta_z, a_prev);
            b_.grad += delta_z;

            const auto W_T = PermuteFromArray<SwapNDims<N_out, N_in>::value>(
                W_.value, std::make_index_sequence<N_out + N_in>{}
            );
            return ΣΠ<N_out>(W_T, delta_z);
        }

        template<size_t Batch>
        Tensor<Batch, OutDims...> BatchedForward(const Tensor<Batch, InDims...> &X) const {
            return Map<Act>(BroadcastMap<0, Add>(ΣΠ<N_in>(X, W_.value), b_.value));
        }

        template<size_t Batch>
        Tensor<Batch, InDims...> BatchedBackward(const Tensor<Batch, OutDims...> &delta_A,
                                                 const Tensor<Batch, OutDims...> &a,
                                                 const Tensor<Batch, InDims...> &a_prev) {
            const float inv_batch = 1.f / static_cast<float>(Batch);
            const auto delta_z = delta_A.zip(a, [](float g, float ai) { return g * Act::prime(ai); });

            W_.grad += Contract<AxisList<0>{}, AxisList<0>{}, Mul, Add>(delta_z, a_prev) * inv_batch;
            b_.grad += Reduce<0, Add>(delta_z) * inv_batch;

            const auto W_T = PermuteFromArray<SwapNDims<N_out, N_in>::value>(
                W_.value, std::make_index_sequence<N_out + N_in>{});
            return ΣΠ<N_out>(delta_z, W_T);
        }
    };


    // DenseMD recipe: specify the output tensor type; input is inferred at chain time via Resolve.
    //
    // Usage:
    //   NetworkBuilder<
    //       Input<3, 4>,                              // Tensor<3,4> input
    //       DenseMD<Tensor<5>>,                       // → Tensor<5>
    //       DenseMD<Tensor<2, 3>, ReLU>               // → Tensor<2,3>
    //   >::type net;
    template<typename OutT, ActivationOp Act_ = Linear>
    struct DenseMD {
        using OutputTensor = OutT;
        template<typename InputT>
        using Resolve = DenseMDBlock<InputT, OutT, Act_>;
    };

    // Dense<N, Act>: rank-1 shorthand for DenseMD<Tensor<N>, Act>
    // @doc: using Dense = DenseMD<Tensor<N>, Act_>
    /** `Dense<128, ReLU>`, `Dense<10, Sigmoid>`, `Dense<10>` (defaults to `Linear`) */
    template<size_t N, ActivationOp Act_ = Linear>
    using Dense = DenseMD<Tensor<N>, Act_>;


    // ═══════════════════════════════════════════════════════════════════════════
    // MAP-DENSE: Dense applied independently along leading "map" axes
    // ═══════════════════════════════════════════════════════════════════════════
    //
    // Splits the input tensor into:
    //   - N_map leading "map" axes (preserved, mapped over independently)
    //   - remaining trailing "contract" axes (transformed by the Dense)
    //
    // Input:  Tensor<MapDims..., ContractDims...>
    // Output: Tensor<MapDims..., PartOutDims...>
    // W:      Tensor<PartOutDims..., ContractDims...>
    // b:      Tensor<PartOutDims...>
    //
    // Forward:  Map<Act>(ΣΠ<N_contract>(X, W_T) + broadcast(b))
    //           ΣΠ naturally maps over the leading MapDims — no per-slice loop.
    //
    // Backward: dW = ΣΠ<N_map>(swap(δz), X)       sum outer products over map positions
    //           dB = Σ_{map} δz[map]              sum over map positions
    //           dX = ΣΠ<N_out>(δz, W)             upstream gradient, maps over MapDims
    //
    // This is the "per-token FFN" in transformer architectures, generalized to any
    // number of map axes and any-rank embedding shapes.

    namespace detail {
        // SplitAt<N, Dims...>: split dimension list at position N into head and tail
        template<size_t N, typename Collected, typename Remaining>
        struct SplitAtImpl;

        template<size_t... Collected, size_t... Remaining>
        struct SplitAtImpl<0,
                    std::integer_sequence<size_t, Collected...>,
                    std::integer_sequence<size_t, Remaining...> > {
            using head = std::integer_sequence<size_t, Collected...>;
            using tail = std::integer_sequence<size_t, Remaining...>;
        };

        template<size_t N, size_t... Collected, size_t D0, size_t... Rest>
            requires (N > 0)
        struct SplitAtImpl<N,
                    std::integer_sequence<size_t, Collected...>,
                    std::integer_sequence<size_t, D0, Rest...> > {
            using next = SplitAtImpl<N - 1,
                std::integer_sequence<size_t, Collected..., D0>,
                std::integer_sequence<size_t, Rest...> >;
            using head = typename next::head;
            using tail = typename next::tail;
        };

        template<size_t N, size_t... Dims>
        struct SplitAt {
            using impl = SplitAtImpl<N,
                std::integer_sequence<size_t>,
                std::integer_sequence<size_t, Dims...> >;
            using head = typename impl::head;
            using tail = typename impl::tail;
        };

        template<typename Seq>
        struct SeqToTensor;

        template<size_t... Ds>
        struct SeqToTensor<std::integer_sequence<size_t, Ds...> > {
            using type = Tensor<Ds...>;
        };

        template<typename A, typename B>
        struct ConcatSeqs;

        template<size_t... As, size_t... Bs>
        struct ConcatSeqs<std::integer_sequence<size_t, As...>,
                    std::integer_sequence<size_t, Bs...> > {
            using type = std::integer_sequence<size_t, As..., Bs...>;
        };

        template<typename Seq>
        struct SeqProduct;

        template<>
        struct SeqProduct<std::integer_sequence<size_t> > {
            static constexpr size_t value = 1;
        };

        template<size_t D0, size_t... Rest>
        struct SeqProduct<std::integer_sequence<size_t, D0, Rest...> > {
            static constexpr size_t value = D0 * SeqProduct<std::integer_sequence<size_t, Rest...> >::value;
        };

        template<size_t N, typename T>
        struct PrependOnes;

        template<size_t... Dims>
        struct PrependOnes<0, Tensor<Dims...> > {
            using type = Tensor<Dims...>;
        };

        template<size_t N, size_t... Dims>
        struct PrependOnes<N, Tensor<Dims...> > {
            using type = typename PrependOnes<N - 1, Tensor<1, Dims...> >::type;
        };
    } // namespace detail


    template<typename InT, typename PartOutT, size_t N_map,
        ActivationOp Act_ = Linear>
    class MapDenseMDBlock;

    template<size_t... InDims, size_t... PartOutDims, size_t N_map, ActivationOp Act_>
    class MapDenseMDBlock<Tensor<InDims...>, Tensor<PartOutDims...>, N_map, Act_> {
        static_assert(N_map < sizeof...(InDims),
                      "N_map must be less than input rank (need at least one contract dim)");

        using Split_ = detail::SplitAt<N_map, InDims...>;
        using MapSeq_ = typename Split_::head;
        using ContractSeq_ = typename Split_::tail;

        using OutDimSeq_ = typename detail::ConcatSeqs<
            MapSeq_, std::integer_sequence<size_t, PartOutDims...> >::type;
        using W_DimSeq_ = typename detail::ConcatSeqs<
            std::integer_sequence<size_t, PartOutDims...>, ContractSeq_>::type;

    public:
        using InputTensor = Tensor<InDims...>;
        using OutputTensor = typename detail::SeqToTensor<OutDimSeq_>::type;
        using W_Type = typename detail::SeqToTensor<W_DimSeq_>::type;
        using BiasType = Tensor<PartOutDims...>;

        using Act = Act_;
        static constexpr size_t N_contract = sizeof...(InDims) - N_map;
        static constexpr size_t N_out_part = sizeof...(PartOutDims);
        static constexpr size_t MapVolume = detail::SeqProduct<MapSeq_>::value;
        static constexpr size_t PartOutSize = (PartOutDims * ...);
        static constexpr size_t ContractSize = detail::SeqProduct<ContractSeq_>::value;

    private:
        Param<W_Type> W_;
        Param<BiasType> b_;

    public:
        auto all_params() { return std::tie(W_, b_); }
        auto all_params() const { return std::tie(W_, b_); }

        MapDenseMDBlock() { XavierInitMD(W_.value, ContractSize, PartOutSize); }

        OutputTensor Forward(const InputTensor &x) const {
            auto z = ΣΠ<N_contract>(x, W_.value);
            for (size_t m = 0; m < MapVolume; ++m)
                for (size_t i = 0; i < PartOutSize; ++i)
                    z.flat(m * PartOutSize + i) += b_.value.flat(i);
            return Map<Act>(z);
        }

        InputTensor Backward(const OutputTensor &delta_A, const OutputTensor &a,
                             const InputTensor &a_prev) {
            const auto delta_z = delta_A.zip(a, [](float g, float ai) { return g * Act::prime(ai); });

            const auto dz_swap = PermuteFromArray<SwapNDims<N_map, N_out_part>::value>(
                delta_z, std::make_index_sequence<N_map + N_out_part>{});
            const auto a_prev_swap = PermuteFromArray<SwapNDims<N_map, N_contract>::value>(
                a_prev, std::make_index_sequence<N_map + N_contract>{});
            W_.grad += ΣΠ<N_map>(dz_swap, a_prev_swap);

            for (size_t m = 0; m < MapVolume; ++m)
                for (size_t i = 0; i < PartOutSize; ++i)
                    b_.grad.flat(i) += delta_z.flat(m * PartOutSize + i);

            const auto W_T = PermuteFromArray<SwapNDims<N_out_part, N_contract>::value>(
                W_.value, std::make_index_sequence<N_out_part + N_contract>{});
            return ΣΠ<N_out_part>(delta_z, W_T);
        }

        template<size_t Batch>
        auto BatchedForward(const typename PrependBatch<Batch, InputTensor>::type &X) const
            -> typename PrependBatch<Batch, OutputTensor>::type {
            auto z = ΣΠ<N_contract>(X, W_.value);
            constexpr size_t slice = MapVolume * PartOutSize;
            for (size_t bi = 0; bi < Batch; ++bi)
                for (size_t m = 0; m < MapVolume; ++m)
                    for (size_t i = 0; i < PartOutSize; ++i)
                        z.flat(bi * slice + m * PartOutSize + i) += b_.value.flat(i);
            return Map<Act>(z);
        }

        template<size_t Batch>
        auto BatchedBackward(
            const typename PrependBatch<Batch, OutputTensor>::type &delta_A,
            const typename PrependBatch<Batch, OutputTensor>::type &a,
            const typename PrependBatch<Batch, InputTensor>::type &a_prev)
            -> typename PrependBatch<Batch, InputTensor>::type {
            const float inv_batch = 1.f / static_cast<float>(Batch);
            const auto delta_z = delta_A.zip(a, [](float g, float ai) { return g * Act::prime(ai); });

            constexpr size_t BmRank = 1 + N_map + N_out_part;
            const auto dz_swap = PermuteFromArray<SwapNDims<1 + N_map, N_out_part>::value>(
                delta_z, std::make_index_sequence<BmRank>{});
            constexpr size_t ApRank = 1 + N_map + N_contract;
            const auto a_prev_swap = PermuteFromArray<SwapNDims<1 + N_map, N_contract>::value>(
                a_prev, std::make_index_sequence<ApRank>{});
            W_.grad += ΣΠ<1 + N_map>(dz_swap, a_prev_swap) * inv_batch;

            BiasType dB_local{};
            constexpr size_t slice = MapVolume * PartOutSize;
            for (size_t bi = 0; bi < Batch; ++bi)
                for (size_t m = 0; m < MapVolume; ++m)
                    for (size_t i = 0; i < PartOutSize; ++i)
                        dB_local.flat(i) += delta_z.flat(bi * slice + m * PartOutSize + i);
            b_.grad += dB_local * inv_batch;

            const auto W_T = PermuteFromArray<SwapNDims<N_out_part, N_contract>::value>(
                W_.value, std::make_index_sequence<N_out_part + N_contract>{});
            return ΣΠ<N_out_part>(delta_z, W_T);
        }
    };


    // MapDense recipe: "preserve the first N_map axes, transform the rest into PartOutT."
    //
    // Usage (transformer per-token FFN):
    //   NetworkBuilder<
    //       Input<SeqLen, EmbDim>,
    //       MHAttention<Heads, EmbDim>,
    //       MapDense<1, Tensor<FFN_Dim>, ReLU>,    // per-token: SeqLen preserved, EmbDim → FFN_Dim
    //       MapDense<1, Tensor<EmbDim>>,           // per-token: SeqLen preserved, FFN_Dim → EmbDim
    //   >::type transformer;
    template<size_t N_map, typename PartOutT,
        ActivationOp Act_ = Linear>
    struct MapDense {
        using OutputTensor = typename detail::PrependOnes<N_map, PartOutT>::type;

        template<typename InputT>
        using Resolve = MapDenseMDBlock<InputT, PartOutT, N_map, Act_>;
    };
}
