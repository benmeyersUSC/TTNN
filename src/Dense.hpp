#pragma once
#include "TensorContract.hpp"
#include "TensorReduce.hpp"
#include "TTTN_ML.hpp"
#include "NetworkUtil.hpp"

namespace TTTN {
    template<typename InT, typename OutT, ActivationOp Act_ = Linear>
    class DenseMDBlock;

    // @doc: template<size_t... InDims, size_t... OutDims, ActivationOp Act_> class DenseMDBlock<Tensor<InDims...>, Tensor<OutDims...>, Act_>
    /**
     * `Block` implementation for a generalized, multidimensional **Dense** layer
     * Input and output dimensions are variadic `size_t...`, letting **Dense ** contraction be far more general than the typical Rank-2 matrix multiplication
     */
    template<size_t... InDims, size_t... OutDims, ActivationOp Act_>
    class DenseMDBlock<Tensor<InDims...>, Tensor<OutDims...>, Act_> {
    public:
        // @doc: using DenseMDBlock::InputTensor
        /** Alias: `InputTensor = Tensor<InDims...>` */
        using InputTensor = Tensor<InDims...>;
        // @doc: using DenseMDBlock::OutputTensor
        /** Alias: `OutputTensor = Tensor<OutDims...>` */
        using OutputTensor = Tensor<OutDims...>;
        // @doc: using DenseMDBlock::Act
        /**
         * Alias: `Act = Act_`
         * `Act_` is a `DenseMDBlock` template argument, forced to be a valid `ActivationOp`
         */
        using Act = Act_;

    private:
        // @doc: static constexpr size_t DenseMDBlock::N_in
        /** Convenience member: `N_in = sizeof...(InDims)` */
        static constexpr size_t N_in = sizeof...(InDims);
        // @doc: static constexpr size_t DenseMDBlock::N_out
        /** Convenience member: `N_out = sizeof...(OutDims)` */
        static constexpr size_t N_out = sizeof...(OutDims);


        // @doc: mutable LearnedContraction<InputTensor, OutputTensor, 0> DenseMDBlock::W_
        /** `LearnedContraction` of all axes of `InputTensor` with all axes of `OutputTensor` */
        mutable LearnedContraction<InputTensor, OutputTensor, 0> W_;

        // @doc: Param<OutputTensor> DenseMDBlock::b_
        /** `Param` whose underlying `Tensor` has type `OutputDims` (because `b_` is added to the output of the `W_` contraction) */
        Param<OutputTensor> b_;

    public:
        // @doc: auto DenseMDBlock::all_params()
        /** Returns `std::tuple` of references to `W_` and `b_` */

        auto all_params() { return std::tie(W_.W_, b_); }
        // @doc: auto DenseMDBlock::all_params() const
        /** Returns `std::tuple` of const references to `W_` and `b_` */
        auto all_params() const { return std::tie(W_.W_, b_); }


        // @doc: DenseMDBlock::DenseMDBlock()
        /** Default constructor */
        DenseMDBlock() = default;


        // @doc: OutputTensor DenseMDBlock::Forward(const InputTensor &x) const
        /** Contracts `W_` and `InputTensor& x` using `ΣΠ`, adds `b_`, maps with `Act` */
        OutputTensor Forward(const InputTensor &x) const {
            auto z = x >> W_;
            z += b_.value;
            return MapMove<Act>(std::move(z));
        }

        // @doc: InputTensor DenseMDBlock::Backward(const OutputTensor &delta_A, const OutputTensor &a, const InputTensor &a_prev)
        /** Computes `delta_z` from `delta_A`, propagates to `W_` and `b_`, passes `InputTensor delta_Input` upstream */
        InputTensor Backward(const OutputTensor &delta_A, const OutputTensor &a, const InputTensor &a_prev) {
            // get dLoss/dz from dLoss/dA with Act::prime
            const auto delta_z = delta_A.zip(a, [](float g, float ai) { return g * Act::prime(ai); });

            // dLoss/db is just dLoss/dz
            b_.grad += delta_z;

            // backpropagate delta_z through weights
            return W_ << delta_z;
        }

        // @doc: template<size_t Batch> Tensor<Batch, OutDims...> DenseMDBlock::BatchedForward(const Tensor<Batch, InDims...> &X) const
        /** Contracts `W_` and `Tensor<Batch, InDims...>& X` using `ΣΠ`, adds `b_`, maps with `Act` */
        template<size_t Batch>
        Tensor<Batch, OutDims...> BatchedForward(const Tensor<Batch, InDims...> &X) const {
            auto z = X >> W_;
            return MapMove<Act>(BroadcastMapMove<0, Add>(std::move(z), b_.value));
        }

        // @doc: template<size_t Batch> Tensor<Batch, InDims...> DenseMDBlock::BatchedBackward(const Tensor<Batch, OutDims...> &delta_A, const Tensor<Batch, OutDims...> &a, const Tensor<Batch, InDims...> &a_prev)
        /** Computes `delta_z` from `delta_A`, propagates to `W_` and `b_` (scaled by `Batch` count), passes `Tensor<Batch, InDims...> delta_Input` upstream */
        template<size_t Batch>
        Tensor<Batch, InDims...> BatchedBackward(const Tensor<Batch, OutDims...> &delta_A,
                                                 const Tensor<Batch, OutDims...> &a,
                                                 const Tensor<Batch, InDims...> &a_prev) {
            const float inv_batch = 1.f / static_cast<float>(Batch);
            // [Batch, OutDims...]
            const auto delta_z = delta_A.zip(a, [](float g, float ai) { return g * Act::prime(ai); });

            b_.grad += Reduce<0, Add>(delta_z);
            b_.grad *= inv_batch;


            // backpropagate delta_z through weights
            return W_ << delta_z;
        }
    };


    // @doc: template<typename OutT, ActivationOp Act_ = Linear> requires IsTensor<OutT> struct DenseMD
    /**
     * `BlockRecipe` for `DenseMDBlock`
     * Customarily, takes in a `Tensor` to be `OutputTensor` as well as an `ActivationOp`
     * `Resolve` takes an `IsTensor<InputT>` and defines the correct full-fledged `DenseMDBlock` type
     */
    template<typename OutT, ActivationOp Act_ = Linear> requires IsTensor<OutT>
    struct DenseMD {
        using OutputTensor = OutT;
        template<typename InputT> requires IsTensor<InputT>
        using Resolve = DenseMDBlock<InputT, OutT, Act_>;
    };


    // @doc: template<size_t N, ActivationOp Act_ = Linear> using Dense
    /**
     * Convenience wrapper around `BlockRecipe DenseMD` for the case where `OutT::Rank == 1`
     * User specifies how large the outgoing vector is to be and gets the corresponding `DenseMD` under the hood
     * Simple as: `using Dense = DenseMD<Tensor<N>, Act_>`
     */
    template<size_t N, ActivationOp Act_ = Linear>
    using Dense = DenseMD<Tensor<N>, Act_>;


    template<typename InT, typename PartOutT, size_t N_map,
        ActivationOp Act_ = Linear>
    class MapDenseMDBlock;

    // @doc: template<size_t... InDims, size_t... PartOutDims, size_t N_map, ActivationOp Act_> class MapDenseMDBlock<Tensor<InDims...>, Tensor<PartOutDims...>, N_map, Act_>
    /** `DenseMDBlock` that preserves first `N_map` axes, using shared weights to independently map the last `Rank - N_map` axes of `InDims...` to `PartOutDims...` */
    template<size_t... InDims, size_t... PartOutDims, size_t N_map, ActivationOp Act_>
    class MapDenseMDBlock<Tensor<InDims...>, Tensor<PartOutDims...>, N_map, Act_> {
        static_assert(N_map < sizeof...(InDims),
                      "N_map must be less than input rank (need at least one contract dim)");

        // @doc: using MapDenseMDBlock::Split_
        /** `SplitAt` resulting object, to be extracted from */
        using Split_ = SplitAt<N_map, InDims...>;
        // @doc: using MapDenseMDBlock::MapSeq_
        /** `std::integer_sequence` with the first `N_map` axes of `InDims...` */
        using MapSeq_ = Split_::head;
        // @doc: using MapDenseMDBlock::ContractSeq_
        /** `std::integer_sequence` with trailing axes to be contracted */
        using ContractSeq_ = Split_::tail;

        // @doc: using MapDenseMDBlock::OutDimSeq_
        /** `std::integer_sequence` representing final output shape: `[MapSeq_..., PartOutDims...]` */
        using OutDimSeq_ = ConcatSeqs<
            MapSeq_, std::integer_sequence<size_t, PartOutDims...> >::type;
        // @doc: using MapDenseMDBlock::W_DimSeq_
        /** `std::integer_sequence` representing weight matrix for each transformation: `[PartOutDims_..., ContractSeq_...]` */
        using W_DimSeq_ = ConcatSeqs<
            std::integer_sequence<size_t, PartOutDims...>, ContractSeq_>::type;

    public:
        // @doc: using MapDenseMDBlock::InputTensor
        /** Unpack `InDims...` to `Tensor<InDims...>` */
        using InputTensor = Tensor<InDims...>;
        // @doc: using MapDenseMDBlock::OutputTensor
        /** Unpack `OutDimSeq_` to `Tensor` using `SeqToTensor` */
        using OutputTensor = SeqToTensor<OutDimSeq_>::type;
        // @doc: using MapDenseMDBlock::W_Type
        /** Unpack `W_Dim_Seq_` to `Tensor` using `SeqToTensor` */
        using W_Type = SeqToTensor<W_DimSeq_>::type;
        // @doc: using MapDenseMDBlock::BiasType
        /** Unpack `PartOutDims...` to `Tensor<PartOutDims...>` */
        using BiasType = Tensor<PartOutDims...>;

        // @doc: using MapDenseMDBlock::Act
        /** Alias for `ActivationOp Act_` */
        using Act = Act_;
        // @doc: static constexpr size_t MapDenseMDBlock::N_contract
        /** `N_contract = sizeof...(InDims) - N_map` */
        static constexpr size_t N_contract = sizeof...(InDims) - N_map;
        // @doc: static constexpr size_t MapDenseMDBlock::N_out_part
        /** `N_out_part = sizeof...(PartOutDims)` */
        static constexpr size_t N_out_part = sizeof...(PartOutDims);
        // @doc: static constexpr size_t MapDenseMDBlock::MapVolume
        /**
         * How many independent sub-`Tensor`s will we contract (one for each **map** index product)
         * `MapVolume = SeqProduct<MapSeq_>::value`
         */
        static constexpr size_t MapVolume = SeqProduct<MapSeq_>::value;
        // @doc: static constexpr size_t MapDenseMDBlock::PartOutSize
        /**
         * Size of each projected copy at each **map** index product
         * `PartOutSize = (PartOutDims * ...)`
         */
        static constexpr size_t PartOutSize = (PartOutDims * ...);
        // @doc: static constexpr size_t MapDenseMDBlock::ContractSize
        /**
         * Size contracted in each copy at each **map** index product
         * `ContractSize = SeqProduct<ContractSeq_>::value`
         */
        static constexpr size_t ContractSize = SeqProduct<ContractSeq_>::value;

    private:
        // @doc: Param<W_Type> MapDenseMDBlock::W_
        /** Wrap `Tensor` of type `W_Type` into `Param` object */
        // Param<W_Type> W_;
        // @doc: Param<BiasType> MapDenseMDBlock::b_
        /** Wrap `Tensor` of type `BiasType` into `Param` object */
        Param<BiasType> b_;

        mutable LearnedContraction<InputTensor, OutputTensor, N_map> lc_;

    public:
        // @doc: auto MapDenseMDBlock::all_params()
        /** Returns `std::tuple` of `Param&`s */
        auto all_params() { return std::tie(lc_.W_, b_); }
        // @doc: auto MapDenseMDBlock::all_params() const
        /** Returns `std::tuple` of `const Param&`s */
        auto all_params() const { return std::tie(lc_.W_, b_); }

        // @doc: MapDenseMDBlock::MapDenseMDBlock()
        /** Default constructor, calls `XavierInitMD` on `W_` */
        MapDenseMDBlock() { /* XavierInitMD handled by LearnedContraction ctor */ }

        // @doc: OutputTensor MapDenseMDBlock::Forward(const InputTensor &x) const
        /** Compute standard `ΣΠ<N_contract>(x, W_.value)`, add `b_`, wrap in `Act` */
        OutputTensor Forward(const InputTensor &x) const {
            // [MapSeq_..., ContractSeq_...] x [PartOutDims..., ContractSeq_...] -> [MapSeq_..., PartOutDims...]
            // auto z = ΣΠ<N_contract>(x, W_.value);
            auto z = x >> lc_;
            // add bias to each mapped copy
            ParForEach(MapVolume * PartOutSize, [&](size_t i) {
                z.flat(i) += b_.value.flat(i % PartOutSize);
            });
            return MapMove<Act>(std::move(z));
        }

        // @doc: InputTensor MapDenseMDBlock::Backward(const OutputTensor &delta_A, const OutputTensor &a, const InputTensor &a_prev)
        /**
         * Backward pass, fills `W_.grad` and `b_.grad`
         * Similar logic as `DenseMD::Backward`
         */
        InputTensor Backward(const OutputTensor &delta_A, const OutputTensor &a, const InputTensor & /*a_prev*/) {
            const auto delta_z = delta_A.zip(a, [](float g, float ai) { return g * Act::prime(ai); });

            // collapse both Tensors into BC_Permute form and contract
            // W_.grad += [&]<size_t... I>(std::index_sequence<I...>) {
            //     return Contract<AxisList<I...>{}, AxisList<I...>{}, Mul, Add>(delta_z, a_prev);
            // }(std::make_index_sequence<N_map>{});

            // need loops to avoid race
            for (size_t m = 0; m < MapVolume; ++m) {
                for (size_t i = 0; i < PartOutSize; ++i) {
                    b_.grad.flat(i) += delta_z.flat(m * PartOutSize + i);
                }
            }

            // const auto W_T = PermuteFromArray<SwapNDims<N_out_part, N_contract>::value>(
            //     W_.value, std::make_index_sequence<N_out_part + N_contract>{});
            // return ΣΠ<N_out_part>(delta_z, W_T);
            return lc_ << delta_z;
        }

        // @doc: template<size_t Batch> auto MapDenseMDBlock::BatchedForward(const PrependBatch<Batch, InputTensor>::type &X) const -> PrependBatch<Batch, OutputTensor>::type
        /** Same internal logic as `MapDenseMDBlock::Forward`, but adds `Batch` axis upfront */
        template<size_t Batch>
        auto BatchedForward(
            const PrependBatch<Batch, InputTensor>::type &X) const -> PrependBatch<Batch, OutputTensor>::type {
            // auto z = ΣΠ<N_contract>(X, W_.value);
            auto z = X >> lc_;
            ParForEach(Batch * MapVolume * PartOutSize, [&](size_t i) {
                z.flat(i) += b_.value.flat(i % PartOutSize);
            });
            return MapMove<Act>(std::move(z));
        }

        // @doc: template<size_t Batch> auto MapDenseMDBlock::BatchedBackward(const PrependBatch<Batch, OutputTensor>::type &delta_A, const PrependBatch<Batch, OutputTensor>::type &a, const PrependBatch<Batch, InputTensor>::type &a_prev) -> PrependBatch<Batch, InputTensor>::type
        /**
         * Batched backward pass, fills `W_.grad` and `b_.grad`
         * Similar logic as `DenseMD::BatchedBackward`
         */
        template<size_t Batch>
        auto BatchedBackward(const PrependBatch<Batch, OutputTensor>::type &delta_A,
                             const PrependBatch<Batch, OutputTensor>::type &a,
                             const PrependBatch<Batch, InputTensor>::type & /*a_prev*/) -> PrependBatch<Batch,
            InputTensor>::type {
            const float inv_batch = 1.f / static_cast<float>(Batch);
            const auto delta_z = delta_A.zip(a, [](float g, float ai) { return g * Act::prime(ai); });

            // auto localWGrad = [&]<size_t... I>(std::index_sequence<I...>) {
            //     return Contract<AxisList<I...>{}, AxisList<I...>{}, Mul, Add>(delta_z, a_prev);
            // }(std::make_index_sequence<1 + N_map>{});
            // localWGrad *= inv_batch;
            // W_.grad += localWGrad;

            BiasType dB_local{};
            constexpr size_t slice = MapVolume * PartOutSize;
            for (size_t bi = 0; bi < Batch; ++bi) {
                for (size_t m = 0; m < MapVolume; ++m) {
                    for (size_t i = 0; i < PartOutSize; ++i) {
                        dB_local.flat(i) += delta_z.flat(bi * slice + m * PartOutSize + i);
                    }
                }
            }
            b_.grad += dB_local * inv_batch;

            // const auto W_T = PermuteFromArray<SwapNDims<N_out_part, N_contract>::value>(
            //     W_.value, std::make_index_sequence<N_out_part + N_contract>{});
            // return ΣΠ<N_out_part>(delta_z, W_T);
            return lc_ << delta_z;
        }
    };


    // @doc: template<size_t N_map, typename PartOutT, ActivationOp Act_ = Linear> requires IsTensor<PartOutT> struct MapDense
    /**
     * `BlockRecipe` for `MapDenseMDBlock`
     * Takes in `N_map` and `Tensor<PartOutDims...> PartOutT`
     */
    template<size_t N_map, typename PartOutT, ActivationOp Act_ = Linear>
        requires IsTensor<PartOutT>
    struct MapDense {
        using OutputTensor = PrependOnes<N_map, PartOutT>::type;

        template<typename InputT> requires IsTensor<InputT>
        using Resolve = MapDenseMDBlock<InputT, PartOutT, N_map, Act_>;
    };
}
