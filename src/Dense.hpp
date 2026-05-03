#pragma once
#include "TensorContract.hpp"
#include "TensorReduce.hpp"
#include "TTTN_ML.hpp"
#include "NetworkUtil.hpp"

namespace TTTN {
    template<typename InT, typename OutT, ActivationOp Act_ = Linear>
    class DenseMDBlock;

    // @doc: template<size_t... InDims, size_t... OutDims, ActivationOp Act_> class DenseMDBlock
    /**
     * Generalized multi-dimensional Dense layer.
     * `TrainingCache` is empty — backward receives `a` and `a_prev` from the trainer's activation record.
     */
    template<size_t... InDims, size_t... OutDims, ActivationOp Act_>
    class DenseMDBlock<Tensor<InDims...>, Tensor<OutDims...>, Act_> {
    public:
        using InputTensor  = Tensor<InDims...>;
        using OutputTensor = Tensor<OutDims...>;
        using Act          = Act_;

        template<size_t> using TrainingCache = std::tuple<>;

    private:
        LearnedContraction<InputTensor, OutputTensor, 0> W_;
        Param<OutputTensor> b_;

    public:
        auto all_params()       { return std::tie(W_.W_, b_); }
        auto all_params() const { return std::tie(W_.W_, b_); }

        DenseMDBlock() = default;

        // @doc: template<size_t Batch> Tensor<Batch, OutDims...> DenseMDBlock::Forward(const Tensor<Batch, InDims...> &X) const
        /** Pure inference forward. */
        template<size_t Batch>
        Tensor<Batch, OutDims...> Forward(const Tensor<Batch, InDims...> &X) const {
            auto z = X >> W_;
            return MapMove<Act>(BroadcastMapMove<0, Add>(std::move(z), b_.value));
        }

        // @doc: template<size_t Batch> Tensor<Batch, OutDims...> DenseMDBlock::Forward(const Tensor<Batch, InDims...> &X, TrainingCache<Batch> &) const
        /** Training forward — cache is empty for Dense, delegates to pure Forward. */
        template<size_t Batch>
        Tensor<Batch, OutDims...> Forward(const Tensor<Batch, InDims...> &X, TrainingCache<Batch> &) const {
            return Forward<Batch>(X);
        }

        // @doc: template<size_t Batch> Tensor<Batch, InDims...> DenseMDBlock::Backward(...)
        /** Backward: delta_z from Act::prime(a), b_.grad accumulated, W_.grad via LC backward with a_prev. */
        template<size_t Batch>
        Tensor<Batch, InDims...> Backward(const Tensor<Batch, OutDims...> &delta_A,
                                          const Tensor<Batch, OutDims...> &a,
                                          const Tensor<Batch, InDims...>  &a_prev,
                                          const TrainingCache<Batch> &) {
            const auto delta_z = delta_A.zip(a, [](float g, float ai) { return g * Act::prime(ai); });
            b_.grad += Reduce<0, Add>(delta_z);
            return W_.backward(delta_z, a_prev);
        }
    };


    // @doc: template<typename OutT, ActivationOp Act_ = Linear> requires IsTensor<OutT> struct DenseMD
    template<typename OutT, ActivationOp Act_ = Linear> requires IsTensor<OutT>
    struct DenseMD {
        using OutputTensor = OutT;
        template<typename InputT> requires IsTensor<InputT>
        using Resolve = DenseMDBlock<InputT, OutT, Act_>;
    };

    // @doc: template<size_t N, ActivationOp Act_ = Linear> using Dense
    template<size_t N, ActivationOp Act_ = Linear>
    using Dense = DenseMD<Tensor<N>, Act_>;


    template<typename InT, typename PartOutT, size_t N_map, ActivationOp Act_ = Linear>
    class MapDenseMDBlock;

    // @doc: template<size_t... InDims, size_t... PartOutDims, size_t N_map, ActivationOp Act_> class MapDenseMDBlock
    /**
     * Dense layer that preserves the first `N_map` axes and independently maps the remainder.
     * `TrainingCache` is empty — backward uses `a` and `a_prev` from the trainer.
     */
    template<size_t... InDims, size_t... PartOutDims, size_t N_map, ActivationOp Act_>
    class MapDenseMDBlock<Tensor<InDims...>, Tensor<PartOutDims...>, N_map, Act_> {
        static_assert(N_map < sizeof...(InDims),
                      "N_map must be less than input rank (need at least one contract dim)");

        using Split_       = SplitAt<N_map, InDims...>;
        using MapSeq_      = Split_::head;
        using ContractSeq_ = Split_::tail;
        using OutDimSeq_   = ConcatSeqs<MapSeq_, std::integer_sequence<size_t, PartOutDims...>>::type;

    public:
        using InputTensor  = Tensor<InDims...>;
        using OutputTensor = SeqToTensor<OutDimSeq_>::type;
        using BiasType     = Tensor<PartOutDims...>;
        using Act          = Act_;

        template<size_t> using TrainingCache = std::tuple<>;

        static constexpr size_t MapVolume   = SeqProduct<MapSeq_>::value;
        static constexpr size_t PartOutSize = (PartOutDims * ...);

    private:
        Param<BiasType> b_;
        LearnedContraction<InputTensor, OutputTensor, N_map> lc_;

    public:
        auto all_params()       { return std::tie(lc_.W_, b_); }
        auto all_params() const { return std::tie(lc_.W_, b_); }

        MapDenseMDBlock() = default;

        // @doc: template<size_t Batch> auto MapDenseMDBlock::Forward(const Tensor<Batch,InDims...> &X) const
        template<size_t Batch>
        auto Forward(const typename PrependBatch<Batch, InputTensor>::type &X) const
            -> typename PrependBatch<Batch, OutputTensor>::type {
            auto z = X >> lc_;
            ParForEach(Batch * MapVolume * PartOutSize, [&](size_t i) {
                z.flat(i) += b_.value.flat(i % PartOutSize);
            });
            return MapMove<Act>(std::move(z));
        }

        // @doc: template<size_t Batch> auto MapDenseMDBlock::Forward(X, cache) const
        template<size_t Batch>
        auto Forward(const typename PrependBatch<Batch, InputTensor>::type &X, TrainingCache<Batch> &) const
            -> typename PrependBatch<Batch, OutputTensor>::type {
            return Forward<Batch>(X);
        }

        // @doc: template<size_t Batch> auto MapDenseMDBlock::Backward(...)
        template<size_t Batch>
        auto Backward(const typename PrependBatch<Batch, OutputTensor>::type &delta_A,
                      const typename PrependBatch<Batch, OutputTensor>::type &a,
                      const typename PrependBatch<Batch, InputTensor>::type  &a_prev,
                      const TrainingCache<Batch> &)
            -> typename PrependBatch<Batch, InputTensor>::type {
            const auto delta_z = delta_A.zip(a, [](float g, float ai) { return g * Act::prime(ai); });

            BiasType dB_local{};
            constexpr size_t slice = MapVolume * PartOutSize;
            for (size_t bi = 0; bi < Batch; ++bi)
                for (size_t m = 0; m < MapVolume; ++m)
                    for (size_t i = 0; i < PartOutSize; ++i)
                        dB_local.flat(i) += delta_z.flat(bi * slice + m * PartOutSize + i);
            b_.grad += dB_local;

            return lc_.backward(delta_z, a_prev);
        }
    };


    // @doc: template<size_t N_map, typename PartOutT, ActivationOp Act_ = Linear> requires IsTensor<PartOutT> struct MapDense
    template<size_t N_map, typename PartOutT, ActivationOp Act_ = Linear>
        requires IsTensor<PartOutT>
    struct MapDense {
        using OutputTensor = PrependOnes<N_map, PartOutT>::type;
        template<typename InputT> requires IsTensor<InputT>
        using Resolve = MapDenseMDBlock<InputT, PartOutT, N_map, Act_>;
    };
}
