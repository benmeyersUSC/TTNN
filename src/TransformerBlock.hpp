#pragma once
#include "BlockSequence.hpp"
#include "Dense.hpp"
#include "Attention.hpp"
#include "MoreNets.hpp"

namespace TTTN {
    template<typename InputT, size_t Heads, size_t FFNHidden>
    class TransformerBlock;

    // @doc: template<size_t SeqLen, size_t EmbDim, size_t Heads, size_t FFNHidden> class TransformerBlock<Tensor<SeqLen, EmbDim>, Heads, FFNHidden>
    /**
     * `Block` that defines standard **Transformer Block***: LayerNorm, Multi-headed Attention, LayerNorm, FFN.
     * Topology:
     * `BlockSequence<`
     * `ResidualBlock<`
     * `BlockSequence<`
     * `LayerNormBlock`
     * `MultiHeadAttentionBlock`
     * `>`
     * `>`
     * `ResidualBlock<`
     * `BlockSequence<`
     * `LayerNormBlock`
     * `MapDenseMDBlock`
     * `MapDenseMDBlock`
     * `>`
     * `>`
     * `>`
     */
    template<size_t SeqLen, size_t EmbDim, size_t Heads, size_t FFNHidden>
    class TransformerBlock<Tensor<SeqLen, EmbDim>, Heads, FFNHidden> {
        // @doc: using TransformerBlock::MHASub_
        /**
         * Multi-head attention sub-sequence, wrapped by the first `ResidualBlock`
         * Topology:
         * `BlockSequence<`
         * `LayerNormBlock`
         * `MultiHeadAttentionBlock`
         * `>`
         */
        using MHASub_ = BlockSequence<
            LayerNormBlock<SeqLen, EmbDim>,
            MultiHeadAttentionBlock<SeqLen, Heads, EmbDim>
        >;
        // @doc: using TransformerBlock::FFNSub_
        /**
         * Feed-forward sub-sequence, wrapped by the second `ResidualBlock`
         * Topology:
         * `BlockSequence<`
         * `LayerNormBlock`
         * `MapDenseMDBlock`
         * `MapDenseMDBlock`
         * `>`
         * Two `MapDenseMDBlock`s: one to map to `FFNHidden`, one to map back down to `EmbDim`
         */
        using FFNSub_ = BlockSequence<
            LayerNormBlock<SeqLen, EmbDim>,
            MapDenseMDBlock<Tensor<SeqLen, EmbDim>, Tensor<FFNHidden>, 1, ReLU>,
            MapDenseMDBlock<Tensor<SeqLen, FFNHidden>, Tensor<EmbDim>, 1>
        >;
        // @doc: using TransformerBlock::Inner_
        /**
         * Topology:
         * `BlockSequence<`
         * `ResidualBlock<`
         * `BlockSequence<`
         * `LayerNormBlock`
         * `MultiHeadAttentionBlock`
         * `>`
         * `>`
         * `ResidualBlock<`
         * `BlockSequence<`
         * `LayerNormBlock`
         * `MapDenseMDBlock`
         * `MapDenseMDBlock`
         * `>`
         * `>`
         * `>`
         */
        using Inner_ = BlockSequence<ResidualBlock<MHASub_>, ResidualBlock<FFNSub_> >;

        // @doc: Inner_ TransformerBlock::inner_
        /** Instance of topology defined in `Inner_` */
        Inner_ inner_;

    public:
        // @doc: using TransformerBlock::InputTensor
        /** Alias for `Tensor<SeqLen, EmbDim>` */
        using InputTensor = Tensor<SeqLen, EmbDim>;
        // @doc: using TransformerBlock::OutputTensor
        /** Alias for `Tensor<SeqLen, EmbDim>` */
        using OutputTensor = Tensor<SeqLen, EmbDim>;

        // @doc: auto TransformerBlock::all_params()
        /** Return `std::tuple` from `inner_.all_params()` */
        auto all_params() { return inner_.all_params(); }
        // @doc: auto TransformerBlock::all_params() const
        /** Return `std::tuple` from `inner_.all_params()` */
        auto all_params() const { return inner_.all_params(); }

        // @doc: OutputTensor TransformerBlock::Forward(const InputTensor &x) const
        /** Forward pass, simply: `inner_.Forward(x)` */
        OutputTensor Forward(const InputTensor &x) const {
            return x >> inner_;
        }

        // @doc: InputTensor TransformerBlock::Backward(const OutputTensor &delta_A, const OutputTensor &a, const InputTensor &a_prev)
        /** Backward pass, simply: `inner_.Backward(delta_A, a, a_prev)` */
        InputTensor Backward(const OutputTensor &delta_A, const OutputTensor &a, const InputTensor &a_prev) {
            return inner_ << BackwardArgs{delta_A, a, a_prev};
        }

        // @doc: template<size_t Batch> auto TransformerBlock::BatchedForward(const PrependBatch<Batch, InputTensor>::type &X) const -> PrependBatch<Batch, OutputTensor>::type
        /** Batched forward pass, simply: `inner_.template BatchedForward<Batch>(X)` */
        template<size_t Batch>
        auto BatchedForward(const PrependBatch<Batch, InputTensor>::type &X) const
            -> PrependBatch<Batch, OutputTensor>::type {
            return X >> inner_;
        }

        // @doc: template<size_t Batch> auto TransformerBlock::BatchedBackward(const PrependBatch<Batch, OutputTensor>::type &delta_A, const PrependBatch<Batch, OutputTensor>::type &a, const PrependBatch<Batch, InputTensor>::type &a_prev) -> PrependBatch<Batch, InputTensor>::type
        /** Batched backward pass, simply: `inner_.template BatchedBackward<Batch>(delta_A, a, a_prev)` */
        template<size_t Batch>
        auto BatchedBackward(const PrependBatch<Batch, OutputTensor>::type &delta_A,
                             const PrependBatch<Batch, OutputTensor>::type &a,
                             const PrependBatch<Batch, InputTensor>::type &a_prev)
            -> PrependBatch<Batch, InputTensor>::type {
            return inner_ << BackwardArgs{delta_A, a, a_prev};
        }
    };


    // @doc: template<size_t Heads, size_t FFNHidden> struct Transformer
    /**
     * `BlockRecipe` struct for building a `TransformerBlock`
     * Parameterized (when in a `NetworkBuilder` sequence) entirely by `Heads` and `FFNHidden`, the rest is deduced
     */
    template<size_t Heads, size_t FFNHidden>
    struct Transformer {
        // @doc: template<IsTensor InputT> using Transformer::Resolve
        /** When given `IsTensor InputT`, create full `TransformerBlock<InputT, Heads, FFNHidden>` */
        template<IsTensor InputT>
        using Resolve = TransformerBlock<InputT, Heads, FFNHidden>;
    };
} // namespace TTTN
