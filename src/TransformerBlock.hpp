#pragma once
#include "BlockSequence.hpp"
#include "Dense.hpp"
#include "Attention.hpp"
#include "MoreNets.hpp"

namespace TTTN {
    template<typename InputT, size_t Heads, size_t FFNHidden, bool PreNorm = true, bool Masked = false>
    class TransformerBlock;

    // @doc: template<size_t SeqLen, size_t EmbDim, size_t Heads, size_t FFNHidden, bool PreNorm, bool Masked> class TransformerBlock<Tensor<SeqLen, EmbDim>, Heads, FFNHidden, PreNorm, Masked>
    /**
     * `Block` that defines standard **Transformer Block**: Multi-headed Attention + FFN with optional LayerNorm.
     * `bool PreNorm` (default `true`): LayerNorm precedes each sub-layer inside the residual (modern default); `false` matches the original 2017 paper — LayerNorm follows each residual addition.
     * `bool Masked` (default `false`): passed through to `MultiHeadAttentionBlock` for causal masking.
     * Pre-norm topology:  `Residual(LN → MHA)`,  `Residual(LN → FFN)`
     * Post-norm topology: `Residual(MHA) → LN`,  `Residual(FFN) → LN`
     */
    template<size_t SeqLen, size_t EmbDim, size_t Heads, size_t FFNHidden, bool PreNorm, bool Masked>
    class TransformerBlock<Tensor<SeqLen, EmbDim>, Heads, FFNHidden, PreNorm, Masked> {
        // @doc: using TransformerBlock::MHABlock_
        /** `MultiHeadAttentionBlock<SeqLen, Heads, Masked, EmbDim>` — shared by both norm orderings */
        using MHABlock_ = MultiHeadAttentionBlock<SeqLen, Heads, Masked, EmbDim>;

        // @doc: using TransformerBlock::MHASub_
        /** Pre-norm MHA sub-sequence: `BlockSequence<LayerNormBlock, MHABlock_>` */
        using MHASub_ = BlockSequence<LayerNormBlock<SeqLen, EmbDim>, MHABlock_>;

        // @doc: using TransformerBlock::FFNOnly_
        /** FFN without LayerNorm (post-norm path): `BlockSequence<MapDenseMDBlock, MapDenseMDBlock>` */
        using FFNOnly_ = BlockSequence<
            MapDenseMDBlock<Tensor<SeqLen, EmbDim>, Tensor<FFNHidden>, 1, ReLU>,
            MapDenseMDBlock<Tensor<SeqLen, FFNHidden>, Tensor<EmbDim>, 1>
        >;

        // @doc: using TransformerBlock::FFNSub_
        /** Pre-norm FFN sub-sequence: `BlockSequence<LayerNormBlock, MapDenseMDBlock, MapDenseMDBlock>` */
        using FFNSub_ = BlockSequence<
            LayerNormBlock<SeqLen, EmbDim>,
            MapDenseMDBlock<Tensor<SeqLen, EmbDim>, Tensor<FFNHidden>, 1, ReLU>,
            MapDenseMDBlock<Tensor<SeqLen, FFNHidden>, Tensor<EmbDim>, 1>
        >;

        // @doc: using TransformerBlock::Inner_
        /**
         * Full topology selected by `PreNorm`:
         * `true`:  `BlockSequence<Residual(LN+MHA),  Residual(LN+FFN)>`
         * `false`: `BlockSequence<Residual(MHA), LN, Residual(FFN), LN>`
         */
        using Inner_ = std::conditional_t<PreNorm,
            BlockSequence<ResidualBlock<MHASub_>, ResidualBlock<FFNSub_>>,
            BlockSequence<ResidualBlock<MHABlock_>, LayerNormBlock<SeqLen, EmbDim>,
                          ResidualBlock<FFNOnly_>,  LayerNormBlock<SeqLen, EmbDim>>
        >;

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

        // @doc: void TransformerBlock::peek(SnapshotMap &out, const std::string &prefix) const
        /** Forwards peek to `inner_`, propagating the prefix so nested peekable blocks (e.g. `MultiHeadAttentionBlock`) are reachable */
        void peek(SnapshotMap &out, const std::string &prefix) const { inner_.peek(out, prefix); }

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


    // @doc: template<size_t Heads, size_t FFNHidden, bool PreNorm, bool Masked> struct Transformer
    /**
     * `BlockRecipe` struct for building a `TransformerBlock`
     * Parameterized by `Heads`, `FFNHidden`, `PreNorm` (default `true`), and `Masked` (default `false`)
     * Remaining dimensions (`SeqLen`, `EmbDim`) are deduced from `InputT` passed to `Resolve`
     */
    template<size_t Heads, size_t FFNHidden, bool PreNorm = true, bool Masked = false>
    struct Transformer {
        // @doc: using Transformer::OutputTensor
        /** Placeholder for `BlockRecipe` concept check: `Tensor<1, Heads>` guarantees `EmbSize % Heads == 0` */
        using OutputTensor = Tensor<1, Heads>;

        // @doc: template<IsTensor InputT> using Transformer::Resolve
        /** When given `IsTensor InputT`, create full `TransformerBlock<InputT, Heads, FFNHidden, PreNorm, Masked>` */
        template<IsTensor InputT>
        using Resolve = TransformerBlock<InputT, Heads, FFNHidden, PreNorm, Masked>;
    };
} // namespace TTTN
