#pragma once

#pragma once
#include "NetworkUtil.hpp"
#include "TensorOps.hpp"

namespace TTTN {
    // =========================================================================
    // Identity Block
    // =========================================================================
    //
    // Pass-through: Forward returns input unchanged.
    // Used as a building block for Residual = Parallel<Block, Identity<T>>.
    //
    // =========================================================================

    // @doc: template<typename T> class IdentityBlock
    /**
     * Pass-through block: `Forward(x) = x`, `Backward(delta, ...) = delta`.
     * No parameters. Satisfies the Block concept.
     * Used internally by `Residual<Block>` = `Parallel<Block, IdentityBlock<InputTensor>>`.
     */
    template<typename T>
    class IdentityBlock {
    public:
        using InputTensor = T;
        using OutputTensor = T;

        auto all_params() { return std::tie(); }
        auto all_params() const { return std::tie(); }

        OutputTensor Forward(const InputTensor &x) const { return x; }

        InputTensor Backward(const OutputTensor &delta_A,
                             const OutputTensor & /*a*/,
                             const InputTensor & /*a_prev*/) {
            return delta_A;
        }

        template<size_t Batch>
        auto BatchedForward(const typename PrependBatch<Batch, InputTensor>::type &X) const
            -> typename PrependBatch<Batch, OutputTensor>::type {
            return X;
        }

        template<size_t Batch>
        auto BatchedBackward(
            const typename PrependBatch<Batch, OutputTensor>::type &delta_A,
            const typename PrependBatch<Batch, OutputTensor>::type & /*a*/,
            const typename PrependBatch<Batch, InputTensor>::type & /*a_prev*/)
            -> typename PrependBatch<Batch, InputTensor>::type {
            return delta_A;
        }
    };


    // =========================================================================
    // Parallel Block
    // =========================================================================
    //
    // Feeds the same input to two sub-blocks, combines outputs with element-wise
    // addition: output = BlockA.Forward(x) + BlockB.Forward(x)
    //
    // Both blocks must have the same InputTensor and OutputTensor types.
    //
    // Backward: gradient flows identically to both branches (since d/dx (a+b) = 1+1).
    // Each branch receives the full upstream gradient and its own cached activation.
    // Upstream gradient = sum of both branches' upstream contributions.
    //
    // =========================================================================

    // @doc: template<typename BlockA, typename BlockB> class ParallelBlock
    /**
     * Parallel composition: `Forward(x) = A.Forward(x) + B.Forward(x)`.
     * Both blocks must share `InputTensor` and `OutputTensor` types.
     * Backward passes gradient to both branches; upstream = `dA + dB`.
     * Caches each branch's output for correct `Backward(delta, a, a_prev)` calls.
     */
    template<typename BlockA, typename BlockB>
    class ParallelBlock {
        static_assert(std::is_same_v<typename BlockA::InputTensor, typename BlockB::InputTensor>,
                      "ParallelBlock: both blocks must have the same InputTensor");
        static_assert(std::is_same_v<typename BlockA::OutputTensor, typename BlockB::OutputTensor>,
                      "ParallelBlock: both blocks must have the same OutputTensor");

    public:
        using InputTensor = typename BlockA::InputTensor;
        using OutputTensor = typename BlockA::OutputTensor;

    private:
        BlockA a_;
        BlockB b_;

        // Cache each branch's output for backward pass
        mutable OutputTensor a_out_{};
        mutable OutputTensor b_out_{};

    public:
        const BlockA& block_a() const { return a_; }
        const BlockB& block_b() const { return b_; }

        auto all_params() {
            return std::tuple_cat(a_.all_params(), b_.all_params());
        }

        auto all_params() const {
            return std::tuple_cat(a_.all_params(), b_.all_params());
        }

        OutputTensor Forward(const InputTensor &x) const {
            return a_.Forward(x) + b_.Forward(x);
        }

        InputTensor Backward(const OutputTensor &delta_A,
                             const OutputTensor & /*a*/,
                             const InputTensor &a_prev) {
            // Both branches receive the full upstream gradient.
            // Each gets its own cached activation for activation derivative.
            return a_.Backward(delta_A, a_out_, a_prev) + b_.Backward(delta_A, b_out_, a_prev);
        }

        template<size_t Batch>
        auto BatchedForward(const typename PrependBatch<Batch, InputTensor>::type &X) const
            -> typename PrependBatch<Batch, OutputTensor>::type {
            // auto a_batch = a_.template BatchedForward<Batch>(X);
            // auto b_batch = b_.template BatchedForward<Batch>(X);
            // element-wise add via zip
            // return a_batch.zip(b_batch, [](float x, float y) { return x + y; });
            return Zip<Add>(a_.template BatchedForward<Batch>(X), b_.template BatchedForward<Batch>(X));
        }

        template<size_t Batch>
        auto BatchedBackward(
            const typename PrependBatch<Batch, OutputTensor>::type &delta_A,
            const typename PrependBatch<Batch, OutputTensor>::type & /*a*/,
            const typename PrependBatch<Batch, InputTensor>::type &a_prev)
            -> typename PrependBatch<Batch, InputTensor>::type {
            // For batched: we don't have cached per-branch outputs.
            // Re-forward to get them. This is the cost of generality.
            auto a_batch_out = a_.template BatchedForward<Batch>(a_prev);
            auto b_batch_out = b_.template BatchedForward<Batch>(a_prev);

            auto grad_a = a_.template BatchedBackward<Batch>(delta_A, a_batch_out, a_prev);
            auto grad_b = b_.template BatchedBackward<Batch>(delta_A, b_batch_out, a_prev);
            return grad_a.zip(grad_b, [](float x, float y) { return x + y; });
        }
    };


    // =========================================================================
    // Residual Block
    // =========================================================================
    //
    // Residual<Block> = Parallel<Block, Identity>
    //
    // output = Block.Forward(x) + x
    //
    // =========================================================================

    // @doc: template<typename Block> using ResidualBlock
    /**
     * `Parallel<Block, IdentityBlock<InputTensor>>` — residual connection.
     * `Forward(x) = Block.Forward(x) + x`. Requires `InputTensor == OutputTensor`.
     */
    template<typename Block>
    using ResidualBlock = ParallelBlock<Block, IdentityBlock<typename Block::InputTensor> >;


    // =========================================================================
    // Recipe wrappers for NetworkBuilder
    // =========================================================================

    // @doc: template<typename RecipeA, typename RecipeB> struct ParallelRecipe
    /**
     * `NetworkBuilder` recipe for `ParallelBlock`.
     * Usage: `Parallel<MHAttention<4, 28>, MHAttention<4, 28>>`
     * Both recipes must resolve to blocks with matching Input/OutputTensor.
     */
    template<typename RecipeA, typename RecipeB>
    struct Parallel {
        // Placeholder — actual OutputTensor depends on input, resolved at chain time
        using OutputTensor = typename RecipeA::OutputTensor;

        template<typename InputT>
        using Resolve = ParallelBlock<
            typename RecipeA::template Resolve<InputT>,
            typename RecipeB::template Resolve<InputT>
        >;
    };

    // @doc: template<typename Recipe> struct Residual
    /**
     * `NetworkBuilder` recipe for `ResidualBlock`.
     * Usage: `Residual<ComposeBlocks<MHAttention<4, 28>, MapDense<1, Tensor<28>>>>`
     * The inner block's output shape must match its input shape.
     */
    template<typename Recipe>
    struct Residual {
        using OutputTensor = typename Recipe::OutputTensor;

        template<typename InputT>
        using Resolve = ResidualBlock<typename Recipe::template Resolve<InputT> >;
    };


    template<typename InnerBlock>
    class TransposeBlock {
    public:
        using InputTensor = typename InnerBlock::InputTensor;
        using OutputTensor = typename InnerBlock::OutputTensor;
        // Only valid for rank-2 tensors where transposing preserves shape
        static_assert(std::is_same_v<InputTensor, OutputTensor>,
                      "TransposeBlock requires InputTensor == OutputTensor");

    private:
        mutable InnerBlock inner_;

    public:
        const InnerBlock& inner() const { return inner_; }

        auto all_params() { return inner_.all_params(); }
        auto all_params() const { return inner_.all_params(); }

        OutputTensor Forward(const InputTensor &x) const {
            return Permute<1, 0>(inner_.Forward(Permute<1, 0>(x)));
        }

        InputTensor Backward(const OutputTensor &delta_A,
                             const OutputTensor &a,
                             const InputTensor &a_prev) {
            // Transpose everything into inner block's perspective, backward, transpose result back
            return Permute<1, 0>(inner_.Backward(
                Permute<1, 0>(delta_A),
                Permute<1, 0>(a),
                Permute<1, 0>(a_prev)));
        }

        template<size_t Batch>
        auto BatchedForward(const typename PrependBatch<Batch, InputTensor>::type &X) const
            -> typename PrependBatch<Batch, OutputTensor>::type {
            // Permute<0, 2, 1> on Tensor<Batch, R, C> → Tensor<Batch, C, R>
            auto X_t = Permute<0, 2, 1>(X);
            auto out_t = inner_.template BatchedForward<Batch>(X_t);
            return Permute<0, 2, 1>(out_t);
        }

        template<size_t Batch>
        auto BatchedBackward(
            const typename PrependBatch<Batch, OutputTensor>::type &delta_A,
            const typename PrependBatch<Batch, OutputTensor>::type &a,
            const typename PrependBatch<Batch, InputTensor>::type &a_prev)
            -> typename PrependBatch<Batch, InputTensor>::type {
            return Permute<0, 2, 1>(inner_.template BatchedBackward<Batch>(
                Permute<0, 2, 1>(delta_A),
                Permute<0, 2, 1>(a),
                Permute<0, 2, 1>(a_prev)));
        }
    };

    // Recipe wrapper for NetworkBuilder
    template<typename InnerRecipe>
    struct Transposed {
        using OutputTensor = typename InnerRecipe::OutputTensor;

        template<typename InputT>
        using Resolve = TransposeBlock<typename InnerRecipe::template Resolve<InputT> >;
    };
}
