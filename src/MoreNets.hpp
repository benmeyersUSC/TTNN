#pragma once
#include "NetworkUtil.hpp"
#include "TensorOps.hpp"
#include "TensorReduce.hpp"

namespace TTTN {
    // @doc: template<IsTensor T> class IdentityBlock
    /**
     * `Block` that performs identity transformation
     * Used by `ResidualBlock`
     */
    template<IsTensor T>
    class IdentityBlock {
    public:
        // @doc: using IdentityBlock::InputTensor
        /** Alias for `IsTensor T` template parameter */
        using InputTensor = T;
        // @doc: using IdentityBlock::OutputTensor
        /** Alias for `IsTensor T` template parameter */
        using OutputTensor = T;

        // @doc: auto IdentityBlock::all_params()
        /** Returns emtpy `std::tuple` */
        auto all_params() { return std::tie(); }
        // @doc: auto IdentityBlock::all_params() const
        /** Returns emtpy `std::tuple` */
        auto all_params() const { return std::tie(); }

        // @doc: static OutputTensor IdentityBlock::Forward(const InputTensor &x)
        /** Return `X` */
        static OutputTensor Forward(const InputTensor &x) { return x; }

        // @doc: static InputTensor IdentityBlock::Backward(const OutputTensor &delta_A, const OutputTensor &a, const InputTensor &a_prev)
        /** Return `delta_A` */
        static InputTensor Backward(const OutputTensor &delta_A,
                                    const OutputTensor & /*a*/,
                                    const InputTensor & /*a_prev*/) {
            return delta_A;
        }

        // @doc: template<size_t Batch> static auto IdentityBlock::BatchedForward(const PrependBatch<Batch, InputTensor>::type &X) -> PrependBatch<Batch, OutputTensor>::type
        /** Return `X` */
        template<size_t Batch>
        static auto BatchedForward(
            const PrependBatch<Batch, InputTensor>::type &X) -> PrependBatch<Batch, OutputTensor>::type {
            return X;
        }

        // @doc: template<size_t Batch> static auto IdentityBlock::BatchedBackward(const PrependBatch<Batch, OutputTensor>::type &delta_A, const PrependBatch<Batch, OutputTensor>::type &a, const PrependBatch<Batch, InputTensor>::type &a_prev) -> PrependBatch<Batch, InputTensor>::type
        /** Return `delta_A` */
        template<size_t Batch>
        static auto BatchedBackward(
            const PrependBatch<Batch, OutputTensor>::type &delta_A,
            const PrependBatch<Batch, OutputTensor>::type & /*a*/,
            const PrependBatch<Batch, InputTensor>::type & /*a_prev*/)
            -> PrependBatch<Batch, InputTensor>::type {
            return delta_A;
        }
    };


    // @doc: template<Block BlockA, Block BlockB> class ParallelBlock
    /**
     * Take two `Block`s who have the same `InputTensor`s and `OutputTensor`s and compute their forward and backward passes, effectively, in parallel
     * Forward pass results are summed
     */
    template<Block BlockA, Block BlockB>
    class ParallelBlock {
        static_assert(std::is_same_v<typename BlockA::InputTensor, typename BlockB::InputTensor>,
                      "ParallelBlock: both blocks must have the same InputTensor");
        static_assert(std::is_same_v<typename BlockA::OutputTensor, typename BlockB::OutputTensor>,
                      "ParallelBlock: both blocks must have the same OutputTensor");

    public:
        // @doc: using ParallelBlock::InputTensor
        /** Alias for `BlockA::`/`BlockB::InputTensor` */
        using InputTensor = BlockA::InputTensor;
        // @doc: using ParallelBlock::OutputTensor
        /** Alias for `BlockA::`/`BlockB::OutputTensor` */
        using OutputTensor = BlockA::OutputTensor;

    private:
        // @doc: BlockA ParallelBlock::a_
        /** `Block` member variable for first `Block BlockA` */
        BlockA a_;
        // @doc: BlockB ParallelBlock::b_
        /** `Block` member variable for second `Block BlockB` */
        BlockB b_;

        // @doc: mutable OutputTensor ParallelBlock::a_out_
        /** `mutable` cache for `a_`'s intermediate resulting `OutputTensor`, used in backward pass */
        mutable OutputTensor a_out_{};
        // @doc: mutable OutputTensor ParallelBlock::b_out_
        /** `mutable` cache for `b_`'s intermediate resulting `OutputTensor`, used in backward pass */
        mutable OutputTensor b_out_{};
        // @doc: mutable std::vector<float> ParallelBlock::a_out_buf_
        /**
         * `mutable` cache for batched intermediate forward results
         * Uses `std::vector<float>` because `Batch` is not a class parameter
         */
        mutable std::vector<float> a_out_buf_;
        // @doc: mutable std::vector<float> ParallelBlock::b_out_buf_
        /**
         * `mutable` cache for batched intermediate forward results
         * Uses `std::vector<float>` because `Batch` is not a class parameter
         */
        mutable std::vector<float> b_out_buf_;

    public:
        // @doc: const BlockA &ParallelBlock::block_a() const
        /** `const &` getter for `BlockA a_` */
        const BlockA &block_a() const { return a_; }
        // @doc: const BlockB &ParallelBlock::block_b() const
        /** `const &` getter for `BlockB b_` */
        const BlockB &block_b() const { return b_; }

        // @doc: auto ParallelBlock::all_params()
        /** `std::tuple_cat` of `a_.all_params()` and `b_.all_params()` */
        auto all_params() {
            return std::tuple_cat(a_.all_params(), b_.all_params());
        }

        // @doc: auto ParallelBlock::all_params() const
        /** `std::tuple_cat` of `a_.all_params()` and `b_.all_params()` */
        auto all_params() const {
            return std::tuple_cat(a_.all_params(), b_.all_params());
        }

        // @doc: OutputTensor ParallelBlock::Forward(const InputTensor &x) const
        /**
         * Call `Forward` on `a_` and `b_` and return their sum
         * Caches intermediate results
         */
        OutputTensor Forward(const InputTensor &x) const {
            a_out_ = x >> a_;
            b_out_ = x >> b_;
            return a_out_ + b_out_;
        }

        // @doc: InputTensor ParallelBlock::Backward(const OutputTensor &delta_A, const OutputTensor &a, const InputTensor &a_prev)
        /** Call `Backward` on `a_out_` and `b_out_` (cached individual results) and pass their sum upstream */
        InputTensor Backward(const OutputTensor &delta_A,
                             const OutputTensor & /*a*/,
                             const InputTensor &a_prev) {
            return (a_ << BackwardArgs{delta_A, a_out_, a_prev}) + (b_ << BackwardArgs{delta_A, b_out_, a_prev});
        }

        // @doc: void ParallelBlock::peek(SnapshotMap &out, const std::string &prefix) const
        /** Forwards peek to `a_` and `b_` if they satisfy `PeekableBlock`; no extra prefix added (transparency layer) */
        void peek(SnapshotMap &out, const std::string &prefix) const {
            if constexpr (PeekableBlock<BlockA>) a_.peek(out, prefix);
            if constexpr (PeekableBlock<BlockB>) b_.peek(out, prefix);
        }

        // @doc: template<size_t Batch> auto ParallelBlock::BatchedForward(const PrependBatch<Batch, InputTensor>::type &X) const -> PrependBatch<Batch, OutputTensor>::type
        /**
         * Call `BatchedForward` on `a_` and `b_` and return their sum
         * Caches intermediate results
         */
        template<size_t Batch>
        auto BatchedForward(const PrependBatch<Batch, InputTensor>::type &X) const
            -> PrependBatch<Batch, OutputTensor>::type {
            using BatchOut = PrependBatch<Batch, OutputTensor>::type;
            const auto af = X >> a_;
            const auto bf = X >> b_;
            a_out_buf_.assign(af.data(), af.data() + BatchOut::Size);
            b_out_buf_.assign(bf.data(), bf.data() + BatchOut::Size);
            return af + bf;
        }

        // @doc: template<size_t Batch> auto ParallelBlock::BatchedBackward(const PrependBatch<Batch, OutputTensor>::type &delta_A, const PrependBatch<Batch, OutputTensor>::type &a, const PrependBatch<Batch, InputTensor>::type &a_prev) -> PrependBatch<Batch, InputTensor>::type
        /** Call `BatchedBackward` on `a_out_buf_` and `b_out_buf_` (cached individual results) and pass their sum upstream */
        template<size_t Batch>
        auto BatchedBackward(
            const PrependBatch<Batch, OutputTensor>::type &delta_A,
            const PrependBatch<Batch, OutputTensor>::type & /*a*/,
            const PrependBatch<Batch, InputTensor>::type &a_prev)
            -> PrependBatch<Batch, InputTensor>::type {
            using BatchOut = PrependBatch<Batch, OutputTensor>::type;
            BatchOut a_batch_out, b_batch_out;
            std::copy(a_out_buf_.begin(), a_out_buf_.begin() + BatchOut::Size, a_batch_out.data());
            std::copy(b_out_buf_.begin(), b_out_buf_.begin() + BatchOut::Size, b_batch_out.data());

            auto grad = a_ << BackwardArgs{delta_A, a_batch_out, a_prev};
            grad += b_ << BackwardArgs{delta_A, b_batch_out, a_prev};
            return grad;
        }
    };


    // @doc: template<Block B> using ResidualBlock
    /**
     * `Block` type alias for residual connections, defined as a `ParallelBlock` with some `Block B` and `IdentityBlock`
     * `ParallelBlock` handles both forward and backward passes!
     */
    template<Block B>
    using ResidualBlock = ParallelBlock<B, IdentityBlock<typename B::InputTensor> >;


    // @doc: template<typename RecipeA, typename RecipeB> struct Parallel
    /** `BlockRecipe` for `ParallelBlock` */
    template<typename RecipeA, typename RecipeB>
    struct Parallel {
        // @doc: using Parallel::OutputTensor
        /** Alias for output `Tensor` type */
        using OutputTensor = RecipeA::OutputTensor;

        // @doc: template<IsTensor InputT> using Parallel::Resolve
        /** Given `IsTensor InputT`, define `ParallelBlock` type that should be created */
        template<IsTensor InputT>
        using Resolve = ParallelBlock<
            typename RecipeA::template Resolve<InputT>,
            typename RecipeB::template Resolve<InputT>
        >;
    };


    // @doc: template<typename Recipe> struct Residual
    /**
     * `Block` type alias for residual connections, defined as a `ParallelBlock` with some `Block B` and `IdentityBlock`
     * `ParallelBlock` handles both forward and backward passes!
     */
    template<typename Recipe>
    struct Residual {
        // @doc: using Residual::OutputTensor
        /** Alias for output `Tensor` type */
        using OutputTensor = Recipe::OutputTensor;

        // @doc: template<IsTensor InputT> using Residual::Resolve
        /** Given `IsTensor T`, define `ResidualBlock` type that should be created */
        template<IsTensor InputT>
        using Resolve = ResidualBlock<typename Recipe::template Resolve<InputT> >;
    };


    // @doc: template<Block InnerBlock> class TransposeBlock
    /**
     * Wraps a `Block InnerBlock` and runs its forward and backward passes on the transposed input, transposing results back out
     * Requires `InputTensor == OutputTensor` (transposing preserves shape)
     */
    template<Block InnerBlock>
    class TransposeBlock {
    public:
        // @doc: using TransposeBlock::InputTensor
        /** Alias for `InnerBlock::InputTensor` */
        using InputTensor = InnerBlock::InputTensor;
        // @doc: using TransposeBlock::OutputTensor
        /** Alias for `InnerBlock::OutputTensor` */
        using OutputTensor = InnerBlock::OutputTensor;
        static_assert(std::is_same_v<InputTensor, OutputTensor>,
                      "TransposeBlock requires InputTensor == OutputTensor");

    private:
        // @doc: mutable InnerBlock TransposeBlock::inner_
        /** The wrapped `Block` */
        mutable InnerBlock inner_;

    public:
        // @doc: const InnerBlock &TransposeBlock::inner() const
        /** `const &` getter for `inner_` */
        const InnerBlock &inner() const { return inner_; }

        // @doc: auto TransposeBlock::all_params()
        /** Delegates to `inner_.all_params()` */
        auto all_params() { return inner_.all_params(); }
        // @doc: auto TransposeBlock::all_params() const
        /** Delegates to `inner_.all_params()` */
        auto all_params() const { return inner_.all_params(); }

        // @doc: OutputTensor TransposeBlock::Forward(const InputTensor &x) const
        /** `Permute<1,0>(inner_.Forward(Permute<1,0>(x)))` */
        OutputTensor Forward(const InputTensor &x) const {
            return Permute<1, 0>(Permute<1, 0>(x) >> inner_);
        }

        // @doc: InputTensor TransposeBlock::Backward(const OutputTensor &delta_A, const OutputTensor &a, const InputTensor &a_prev)
        /** Transposes all arguments into `inner_`'s perspective, calls `inner_.Backward`, transposes result back */
        InputTensor Backward(const OutputTensor &delta_A,
                             const OutputTensor &a,
                             const InputTensor &a_prev) {
            return Permute<1, 0>(inner_ << BackwardArgs{
                Permute<1, 0>(delta_A),
                Permute<1, 0>(a),
                Permute<1, 0>(a_prev)});
        }

        // @doc: template<size_t Batch> auto TransposeBlock::BatchedForward(const PrependBatch<Batch, InputTensor>::type &X) const -> PrependBatch<Batch, OutputTensor>::type
        /** `Permute<0,2,1>` on `Tensor<Batch, R, C>` gives `Tensor<Batch, C, R>`, forward through `inner_`, permute back */
        template<size_t Batch>
        auto BatchedForward(const PrependBatch<Batch, InputTensor>::type &X) const
            -> PrependBatch<Batch, OutputTensor>::type {
            // Permute<0, 2, 1> on Tensor<Batch, R, C> → Tensor<Batch, C, R>
            auto X_t = Permute<0, 2, 1>(X);
            return Permute<0, 2, 1>(X_t >> inner_);
        }

        // @doc: template<size_t Batch> auto TransposeBlock::BatchedBackward(const PrependBatch<Batch, OutputTensor>::type &delta_A, const PrependBatch<Batch, OutputTensor>::type &a, const PrependBatch<Batch, InputTensor>::type &a_prev) -> PrependBatch<Batch, InputTensor>::type
        /** Same as `Backward` with batched `Permute<0,2,1>` on all arguments */
        template<size_t Batch>
        auto BatchedBackward(
            const PrependBatch<Batch, OutputTensor>::type &delta_A,
            const PrependBatch<Batch, OutputTensor>::type &a,
            const PrependBatch<Batch, InputTensor>::type &a_prev)
            -> PrependBatch<Batch, InputTensor>::type {
            return Permute<0, 2, 1>(inner_ << BackwardArgs{
                Permute<0, 2, 1>(delta_A),
                Permute<0, 2, 1>(a),
                Permute<0, 2, 1>(a_prev)});
        }
    };

    // @doc: template<typename InnerRecipe> struct Transposed
    /** `BlockRecipe` for `TransposeBlock` */
    template<typename InnerRecipe>
    struct Transposed {
        // @doc: using Transposed::OutputTensor
        /** Alias for `InnerRecipe::OutputTensor` */
        using OutputTensor = InnerRecipe::OutputTensor;

        // @doc: template<IsTensor InputT> using Transposed::Resolve
        /** Given `IsTensor InputT`, define `TransposeBlock` type wrapping the resolved inner block */
        template<IsTensor InputT>
        using Resolve = TransposeBlock<typename InnerRecipe::template Resolve<InputT> >;
    };


    // @doc: template<size_t SeqLen, size_t... EmbDims> class LayerNormBlock
    /** Subtract mean from each token, divide by standard deviation, apply learned elementwise `gamma_` and `beta_` transformation */
    template<size_t SeqLen, size_t... EmbDims>
    class LayerNormBlock {
        static_assert(sizeof...(EmbDims) == 1, "LayerNormBlock: only 1D embeddings supported");
        // @doc: static constexpr size_t LayerNormBlock::EmbSize
        /** Total element count of (Rank-1 mandated) `EmbDims...` */
        static constexpr size_t EmbSize = TensorDimsProduct<EmbDims...>::value;

        // @doc: static constexpr float LayerNormBlock::inv_emb
        /** Precompute `1.f / EmbSize` */
        static constexpr float inv_emb = 1.f / static_cast<float>(EmbSize);

    public:
        // @doc: using LayerNormBlock::InputTensor
        /** Alias around `Tensor<SeqLen, EmbDims...>` */
        using InputTensor = Tensor<SeqLen, EmbDims...>;
        // @doc: using LayerNormBlock::OutputTensor
        /** Alias around `Tensor<SeqLen, EmbDims...>` */
        using OutputTensor = Tensor<SeqLen, EmbDims...>;
        // @doc: using LayerNormBlock::Scale_Type
        /** Alias around the type to be scaled (each `Tensor<EmbDims...>`) */
        using Scale_Type = Tensor<EmbDims...>;
        // @doc: using LayerNormBlock::Sigma_Type
        /** Scalar multiple for each of the `SeqLen` sub`Tensor<EmbDims...>`s */
        using Sigma_Type = Tensor<SeqLen>;

    private:
        // @doc: Param<Scale_Type> LayerNormBlock::gamma_
        /** Learned `Scale_Type` parameter used in Hadamard product to stretch distribution for each token */
        // @doc: Param<Scale_Type> LayerNormBlock::beta_
        /** Learned `Scale_Type` parameter used in elementwise sum to translate/shift mean for each token */
        Param<Scale_Type> gamma_, beta_;
        // @doc: mutable InputTensor LayerNormBlock::x_hat_
        /** `mutable` cache for pre-`gamma_`- and -`beta_`-transformed tokens */
        mutable InputTensor x_hat_{};
        // @doc: mutable Sigma_Type LayerNormBlock::inv_sigma_
        /** `mutable` cache for precomputed inverse of standard deviation */
        mutable Sigma_Type inv_sigma_{};

    public:
        // @doc: LayerNormBlock::LayerNormBlock()
        /** Default construct, fill `gamma_` with `1.f` */
        LayerNormBlock() { gamma_.value.apply([](float) { return 1.f; }); }

        // @doc: auto LayerNormBlock::all_params()
        /** Return `std::tuple` of `Param&` for `gamma_` and `beta_` */
        auto all_params() { return std::tie(gamma_, beta_); }
        // @doc: auto LayerNormBlock::all_params() const
        /** Return `std::tuple` of `const Param&` for `gamma_` and `beta_` */
        auto all_params() const { return std::tie(gamma_, beta_); }

        // @doc: OutputTensor LayerNormBlock::Forward(const InputTensor &X) const
        /** Subtract mean from each token, divide by standard deviation, apply learned elementwise `gamma_` and `beta_` transformation */
        OutputTensor Forward(const InputTensor &X) const {
            auto centered = BroadcastReduce<1, SubMean<EmbSize>, Add>(X);
            inv_sigma_ = Reduce<1, SqAdd>(centered);
            inv_sigma_.apply([](const float s) {
                return 1.f / std::sqrt(s * inv_emb + EPS);
            });
            x_hat_ = BroadcastMap<1, Mul>(centered, inv_sigma_);
            return BroadcastMapMove<0, Add>(BroadcastMap<0, Mul>(x_hat_, gamma_.value), beta_.value);
        }

        // @doc: InputTensor LayerNormBlock::Backward(const OutputTensor &delta_A, const OutputTensor &a, const InputTensor &a_prev)
        /**
         * Sum `delta_A` over all tokens for `beta_.grad`
         * Sum `delta_A * x_hat_` over all tokens for `gamma_.grad` (product rule 101)
         * Reverse `gamma_`- and `beta_`-transformations, reverse z-score, pass gradient back upstream
         */
        InputTensor Backward(const OutputTensor &delta_A,
                             const OutputTensor & /*a*/,
                             const InputTensor & /*a_prev*/) {
            beta_.grad += Reduce<0, Add>(delta_A);
            gamma_.grad += Reduce<0, Add>(delta_A * x_hat_);

            // broadcast delta * gamma along all tokens in sequence
            auto ds = BroadcastMap<0, Mul>(delta_A, gamma_.value);
            // reduce each embedding to sum of delta * gamma * x_hat_
            auto cov_ds = Reduce<1, Add>(ds * x_hat_);
            cov_ds *= inv_emb;
            // sum delta * gamma along each token, then broadcast up with mean subtraction
            auto ds_c = BroadcastReduceMove<1, SubMean<EmbSize>, Add>(std::move(ds));
            return BroadcastMapMove<1, Mul>(
                ds_c - BroadcastMap<1, Mul>(x_hat_, cov_ds),
                inv_sigma_);
        }

        // @doc: template<size_t Batch> auto LayerNormBlock::BatchedForward(const PrependBatch<Batch, InputTensor>::type &X) const -> PrependBatch<Batch, OutputTensor>::type
        /**
         * Slices each sample out of the batch, calls `Forward`, writes result back
         * Layer norm has no cross-sample dependencies so per-sample dispatch is correct
         */
        template<size_t Batch>
        auto BatchedForward(const PrependBatch<Batch, InputTensor>::type &X) const
            -> PrependBatch<Batch, OutputTensor>::type {
            typename PrependBatch<Batch, OutputTensor>::type result;
            constexpr size_t ss = InputTensor::Size;
            for (size_t b = 0; b < Batch; ++b) {
                InputTensor x_b;
                for (size_t i = 0; i < ss; ++i) x_b.flat(i) = X.flat(b * ss + i);
                const auto out = Forward(x_b);
                for (size_t i = 0; i < ss; ++i) result.flat(b * ss + i) = out.flat(i);
            }
            return result;
        }

        // @doc: template<size_t Batch> auto LayerNormBlock::BatchedBackward(const PrependBatch<Batch, OutputTensor>::type &delta_A, const PrependBatch<Batch, OutputTensor>::type &a, const PrependBatch<Batch, InputTensor>::type &a_prev) -> PrependBatch<Batch, InputTensor>::type
        /**
         * Slices each sample out of the batch, calls `Backward`, writes result back
         * `gamma_.grad` and `beta_.grad` accumulate across the loop then are scaled by `inv_batch`
         */
        template<size_t Batch>
        auto BatchedBackward(
            const PrependBatch<Batch, OutputTensor>::type &delta_A,
            const PrependBatch<Batch, OutputTensor>::type &a,
            const PrependBatch<Batch, InputTensor>::type &a_prev)
            -> PrependBatch<Batch, InputTensor>::type {
            typename PrependBatch<Batch, InputTensor>::type result;
            constexpr size_t ss = InputTensor::Size;
            for (size_t b = 0; b < Batch; ++b) {
                InputTensor dA_b, a_b, ap_b;
                for (size_t i = 0; i < ss; ++i) {
                    dA_b.flat(i) = delta_A.flat(b * ss + i);
                    a_b.flat(i) = a.flat(b * ss + i);
                    ap_b.flat(i) = a_prev.flat(b * ss + i);
                }
                const auto up = Backward(dA_b, a_b, ap_b);
                for (size_t i = 0; i < ss; ++i) result.flat(b * ss + i) = up.flat(i);
            }
            return result;
        }
    };


    // @doc: template<size_t... EmbDims> struct LayerNorm
    /** `BlockRecipe` for `LayerNormBlock` */
    template<size_t... EmbDims>
    struct LayerNorm {
        // @doc: using LayerNorm::OutputTensor
        /** Placeholder output type for recipe chain resolution */
        using OutputTensor = Tensor<1, EmbDims...>;

        // @doc: template<IsTensor InputT> using LayerNorm::Resolve
        /** Extracts `SeqLen` from `InputT` via `TensorFirstDim` and resolves to `LayerNormBlock<SeqLen, EmbDims...>` */
        template<IsTensor InputT>
        using Resolve = LayerNormBlock<TensorFirstDim<InputT>::value, EmbDims...>;
    };
}
