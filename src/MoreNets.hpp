#pragma once
#include "NetworkUtil.hpp"
#include "TensorOps.hpp"
#include "TensorReduce.hpp"
#include "TTTN_ML.hpp"

namespace TTTN {
    // @doc: template<IsTensor T> class IdentityBlock
    /** Identity pass-through block. TrainingCache is empty. */
    template<IsTensor T>
    class IdentityBlock {
    public:
        using InputTensor  = T;
        using OutputTensor = T;

        template<size_t> using TrainingCache = std::tuple<>;

        auto all_params()       { return std::tie(); }
        auto all_params() const { return std::tie(); }

        template<size_t Batch>
        static auto Forward(const typename PrependBatch<Batch, T>::type &X)
            -> typename PrependBatch<Batch, T>::type { return X; }

        template<size_t Batch>
        static auto Forward(const typename PrependBatch<Batch, T>::type &X, TrainingCache<Batch> &)
            -> typename PrependBatch<Batch, T>::type { return X; }

        template<size_t Batch>
        static auto Backward(const typename PrependBatch<Batch, T>::type &delta_A,
                             const typename PrependBatch<Batch, T>::type & /*a*/,
                             const typename PrependBatch<Batch, T>::type & /*a_prev*/,
                             const TrainingCache<Batch> &)
            -> typename PrependBatch<Batch, T>::type { return delta_A; }
    };


    // @doc: template<Block BlockA, Block BlockB> class ParallelBlock
    /**
     * Runs two blocks in parallel on the same input; sums their outputs.
     * TrainingCache holds the individual outputs of A and B (needed for their separate backwards)
     * plus each block's own training sub-cache.
     */
    template<Block BlockA, Block BlockB>
    class ParallelBlock {
        static_assert(std::is_same_v<typename BlockA::InputTensor,  typename BlockB::InputTensor>,
                      "ParallelBlock: InputTensor mismatch");
        static_assert(std::is_same_v<typename BlockA::OutputTensor, typename BlockB::OutputTensor>,
                      "ParallelBlock: OutputTensor mismatch");

    public:
        using InputTensor  = typename BlockA::InputTensor;
        using OutputTensor = typename BlockA::OutputTensor;

        template<size_t Batch>
        struct TrainingCacheData {
            typename PrependBatch<Batch, OutputTensor>::type a_out;
            typename PrependBatch<Batch, OutputTensor>::type b_out;
            typename BlockA::template TrainingCache<Batch> cache_a;
            typename BlockB::template TrainingCache<Batch> cache_b;
        };
        template<size_t Batch> using TrainingCache = TrainingCacheData<Batch>;

    private:
        BlockA a_;
        BlockB b_;

    public:
        const BlockA &block_a() const { return a_; }
        const BlockB &block_b() const { return b_; }

        auto all_params()       { return std::tuple_cat(a_.all_params(), b_.all_params()); }
        auto all_params() const { return std::tuple_cat(a_.all_params(), b_.all_params()); }

        void peek(SnapshotMap &out, const std::string &prefix) const {
            if constexpr (PeekableBlock<BlockA>) a_.peek(out, prefix);
            if constexpr (PeekableBlock<BlockB>) b_.peek(out, prefix);
        }

        // @doc: template<size_t Batch> auto ParallelBlock::Forward(X) const
        /** Pure inference: sums A and B outputs without populating any cache. */
        template<size_t Batch>
        auto Forward(const typename PrependBatch<Batch, InputTensor>::type &X) const
            -> typename PrependBatch<Batch, OutputTensor>::type {
            return a_.template Forward<Batch>(X) + b_.template Forward<Batch>(X);
        }

        // @doc: template<size_t Batch> auto ParallelBlock::Forward(X, cache) const
        template<size_t Batch>
        auto Forward(const typename PrependBatch<Batch, InputTensor>::type &X,
                     TrainingCache<Batch> &cache) const
            -> typename PrependBatch<Batch, OutputTensor>::type {
            cache.a_out = a_.template Forward<Batch>(X, cache.cache_a);
            cache.b_out = b_.template Forward<Batch>(X, cache.cache_b);
            return cache.a_out + cache.b_out;
        }

        // @doc: template<size_t Batch> auto ParallelBlock::Backward(...)
        template<size_t Batch>
        auto Backward(const typename PrependBatch<Batch, OutputTensor>::type &delta_A,
                      const typename PrependBatch<Batch, OutputTensor>::type & /*a*/,
                      const typename PrependBatch<Batch, InputTensor>::type  &a_prev,
                      const TrainingCache<Batch> &cache)
            -> typename PrependBatch<Batch, InputTensor>::type {
            // each sub-block gets its individual output as 'a', not the combined sum
            auto dX = a_.template Backward<Batch>(delta_A, cache.a_out, a_prev, cache.cache_a);
            dX += b_.template Backward<Batch>(delta_A, cache.b_out, a_prev, cache.cache_b);
            return dX;
        }
    };


    // @doc: template<Block B> using ResidualBlock
    template<Block B>
    using ResidualBlock = ParallelBlock<B, IdentityBlock<typename B::InputTensor>>;

    // @doc: template<typename RecipeA, typename RecipeB> struct Parallel
    template<typename RecipeA, typename RecipeB>
    struct Parallel {
        using OutputTensor = RecipeA::OutputTensor;
        template<IsTensor InputT>
        using Resolve = ParallelBlock<
            typename RecipeA::template Resolve<InputT>,
            typename RecipeB::template Resolve<InputT>>;
    };

    // @doc: template<typename Recipe> struct Residual
    template<typename Recipe>
    struct Residual {
        using OutputTensor = Recipe::OutputTensor;
        template<IsTensor InputT>
        using Resolve = ResidualBlock<typename Recipe::template Resolve<InputT>>;
    };


    // @doc: template<Block InnerBlock> class TransposeBlock
    /**
     * Wraps a block and runs it on the transposed (Permute<0,2,1>) input, transposing results back.
     * TrainingCache is the inner block's TrainingCache — no extra state needed since a/a_prev
     * are passed through permutation from the outer activations.
     */
    template<Block InnerBlock>
    class TransposeBlock {
    public:
        using InputTensor  = typename InnerBlock::InputTensor;
        using OutputTensor = typename InnerBlock::OutputTensor;
        static_assert(std::is_same_v<InputTensor, OutputTensor>,
                      "TransposeBlock requires InputTensor == OutputTensor");

        template<size_t Batch> using TrainingCache = typename InnerBlock::template TrainingCache<Batch>;

    private:
        InnerBlock inner_;

    public:
        const InnerBlock &inner() const { return inner_; }

        auto all_params()       { return inner_.all_params(); }
        auto all_params() const { return inner_.all_params(); }

        // @doc: template<size_t Batch> auto TransposeBlock::Forward(X) const
        template<size_t Batch>
        auto Forward(const typename PrependBatch<Batch, InputTensor>::type &X) const
            -> typename PrependBatch<Batch, OutputTensor>::type {
            return Permute<0, 2, 1>(inner_.template Forward<Batch>(Permute<0, 2, 1>(X)));
        }

        // @doc: template<size_t Batch> auto TransposeBlock::Forward(X, cache) const
        template<size_t Batch>
        auto Forward(const typename PrependBatch<Batch, InputTensor>::type &X,
                     TrainingCache<Batch> &cache) const
            -> typename PrependBatch<Batch, OutputTensor>::type {
            return Permute<0, 2, 1>(inner_.template Forward<Batch>(Permute<0, 2, 1>(X), cache));
        }

        // @doc: template<size_t Batch> auto TransposeBlock::Backward(...)
        template<size_t Batch>
        auto Backward(const typename PrependBatch<Batch, OutputTensor>::type &delta_A,
                      const typename PrependBatch<Batch, OutputTensor>::type &a,
                      const typename PrependBatch<Batch, InputTensor>::type  &a_prev,
                      const TrainingCache<Batch> &cache)
            -> typename PrependBatch<Batch, InputTensor>::type {
            return Permute<0, 2, 1>(
                inner_.template Backward<Batch>(
                    Permute<0, 2, 1>(delta_A),
                    Permute<0, 2, 1>(a),
                    Permute<0, 2, 1>(a_prev),
                    cache));
        }
    };

    // @doc: template<typename InnerRecipe> struct Transposed
    template<typename InnerRecipe>
    struct Transposed {
        using OutputTensor = InnerRecipe::OutputTensor;
        template<IsTensor InputT>
        using Resolve = TransposeBlock<typename InnerRecipe::template Resolve<InputT>>;
    };


    // @doc: template<size_t SeqLen, size_t... EmbDims> class LayerNormBlock
    /**
     * Per-token layer normalisation.
     * TrainingCache stores per-sample x_hat and inv_sigma so that batched backward is correct
     * (the old mutable-member approach was buggy: Forward overwrote the cache on each sample).
     */
    template<size_t SeqLen, size_t... EmbDims>
    class LayerNormBlock {
        static_assert(sizeof...(EmbDims) == 1, "LayerNormBlock: only 1D embeddings supported");
        static constexpr size_t EmbSize = TensorDimsProduct<EmbDims...>::value;
        static constexpr float  inv_emb = 1.f / static_cast<float>(EmbSize);

    public:
        using InputTensor  = Tensor<SeqLen, EmbDims...>;
        using OutputTensor = Tensor<SeqLen, EmbDims...>;
        using Scale_Type   = Tensor<EmbDims...>;
        using Sigma_Type   = Tensor<SeqLen>;

        template<size_t Batch>
        struct TrainingCacheData {
            typename PrependBatch<Batch, InputTensor>::type x_hat;      // [Batch, SeqLen, EmbDim]
            typename PrependBatch<Batch, Sigma_Type>::type  inv_sigma;   // [Batch, SeqLen]
        };
        template<size_t Batch> using TrainingCache = TrainingCacheData<Batch>;

    private:
        Param<Scale_Type> gamma_, beta_;

        // per-sample helper used by both Forward overloads
        struct SampleResult { OutputTensor out; InputTensor x_hat; Sigma_Type inv_sigma; };

        SampleResult forward_sample(const InputTensor &X) const {
            auto centered  = BroadcastReduce<1, SubMean<EmbSize>, Add>(X);
            Sigma_Type inv_s = Reduce<1, SqAdd>(centered);
            inv_s.apply([](float s) { return 1.f / std::sqrt(s * inv_emb + EPS); });
            const auto x_hat = BroadcastMap<1, Mul>(centered, inv_s);
            const auto out   = BroadcastMapMove<0, Add>(BroadcastMap<0, Mul>(x_hat, gamma_.value), beta_.value);
            return {out, x_hat, inv_s};
        }

        InputTensor backward_sample(const OutputTensor &dA,
                                    const InputTensor  &x_hat,
                                    const Sigma_Type   &inv_sigma) {
            beta_.grad  += Reduce<0, Add>(dA);
            gamma_.grad += Reduce<0, Add>(dA * x_hat);

            auto ds      = BroadcastMap<0, Mul>(dA, gamma_.value);
            auto cov_ds  = Reduce<1, Add>(ds * x_hat);
            cov_ds      *= inv_emb;
            auto ds_c    = BroadcastReduceMove<1, SubMean<EmbSize>, Add>(std::move(ds));
            return BroadcastMapMove<1, Mul>(ds_c - BroadcastMap<1, Mul>(x_hat, cov_ds), inv_sigma);
        }

    public:
        LayerNormBlock() { gamma_.value.apply([](float) { return 1.f; }); }

        auto all_params()       { return std::tie(gamma_, beta_); }
        auto all_params() const { return std::tie(gamma_, beta_); }

        // @doc: template<size_t Batch> auto LayerNormBlock::Forward(X) const
        /** Pure inference forward — no cache. */
        template<size_t Batch>
        auto Forward(const typename PrependBatch<Batch, InputTensor>::type &X) const
            -> typename PrependBatch<Batch, OutputTensor>::type {
            typename PrependBatch<Batch, OutputTensor>::type result;
            constexpr size_t ss = InputTensor::Size;
            for (size_t b = 0; b < Batch; ++b) {
                InputTensor x_b;
                for (size_t i = 0; i < ss; ++i) x_b.flat(i) = X.flat(b * ss + i);
                const auto [out, xh, is] = forward_sample(x_b);
                for (size_t i = 0; i < ss; ++i) result.flat(b * ss + i) = out.flat(i);
            }
            return result;
        }

        // @doc: template<size_t Batch> auto LayerNormBlock::Forward(X, cache) const
        /** Training forward — populates cache with per-sample x_hat and inv_sigma. */
        template<size_t Batch>
        auto Forward(const typename PrependBatch<Batch, InputTensor>::type &X,
                     TrainingCache<Batch> &cache) const
            -> typename PrependBatch<Batch, OutputTensor>::type {
            typename PrependBatch<Batch, OutputTensor>::type result;
            constexpr size_t ss  = InputTensor::Size;
            constexpr size_t sss = Sigma_Type::Size;
            for (size_t b = 0; b < Batch; ++b) {
                InputTensor x_b;
                for (size_t i = 0; i < ss; ++i) x_b.flat(i) = X.flat(b * ss + i);
                const auto [out, xh, inv_s] = forward_sample(x_b);
                for (size_t i = 0; i < ss; ++i)  result.flat(b * ss + i)   = out.flat(i);
                for (size_t i = 0; i < ss; ++i)  cache.x_hat.flat(b * ss + i) = xh.flat(i);
                for (size_t i = 0; i < sss; ++i) cache.inv_sigma.flat(b * sss + i) = inv_s.flat(i);
            }
            return result;
        }

        // @doc: template<size_t Batch> auto LayerNormBlock::Backward(...)
        /** Backward — reads per-sample x_hat and inv_sigma from cache. */
        template<size_t Batch>
        auto Backward(const typename PrependBatch<Batch, OutputTensor>::type &delta_A,
                      const typename PrependBatch<Batch, OutputTensor>::type & /*a*/,
                      const typename PrependBatch<Batch, InputTensor>::type  & /*a_prev*/,
                      const TrainingCache<Batch> &cache)
            -> typename PrependBatch<Batch, InputTensor>::type {
            typename PrependBatch<Batch, InputTensor>::type result;
            constexpr size_t ss  = InputTensor::Size;
            constexpr size_t sss = Sigma_Type::Size;
            for (size_t b = 0; b < Batch; ++b) {
                OutputTensor dA_b;
                InputTensor  xh_b;
                Sigma_Type   is_b;
                for (size_t i = 0; i < ss; ++i)  dA_b.flat(i) = delta_A.flat(b * ss + i);
                for (size_t i = 0; i < ss; ++i)  xh_b.flat(i) = cache.x_hat.flat(b * ss + i);
                for (size_t i = 0; i < sss; ++i) is_b.flat(i)  = cache.inv_sigma.flat(b * sss + i);
                const auto up = backward_sample(dA_b, xh_b, is_b);
                for (size_t i = 0; i < ss; ++i) result.flat(b * ss + i) = up.flat(i);
            }
            return result;
        }
    };


    // @doc: template<size_t... EmbDims> struct LayerNorm
    template<size_t... EmbDims>
    struct LayerNorm {
        using OutputTensor = Tensor<1, EmbDims...>;
        template<IsTensor InputT>
        using Resolve = LayerNormBlock<TensorFirstDim<InputT>::value, EmbDims...>;
    };
}
