#pragma once
#include <limits>
#include "TensorContract.hpp"
#include "NetworkUtil.hpp"

namespace TTTN {
    // @doc: template<size_t SeqLen, size_t Heads, bool Masked, size_t... EmbDims> class MultiHeadAttentionBlock
    /**
     * Multi-head self-attention over sequences.
     * TrainingCache holds Q, K, V, attn weights, and attended_perm (input to lc_O_).
     * attn_weights_ mutable member is kept solely for peek()/snapshot visualization.
     */
    template<size_t SeqLen, size_t Heads, bool Masked, size_t... EmbDims>
    class MultiHeadAttentionBlock {
    public:
        using InputTensor   = Tensor<SeqLen, EmbDims...>;
        using OutputTensor  = Tensor<SeqLen, EmbDims...>;

        static constexpr size_t N_emb   = sizeof...(EmbDims);
        static constexpr size_t EmbSize = TensorDimsProduct<EmbDims...>::value;
        static constexpr size_t HeadDim = EmbSize / Heads;
        static_assert(EmbSize % Heads == 0, "Heads must divide EmbSize");

        using W_QKV_Type    = Tensor<Heads, HeadDim, EmbDims...>;
        using W_O_Type      = Tensor<EmbDims..., Heads, HeadDim>;
        using QKV_Type      = Tensor<SeqLen, Heads, HeadDim>;
        using Scores_Type   = Tensor<Heads, SeqLen, SeqLen>;

        template<size_t Batch>
        struct TrainingCacheData {
            Tensor<Batch, SeqLen, Heads, HeadDim> Q, K, V;
            Tensor<Batch, Heads, SeqLen, SeqLen>  attn;
            Tensor<Batch, SeqLen, Heads, HeadDim> attended_perm; // permuted attended; input to lc_O_
        };
        template<size_t Batch> using TrainingCache = TrainingCacheData<Batch>;

    private:
        // QKV: [SeqLen,EmbDims...] x [Heads,HeadDim,EmbDims...] -> [SeqLen,Heads,HeadDim], NFree=1
        LearnedContraction<InputTensor, QKV_Type,  1> lc_Q_, lc_K_, lc_V_;
        // Out: [SeqLen,Heads,HeadDim] x [EmbDims...,Heads,HeadDim] -> [SeqLen,EmbDims...], NFree=1
        LearnedContraction<QKV_Type, OutputTensor, 1> lc_O_;

        mutable Scores_Type attn_weights_{}; // visualization only

    public:
        auto all_params()       { return std::tie(lc_Q_.W_, lc_K_.W_, lc_V_.W_, lc_O_.W_); }
        auto all_params() const { return std::tie(lc_Q_.W_, lc_K_.W_, lc_V_.W_, lc_O_.W_); }

        const Scores_Type &attn_weights() const { return attn_weights_; }
        void peek(SnapshotMap &out, const std::string &prefix) const {
            snap_add(out, prefix + "attn_weights", attn_weights_);
        }

        MultiHeadAttentionBlock() = default;

        // @doc: template<size_t Batch> Forward(X) -- pure inference.
        template<size_t Batch>
        Tensor<Batch, SeqLen, EmbDims...> Forward(const Tensor<Batch, SeqLen, EmbDims...> &X) const {
            const float inv_sqrt = 1.f / std::sqrt(static_cast<float>(HeadDim));
            const auto bQ = X >> lc_Q_;
            const auto bK = X >> lc_K_;
            const auto bV = X >> lc_V_;
            auto scores = BatchContract<AxisList<0,2>{}, AxisList<0,2>{}, AxisList<3>{}, AxisList<3>{}, Mul, Add>(bQ, bK);
            scores *= inv_sqrt;
            if constexpr (Masked) {
                constexpr float neg_inf = -std::numeric_limits<float>::infinity();
                for (size_t b = 0; b < Batch; ++b)
                    for (size_t h = 0; h < Heads; ++h)
                        for (size_t q = 0; q < SeqLen - 1; ++q)
                            std::fill_n(&scores(b, h, q, q + 1), SeqLen - q - 1, neg_inf);
            }
            const auto b_attn     = Softmax<3>(scores);
            attn_weights_         = TensorIndex<0, 0>(b_attn);
            const auto b_attended = BatchContract<AxisList<0,1>{}, AxisList<0,2>{}, AxisList<3>{}, AxisList<1>{}, Mul, Add>(b_attn, bV);
            return Permute<0, 2, 1, 3>(b_attended) >> lc_O_;
        }

        // @doc: template<size_t Batch> Forward(X, cache) -- training forward; populates cache.
        template<size_t Batch>
        Tensor<Batch, SeqLen, EmbDims...> Forward(const Tensor<Batch, SeqLen, EmbDims...> &X,
                                                  TrainingCache<Batch> &cache) const {
            const float inv_sqrt = 1.f / std::sqrt(static_cast<float>(HeadDim));
            cache.Q = X >> lc_Q_;
            cache.K = X >> lc_K_;
            cache.V = X >> lc_V_;
            auto scores = BatchContract<AxisList<0,2>{}, AxisList<0,2>{}, AxisList<3>{}, AxisList<3>{}, Mul, Add>(cache.Q, cache.K);
            scores *= inv_sqrt;
            if constexpr (Masked) {
                constexpr float neg_inf = -std::numeric_limits<float>::infinity();
                for (size_t b = 0; b < Batch; ++b)
                    for (size_t h = 0; h < Heads; ++h)
                        for (size_t q = 0; q < SeqLen - 1; ++q)
                            std::fill_n(&scores(b, h, q, q + 1), SeqLen - q - 1, neg_inf);
            }
            cache.attn          = Softmax<3>(scores);
            attn_weights_       = TensorIndex<0, 0>(cache.attn);
            const auto b_att    = BatchContract<AxisList<0,1>{}, AxisList<0,2>{}, AxisList<3>{}, AxisList<1>{}, Mul, Add>(cache.attn, cache.V);
            cache.attended_perm = Permute<0, 2, 1, 3>(b_att);
            return cache.attended_perm >> lc_O_;
        }

        // @doc: template<size_t Batch> Backward(dY, a, a_prev, cache).
        template<size_t Batch>
        Tensor<Batch, SeqLen, EmbDims...> Backward(const Tensor<Batch, SeqLen, EmbDims...> &delta_A,
                                                   const Tensor<Batch, SeqLen, EmbDims...> & /*a*/,
                                                   const Tensor<Batch, SeqLen, EmbDims...> &a_prev,
                                                   const TrainingCache<Batch> &cache) {
            const float inv_sqrt = 1.f / std::sqrt(static_cast<float>(HeadDim));

            // W_O grad: attended_perm was the input to lc_O_
            const auto d_attended = lc_O_.backward(delta_A, cache.attended_perm);

            const auto d_attn = BatchContract<AxisList<0,2>{}, AxisList<0,2>{}, AxisList<3>{}, AxisList<3>{}, Mul, Add>(d_attended, cache.V);
            const auto d_V    = BatchContract<AxisList<0,1>{}, AxisList<0,2>{}, AxisList<2>{}, AxisList<1>{}, Mul, Add>(cache.attn, d_attended);

            auto d_scores = SoftmaxPrime<3>(d_attn, cache.attn);
            d_scores     *= inv_sqrt;

            const auto d_Q = BatchContract<AxisList<0,1>{}, AxisList<0,2>{}, AxisList<3>{}, AxisList<1>{}, Mul, Add>(d_scores, cache.K);
            const auto d_K = BatchContract<AxisList<0,1>{}, AxisList<0,2>{}, AxisList<2>{}, AxisList<1>{}, Mul, Add>(d_scores, cache.Q);

            // W_Q/K/V grads: a_prev is the input to all three projections
            auto result  = lc_Q_.backward(Permute<0, 2, 1, 3>(d_Q), a_prev);
            result      += lc_K_.backward(Permute<0, 2, 1, 3>(d_K), a_prev);
            result      += lc_V_.backward(Permute<0, 2, 1, 3>(d_V), a_prev);
            return result;
        }
    };


    // @doc: template<size_t SeqLenQ, size_t SeqLenKV, size_t Heads, size_t EmbDim> class MultiHeadCrossAttentionBlock
    /**
     * Multi-head cross-attention. Input is packed [Q-side | KV-side] along the sequence axis.
     * TrainingCache holds Q, K, V, attn, attended_perm.
     */
    template<size_t SeqLenQ, size_t SeqLenKV, size_t Heads, size_t EmbDim>
    class MultiHeadCrossAttentionBlock {
    public:
        using InputTensor   = Tensor<SeqLenQ + SeqLenKV, EmbDim>;
        using OutputTensor  = Tensor<SeqLenQ, EmbDim>;
        using QSideTensor   = Tensor<SeqLenQ, EmbDim>;
        using KVSideTensor  = Tensor<SeqLenKV, EmbDim>;

        static constexpr size_t EmbSize = EmbDim;
        static constexpr size_t HeadDim = EmbDim / Heads;
        static_assert(EmbDim % Heads == 0, "Heads must divide EmbDim");

        using W_Q_Type     = Tensor<Heads, HeadDim, EmbDim>;
        using W_KV_Type    = Tensor<Heads, HeadDim, EmbDim>;
        using W_O_Type     = Tensor<EmbDim, Heads, HeadDim>;
        using QProj_Type   = Tensor<SeqLenQ,  Heads, HeadDim>;
        using KVProj_Type  = Tensor<SeqLenKV, Heads, HeadDim>;
        using Scores_Type  = Tensor<Heads, SeqLenQ, SeqLenKV>;
        using Attended_Type= Tensor<Heads, SeqLenQ, HeadDim>;

        template<size_t Batch>
        struct TrainingCacheData {
            Tensor<Batch, SeqLenQ,  Heads, HeadDim> Q;
            Tensor<Batch, SeqLenKV, Heads, HeadDim> K, V;
            Tensor<Batch, Heads, SeqLenQ, SeqLenKV> attn;
            Tensor<Batch, SeqLenQ, Heads, HeadDim>  attended_perm;
        };
        template<size_t Batch> using TrainingCache = TrainingCacheData<Batch>;

    private:
        LearnedContraction<QSideTensor,  QProj_Type,  1> lc_Q_;
        LearnedContraction<KVSideTensor, KVProj_Type, 1> lc_K_, lc_V_;
        LearnedContraction<QProj_Type,   OutputTensor, 1> lc_O_;

        mutable Scores_Type attn_weights_{}; // visualization only

    public:
        auto all_params()       { return std::tie(lc_Q_.W_, lc_K_.W_, lc_V_.W_, lc_O_.W_); }
        auto all_params() const { return std::tie(lc_Q_.W_, lc_K_.W_, lc_V_.W_, lc_O_.W_); }

        const Scores_Type &attn_weights() const { return attn_weights_; }
        void peek(SnapshotMap &out, const std::string &prefix) const {
            snap_add(out, prefix + "cross_attn_weights", attn_weights_);
        }

        MultiHeadCrossAttentionBlock() = default;

        // @doc: template<size_t Batch> Forward(X) -- pure inference.
        template<size_t Batch>
        Tensor<Batch, SeqLenQ, EmbDim> Forward(const Tensor<Batch, SeqLenQ + SeqLenKV, EmbDim> &X) const {
            const float inv_sqrt = 1.f / std::sqrt(static_cast<float>(HeadDim));
            Tensor<Batch, SeqLenQ, EmbDim>  inQ;
            Tensor<Batch, SeqLenKV, EmbDim> inKV;
            std::tie(inQ, inKV) = SplitAxis<1, SeqLenQ>(X);
            const auto bQ = inQ  >> lc_Q_;
            const auto bK = inKV >> lc_K_;
            const auto bV = inKV >> lc_V_;
            auto scores = BatchContract<AxisList<0,2>{}, AxisList<0,2>{}, AxisList<3>{}, AxisList<3>{}, Mul, Add>(bQ, bK);
            scores *= inv_sqrt;
            const auto b_attn   = Softmax<3>(scores);
            attn_weights_       = TensorIndex<0, 0>(b_attn);
            const auto b_att    = BatchContract<AxisList<0,1>{}, AxisList<0,2>{}, AxisList<3>{}, AxisList<1>{}, Mul, Add>(b_attn, bV);
            return Permute<0, 2, 1, 3>(b_att) >> lc_O_;
        }

        // @doc: template<size_t Batch> Forward(X, cache) -- training forward.
        template<size_t Batch>
        Tensor<Batch, SeqLenQ, EmbDim> Forward(const Tensor<Batch, SeqLenQ + SeqLenKV, EmbDim> &X,
                                               TrainingCache<Batch> &cache) const {
            const float inv_sqrt = 1.f / std::sqrt(static_cast<float>(HeadDim));
            Tensor<Batch, SeqLenQ, EmbDim>  inQ;
            Tensor<Batch, SeqLenKV, EmbDim> inKV;
            std::tie(inQ, inKV) = SplitAxis<1, SeqLenQ>(X);
            cache.Q = inQ  >> lc_Q_;
            cache.K = inKV >> lc_K_;
            cache.V = inKV >> lc_V_;
            auto scores = BatchContract<AxisList<0,2>{}, AxisList<0,2>{}, AxisList<3>{}, AxisList<3>{}, Mul, Add>(cache.Q, cache.K);
            scores *= inv_sqrt;
            cache.attn        = Softmax<3>(scores);
            attn_weights_     = TensorIndex<0, 0>(cache.attn);
            const auto b_att  = BatchContract<AxisList<0,1>{}, AxisList<0,2>{}, AxisList<3>{}, AxisList<1>{}, Mul, Add>(cache.attn, cache.V);
            cache.attended_perm = Permute<0, 2, 1, 3>(b_att);
            return cache.attended_perm >> lc_O_;
        }

        // @doc: template<size_t Batch> Backward(dY, a, a_prev, cache).
        template<size_t Batch>
        Tensor<Batch, SeqLenQ + SeqLenKV, EmbDim> Backward(
            const Tensor<Batch, SeqLenQ, EmbDim> &delta_A,
            const Tensor<Batch, SeqLenQ, EmbDim> & /*a*/,
            const Tensor<Batch, SeqLenQ + SeqLenKV, EmbDim> &a_prev,
            const TrainingCache<Batch> &cache) {
            const float inv_sqrt = 1.f / std::sqrt(static_cast<float>(HeadDim));

            Tensor<Batch, SeqLenQ, EmbDim>  inQ;
            Tensor<Batch, SeqLenKV, EmbDim> inKV;
            std::tie(inQ, inKV) = SplitAxis<1, SeqLenQ>(a_prev);

            const auto d_attended = lc_O_.backward(delta_A, cache.attended_perm);

            const auto d_attn = BatchContract<AxisList<0,2>{}, AxisList<0,2>{}, AxisList<3>{}, AxisList<3>{}, Mul, Add>(d_attended, cache.V);
            const auto d_V    = Permute<0,2,1,3>(BatchContract<AxisList<0,1>{}, AxisList<0,2>{}, AxisList<2>{}, AxisList<1>{}, Mul, Add>(cache.attn, d_attended));

            auto d_scores = SoftmaxPrime<3>(d_attn, cache.attn);
            d_scores     *= inv_sqrt;

            const auto d_Q = Permute<0,2,1,3>(BatchContract<AxisList<0,1>{}, AxisList<0,2>{}, AxisList<3>{}, AxisList<1>{}, Mul, Add>(d_scores, cache.K));
            const auto d_K = Permute<0,2,1,3>(BatchContract<AxisList<0,1>{}, AxisList<0,2>{}, AxisList<2>{}, AxisList<1>{}, Mul, Add>(d_scores, cache.Q));

            const auto d_Q_in = lc_Q_.backward(d_Q, inQ);
            auto d_KV_in      = lc_K_.backward(d_K, inKV);
            d_KV_in          += lc_V_.backward(d_V, inKV);

            return ConcatAxis<1>(d_Q_in, d_KV_in);
        }
    };


    // @doc: template<size_t Heads, size_t... EmbDims> struct MHAttention
    template<size_t Heads, size_t... EmbDims>
    struct MHAttention {
        using OutputTensor = Tensor<1, EmbDims...>;
        template<typename InputT> requires IsTensor<InputT>
        using Resolve = MultiHeadAttentionBlock<TensorFirstDim<InputT>::value, Heads, false, EmbDims...>;
    };

    // @doc: template<size_t Heads, size_t... EmbDims> struct MHCausalAttention
    template<size_t Heads, size_t... EmbDims>
    struct MHCausalAttention {
        using OutputTensor = Tensor<1, EmbDims...>;
        template<typename InputT> requires IsTensor<InputT>
        using Resolve = MultiHeadAttentionBlock<TensorFirstDim<InputT>::value, Heads, true, EmbDims...>;
    };
} // namespace TTTN
