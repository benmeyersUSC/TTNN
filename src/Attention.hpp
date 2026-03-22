#pragma once
#include "NetworkUtil.hpp"

namespace TTTN {

    // MULTI-HEAD SELF-ATTENTION BLOCK (generalized embedding rank)
    //
    // Template parameters:
    //   SeqLen  — sequence length (tokens)
    //   Heads   — number of attention heads
    //   HeadDim — dimension per head
    //   EmbDims — embedding shape (variadic; Tensor<EmbDim> for standard LLM, any rank supported)
    //
    // Constraint: Heads * HeadDim == TensorDimsProduct<EmbDims...>::value  (= EmbSize)
    //
    // Types:
    //   InputTensor  = OutputTensor = Tensor<SeqLen, EmbDims...>  (shape-preserving)
    //   W_Q, W_K, W_V : Tensor<Heads, HeadDim, EmbDims...>   project EmbDims → (Heads, HeadDim)
    //   W_O            : Tensor<EmbDims..., Heads, HeadDim>   project (Heads, HeadDim) → EmbDims
    //
    // Forward:
    //   For each token s:
    //       x_s = TensorIndex<0>(X, s)                                 → Tensor<EmbDims...>
    //       q_s = ΣΠ<N_emb>(W_Q, x_s)                                 → Tensor<Heads, HeadDim>
    //       (same for K, V)
    //   For each head h:
    //       scores_h  = ΣΠ<1>(Q_h, Permute<1,0>(K_h)) / √HeadDim      → Tensor<SeqLen, SeqLen>
    //       weights_h = Softmax<1>(scores_h)                            → Tensor<SeqLen, SeqLen>
    //       attended_h = ΣΠ<1>(weights_h, V_h)                         → Tensor<SeqLen, HeadDim>
    //   For each token s:
    //       out_s = ΣΠ<2>(W_O, attended_s)                             → Tensor<EmbDims...>
    //
    // Backward (chain rule through all of the above):
    //   Caches X, Q, K, V, attn_weights, attended from the forward pass.
    //   W_Q_T = WTBlockSwapPerm<2, N_emb>(W_Q) → Tensor<EmbDims..., Heads, HeadDim>
    //   W_O_T = WTBlockSwapPerm<N_emb, 2>(W_O) → Tensor<Heads, HeadDim, EmbDims...>
    //   upstream dX: ΣΠ<2>(W_Q_T, dq_s) + ΣΠ<2>(W_K_T, dk_s) + ΣΠ<2>(W_V_T, dv_s)

    template<size_t SeqLen, size_t Heads, size_t... EmbDims>
    class MultiHeadAttentionBlock {
    public:
        using InputTensor  = Tensor<SeqLen, EmbDims...>;
        using OutputTensor = Tensor<SeqLen, EmbDims...>;

        static constexpr size_t N_emb   = sizeof...(EmbDims);
        static constexpr size_t EmbSize = TensorDimsProduct<EmbDims...>::value;
        static constexpr size_t HeadDim = EmbSize / Heads;
        static_assert(EmbSize % Heads == 0,
                      "MultiHeadAttentionBlock: EmbSize must be divisible by Heads");

        using W_QKV_Type  = Tensor<Heads, HeadDim, EmbDims...>;   // shape of W_Q, W_K, W_V
        using W_O_Type    = Tensor<EmbDims..., Heads, HeadDim>;    // shape of W_O
        using QKV_Type    = Tensor<SeqLen, Heads, HeadDim>;        // projected Q, K, V
        using Scores_Type = Tensor<Heads, SeqLen, SeqLen>;         // attention weight matrix

        static constexpr size_t ParamCount =
            3 * W_QKV_Type::Size + W_O_Type::Size;

    private:
        // weights
        W_QKV_Type W_Q_, W_K_, W_V_;
        W_O_Type   W_O_;

        // Adam moments
        W_QKV_Type mW_Q_{}, vW_Q_, mW_K_{}, vW_K_{}, mW_V_{}, vW_V_{};
        W_O_Type   mW_O_{}, vW_O_{};

        // gradients
        W_QKV_Type dW_Q_{}, dW_K_{}, dW_V_{};
        W_O_Type   dW_O_{};

        // forward-pass cache (needed by Backward) — mutable so Forward can be const
        mutable InputTensor  X_cache_{};
        mutable QKV_Type     Q_{}, K_{}, V_{};
        mutable Scores_Type  attn_weights_{};
        mutable QKV_Type     attended_{};

    public:
        MultiHeadAttentionBlock() {
            XavierInitMD(W_Q_, EmbSize, HeadDim);
            XavierInitMD(W_K_, EmbSize, HeadDim);
            XavierInitMD(W_V_, EmbSize, HeadDim);
            XavierInitMD(W_O_, EmbSize, EmbSize);
        }

        // ─── FORWARD ────────────────────────────────────────────────────────────

        OutputTensor Forward(const InputTensor& X) const {
            const float inv_sqrt = 1.f / std::sqrt(static_cast<float>(HeadDim));
            X_cache_ = X;

            // 1. Project each token to (Heads, HeadDim) Q, K, V
            Q_.fill(0.f); K_.fill(0.f); V_.fill(0.f);
            for (size_t s = 0; s < SeqLen; ++s) {
                const auto x_s = TensorIndex<0>(X, s);
                TensorIndexAdd<0>(Q_, s, ΣΠ<N_emb>(W_Q_, x_s));
                TensorIndexAdd<0>(K_, s, ΣΠ<N_emb>(W_K_, x_s));
                TensorIndexAdd<0>(V_, s, ΣΠ<N_emb>(W_V_, x_s));
            }

            // 2. Per-head: scale dot-product scores → softmax weights → weighted V
            attn_weights_.fill(0.f);
            attended_.fill(0.f);
            for (size_t h = 0; h < Heads; ++h) {
                const auto Q_h   = TensorIndex<1>(Q_, h);             // Tensor<SeqLen, HeadDim>
                const auto K_h_T = Permute<1,0>(TensorIndex<1>(K_, h)); // Tensor<HeadDim, SeqLen>
                const auto V_h   = TensorIndex<1>(V_, h);             // Tensor<SeqLen, HeadDim>

                // scores_h[q,k] = dot(Q_h[q,:], K_h[:,k]) / sqrt(HeadDim)
                const auto scores_h  = ΣΠ<1>(Q_h, K_h_T) * inv_sqrt; // Tensor<SeqLen, SeqLen>
                const auto weights_h = Softmax<1>(scores_h);           // Tensor<SeqLen, SeqLen>

                // cache weights for backward
                TensorIndexAdd<0>(attn_weights_, h, weights_h);

                // attended_h = weights_h @ V_h: Tensor<SeqLen,SeqLen> × Tensor<SeqLen,HeadDim>
                TensorIndexAdd<1>(attended_, h, ΣΠ<1>(weights_h, V_h));
            }

            // 3. Output projection: contract (Heads, HeadDim) → EmbDims per token
            OutputTensor output;
            output.fill(0.f);
            for (size_t s = 0; s < SeqLen; ++s) {
                const auto att_s = TensorIndex<0>(attended_, s);   // Tensor<Heads, HeadDim>
                TensorIndexAdd<0>(output, s, ΣΠ<2>(W_O_, att_s)); // Tensor<EmbDims...>
            }
            return output;
        }

        // ─── BACKWARD ───────────────────────────────────────────────────────────

        void ZeroGrad() {
            dW_Q_.fill(0.f); dW_K_.fill(0.f); dW_V_.fill(0.f); dW_O_.fill(0.f);
        }

        // delta_A: dL/dOutput.  a: output (unused — attention has no simple pointwise activation).
        // a_prev: input X (same as X_cache_, provided for interface compliance).
        InputTensor Backward(const OutputTensor& delta_A,
                             const OutputTensor& /*a*/,
                             const InputTensor&  /*a_prev*/) {
            const float inv_sqrt = 1.f / std::sqrt(static_cast<float>(HeadDim));

            // W_Q_T: Tensor<EmbDims..., Heads, HeadDim>  (block-swap W_Q's two halves)
            // W_K_T, W_V_T: same permutation
            // W_O_T: Tensor<Heads, HeadDim, EmbDims...>  (block-swap W_O's two halves)
            const auto W_Q_T = PermuteFromHolder<WTBlockSwapPerm<2, N_emb>>(
                W_Q_, std::make_index_sequence<2 + N_emb>{});
            const auto W_K_T = PermuteFromHolder<WTBlockSwapPerm<2, N_emb>>(
                W_K_, std::make_index_sequence<2 + N_emb>{});
            const auto W_V_T = PermuteFromHolder<WTBlockSwapPerm<2, N_emb>>(
                W_V_, std::make_index_sequence<2 + N_emb>{});
            const auto W_O_T = PermuteFromHolder<WTBlockSwapPerm<N_emb, 2>>(
                W_O_, std::make_index_sequence<N_emb + 2>{});

            // Step 1: backward through output projection W_O
            // out_s = ΣΠ<2>(W_O, att_s)  →  dW_O += outer(dout_s, att_s), d_att_s = ΣΠ<N_emb>(W_O_T, dout_s)
            QKV_Type d_attended;
            d_attended.fill(0.f);
            for (size_t s = 0; s < SeqLen; ++s) {
                const auto dout_s = TensorIndex<0>(delta_A,  s);   // Tensor<EmbDims...>
                const auto att_s  = TensorIndex<0>(attended_, s);   // Tensor<Heads, HeadDim>
                dW_O_ += ΣΠ<0>(dout_s, att_s);                     // outer → Tensor<EmbDims..., Heads, HeadDim>
                TensorIndexAdd<0>(d_attended, s, ΣΠ<N_emb>(W_O_T, dout_s));
            }

            // Steps 2–4 (one head loop): attended, softmax, scores
            // attended_h = weights_h @ V_h
            // weights_h  = Softmax<1>(scores_h * inv_sqrt)
            // scores_h   = Q_h @ K_h^T
            QKV_Type d_Q, d_K, d_V;
            d_Q.fill(0.f); d_K.fill(0.f); d_V.fill(0.f);
            for (size_t h = 0; h < Heads; ++h) {
                const auto weights_h  = TensorIndex<0>(attn_weights_, h); // Tensor<SeqLen, SeqLen>
                const auto V_h        = TensorIndex<1>(V_,            h); // Tensor<SeqLen, HeadDim>
                const auto Q_h        = TensorIndex<1>(Q_,            h); // Tensor<SeqLen, HeadDim>
                const auto K_h        = TensorIndex<1>(K_,            h); // Tensor<SeqLen, HeadDim>
                const auto d_att_h    = TensorIndex<1>(d_attended,    h); // Tensor<SeqLen, HeadDim>

                // Step 2: d_weights_h = d_att_h @ V_h^T,   d_V_h = weights_h^T @ d_att_h
                const auto d_weights_h = ΣΠ<1>(d_att_h, Permute<1,0>(V_h));         // Tensor<SeqLen, SeqLen>
                TensorIndexAdd<1>(d_V, h, ΣΠ<1>(Permute<1,0>(weights_h), d_att_h)); // Tensor<SeqLen, HeadDim>

                // Step 3: peel off softmax (and the 1/√HeadDim scale)
                const auto d_scores_h = SoftmaxPrime<1>(d_weights_h, weights_h) * inv_sqrt; // Tensor<SeqLen, SeqLen>

                // Step 4: d_Q_h = d_scores_h @ K_h,   d_K_h = d_scores_h^T @ Q_h
                TensorIndexAdd<1>(d_Q, h, ΣΠ<1>(d_scores_h,           K_h)); // Tensor<SeqLen, HeadDim>
                TensorIndexAdd<1>(d_K, h, ΣΠ<1>(Permute<1,0>(d_scores_h), Q_h)); // Tensor<SeqLen, HeadDim>
            }

            // Step 5: backward through Q, K, V projections, accumulate dW and upstream dX
            InputTensor dX;
            dX.fill(0.f);
            for (size_t s = 0; s < SeqLen; ++s) {
                const auto x_s  = TensorIndex<0>(X_cache_, s);  // Tensor<EmbDims...>
                const auto dq_s = TensorIndex<0>(d_Q, s);        // Tensor<Heads, HeadDim>
                const auto dk_s = TensorIndex<0>(d_K, s);
                const auto dv_s = TensorIndex<0>(d_V, s);

                dW_Q_ += ΣΠ<0>(dq_s, x_s);   // outer → Tensor<Heads, HeadDim, EmbDims...>
                dW_K_ += ΣΠ<0>(dk_s, x_s);
                dW_V_ += ΣΠ<0>(dv_s, x_s);

                // upstream: contract (Heads, HeadDim) axes of transposed weights with per-token grad
                const auto dx_s = ΣΠ<2>(W_Q_T, dq_s) + ΣΠ<2>(W_K_T, dk_s) + ΣΠ<2>(W_V_T, dv_s);
                TensorIndexAdd<0>(dX, s, dx_s);
            }
            return dX;
        }

        // ─── BATCHED (loop over leading Batch dimension) ─────────────────────────

        template<size_t Batch>
        Tensor<Batch, SeqLen, EmbDims...> BatchedForward(const Tensor<Batch, SeqLen, EmbDims...>& X) {
            Tensor<Batch, SeqLen, EmbDims...> result;
            constexpr size_t sample_size = InputTensor::Size;
            for (size_t b = 0; b < Batch; ++b) {
                InputTensor x_b;
                for (size_t i = 0; i < sample_size; ++i)
                    x_b.flat(i) = X.flat(b * sample_size + i);
                const auto out = Forward(x_b);
                for (size_t i = 0; i < sample_size; ++i)
                    result.flat(b * sample_size + i) = out.flat(i);
            }
            return result;
        }

        template<size_t Batch>
        Tensor<Batch, SeqLen, EmbDims...> BatchedBackward(
            const Tensor<Batch, SeqLen, EmbDims...>& delta_A,
            const Tensor<Batch, SeqLen, EmbDims...>& a,
            const Tensor<Batch, SeqLen, EmbDims...>& a_prev)
        {
            Tensor<Batch, SeqLen, EmbDims...> result;
            const float batch_adj = 1.f / static_cast<float>(Batch);
            constexpr size_t sample_size = InputTensor::Size;
            for (size_t b = 0; b < Batch; ++b) {
                InputTensor dA_b, a_b, ap_b;
                for (size_t i = 0; i < sample_size; ++i) {
                    dA_b.flat(i) = delta_A.flat(b * sample_size + i);
                    a_b.flat(i)  = a.flat(b * sample_size + i);
                    ap_b.flat(i) = a_prev.flat(b * sample_size + i);
                }
                const auto upstream = Backward(dA_b, a_b, ap_b);
                // scale each sample's weight gradients by 1/Batch
                dW_Q_ *= batch_adj; dW_K_ *= batch_adj; dW_V_ *= batch_adj; dW_O_ *= batch_adj;
                for (size_t i = 0; i < sample_size; ++i)
                    result.flat(b * sample_size + i) = upstream.flat(i);
            }
            return result;
        }

        // ─── ADAM UPDATE ────────────────────────────────────────────────────────

        void Update(float adamBeta1, float adamBeta2, float lr,
                    float mCorr, float vCorr, float eps = 1e-8f) {
            auto update_param = [&](auto& W, auto& m, auto& v, const auto& dW) {
                for (size_t i = 0; i < std::decay_t<decltype(W)>::Size; ++i) {
                    const float g = dW.flat(i);
                    m.flat(i) = adamBeta1 * m.flat(i) + (1.f - adamBeta1) * g;
                    v.flat(i) = adamBeta2 * v.flat(i) + (1.f - adamBeta2) * g * g;
                    W.flat(i) -= lr * (m.flat(i) * mCorr) / (std::sqrt(v.flat(i) * vCorr) + eps);
                }
            };
            update_param(W_Q_, mW_Q_, vW_Q_, dW_Q_);
            update_param(W_K_, mW_K_, vW_K_, dW_K_);
            update_param(W_V_, mW_V_, vW_V_, dW_V_);
            update_param(W_O_, mW_O_, vW_O_, dW_O_);
        }

        // ─── SAVE / LOAD ─────────────────────────────────────────────────────────

        void Save(std::ofstream& f) const {
            W_Q_.Save(f); W_K_.Save(f); W_V_.Save(f); W_O_.Save(f);
        }
        void Load(std::ifstream& f) {
            W_Q_.Load(f); W_K_.Load(f); W_V_.Load(f); W_O_.Load(f);
        }
    };


    // Helper: extract the first (SeqLen) dimension from Tensor<SeqLen, EmbDims...>
    template<typename T>
    struct TensorFirstDim;

    template<size_t D0, size_t... Rest>
    struct TensorFirstDim<Tensor<D0, Rest...>> {
        static constexpr size_t value = D0;
    };


    // MHAttention<Heads, EmbDims...>: recipe for MultiHeadAttentionBlock.
    // HeadDim = EmbSize / Heads is derived automatically; EmbSize % Heads == 0 is asserted.
    //
    // Usage in NetworkBuilder:
    //   NetworkBuilder<
    //       Input<SeqLen, EmbDim>,
    //       MHAttention<Heads, EmbDim>,
    //       Dense<EmbDim, ActivationFunction::ReLU>
    //   >::type transformer_layer;
    //
    // OutputTensor = Tensor<1, EmbDims...> is a SeqLen=1 placeholder for the Block concept
    // self-check only; actual SeqLen is inferred from the input type via Resolve.
    template<size_t Heads, size_t... EmbDims>
    struct MHAttention {
        using OutputTensor = Tensor<1, EmbDims...>;

        template<typename InputT>
        using Resolve = MultiHeadAttentionBlock<
            TensorFirstDim<InputT>::value,
            Heads, EmbDims...
        >;
    };

} // namespace TTTN
