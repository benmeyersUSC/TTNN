#pragma once
#include "TensorContract.hpp"
#include "TensorReduce.hpp"
#include "NetworkUtil.hpp"
#include "Params.hpp"

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
        using InputTensor = Tensor<SeqLen, EmbDims...>;
        using OutputTensor = Tensor<SeqLen, EmbDims...>;

        static constexpr size_t N_emb = sizeof...(EmbDims);
        static constexpr size_t EmbSize = TensorDimsProduct<EmbDims...>::value;
        static constexpr size_t HeadDim = EmbSize / Heads;
        static_assert(EmbSize % Heads == 0,
                      "Heads must be a factor of EmbSize (the product of all EmbDims)");

        using W_QKV_Type = Tensor<Heads, HeadDim, EmbDims...>; // shape of W_Q, W_K, W_V
        using W_O_Type = Tensor<EmbDims..., Heads, HeadDim>; // shape of W_O
        using QKV_Type = Tensor<SeqLen, Heads, HeadDim>; // projected Q, K, V
        using Scores_Type = Tensor<Heads, SeqLen, SeqLen>; // attention weight matrix

    private:
        Param<W_QKV_Type> WQ_, WK_, WV_;
        Param<W_O_Type> WO_;

        // forward-pass cache (needed by Backward) — mutable so Forward can be const
        mutable InputTensor X_cache_{};
        mutable QKV_Type Q_{}, K_{}, V_{};
        mutable Scores_Type attn_weights_{};
        mutable QKV_Type attended_{};

    public:
        // @doc: auto all_params()
        /** Returns `std::tie(WQ_, WK_, WV_, WO_)`; TTN drives `ZeroGrad`, `Update`, `Save`, `Load` from this */
        auto all_params() { return std::tie(WQ_, WK_, WV_, WO_); }
        auto all_params() const { return std::tie(WQ_, WK_, WV_, WO_); }

        /** Cached attention weights `Tensor<Heads, SeqLen, SeqLen>` from the most recent `Forward` call. */
        const Scores_Type& attn_weights() const { return attn_weights_; }

        // @doc: MultiHeadAttentionBlock()
        /** Xavier-initializes `WQ`, `WK`, `WV`, `WO` */
        MultiHeadAttentionBlock() {
            XavierInitMD(WQ_.value, EmbSize, HeadDim);
            XavierInitMD(WK_.value, EmbSize, HeadDim);
            XavierInitMD(WV_.value, EmbSize, HeadDim);
            XavierInitMD(WO_.value, EmbSize, EmbSize);
        }

        // ─── FORWARD ────────────────────────────────────────────────────────────

        // OutputTensor Forward(const InputTensor& X) const {
        //     const float inv_sqrt = 1.f / std::sqrt(static_cast<float>(HeadDim));
        //     X_cache_ = X;
        //
        //     // 1. Project each token to (Heads, HeadDim) Q, K, V
        //     Q_.fill(0.f); K_.fill(0.f); V_.fill(0.f);
        //     for (size_t s = 0; s < SeqLen; ++s) {
        //         const auto x_s = TensorIndex<0>(X, s);
        //         TensorIndexApply<0>(Q_, s, ΣΠ<N_emb>(WQ_.value, x_s), [](float a, float b){ return a + b; });
        //         TensorIndexApply<0>(K_, s, ΣΠ<N_emb>(WK_.value, x_s), [](float a, float b){ return a + b; });
        //         TensorIndexApply<0>(V_, s, ΣΠ<N_emb>(WV_.value, x_s), [](float a, float b){ return a + b; });
        //     }
        //
        //     // 2. Per-head: scale dot-product scores → softmax weights → weighted V
        //     attn_weights_.fill(0.f);
        //     attended_.fill(0.f);
        //     for (size_t h = 0; h < Heads; ++h) {
        //         const auto Q_h   = TensorIndex<1>(Q_, h);             // Tensor<SeqLen, HeadDim>
        //         const auto K_h_T = Permute<1,0>(TensorIndex<1>(K_, h)); // Tensor<HeadDim, SeqLen>
        //         const auto V_h   = TensorIndex<1>(V_, h);             // Tensor<SeqLen, HeadDim>
        //
        //         // scores_h[q,k] = dot(Q_h[q,:], K_h[:,k]) / sqrt(HeadDim)
        //         const auto scores_h  = ΣΠ<1>(Q_h, K_h_T) * inv_sqrt; // Tensor<SeqLen, SeqLen>
        //         const auto weights_h = Softmax<1>(scores_h);           // Tensor<SeqLen, SeqLen>
        //
        //         // cache weights for backward
        //         TensorIndexApply<0>(attn_weights_, h, weights_h, [](float a, float b){ return a + b; });
        //
        //         // attended_h = weights_h @ V_h: Tensor<SeqLen,SeqLen> × Tensor<SeqLen,HeadDim>
        //         TensorIndexApply<1>(attended_, h, ΣΠ<1>(weights_h, V_h), [](float a, float b){ return a + b; });
        //     }
        //
        //     // 3. Output projection: contract (Heads, HeadDim) → EmbDims per token
        //     OutputTensor output;
        //     output.fill(0.f);
        //     for (size_t s = 0; s < SeqLen; ++s) {
        //         const auto att_s = TensorIndex<0>(attended_, s);   // Tensor<Heads, HeadDim>
        //         TensorIndexApply<0>(output, s, ΣΠ<2>(WO_.value, att_s), [](float a, float b){ return a + b; }); // Tensor<EmbDims...>
        //     }
        //     return output;
        // }


        OutputTensor Forward(const InputTensor &X) const {
            const float inv_sqrt = 1.f / std::sqrt(static_cast<float>(HeadDim));
            X_cache_ = X;

            const auto WQ_T = PermuteFromHolder<WTBlockSwapPerm<2, N_emb> >(
                WQ_.value, std::make_index_sequence<2 + N_emb>{});
            const auto WK_T = PermuteFromHolder<WTBlockSwapPerm<2, N_emb> >(
                WK_.value, std::make_index_sequence<2 + N_emb>{});
            const auto WV_T = PermuteFromHolder<WTBlockSwapPerm<2, N_emb> >(
                WV_.value, std::make_index_sequence<2 + N_emb>{});

            // Project: Tensor<SeqLen, Heads, HeadDim>
            Q_ = ΣΠ<N_emb>(X, WQ_T);
            K_ = ΣΠ<N_emb>(X, WK_T);
            V_ = ΣΠ<N_emb>(X, WV_T);

            // Permute to heads-first: Tensor<Heads, SeqLen, HeadDim>
            auto Qp = Permute<1, 0, 2>(Q_);
            auto Kp = Permute<1, 0, 2>(K_);
            auto Vp = Permute<1, 0, 2>(V_);

            // Batched over Heads: scores[h,q,k] = Σ_d Q[h,q,d] * K[h,k,d]
            auto scores = BatchΣΠ<1, 1>(Qp, Permute<0, 2, 1>(Kp)) * inv_sqrt;

            attn_weights_ = Softmax<2>(scores);

            // In Forward, after the BatchΣΠ:
            auto attended_hp = BatchΣΠ<1, 1>(attn_weights_, Vp); // Tensor<Heads, SeqLen, HeadDim>
            attended_ = Permute<1, 0, 2>(attended_hp); // back to Tensor<SeqLen, Heads, HeadDim>
            const auto WO_T = PermuteFromHolder<WTBlockSwapPerm<N_emb, 2> >(
                WO_.value, std::make_index_sequence<N_emb + 2>{});
            // Then att_seq IS attended_ — no second permute needed:
            return ΣΠ<2>(attended_, WO_T);
        }

        // ─── BACKWARD ───────────────────────────────────────────────────────────

        // // delta_A: dL/dOutput.  a: output (unused — attention has no simple pointwise activation).
        // // a_prev: input X (same as X_cache_, provided for interface compliance).
        // InputTensor Backward(const OutputTensor &delta_A,
        //                      const OutputTensor & /*a*/,
        //                      const InputTensor & /*a_prev*/) {
        //     const float inv_sqrt = 1.f / std::sqrt(static_cast<float>(HeadDim));
        //
        //     // W_Q_T: Tensor<EmbDims..., Heads, HeadDim>  (block-swap W_Q's two halves)
        //     // W_K_T, W_V_T: same permutation
        //     // W_O_T: Tensor<Heads, HeadDim, EmbDims...>  (block-swap W_O's two halves)
        //     const auto W_Q_T = PermuteFromHolder<WTBlockSwapPerm<2, N_emb> >(
        //         WQ_.value, std::make_index_sequence<2 + N_emb>{});
        //     const auto W_K_T = PermuteFromHolder<WTBlockSwapPerm<2, N_emb> >(
        //         WK_.value, std::make_index_sequence<2 + N_emb>{});
        //     const auto W_V_T = PermuteFromHolder<WTBlockSwapPerm<2, N_emb> >(
        //         WV_.value, std::make_index_sequence<2 + N_emb>{});
        //     const auto W_O_T = PermuteFromHolder<WTBlockSwapPerm<N_emb, 2> >(
        //         WO_.value, std::make_index_sequence<N_emb + 2>{});
        //
        //     // Step 1: backward through output projection W_O
        //     // out_s = ΣΠ<2>(W_O, att_s)  →  dW_O += outer(dout_s, att_s), d_att_s = ΣΠ<N_emb>(W_O_T, dout_s)
        //     QKV_Type d_attended;
        //     d_attended.fill(0.f);
        //     for (size_t s = 0; s < SeqLen; ++s) {
        //         const auto dout_s = TensorIndex<0>(delta_A, s); // Tensor<EmbDims...>
        //         const auto att_s = TensorIndex<0>(attended_, s); // Tensor<Heads, HeadDim>
        //         WO_.grad += ΣΠ<0>(dout_s, att_s); // outer → Tensor<EmbDims..., Heads, HeadDim>
        //         TensorIndexApply<0>(d_attended, s, ΣΠ<N_emb>(W_O_T, dout_s), [](float a, float b) { return a + b; });
        //     }
        //
        //     // Steps 2–4 (one head loop): attended, softmax, scores
        //     // attended_h = weights_h @ V_h
        //     // weights_h  = Softmax<1>(scores_h * inv_sqrt)
        //     // scores_h   = Q_h @ K_h^T
        //     QKV_Type d_Q, d_K, d_V;
        //     d_Q.fill(0.f);
        //     d_K.fill(0.f);
        //     d_V.fill(0.f);
        //     for (size_t h = 0; h < Heads; ++h) {
        //         const auto weights_h = TensorIndex<0>(attn_weights_, h); // Tensor<SeqLen, SeqLen>
        //         const auto V_h = TensorIndex<1>(V_, h); // Tensor<SeqLen, HeadDim>
        //         const auto Q_h = TensorIndex<1>(Q_, h); // Tensor<SeqLen, HeadDim>
        //         const auto K_h = TensorIndex<1>(K_, h); // Tensor<SeqLen, HeadDim>
        //         const auto d_att_h = TensorIndex<1>(d_attended, h); // Tensor<SeqLen, HeadDim>
        //
        //         // Step 2: d_weights_h = d_att_h @ V_h^T,   d_V_h = weights_h^T @ d_att_h
        //         const auto d_weights_h = ΣΠ<1>(d_att_h, Permute<1, 0>(V_h)); // Tensor<SeqLen, SeqLen>
        //         TensorIndexApply<1>(d_V, h, ΣΠ<1>(Permute<1, 0>(weights_h), d_att_h),
        //                             [](float a, float b) { return a + b; }); // Tensor<SeqLen, HeadDim>
        //
        //         // Step 3: peel off softmax (and the 1/√HeadDim scale)
        //         const auto d_scores_h = SoftmaxPrime<1>(d_weights_h, weights_h) * inv_sqrt; // Tensor<SeqLen, SeqLen>
        //
        //         // Step 4: d_Q_h = d_scores_h @ K_h,   d_K_h = d_scores_h^T @ Q_h
        //         TensorIndexApply<1>(d_Q, h, ΣΠ<1>(d_scores_h, K_h), [](float a, float b) { return a + b; });
        //         // Tensor<SeqLen, HeadDim>
        //         TensorIndexApply<1>(d_K, h, ΣΠ<1>(Permute<1, 0>(d_scores_h), Q_h),
        //                             [](float a, float b) { return a + b; }); // Tensor<SeqLen, HeadDim>
        //     }
        //
        //     // Step 5: backward through Q, K, V projections, accumulate dW and upstream dX
        //     InputTensor dX;
        //     dX.fill(0.f);
        //     for (size_t s = 0; s < SeqLen; ++s) {
        //         const auto x_s = TensorIndex<0>(X_cache_, s); // Tensor<EmbDims...>
        //         const auto dq_s = TensorIndex<0>(d_Q, s); // Tensor<Heads, HeadDim>
        //         const auto dk_s = TensorIndex<0>(d_K, s);
        //         const auto dv_s = TensorIndex<0>(d_V, s);
        //
        //         WQ_.grad += ΣΠ<0>(dq_s, x_s); // outer → Tensor<Heads, HeadDim, EmbDims...>
        //         WK_.grad += ΣΠ<0>(dk_s, x_s);
        //         WV_.grad += ΣΠ<0>(dv_s, x_s);
        //
        //         // upstream: contract (Heads, HeadDim) axes of transposed weights with per-token grad
        //         const auto dx_s = ΣΠ<2>(W_Q_T, dq_s) + ΣΠ<2>(W_K_T, dk_s) + ΣΠ<2>(W_V_T, dv_s);
        //         TensorIndexApply<0>(dX, s, dx_s, [](float a, float b) { return a + b; });
        //     }
        //     return dX;
        // }

        InputTensor Backward(const OutputTensor &delta_A,
                             const OutputTensor & /*a*/,
                             const InputTensor & /*a_prev*/) {
            const float inv_sqrt = 1.f / std::sqrt(static_cast<float>(HeadDim));

            // --- Output projection backward ---
            // const auto delta_A_swap = PermuteFromHolder<MoveToLastPerm<0, 1 + N_emb> >(
            //     delta_A, std::make_index_sequence<1 + N_emb>{});
            // WO_.grad += ΣΠ<1>(delta_A_swap, attended_);
            WO_.grad += Einsum<0, 0>(delta_A, attended_);

            auto d_attended = ΣΠ<N_emb>(delta_A, WO_.value);

            // --- Heads-first layouts ---
            auto d_att_hp = Permute<1, 0, 2>(d_attended); // Tensor<Heads, SeqLen, HeadDim>
            auto Qp = Permute<1, 0, 2>(Q_); // Tensor<Heads, SeqLen, HeadDim>
            auto Kp = Permute<1, 0, 2>(K_);
            auto Vp = Permute<1, 0, 2>(V_);

            // --- Weighted V backward (batched over Heads) ---
            // forward: attended = BatchΣΠ<1,1>(weights, Vp)
            //
            // d_weights = BatchΣΠ<1,1>(d_att_hp, Vp^T)
            //   Tensor<Heads, SeqLen, HeadDim> × Tensor<Heads, HeadDim, SeqLen>
            //   → Tensor<Heads, SeqLen, SeqLen>
            auto d_weights = BatchΣΠ<1, 1>(d_att_hp, Permute<0, 2, 1>(Vp));

            // d_Vp = BatchΣΠ<1,1>(weights^T, d_att_hp)
            //   Tensor<Heads, SeqLen, SeqLen> × Tensor<Heads, SeqLen, HeadDim>
            //   → Tensor<Heads, SeqLen, HeadDim>
            auto d_Vp = BatchΣΠ<1, 1>(Permute<0, 2, 1>(attn_weights_), d_att_hp);

            // --- Softmax backward (operates on full Tensor<Heads, SeqLen, SeqLen>) ---
            auto d_scores = SoftmaxPrime<2>(d_weights, attn_weights_) * inv_sqrt;

            // --- Scores backward (batched over Heads) ---
            // forward: scores = BatchΣΠ<1,1>(Qp, Kp^T)
            //
            // d_Qp = BatchΣΠ<1,1>(d_scores, Kp)
            //   Tensor<Heads, SeqLen, SeqLen> × Tensor<Heads, SeqLen, HeadDim>
            //   → Tensor<Heads, SeqLen, HeadDim>
            auto d_Qp = BatchΣΠ<1, 1>(d_scores, Kp);

            // d_Kp = BatchΣΠ<1,1>(d_scores^T, Qp)
            //   Tensor<Heads, SeqLen, SeqLen> × Tensor<Heads, SeqLen, HeadDim>
            //   → Tensor<Heads, SeqLen, HeadDim>
            auto d_Kp = BatchΣΠ<1, 1>(Permute<0, 2, 1>(d_scores), Qp);

            // --- Permute gradients back to Tensor<SeqLen, Heads, HeadDim> ---
            auto d_Q = Permute<1, 0, 2>(d_Qp);
            auto d_K = Permute<1, 0, 2>(d_Kp);
            auto d_V = Permute<1, 0, 2>(d_Vp);

            // --- Q/K/V projection backward (no loops) ---
            // dW = ΣΠ<1>(dGrad_swap, X_cache_)
            //   Tensor<Heads, HeadDim, SeqLen> × Tensor<SeqLen, EmbDims...>
            //   contracts SeqLen → Tensor<Heads, HeadDim, EmbDims...>
            // auto dQ_swap = Permute<1, 2, 0>(d_Q);
            // auto dK_swap = Permute<1, 2, 0>(d_K);
            // auto dV_swap = Permute<1, 2, 0>(d_V);

            // WQ_.grad += ΣΠ<1>(dQ_swap, X_cache_);
            // WK_.grad += ΣΠ<1>(dK_swap, X_cache_);
            // WV_.grad += ΣΠ<1>(dV_swap, X_cache_);

            WQ_.grad += Einsum<0, 0>(d_Q, X_cache_);
            WK_.grad += Einsum<0, 0>(d_K, X_cache_);
            WV_.grad += Einsum<0, 0>(d_V, X_cache_);

            // dX = ΣΠ<2>(d_Q, WQ) + ΣΠ<2>(d_K, WK) + ΣΠ<2>(d_V, WV)
            //   Tensor<SeqLen, Heads, HeadDim> × Tensor<Heads, HeadDim, EmbDims...>
            //   contracts (Heads, HeadDim) → Tensor<SeqLen, EmbDims...>
            return ΣΠ<2>(d_Q, WQ_.value) + ΣΠ<2>(d_K, WK_.value) + ΣΠ<2>(d_V, WV_.value);
        }

        // ─── BATCHED (loop over leading Batch dimension) ─────────────────────────

        template<size_t Batch>
        Tensor<Batch, SeqLen, EmbDims...> BatchedForward(const Tensor<Batch, SeqLen, EmbDims...> &X) const {
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
            const Tensor<Batch, SeqLen, EmbDims...> &delta_A,
            const Tensor<Batch, SeqLen, EmbDims...> &a,
            const Tensor<Batch, SeqLen, EmbDims...> &a_prev) {
            Tensor<Batch, SeqLen, EmbDims...> result;
            constexpr size_t sample_size = InputTensor::Size;
            for (size_t b = 0; b < Batch; ++b) {
                InputTensor dA_b, a_b, ap_b;
                for (size_t i = 0; i < sample_size; ++i) {
                    dA_b.flat(i) = delta_A.flat(b * sample_size + i);
                    a_b.flat(i) = a.flat(b * sample_size + i);
                    ap_b.flat(i) = a_prev.flat(b * sample_size + i);
                }
                const auto upstream = Backward(dA_b, a_b, ap_b);
                for (size_t i = 0; i < sample_size; ++i)
                    result.flat(b * sample_size + i) = upstream.flat(i);
            }
            // scale weight gradients by 1/Batch ONCE after all samples accumulate
            const float batch_adj = 1.f / static_cast<float>(Batch);
            WQ_.grad = WQ_.grad * batch_adj;
            WK_.grad = WK_.grad * batch_adj;
            WV_.grad = WV_.grad * batch_adj;
            WO_.grad = WO_.grad * batch_adj;
            return result;
        }

        // ─── ADAM UPDATE ────────────────────────────────────────────────────────
    };


    // Helper: extract the first (SeqLen) dimension from Tensor<SeqLen, EmbDims...>
    template<typename T>
    struct TensorFirstDim;

    template<size_t D0, size_t... Rest>
    struct TensorFirstDim<Tensor<D0, Rest...> > {
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
