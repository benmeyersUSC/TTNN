#pragma once
#include "TensorContract.hpp"
#include "TensorReduce.hpp"
#include "NetworkUtil.hpp"
#include "Params.hpp"

namespace TTTN {
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

        // shape of W_Q, W_K, W_V
        // [Heads, HeadDim, EmbDims...]
        //      for each head, contract EmbDims from Input to get QKV_Type
        //      [Heads, HeadDim, EmbDims...] x [SeqLen, EmbDims...]
        using W_QKV_Type = Tensor<Heads, HeadDim, EmbDims...>;
        // shape of W_O
        // [EmbDims..., Heads, HeadDim]
        using W_O_Type = Tensor<EmbDims..., Heads, HeadDim>; // projected Q, K, V
        // [SeqLen, Heads, HeadDim]
        using QKV_Type = Tensor<SeqLen, Heads, HeadDim>;
        // attention weight matrix
        // [Heads, SeqLen, SeqLen]
        using Scores_Type = Tensor<Heads, SeqLen, SeqLen>;

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
        const Scores_Type &attn_weights() const { return attn_weights_; }

        /** PeekableBlock: expose attn_weights into a SnapshotMap under `prefix`. */
        void peek(SnapshotMap &out, const std::string &prefix) const {
            snap_add(out, prefix + "attn_weights", attn_weights_);
        }

        // @doc: MultiHeadAttentionBlock()
        /** Xavier-initializes `WQ`, `WK`, `WV`, `WO` */
        MultiHeadAttentionBlock() {
            XavierInitMD(WQ_.value, EmbSize, HeadDim);
            XavierInitMD(WK_.value, EmbSize, HeadDim);
            XavierInitMD(WV_.value, EmbSize, HeadDim);
            XavierInitMD(WO_.value, EmbSize, EmbSize);
        }


        // @doc: OutputTensor Forward(const InputTensor& X) const
        /** ######### */
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


        // @doc: InputTensor Backward(const OutputTensor& delta_A, const OutputTensor& a, const InputTensor& a_prev)
        /** ######### */
        InputTensor Backward(const OutputTensor &delta_A,
                             const OutputTensor & /*a*/,
                             const InputTensor & /*a_prev*/) {
            const float inv_sqrt = 1.f / std::sqrt(static_cast<float>(HeadDim));

            // --- Output projection backward ---
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

        // @doc: template<size_t Batch> Tensor<Batch, SeqLen, EmbDims...> BatchedForward(...)
        /** ######### */
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

    // template<size_t D0, size_t... Rest>
    // struct TensorFirstDim<Tensor<D0, Rest...> > {
    //     static constexpr size_t value = D0;
    // };


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
