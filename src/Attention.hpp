#pragma once
#include <vector>
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
    //   Project: Q_,K_,V_ = ΣΠ<N_emb>(X, W_T)         → Tensor<SeqLen, Heads, HeadDim>  (seq-major)
    //   scores[h,q,k] = BatchContract<H,H,D,D>(Q_,K_) / √HeadDim → Tensor<Heads, SeqLen, SeqLen>
    //   attn_weights  = Softmax<2>(scores)
    //   attended_[h,q,d] = BatchContract<H,H,S_k,S_k>(weights,V_) → Tensor<Heads, SeqLen, HeadDim>
    //   out = Contract<(H,D),(N_emb,N_emb+1)>(attended_, WO_)      → Tensor<SeqLen, EmbDims...>
    //
    // Backward: chain rule through all of the above.
    //   Q_,K_,V_ remain seq-major; d_Q,d_K,d_V likewise.
    //   d_att_hp [H,S,D] = Permute<1,0,2>(ΣΠ<N_emb>(delta_A, WO_))
    //   BatchContract used throughout to avoid explicit ^T permutes on activations.


    template<size_t SeqLen, size_t Heads, size_t... EmbDims>
    class MultiHeadAttentionBlock {
    public:
        using InputTensor   = Tensor<SeqLen, EmbDims...>;
        using OutputTensor  = Tensor<SeqLen, EmbDims...>;

        static constexpr size_t N_emb   = sizeof...(EmbDims);
        static constexpr size_t EmbSize = TensorDimsProduct<EmbDims...>::value;
        static constexpr size_t HeadDim = EmbSize / Heads;
        static_assert(EmbSize % Heads == 0,
                      "Heads must be a factor of EmbSize (the product of all EmbDims)");

        using W_QKV_Type    = Tensor<Heads, HeadDim, EmbDims...>;   // W_Q, W_K, W_V
        using W_O_Type      = Tensor<EmbDims..., Heads, HeadDim>;   // W_O
        using QKV_Type      = Tensor<SeqLen, Heads, HeadDim>;       // seq-major  [S,H,D]
        using Scores_Type   = Tensor<Heads, SeqLen, SeqLen>;        // [H,S_q,S_k]
        using Attended_Type = Tensor<Heads, SeqLen, HeadDim>;       // heads-first [H,S,D]

    private:
        Param<W_QKV_Type> WQ_, WK_, WV_;
        Param<W_O_Type>   WO_;

        // forward-pass cache (needed by Backward) — mutable so Forward can be const
        mutable InputTensor    X_cache_{};
        mutable QKV_Type       Q_{}, K_{}, V_{};
        mutable Scores_Type    attn_weights_{};
        mutable Attended_Type  attended_{};

        // batched cache — flat float buffers; typed Tensors can't be members when Batch is a fn template param
        mutable std::vector<float> bQ_buf_, bK_buf_, bV_buf_, battn_w_buf_, battended_buf_;

        template<typename T>
        static void bcache_store(const T& t, std::vector<float>& buf) {
            buf.resize(T::Size);
            for (size_t i = 0; i < T::Size; ++i) buf[i] = t.flat(i);
        }
        template<typename T>
        static T bcache_load(const std::vector<float>& buf) {
            T t; for (size_t i = 0; i < T::Size; ++i) t.flat(i) = buf[i]; return t;
        }

    public:
        // @doc: auto all_params()
        /** Returns `std::tie(WQ_, WK_, WV_, WO_)`; TTN drives `ZeroGrad`, `Update`, `Save`, `Load` from this */
        auto all_params() { return std::tie(WQ_, WK_, WV_, WO_); }
        auto all_params() const { return std::tie(WQ_, WK_, WV_, WO_); }

        /** Cached attention weights `Tensor<Heads, SeqLen, SeqLen>` from the most recent `Forward` call. */
        const Scores_Type& attn_weights() const { return attn_weights_; }

        /** PeekableBlock: expose attn_weights into a SnapshotMap under `prefix`. */
        void peek(SnapshotMap& out, const std::string& prefix) const {
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

            const auto WQ_T = PermuteFromHolder<WTBlockSwapPerm<2, N_emb>>(
                WQ_.value, std::make_index_sequence<2 + N_emb>{});
            const auto WK_T = PermuteFromHolder<WTBlockSwapPerm<2, N_emb>>(
                WK_.value, std::make_index_sequence<2 + N_emb>{});
            const auto WV_T = PermuteFromHolder<WTBlockSwapPerm<2, N_emb>>(
                WV_.value, std::make_index_sequence<2 + N_emb>{});

            // Project: seq-major Tensor<SeqLen, Heads, HeadDim>
            Q_ = ΣΠ<N_emb>(X, WQ_T);
            K_ = ΣΠ<N_emb>(X, WK_T);
            V_ = ΣΠ<N_emb>(X, WV_T);

            // scores[h,q,k] = Σ_d Q_[q,h,d] * K_[k,h,d]
            // batch H (axis 1 of both Q_,K_), contract D (axis 2 of both)
            auto scores = BatchContract<AxisList<1>{}, AxisList<1>{},
                                        AxisList<2>{}, AxisList<2>{}, Mul, Add>(Q_, K_) * inv_sqrt;

            attn_weights_ = Softmax<2>(scores);

            // attended_[h,q,d] = Σ_k attn_weights_[h,q,k] * V_[k,h,d]
            // batch H (axis 0 of weights, axis 1 of V_), contract S_k (axis 2 of weights, axis 0 of V_)
            attended_ = BatchContract<AxisList<0>{}, AxisList<1>{},
                                      AxisList<2>{}, AxisList<0>{}, Mul, Add>(attn_weights_, V_);

            // out[s,emb] = Σ_{h,d} attended_[h,s,d] * WO_[emb,h,d]
            // contract (H,D) = axes (0,2) of attended_ with axes (N_emb, N_emb+1) of WO_
            return Contract<AxisList<0, 2>{}, AxisList<N_emb, N_emb + 1>{}, Mul, Add>(attended_, WO_.value);
        }


        // @doc: InputTensor Backward(const OutputTensor& delta_A, const OutputTensor& a, const InputTensor& a_prev)
        /** ######### */
        InputTensor Backward(const OutputTensor &delta_A,
                             const OutputTensor & /*a*/,
                             const InputTensor & /*a_prev*/) {
            const float inv_sqrt = 1.f / std::sqrt(static_cast<float>(HeadDim));

            // --- Output projection backward ---
            // dWO[emb,h,d] = Σ_s delta_A[s,emb] * attended_[h,s,d]
            // contract S: axis 0 of delta_A, axis 1 of attended_
            WO_.grad += Einsum<0, 1>(delta_A, attended_);

            // d_attended[s,h,d] = Σ_emb delta_A[s,emb] * WO_[emb,h,d]  → [S,H,D]
            // permute to heads-first [H,S,D] for subsequent ops
            auto d_att_hp = Permute<1, 0, 2>(ΣΠ<N_emb>(delta_A, WO_.value));

            // --- Heads-first layouts for Q, K, V ---
            auto Qp = Permute<1, 0, 2>(Q_);
            auto Kp = Permute<1, 0, 2>(K_);
            auto Vp = Permute<1, 0, 2>(V_);

            // --- d_weights[h,q,k] = Σ_d d_att[h,q,d] * Vp[h,k,d]
            // batch H (axis 0 of both), contract D (axis 2 of both)
            auto d_weights = BatchContract<AxisList<0>{}, AxisList<0>{},
                                           AxisList<2>{}, AxisList<2>{}, Mul, Add>(d_att_hp, Vp);

            // d_Vp[h,k,d] = Σ_q attn_weights_[h,q,k] * d_att[h,q,d]
            // batch H (axis 0 of both), contract S_q (axis 1 of both)
            auto d_Vp = BatchContract<AxisList<0>{}, AxisList<0>{},
                                      AxisList<1>{}, AxisList<1>{}, Mul, Add>(attn_weights_, d_att_hp);

            // --- Softmax backward ---
            auto d_scores = SoftmaxPrime<2>(d_weights, attn_weights_) * inv_sqrt;

            // d_Qp[h,q,d] = Σ_k d_scores[h,q,k] * Kp[h,k,d]
            auto d_Qp = BatchΣΠ<1, 1>(d_scores, Kp);

            // d_Kp[h,k,d] = Σ_q d_scores[h,q,k] * Qp[h,q,d]
            // batch H (axis 0 of both), contract S_q (axis 1 of both)
            auto d_Kp = BatchContract<AxisList<0>{}, AxisList<0>{},
                                      AxisList<1>{}, AxisList<1>{}, Mul, Add>(d_scores, Qp);

            // --- Permute gradients back to seq-major Tensor<SeqLen, Heads, HeadDim> ---
            auto d_Q = Permute<1, 0, 2>(d_Qp);
            auto d_K = Permute<1, 0, 2>(d_Kp);
            auto d_V = Permute<1, 0, 2>(d_Vp);

            // --- Q/K/V weight gradients ---
            WQ_.grad += Einsum<0, 0>(d_Q, X_cache_);
            WK_.grad += Einsum<0, 0>(d_K, X_cache_);
            WV_.grad += Einsum<0, 0>(d_V, X_cache_);

            // --- Input gradient ---
            return ΣΠ<2>(d_Q, WQ_.value) + ΣΠ<2>(d_K, WK_.value) + ΣΠ<2>(d_V, WV_.value);
        }

        // ─── BATCHED (true batched ops — no loop over B) ────────────────────────

        // @doc: template<size_t Batch> Tensor<Batch, SeqLen, EmbDims...> BatchedForward(...)
        /** ######### */
        template<size_t Batch>
        Tensor<Batch, SeqLen, EmbDims...> BatchedForward(const Tensor<Batch, SeqLen, EmbDims...> &X) const {
            using BQV  = Tensor<Batch, SeqLen, Heads, HeadDim>;   // [B,S,H,D] seq-major
            using BSc  = Tensor<Batch, Heads, SeqLen, SeqLen>;    // [B,H,S_q,S_k]
            using BAtt = Tensor<Batch, Heads, SeqLen, HeadDim>;   // [B,H,S,D] heads-first

            const float inv_sqrt = 1.f / std::sqrt(static_cast<float>(HeadDim));

            const auto WQ_T = PermuteFromHolder<WTBlockSwapPerm<2, N_emb>>(WQ_.value, std::make_index_sequence<2 + N_emb>{});
            const auto WK_T = PermuteFromHolder<WTBlockSwapPerm<2, N_emb>>(WK_.value, std::make_index_sequence<2 + N_emb>{});
            const auto WV_T = PermuteFromHolder<WTBlockSwapPerm<2, N_emb>>(WV_.value, std::make_index_sequence<2 + N_emb>{});

            // [B,S,EmbDims...] × [EmbDims...,H,D] → [B,S,H,D]  (ΣΠ treats B,S as free dims)
            const auto bQ = ΣΠ<N_emb>(X, WQ_T);
            const auto bK = ΣΠ<N_emb>(X, WK_T);
            const auto bV = ΣΠ<N_emb>(X, WV_T);

            // scores[b,h,q,k]: batch B(0),H(2) of Q/K; contract D(3)  →  [B,H,S_q,S_k]
            const auto scores = BatchContract<AxisList<0,2>{}, AxisList<0,2>{},
                                              AxisList<3>{},   AxisList<3>{}, Mul, Add>(bQ, bK) * inv_sqrt;

            const auto battn = Softmax<3>(scores);

            // attended[b,h,q,d]: batch B(0),H(1 of weights / 2 of V); contract S_k(3 of weights / 1 of V)
            const auto batt = BatchContract<AxisList<0,1>{}, AxisList<0,2>{},
                                            AxisList<3>{},   AxisList<1>{}, Mul, Add>(battn, bV);

            // out[b,s,emb]: contract H(1),D(3) of batt with (N_emb,N_emb+1) of WO_  →  [B,S,EmbDims...]
            const auto out = Contract<AxisList<1,3>{}, AxisList<N_emb, N_emb + 1>{}, Mul, Add>(batt, WO_.value);

            bcache_store(bQ,   bQ_buf_);
            bcache_store(bK,   bK_buf_);
            bcache_store(bV,   bV_buf_);
            bcache_store(battn, battn_w_buf_);
            bcache_store(batt,  battended_buf_);

            return out;
        }

        template<size_t Batch>
        Tensor<Batch, SeqLen, EmbDims...> BatchedBackward(
            const Tensor<Batch, SeqLen, EmbDims...> &delta_A,
            const Tensor<Batch, SeqLen, EmbDims...> & /*a*/,
            const Tensor<Batch, SeqLen, EmbDims...> &a_prev) {

            using BQV  = Tensor<Batch, SeqLen, Heads, HeadDim>;   // [B,S,H,D] seq-major
            using BQVp = Tensor<Batch, Heads, SeqLen, HeadDim>;   // [B,H,S,D] heads-first
            using BSc  = Tensor<Batch, Heads, SeqLen, SeqLen>;    // [B,H,S_q,S_k]
            using BAtt = Tensor<Batch, Heads, SeqLen, HeadDim>;   // [B,H,S,D]

            const float inv_sqrt  = 1.f / std::sqrt(static_cast<float>(HeadDim));
            const float inv_batch = 1.f / static_cast<float>(Batch);

            const auto bQ   = bcache_load<BQV>(bQ_buf_);
            const auto bK   = bcache_load<BQV>(bK_buf_);
            const auto bV   = bcache_load<BQV>(bV_buf_);
            const auto battn = bcache_load<BSc>(battn_w_buf_);
            const auto batt  = bcache_load<BAtt>(battended_buf_);

            // dWO[emb,h,d] = Σ_{b,s} delta_A[b,s,emb] * batt[b,h,s,d]
            WO_.grad += Contract<AxisList<0,1>{}, AxisList<0,2>{}, Mul, Add>(delta_A, batt) * inv_batch;

            // d_att_hp[b,h,s,d]: project delta_A through WO_, permute to heads-first
            const auto d_att_hp = Permute<0,2,1,3>(ΣΠ<N_emb>(delta_A, WO_.value));

            // heads-first views of Q,K,V
            const auto Qp = Permute<0,2,1,3>(bQ);
            const auto Kp = Permute<0,2,1,3>(bK);
            const auto Vp = Permute<0,2,1,3>(bV);

            // d_weights[b,h,q,k]: batch B(0),H(1); contract D(3)
            const auto d_weights = BatchContract<AxisList<0,1>{}, AxisList<0,1>{},
                                                 AxisList<3>{},   AxisList<3>{}, Mul, Add>(d_att_hp, Vp);

            // d_Vp[b,h,k,d]: batch B(0),H(1); contract S_q(2)
            const auto d_Vp = BatchContract<AxisList<0,1>{}, AxisList<0,1>{},
                                            AxisList<2>{},   AxisList<2>{}, Mul, Add>(battn, d_att_hp);

            const auto d_scores = SoftmaxPrime<3>(d_weights, battn) * inv_sqrt;

            // d_Qp[b,h,q,d]: batch B(0),H(1); contract S_k(3 of d_scores / 2 of Kp)
            const auto d_Qp = BatchContract<AxisList<0,1>{}, AxisList<0,1>{},
                                            AxisList<3>{},   AxisList<2>{}, Mul, Add>(d_scores, Kp);

            // d_Kp[b,h,k,d]: batch B(0),H(1); contract S_q(2)
            const auto d_Kp = BatchContract<AxisList<0,1>{}, AxisList<0,1>{},
                                            AxisList<2>{},   AxisList<2>{}, Mul, Add>(d_scores, Qp);

            // permute back to seq-major [B,S,H,D]
            const auto d_Q = Permute<0,2,1,3>(d_Qp);
            const auto d_K = Permute<0,2,1,3>(d_Kp);
            const auto d_V = Permute<0,2,1,3>(d_Vp);

            // weight gradients: contract B(0),S(1) from d_Q/d_K/d_V and a_prev
            WQ_.grad += Contract<AxisList<0,1>{}, AxisList<0,1>{}, Mul, Add>(d_Q, a_prev) * inv_batch;
            WK_.grad += Contract<AxisList<0,1>{}, AxisList<0,1>{}, Mul, Add>(d_K, a_prev) * inv_batch;
            WV_.grad += Contract<AxisList<0,1>{}, AxisList<0,1>{}, Mul, Add>(d_V, a_prev) * inv_batch;

            // input gradient: contract H,D (last 2) of d_Q/K/V against WQ/K/V_ (first 2 dims)
            return ΣΠ<2>(d_Q, WQ_.value) + ΣΠ<2>(d_K, WK_.value) + ΣΠ<2>(d_V, WV_.value);
        }

        // ─── ADAM UPDATE ────────────────────────────────────────────────────────
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
