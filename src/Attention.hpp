#pragma once
#include "TensorContract.hpp"
#include "TensorReduce.hpp"
#include "NetworkUtil.hpp"

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
        //      [Heads, HeadDim, EmbDims...] x [SeqLen, EmbDims...] -> [SeqLen, Heads, HeadDim]
        using W_QKV_Type = Tensor<Heads, HeadDim, EmbDims...>;

        // shape of W_O
        // scores is [Heads, SeqLen, SeqLen]...need to go to [SeqLen, EmbDims...] for output
        // so W_O is [EmbDims..., Heads, HeadDim]...[EmbDims...
        using W_O_Type = Tensor<EmbDims..., Heads, HeadDim>;

        // shape of Q, K, V
        // [SeqLen, Heads, HeadDim] (right now, at each head, each element of the sequence is represented in HeadDims)
        using QKV_Type = Tensor<SeqLen, Heads, HeadDim>;

        // attention weight matrix
        // [Heads, SeqLen, SeqLen]
        using Scores_Type = Tensor<Heads, SeqLen, SeqLen>;

        // attended values after softmax-weighted sum
        // [Heads, SeqLen, HeadDim]
        using Attended_Type = Tensor<Heads, SeqLen, HeadDim>;

    private:
        Param<W_QKV_Type> WQ_, WK_, WV_;
        Param<W_O_Type> WO_;

        // forward-pass cache (needed by Backward) — mutable so Forward can be const
        mutable InputTensor X_cache_{};
        mutable QKV_Type Q_{}, K_{}, V_{};
        mutable Scores_Type attn_weights_{};
        mutable Attended_Type attended_{};

        // batched forward-pass cache — float vectors because Batch is a template param, not a class param
        mutable std::vector<float> bX_buf_, bQ_buf_, bK_buf_, bV_buf_, battn_buf_, battended_buf_;

        template<typename T>
        static void bcache_store(const T &t, std::vector<float> &buf) {
            buf.assign(t.data(), t.data() + T::Size);
        }

        template<typename T>
        static T bcache_load(const std::vector<float> &buf) {
            T t;
            std::copy(buf.begin(), buf.begin() + T::Size, t.data());
            return t;
        }

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

        template<size_t... Is>
        static constexpr auto QKV_Contract(const InputTensor &X, const W_QKV_Type &wm, std::index_sequence<Is...>) {
            return Contract<AxisList<(1 + Is)...>{}, AxisList<(2 + Is)...>{}, Mul, Add>(X, wm);
        };


        // @doc: OutputTensor Forward(const InputTensor& X) const
        /** ######### */
        OutputTensor Forward(const InputTensor &X) const {
            const float inv_sqrt = 1.f / std::sqrt(static_cast<float>(HeadDim));
            X_cache_ = X;

            // [SeqLen, EmbDims...] x [Heads, HeadDim, EmbDims...] -> [SeqLen, Heads, HeadDim]
            Q_ = QKV_Contract(X, WQ_.value, std::make_index_sequence<N_emb>{});
            K_ = QKV_Contract(X, WK_.value, std::make_index_sequence<N_emb>{});
            V_ = QKV_Contract(X, WV_.value, std::make_index_sequence<N_emb>{});

            // scores[h,s_q,s_k] = Σ_d Q[s_q,h,d] * K[s_k,h,d]
            // Batch H(1,1)  Contract D(2,2)  Free S_q(0) S_k(0)  -> [H,S_q,S_k]
            attn_weights_ = Softmax<2>(
                BatchContract<AxisList<1>{}, AxisList<1>{}, AxisList<2>{}, AxisList<2>{}, Mul, Add>(Q_, K_) * inv_sqrt);

            // attended[h,s_q,d] = Σ_{s_k} attn[h,s_q,s_k] * V[s_k,h,d]
            // Batch H(0,1)  Contract S_k(2,0)  Free S_q(1) D(2)  -> [H,S,D]
            attended_ = BatchContract<AxisList<0>{}, AxisList<1>{}, AxisList<2>{}, AxisList<0>{}, Mul, Add>(
                attn_weights_, V_);

            // out[s,e...] = Σ_{h,d} attended[h,s,d] * WO[e...,h,d]
            // Contract H(0,N_emb) D(2,N_emb+1)  Free S(1) and E...  -> [S,E...]
            return Contract<AxisList<0, 2>{}, AxisList<N_emb, N_emb + 1>{}, Mul, Add>(attended_, WO_.value);
        }


        // @doc: InputTensor Backward(const OutputTensor& delta_A, const OutputTensor& a, const InputTensor& a_prev)
        /** ######### */
        InputTensor Backward(const OutputTensor &delta_A,
                             const OutputTensor & /*a*/,
                             const InputTensor & /*a_prev*/) {
            const float inv_sqrt = 1.f / std::sqrt(static_cast<float>(HeadDim));

            // --- WO_ grad ---
            // dWO[e...,h,d] = Σ_s delta_A[s,e...] * attended[h,s,d]
            // Contract S: axis 0 in delta_A, axis 1 in attended  -> [E...,H,D]
            WO_.grad += Einsum<0, 1>(delta_A, attended_);

            // d_att[s,h,d] = Σ_{e...} delta_A[s,e...] * WO[e...,h,d]   -> [S,H,D]
            // Contracts last N_emb of delta_A against first N_emb of WO_
            const auto d_att = ΣΠ<N_emb>(delta_A, WO_.value);

            // --- Attended backward ---
            // d_attn[h,s_q,s_k] = Σ_d d_att[s_q,h,d] * V[s_k,h,d]
            // Batch H(1,1)  Contract D(2,2)  Free S_q(0) S_k(0)  -> [H,S_q,S_k]
            const auto d_attn = BatchContract<AxisList<1>{}, AxisList<1>{}, AxisList<2>{}, AxisList<2>{}, Mul, Add>(
                d_att, V_);

            // d_V[h,s_k,d] = Σ_{s_q} attn[h,s_q,s_k] * d_att[s_q,h,d]
            // Batch H(0,1)  Contract S_q(1,0)  Free S_k(2) D(2)  -> [H,S_k,D]
            const auto d_V = BatchContract<AxisList<0>{}, AxisList<1>{}, AxisList<1>{}, AxisList<0>{}, Mul, Add>(
                attn_weights_, d_att);

            // --- Softmax backward ---
            const auto d_scores = SoftmaxPrime<2>(d_attn, attn_weights_) * inv_sqrt;

            // --- Scores backward ---
            // d_Q[h,s_q,d] = Σ_{s_k} d_scores[h,s_q,s_k] * K[s_k,h,d]
            // Batch H(0,1)  Contract S_k(2,0)  Free S_q(1) D(2)  -> [H,S_q,D]
            const auto d_Q = BatchContract<AxisList<0>{}, AxisList<1>{}, AxisList<2>{}, AxisList<0>{}, Mul, Add>(
                d_scores, K_);

            // d_K[h,s_k,d] = Σ_{s_q} d_scores[h,s_q,s_k] * Q[s_q,h,d]
            // Batch H(0,1)  Contract S_q(1,0)  Free S_k(2) D(2)  -> [H,S_k,D]
            const auto d_K = BatchContract<AxisList<0>{}, AxisList<1>{}, AxisList<1>{}, AxisList<0>{}, Mul, Add>(
                d_scores, Q_);

            // --- W grads: d_[H,S,D] × X[S,E...] -> [H,D,E...]
            // Contract S: axis 1 in d_, axis 0 in X
            WQ_.grad += Einsum<1, 0>(d_Q, X_cache_);
            WK_.grad += Einsum<1, 0>(d_K, X_cache_);
            WV_.grad += Einsum<1, 0>(d_V, X_cache_);

            // --- dX: d_[H,S,D] × W[H,D,E...] -> [S,E...]
            // Contract (H,D): axes {0,2} in d_, axes {0,1} in W
            return Contract<AxisList<0, 2>{}, AxisList<0, 1>{}, Mul, Add>(d_Q, WQ_.value)
                   + Contract<AxisList<0, 2>{}, AxisList<0, 1>{}, Mul, Add>(d_K, WK_.value)
                   + Contract<AxisList<0, 2>{}, AxisList<0, 1>{}, Mul, Add>(d_V, WV_.value);
        }

        // ─── BATCHED (no loop — fully batched tensor ops via BatchContract/Contract) ──

        // Helper: project [B,S,E...] through [H,D,E...] → [B,S,H,D]
        // Contracts E... at axes (2+Is) in X against axes (2+Is) in W.
        template<size_t Batch, size_t... Is>
        static auto BatchedQKV_Contract(const Tensor<Batch, SeqLen, EmbDims...> &X, const W_QKV_Type &W,
                                        std::index_sequence<Is...>) {
            return Contract<AxisList<(2 + Is)...>{}, AxisList<(2 + Is)...>{}, Mul, Add>(X, W);
        }

        // Helper: backward through WO_ — contracts E... at axes (2+Is) in dA against axes (Is) in WO_.
        // [B,S,E...] × [E...,H,D] → [B,S,H,D]
        template<size_t Batch, size_t... Is>
        static auto BatchedDAttended(const Tensor<Batch, SeqLen, EmbDims...> &dA, const W_O_Type &WO,
                                     std::index_sequence<Is...>) {
            return Contract<AxisList<(2 + Is)...>{}, AxisList<Is...>{}, Mul, Add>(dA, WO);
        }

        // @doc: template<size_t Batch> Tensor<Batch, SeqLen, EmbDims...> BatchedForward(...)
        /** ######### */
        template<size_t Batch>
        Tensor<Batch, SeqLen, EmbDims...> BatchedForward(const Tensor<Batch, SeqLen, EmbDims...> &X) const {
            const float inv_sqrt = 1.f / std::sqrt(static_cast<float>(HeadDim));
            bcache_store(X, bX_buf_);

            // [B,S,E...] × [H,D,E...] → [B,S,H,D]
            const auto bQ = BatchedQKV_Contract<Batch>(X, WQ_.value, std::make_index_sequence<N_emb>{});
            const auto bK = BatchedQKV_Contract<Batch>(X, WK_.value, std::make_index_sequence<N_emb>{});
            const auto bV = BatchedQKV_Contract<Batch>(X, WV_.value, std::make_index_sequence<N_emb>{});
            bcache_store(bQ, bQ_buf_);
            bcache_store(bK, bK_buf_);
            bcache_store(bV, bV_buf_);

            // scores[b,h,s_q,s_k] = Σ_d Q[b,s_q,h,d] * K[b,s_k,h,d]
            // Batch: B(0,0) H(2,2)  Contract: D(3,3)  Free: S_q(1) S_k(1)
            // → [B,H,S_q,S_k]
            const auto scores = BatchContract<AxisList<0, 2>{}, AxisList<0, 2>{}, AxisList<3>{}, AxisList<3>{}, Mul,
                                    Add>(bQ, bK) * inv_sqrt;

            const auto battn = Softmax<3>(scores);
            bcache_store(battn, battn_buf_);
            attn_weights_ = TensorIndex<0, 0>(battn); // snap() support: expose first-sample head weights

            // attended[b,h,s_q,d] = Σ_{s_k} attn[b,h,s_q,s_k] * V[b,s_k,h,d]
            // Batch: B(0,0) H(1,2)  Contract: S_k(3,1)  Free: S_q(2) D(3)
            // → [B,H,S_q,D]
            const auto battended = BatchContract<AxisList<0, 1>{}, AxisList<0, 2>{}, AxisList<3>{}, AxisList<1>{}, Mul,
                Add>(battn, bV);
            bcache_store(battended, battended_buf_);

            // out[b,s,e...] = Σ_{h,d} attended[b,h,s,d] * WO[e...,h,d]
            // Contract: H(1,N_emb) D(3,N_emb+1)  Free: (B,S) and E...
            // → [B,S,E...]
            return Contract<AxisList<1, 3>{}, AxisList<N_emb, N_emb + 1>{}, Mul, Add>(battended, WO_.value);
        }

        template<size_t Batch>
        Tensor<Batch, SeqLen, EmbDims...> BatchedBackward(
            const Tensor<Batch, SeqLen, EmbDims...> &delta_A,
            const Tensor<Batch, SeqLen, EmbDims...> & /*a*/,
            const Tensor<Batch, SeqLen, EmbDims...> & /*a_prev*/) {
            const float inv_sqrt = 1.f / std::sqrt(static_cast<float>(HeadDim));
            const float inv_batch = 1.f / static_cast<float>(Batch);

            const auto bQ = bcache_load<Tensor<Batch, SeqLen, Heads, HeadDim> >(bQ_buf_);
            const auto bK = bcache_load<Tensor<Batch, SeqLen, Heads, HeadDim> >(bK_buf_);
            const auto bV = bcache_load<Tensor<Batch, SeqLen, Heads, HeadDim> >(bV_buf_);
            const auto battn = bcache_load<Tensor<Batch, Heads, SeqLen, SeqLen> >(battn_buf_);
            const auto battended = bcache_load<Tensor<Batch, Heads, SeqLen, HeadDim> >(battended_buf_);
            const auto bX = bcache_load<Tensor<Batch, SeqLen, EmbDims...> >(bX_buf_);

            // --- WO_ grad ---
            // dWO[e...,h,d] = Σ_{b,s} dA[b,s,e...] * attended[b,h,s,d]
            // Contract (B,S): axes {0,1} in dA, axes {0,2} in attended
            WO_.grad += Contract<AxisList<0, 1>{}, AxisList<0, 2>{}, Mul, Add>(delta_A, battended) * inv_batch;

            // d_attended[b,s,h,d] = Σ_{e...} dA[b,s,e...] * WO[e...,h,d]   → [B,S,H,D]
            const auto d_attended = BatchedDAttended<Batch>(delta_A, WO_.value, std::make_index_sequence<N_emb>{});

            // --- Attended backward ---
            // d_attn[b,h,s_q,s_k] = Σ_d d_att[b,s_q,h,d] * V[b,s_k,h,d]
            // Batch: B(0,0) H(2,2)  Contract: D(3,3)  Free: S_q(1) S_k(1)  → [B,H,S_q,S_k]
            const auto d_attn = BatchContract<AxisList<0, 2>{}, AxisList<0, 2>{}, AxisList<3>{}, AxisList<3>{}, Mul,
                Add>(d_attended, bV);

            // d_V[b,h,s_k,d] = Σ_{s_q} attn[b,h,s_q,s_k] * d_att[b,s_q,h,d]
            // Batch: B(0,0) H(1,2)  Contract: S_q(2,1)  Free: S_k(3) D(3)  → [B,H,S_k,D]
            const auto d_V = BatchContract<AxisList<0, 1>{}, AxisList<0, 2>{}, AxisList<2>{}, AxisList<1>{}, Mul, Add>(
                battn, d_attended);

            // --- Softmax backward (over axis 3) ---
            const auto d_scores = SoftmaxPrime<3>(d_attn, battn) * inv_sqrt;

            // --- Scores backward ---
            // d_Q[b,h,s_q,d] = Σ_{s_k} d_scores[b,h,s_q,s_k] * K[b,s_k,h,d]
            // Batch: B(0,0) H(1,2)  Contract: S_k(3,1)  Free: S_q(2) D(3)  → [B,H,S_q,D]
            const auto d_Q = BatchContract<AxisList<0, 1>{}, AxisList<0, 2>{}, AxisList<3>{}, AxisList<1>{}, Mul, Add>(
                d_scores, bK);

            // d_K[b,h,s_k,d] = Σ_{s_q} d_scores[b,h,s_q,s_k] * Q[b,s_q,h,d]
            // Batch: B(0,0) H(1,2)  Contract: S_q(2,1)  Free: S_k(3) D(3)  → [B,H,S_k,D]
            const auto d_K = BatchContract<AxisList<0, 1>{}, AxisList<0, 2>{}, AxisList<2>{}, AxisList<1>{}, Mul, Add>(
                d_scores, bQ);

            // --- W grads: d_[B,H,S,D] × bX[B,S,E...] → [H,D,E...]
            // Contract (B,S): axes {0,2} in d_, axes {0,1} in bX
            WQ_.grad += Contract<AxisList<0, 2>{}, AxisList<0, 1>{}, Mul, Add>(d_Q, bX) * inv_batch;
            WK_.grad += Contract<AxisList<0, 2>{}, AxisList<0, 1>{}, Mul, Add>(d_K, bX) * inv_batch;
            WV_.grad += Contract<AxisList<0, 2>{}, AxisList<0, 1>{}, Mul, Add>(d_V, bX) * inv_batch;

            // --- dX: d_[B,H,S,D] × W[H,D,E...] → [B,S,E...]
            // Contract (H,D): axes {1,3} in d_, axes {0,1} in W
            return Contract<AxisList<1, 3>{}, AxisList<0, 1>{}, Mul, Add>(d_Q, WQ_.value)
                   + Contract<AxisList<1, 3>{}, AxisList<0, 1>{}, Mul, Add>(d_K, WK_.value)
                   + Contract<AxisList<1, 3>{}, AxisList<0, 1>{}, Mul, Add>(d_V, WV_.value);
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
