#pragma once
#include "TensorContract.hpp"
#include "TensorReduce.hpp"
#include "NetworkUtil.hpp"

namespace TTTN {
    // @doc: template<size_t SeqLen, size_t Heads, size_t... EmbDims> class MultiHeadAttentionBlock
    /**
     * `Block` implementation for **mult-head self-attention** over sequences of arbitrary-rank token embeddings
     * Parameterized by `size_t SeqLen`, `size_t Heads`, and `size_t...EmbDims`
     */
    template<size_t SeqLen, size_t Heads, size_t... EmbDims>
    class MultiHeadAttentionBlock {
    public:
        // @doc: using MultiHeadAttentionBlock::InputTensor
        /** `InputTensor = Tensor<SeqLen, EmbDims...>` */
        using InputTensor = Tensor<SeqLen, EmbDims...>;
        // @doc: using MultiHeadAttentionBlock::OutputTensor
        /** `OutputTensor = Tensor<SeqLen, EmbDims...>` */
        using OutputTensor = Tensor<SeqLen, EmbDims...>;

        // @doc: static constexpr size_t MultiHeadAttentionBlock::N_emb
        /** Embeddings rank (`sizeof...(EmbDims)`) */
        static constexpr size_t N_emb = sizeof...(EmbDims);
        // @doc: static constexpr size_t MultiHeadAttentionBlock::EmbSize
        /** Embeddings net size (`TensorDimsProduct<EmbDims...>::value`) */
        static constexpr size_t EmbSize = TensorDimsProduct<EmbDims...>::value;
        // @doc: static constexpr size_t MultiHeadAttentionBlock::HeadDim
        /**
         * `HeadDim = EmbSize / Heads`
         * `static_assert` that `EmbSize % Heads == 0`
         */
        static constexpr size_t HeadDim = EmbSize / Heads;
        static_assert(EmbSize % Heads == 0,
                      "Heads must be a factor of EmbSize (the product of all EmbDims)");


        // @doc: using MultiHeadAttentionBlock::W_QKV_Type
        /**
         * Shape of `W_Q`, `W_K`, `W_V`
         * `[Heads, HeadDim, EmbDims...]`
         * For each head, contract `EmbDims...` from Input to get `QKV_Type`
         * `[Heads, HeadDim, EmbDims...] x [SeqLen, EmbDims...] -> [SeqLen, Heads, HeadDim]`
         */
        using W_QKV_Type = Tensor<Heads, HeadDim, EmbDims...>;

        // @doc: using MultiHeadAttentionBlock::W_O_Type
        /**
         * Shape of `W_O`
         * `[EmbDims..., Heads, HeadDim]`
         * Scores x `W_O` -> `OutputTensor`:
         * `[Heads, SeqLen, SeqLen] x [EmbDims..., Heads, HeadDim] -> [SeqLen, EmbDims...]`
         */
        using W_O_Type = Tensor<EmbDims..., Heads, HeadDim>;


        // @doc: using MultiHeadAttentionBlock::QKV_Type
        /**
         * Shape of `Q`, `K`, `V`
         * `[SeqLen, Heads, HeadDim]`
         * Right now, at each head, each element of the sequence is represented by `HeadDim`
         */
        using QKV_Type = Tensor<SeqLen, Heads, HeadDim>;


        // @doc: using MultiHeadAttentionBlock::Scores_Type
        /**
         * Attention pattern matrix
         * `[Heads, SeqLen, SeqLen]`
         */
        using Scores_Type = Tensor<Heads, SeqLen, SeqLen>;


        // @doc: using MultiHeadAttentionBlock::Attended_Type
        /**
         * Attended values after softmax-weighted sum (between attention pattern and `W_O` transformation)
         * `[Heads, SeqLen, HeadDim]`
         */
        using Attended_Type = Tensor<Heads, SeqLen, HeadDim>;

    private:
        // @doc: Param<W_QKV_Type> MultiHeadAttentionBlock::WQ_
        /** Query Weight Parameter (`Param<W_QKV_Type>`) */
        Param<W_QKV_Type> WQ_;
        // @doc: Param<W_QKV_Type> MultiHeadAttentionBlock::WK_
        /** Key Weight Parameter (`Param<W_QKV_Type>`) */
        Param<W_QKV_Type> WK_;
        // @doc: Param<W_QKV_Type> MultiHeadAttentionBlock::WV_
        /** Value Weight Parameter (`Param<W_QKV_Type>`) */
        Param<W_QKV_Type> WV_;
        // @doc: Param<W_O_Type> MultiHeadAttentionBlock::WO_
        /** Out projection matrix Parameter (`Param<W_O_Type>`) */
        Param<W_O_Type> WO_;

        // @doc: mutable InputTensor MultiHeadAttentionBlock::X_cache_
        /** Cached `mutable` `Tensor` for `InputTensor x`, used by `Backward` */
        mutable InputTensor X_cache_{};
        // @doc: mutable QKV_Type MultiHeadAttentionBlock::Q_
        /** Cached `mutable` `Tensor` for `Q`, used by `Backward` */
        mutable QKV_Type Q_{};
        // @doc: mutable QKV_Type MultiHeadAttentionBlock::K_
        /** Cached `mutable` `Tensor` for `K`, used by `Backward` */
        mutable QKV_Type K_{};
        // @doc: mutable QKV_Type MultiHeadAttentionBlock::V_
        /** Cached `mutable` `Tensor` for `V`, used by `Backward` */
        mutable QKV_Type V_{};
        // @doc: mutable Scores_Type MultiHeadAttentionBlock::attn_weights_
        /** Cached `mutable` `Tensor` for attention matrix, used by `Backward` */
        mutable Scores_Type attn_weights_{};
        // @doc: mutable Attended_Type MultiHeadAttentionBlock::attended_
        /** Cached `mutable` `Tensor` for attended embeddings, used by `Backward` */
        mutable Attended_Type attended_{};

        // @doc: mutable std::vector<float> MultiHeadAttentionBlock::bX_buf_
        /** Cached `std::vector<float>` for batched `InputTensor x`, used by `BatchedBackward` (not a `Tensor` because `Batch` is a function template parameter, not a class paramater) */
        mutable std::vector<float> bX_buf_;
        // @doc: mutable std::vector<float> MultiHeadAttentionBlock::bQ_buf_
        /** Cached `std::vector<float>` for batched `Q`, used by `BatchedBackward` (not a `Tensor` because `Batch` is a function template parameter, not a class paramater) */
        mutable std::vector<float> bQ_buf_;
        // @doc: mutable std::vector<float> MultiHeadAttentionBlock::bK_buf_
        /** Cached `std::vector<float>` for batched `K`, used by `BatchedBackward` (not a `Tensor` because `Batch` is a function template parameter, not a class paramater) */
        mutable std::vector<float> bK_buf_;
        // @doc: mutable std::vector<float> MultiHeadAttentionBlock::bV_buf_
        /** Cached `std::vector<float>` for batched `V`, used by `BatchedBackward` (not a `Tensor` because `Batch` is a function template parameter, not a class paramater) */
        mutable std::vector<float> bV_buf_;
        // @doc: mutable std::vector<float> MultiHeadAttentionBlock::battn_buf_
        /** Cached `std::vector<float>` for batched attention matrix, used by `BatchedBackward` (not a `Tensor` because `Batch` is a function template parameter, not a class paramater) */
        mutable std::vector<float> battn_buf_;
        // @doc: mutable std::vector<float> MultiHeadAttentionBlock::battended_buf_
        /** Cached `std::vector<float>` for batched attended embeddings, used by `BatchedBackward` (not a `Tensor` because `Batch` is a function template parameter, not a class paramater) */
        mutable std::vector<float> battended_buf_;

        // @doc: template<typename T> static void MultiHeadAttentionBlock::bcache_store(const T &t, std::vector<float> &buf)
        /** Setter for batch cache `std::vector<float>`s */
        template<typename T>
        static void bcache_store(const T &t, std::vector<float> &buf) {
            // assign handles efficiently allocating + filling
            buf.assign(t.data(), t.data() + T::Size);
        }

        // @doc: template<typename T> static T MultiHeadAttentionBlock::bcache_load(const std::vector<float> &buf)
        /** Setter for batch cache `std::vector<float>`s */
        template<typename T>
        static T bcache_load(const std::vector<float> &buf) {
            T t;
            std::copy(buf.begin(), buf.begin() + T::Size, t.data());
            return t;
        }

    public:
        // @doc: auto MultiHeadAttentionBlock::all_params()
        /** Returns `std::tuple` of `Param&` */
        auto all_params() { return std::tie(WQ_, WK_, WV_, WO_); }
        // @doc: auto MultiHeadAttentionBlock::all_params() const
        /** Returns `std::tuple` of `const Param&` */
        auto all_params() const { return std::tie(WQ_, WK_, WV_, WO_); }

        // @doc: const Scores_Type &MultiHeadAttentionBlock::attn_weights() const
        /** Getter for attention pattern matrix */
        const Scores_Type &attn_weights() const { return attn_weights_; }

        // @doc: void MultiHeadAttentionBlock::peek(SnapshotMap &out, const std::string &prefix) const
        /** `PeekableBlock` satisfier function; returns `attn_weights_` */
        void peek(SnapshotMap &out, const std::string &prefix) const {
            snap_add(out, prefix + "attn_weights", attn_weights_);
        }

        // @doc: MultiHeadAttentionBlock::MultiHeadAttentionBlock()
        /** Calls `XavierInitMD` on all weight `Tensor`s (`WQ_`, `WK_`, `WV_`, `WO_`) */
        MultiHeadAttentionBlock() {
            XavierInitMD(WQ_.value, EmbSize, HeadDim);
            XavierInitMD(WK_.value, EmbSize, HeadDim);
            XavierInitMD(WV_.value, EmbSize, HeadDim);
            XavierInitMD(WO_.value, EmbSize, EmbSize);
        }

        // @doc: static constexpr auto MultiHeadAttentionBlock::QKV_Contract(const InputTensor &X, const W_QKV_Type &wm)
        /** ######### */
        static constexpr auto QKV_Contract(const InputTensor &X, const W_QKV_Type &wm) {
            return []<size_t... Is>(std::index_sequence<Is...>, const InputTensor &x, const W_QKV_Type &w) {
                return Contract<AxisList<(1 + Is)...>{}, AxisList<(2 + Is)...>{}, Mul, Add>(x, w);
            }(std::make_index_sequence<N_emb>{}, X, wm);
        }

        // @doc: OutputTensor MultiHeadAttentionBlock::Forward(const InputTensor &X) const
        /** ######### */
        OutputTensor Forward(const InputTensor &X) const {
            const float inv_sqrt = 1.f / std::sqrt(static_cast<float>(HeadDim));
            X_cache_ = X;

            // [SeqLen, EmbDims...] x [Heads, HeadDim, EmbDims...] -> [SeqLen, Heads, HeadDim]
            Q_ = QKV_Contract(X, WQ_.value);
            K_ = QKV_Contract(X, WK_.value);
            V_ = QKV_Contract(X, WV_.value);

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


        // @doc: InputTensor MultiHeadAttentionBlock::Backward(const OutputTensor &delta_A, const OutputTensor &a, const InputTensor &a_prev)
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
        // @doc: template<size_t Batch> static auto MultiHeadAttentionBlock::BatchedQKV_Contract(const Tensor<Batch, SeqLen, EmbDims...> &X, const W_QKV_Type &W)
        /** ######### */
        template<size_t Batch>
        static auto BatchedQKV_Contract(const Tensor<Batch, SeqLen, EmbDims...> &X, const W_QKV_Type &W) {
            return []<size_t... Is>(std::index_sequence<Is...>,
                                    const Tensor<Batch, SeqLen, EmbDims...> &x, const W_QKV_Type &w) {
                return Contract<AxisList<(2 + Is)...>{}, AxisList<(2 + Is)...>{}, Mul, Add>(x, w);
            }(std::make_index_sequence<N_emb>{}, X, W);
        }

        // Helper: backward through WO_ — contracts E... at axes (2+Is) in dA against axes (Is) in WO_.
        // [B,S,E...] × [E...,H,D] → [B,S,H,D]
        // @doc: template<size_t Batch> static auto MultiHeadAttentionBlock::BatchedDAttended(const Tensor<Batch, SeqLen, EmbDims...> &dA, const W_O_Type &WO)
        /** ######### */
        template<size_t Batch>
        static auto BatchedDAttended(const Tensor<Batch, SeqLen, EmbDims...> &dA, const W_O_Type &WO) {
            return []<size_t... Is>(std::index_sequence<Is...>,
                                    const Tensor<Batch, SeqLen, EmbDims...> &da, const W_O_Type &wo) {
                return Contract<AxisList<(2 + Is)...>{}, AxisList<Is...>{}, Mul, Add>(da, wo);
            }(std::make_index_sequence<N_emb>{}, dA, WO);
        }


        // @doc: template<size_t Batch> Tensor<Batch, SeqLen, EmbDims...> MultiHeadAttentionBlock::BatchedForward(const Tensor<Batch, SeqLen, EmbDims...> &X) const
        /** ######### */
        template<size_t Batch>
        Tensor<Batch, SeqLen, EmbDims...> BatchedForward(const Tensor<Batch, SeqLen, EmbDims...> &X) const {
            const float inv_sqrt = 1.f / std::sqrt(static_cast<float>(HeadDim));
            bcache_store(X, bX_buf_);

            // [B,S,E...] × [H,D,E...] → [B,S,H,D]
            const auto bQ = BatchedQKV_Contract<Batch>(X, WQ_.value);
            const auto bK = BatchedQKV_Contract<Batch>(X, WK_.value);
            const auto bV = BatchedQKV_Contract<Batch>(X, WV_.value);
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

        // @doc: template<size_t Batch> Tensor<Batch, SeqLen, EmbDims...> MultiHeadAttentionBlock::BatchedBackward(const Tensor<Batch, SeqLen, EmbDims...> &delta_A, const Tensor<Batch, SeqLen, EmbDims...> &a, const Tensor<Batch, SeqLen, EmbDims...> &a_prev)
        /** ######### */
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
            const auto d_attended = BatchedDAttended<Batch>(delta_A, WO_.value);

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
    };


    // @doc: template<size_t Heads, size_t... EmbDims> struct MHAttention
    /**
     * `BlockRecipe` for `MultiHeadAttentionBlock`
     * Takes in `size_t Heads` for head count and `size_t...EmbDims` indicating the dimensionality of the embeddings
     * `InputT` passed to `Resolve` should have its first axis be `SeqLen`
     * `Resolve = MultiHeadAttentionBlock<TensorFirstDim<InputT>::value, Heads, EmbDims...>`
     */
    template<size_t Heads, size_t... EmbDims>
    struct MHAttention {
        using OutputTensor = Tensor<1, EmbDims...>;

        template<typename InputT> requires IsTensor<InputT>
        using Resolve = MultiHeadAttentionBlock<TensorFirstDim<InputT>::value, Heads, EmbDims...>;
    };
} // namespace TTTN
