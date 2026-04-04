#pragma once
#include "TensorContract.hpp"
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
         * `[Heads, SeqLen, HeadDim] x [EmbDims..., Heads, HeadDim] -> [SeqLen, EmbDims...]`
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
        /** Cached `std::vector<float>` for batched `InputTensor x`, used by `BatchedBackward` (not a `Tensor` because `Batch` is a function template parameter, not a class parameter) */
        mutable std::vector<float> bX_buf_;
        // @doc: mutable std::vector<float> MultiHeadAttentionBlock::bQ_buf_
        /** Cached `std::vector<float>` for batched `Q`, used by `BatchedBackward` (not a `Tensor` because `Batch` is a function template parameter, not a class parameter) */
        mutable std::vector<float> bQ_buf_;
        // @doc: mutable std::vector<float> MultiHeadAttentionBlock::bK_buf_
        /** Cached `std::vector<float>` for batched `K`, used by `BatchedBackward` (not a `Tensor` because `Batch` is a function template parameter, not a class parameter) */
        mutable std::vector<float> bK_buf_;
        // @doc: mutable std::vector<float> MultiHeadAttentionBlock::bV_buf_
        /** Cached `std::vector<float>` for batched `V`, used by `BatchedBackward` (not a `Tensor` because `Batch` is a function template parameter, not a class parameter) */
        mutable std::vector<float> bV_buf_;
        // @doc: mutable std::vector<float> MultiHeadAttentionBlock::battn_buf_
        /** Cached `std::vector<float>` for batched attention matrix, used by `BatchedBackward` (not a `Tensor` because `Batch` is a function template parameter, not a class parameter) */
        mutable std::vector<float> battn_buf_;
        // @doc: mutable std::vector<float> MultiHeadAttentionBlock::battended_buf_
        /** Cached `std::vector<float>` for batched attended embeddings, used by `BatchedBackward` (not a `Tensor` because `Batch` is a function template parameter, not a class parameter) */
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
        /** Abstraction of the contraction that occurs between `InputTensor x` and `W_QKV_Type w` (for **query**, **key**, and **value**) */
        static constexpr auto QKV_Contract(const InputTensor &X, const W_QKV_Type &wm) {
            return []<size_t... Is>(std::index_sequence<Is...>, const InputTensor &x, const W_QKV_Type &w) {
                // `Is` iterates over embedding rank (not size)
                // sumproduct-contract embedding dimensions: all but first from X, all but first 2 from W
                return Contract<AxisList<(1 + Is)...>{}, AxisList<(2 + Is)...>{}, Mul, Add>(x, w);
            }(std::make_index_sequence<N_emb>{}, X, wm);
        }

        // @doc: OutputTensor MultiHeadAttentionBlock::Forward(const InputTensor &X) const
        /**
         * Forward pass:
         * Take `InputTensor x` as input
         * Map to **heads**
         * Within each head, map to `Q`, `K`, and `V` representations
         * Compute attention matrix for each head (`softmax(Q . K)`)
         * Contract `V` with `x` according to attention matrix (attend)
         * Map from **heads** to normal embedding dimension, return
         * Excellent comments in the code
         */
        OutputTensor Forward(const InputTensor &X) const {
            const float inv_sqrt = 1.f / std::sqrt(static_cast<float>(HeadDim));
            X_cache_ = X;

            // [SeqLen, EmbDims...] x [Heads, HeadDim, EmbDims...] -> [SeqLen, Heads, HeadDim]
            Q_ = QKV_Contract(X, WQ_.value);
            K_ = QKV_Contract(X, WK_.value);
            V_ = QKV_Contract(X, WV_.value);

            // Q: [SeqLen, Heads, HeadDim]
            // K: [SeqLen, Heads, HeadDim]
            // ?(Q, K) -> [Heads, SeqLen, SeqLen]
            // ? :
            //      batch axes: Q/K[1] = 'Heads`
            //      contract axes: Q/K[2] = 'HeadDim'
            auto rawQK = BatchContract<AxisList<1>{}, AxisList<1>{}, AxisList<2>{}, AxisList<2>{}, Mul, Add>(Q_, K_);
            rawQK *= inv_sqrt;
            // rawQK: [Heads, SeqLen_Q, SeqLen_K]
            // softmax needs to apply 'over the keys', so index 2
            attn_weights_ = Softmax<2>(rawQK);

            // attn_weights_: [Heads, SeqLen_Q, SeqLen_K]
            // ?(attn_weights_, V_) -> [Heads, SeqLen, HeadDim]
            // ? :
            //      batch axes: attn_weights_/V_[0] = 'Heads'
            //      contract axes: attn_weights_[2] = 'SeqLen_K', V_[0] = 'SeqLen'
            attended_ = BatchContract<AxisList<0>{}, AxisList<1>{}, AxisList<2>{}, AxisList<0>{}, Mul, Add>(
                attn_weights_, V_);

            // attended_: [Heads, SeqLen, HeadDim]
            // WO_: [EmbDims..., Heads, HeadDim]
            // ?(attended_, WO_) -> [SeqLen, EmbDims...]
            //      contract axes: attended_[0, 2] = ['Heads', 'HeadDim'], WO_[N_emb, N_emb+1] = ['Heads', 'HeadDim']
            return Contract<AxisList<0, 2>{}, AxisList<N_emb, N_emb + 1>{}, Mul, Add>(attended_, WO_.value);
        }


        // @doc: InputTensor MultiHeadAttentionBlock::Backward(const OutputTensor &delta_A, const OutputTensor &a, const InputTensor &a_prev)
        /**
         * Backpropagates gradients through attended, attention pattern,`Q`, `K`, and `V` representations, passing same-shape gradient upstream
         * Extensive commenting below and in the code
         */
        InputTensor Backward(const OutputTensor &delta_A,
                             const OutputTensor & /*a*/,
                             const InputTensor & /*a_prev*/) {
            const float inv_sqrt = 1.f / std::sqrt(static_cast<float>(HeadDim));

            // delta_A: [SeqLen, EmbDims...]
            // attended_: [Heads, SeqLen, HeadDim]
            // ?(delta_A, attended_) -> [EmbDims..., Heads, HeadDim]
            //      contract delta_A[0] = 'SeqLen' with attended_[1] = 'SeqLen'
            // this collapses the gradient coming back to this Block along SeqLen
            // ie WO_ is punished at each free index for its effect on *all* tokens in the sequence
            // when we strip SeqLen, we have [EmbDims...] x [Heads, HeadDim]
            // the result is a sum of: Outer(delta_A[s, EmbDims...], attended_[Heads, s, HeadDim]) for s in SeqLen
            WO_.grad += Einsum<0, 1>(delta_A, attended_);

            // now need dLoss/d_attended_
            // delta_A: [SeqLen, EmbDims...]
            // WO_: [EmbDims..., Heads, HeadDim]
            // ?(delta_A, WO_) -> [SeqLen, Heads, HeadDim]
            //      ΣΠ contracted over EmbDims... is exactly what we want
            // NOTE: d_att ([SeqLen, Heads, HeadDim]) does not have the same shape as attended_ ([Heads, SeqLen, HeadDim])
            // this is because attended_ itself is an intermediary (no weights) and we can use flexible contraction below to grab the axes we want
            const auto d_attended = ΣΠ<N_emb>(delta_A, WO_.value);

            // now propagate gradients from d_attended to the attention pattern itself
            // d_attended: [SeqLen, Heads, HeadDim]
            // V_: [SeqLen, Heads, HeadDim]
            // ?(d_attended, V_) -> [Heads, SeqLen_Q, SeqLen_K]
            //      batch axes: d_attended[1] = 'Heads', V_[1] = 'Heads'
            //      contract axes: d_attended[2] = 'HeadDim', V_[2] = 'HeadDim'
            // NOTE: d_attended is 'A' in the BatchContract call, meaning its free axes (SeqLen) go first
            // so SeqLen_Q comes from d_attended, SeqLen_K comes from V_.

            // d_attended[s_q, h, d] carries 'how much did attended token s_q need to change?'
            // V_[s_k, h, d] carries 'what did key s_k contribute to values it attended?'
            // so d_attention[h, s_q, s_k] carries 'how much should the WEIGHT on key s_k change for query s_q?'
            //      -> this is exactly the dot product over d (over the whole HeadDim)
            // Why? because s_k's WEIGHT needs to change for s_q exactly according to:
            //      (A) how much the *attended* s_q should change (post s_k)
            //      (B) what s_k passes to s_q (regardless of weight)
            const auto d_attention = BatchContract<AxisList<1>{}, AxisList<1>{}, AxisList<2>{}, AxisList<2>{}, Mul,
                Add>(
                d_attended, V_);

            // V_[s_k, h, d] carries 'what did key s_k contribute to values it attended?'
            // V_ is combined with attn_weights_ to make attended_...
            //      V_ x attn_weights_ = attended_; d_attended_/d_V_ = attn_weights_... so feed grad to attn_weights_:
            // attn_weights_: [Heads, SeqLen_Q, SeqLen_K]
            // d_attended: [SeqLen, Heads, HeadDim]
            // ?(attn_weights_, d_attended) -> [Heads, SeqLen, HeadDim]
            //      (...d_V_ is a temp, so doesn't need V_'s exact shape, just the same info!)
            //      batch axes: attn_weights_[0] = 'Heads', d_attended[1] = 'Heads'
            //      contract axes: attn_weights_[1] = 'SeqLen_Q', d_attended[0] = 'SeqLen'
            const auto d_V = BatchContract<AxisList<0>{}, AxisList<1>{}, AxisList<1>{}, AxisList<0>{}, Mul, Add>(
                attn_weights_, d_attended);

            // take softmax derivative
            // again, axis = 2, over SeqLen_K (which is the axis over which Softmax was computed)
            auto d_scores = SoftmaxPrime<2>(d_attention, attn_weights_);
            d_scores *= inv_sqrt;

            // d_scores: [Heads, SeqLen_Q, SeqLen_K]
            // K_: [SeqLen, Heads, HeadDim]
            // ?(d_scores, K_) -> [Heads, SeqLen, HeadDim] (d_Q is also a temp...when WQ_.grad is updated, shapes will agree)
            //      batch axes: d_scores[0] = 'Heads', K_[1] = 'Heads'
            //      contract axes: d_scores[2] = 'SeqLen_K', K_[0] = 'SeqLen'
            const auto d_Q = BatchContract<AxisList<0>{}, AxisList<1>{}, AxisList<2>{}, AxisList<0>{}, Mul, Add>(
                d_scores, K_);
            // same logic for grad wrt K
            const auto d_K = BatchContract<AxisList<0>{}, AxisList<1>{}, AxisList<1>{}, AxisList<0>{}, Mul, Add>(
                d_scores, Q_);

            // now for WQ_, we have d_Q (coming back) times X (which multiplies with WQ_ to make Q!)
            // d_Q: [Heads, SeqLen, HeadDim]
            // X_cache_: [SeqLen, EmbDims...]
            // ?(d_Q, X_cache_) -> [Heads, HeadDim, EmbDims...] (this must be actual WQ_ shape)
            //      contract: d_Q[1] = 'SeqLen' with X_cache_[0] = 'SeqLen'
            // these weights are duly punished for all tokens!
            WQ_.grad += Einsum<1, 0>(d_Q, X_cache_);
            WK_.grad += Einsum<1, 0>(d_K, X_cache_);
            WV_.grad += Einsum<1, 0>(d_V, X_cache_);

            // now derivatives of all Q, K, V representations of the sequence need to be added
            // to represent what they all depend on: the input tokens

            // d_Q: [Heads, SeqLen, HeadDim]
            // WQ_: [Heads, HeadDim, EmbDims...]
            // ?(d_Q, WQ_) -> [SeqLen, EmbDims...]
            //      contract d_Q[0, 2] = ['Heads', 'HeadDim'] with WQ_[0, 1] = ['Heads', 'HeadDim']
            auto result = Contract<AxisList<0, 2>{}, AxisList<0, 1>{}, Mul, Add>(d_Q, WQ_.value);
            result += Contract<AxisList<0, 2>{}, AxisList<0, 1>{}, Mul, Add>(d_K, WK_.value);
            result += Contract<AxisList<0, 2>{}, AxisList<0, 1>{}, Mul, Add>(d_V, WV_.value);
            return result;
        }


        // @doc: template<size_t Batch> static auto MultiHeadAttentionBlock::BatchedQKV_Contract(const Tensor<Batch, SeqLen, EmbDims...> &X, const W_QKV_Type &W)
        /** Abstraction of the batched contraction that occurs between `PrependBatch<InputTensor> x` and `W_QKV_Type w` (for * *query**, **key**, and **value**) */
        template<size_t Batch>
        static auto BatchedQKV_Contract(const Tensor<Batch, SeqLen, EmbDims...> &X, const W_QKV_Type &W) {
            // same logic as QKV_Contract but now with Batch
            return []<size_t... Is>(std::index_sequence<Is...>,
                                    const Tensor<Batch, SeqLen, EmbDims...> &x, const W_QKV_Type &w) {
                // 2 + instead of 1 + (in QKV_Contract) because of Batch dim
                return Contract<AxisList<(2 + Is)...>{}, AxisList<(2 + Is)...>{}, Mul, Add>(x, w);
            }(std::make_index_sequence<N_emb>{}, X, W);
        }

        // @doc: template<size_t Batch> static auto MultiHeadAttentionBlock::BatchedDAttended(const Tensor<Batch, SeqLen, EmbDims...> &dA, const W_O_Type &WO)
        /** Abstraction of the contraction that occurs between `delta_A` coming back and `WO_` to produce the attended tokens pre out-projection */
        template<size_t Batch>
        static auto BatchedDAttended(const Tensor<Batch, SeqLen, EmbDims...> &dA, const W_O_Type &WO) {
            // d_attended contracts the EmbDims..., so we use index sequence to cover those, all other dims being free
            return []<size_t... Is>(std::index_sequence<Is...>,
                                    const Tensor<Batch, SeqLen, EmbDims...> &da, const W_O_Type &wo) {
                return Contract<AxisList<(2 + Is)...>{}, AxisList<Is...>{}, Mul, Add>(da, wo);
            }(std::make_index_sequence<N_emb>{}, dA, WO);
        }


        // @doc: template<size_t Batch> Tensor<Batch, SeqLen, EmbDims...> MultiHeadAttentionBlock::BatchedForward(const Tensor<Batch, SeqLen, EmbDims...> &X) const
        /**
         * Echoes `Forward`, but incorporates `Batch`
         * Batched forward pass goes from `[Batch, SeqLen, EmbDims...]` -> `[Batch, SeqLen, EmbDims...]`
         * Extensive comments in code
         */
        template<size_t Batch>
        Tensor<Batch, SeqLen, EmbDims...> BatchedForward(const Tensor<Batch, SeqLen, EmbDims...> &X) const {
            // more extensive commenting exists in Forward()
            const float inv_sqrt = 1.f / std::sqrt(static_cast<float>(HeadDim));
            bcache_store(X, bX_buf_);

            // [Batch, SeqLen, EmbDims...] x [Batch, Heads, HeadDim, EmbDims...] -> [Batch, SeqLen, Heads, HeadDim]
            const auto bQ = BatchedQKV_Contract<Batch>(X, WQ_.value);
            const auto bK = BatchedQKV_Contract<Batch>(X, WK_.value);
            const auto bV = BatchedQKV_Contract<Batch>(X, WV_.value);
            bcache_store(bQ, bQ_buf_);
            bcache_store(bK, bK_buf_);
            bcache_store(bV, bV_buf_);

            // [Batch, Heads, SeqLen_Q, SeqLen_K]
            auto scores = BatchContract<AxisList<0, 2>{}, AxisList<0, 2>{}, AxisList<3>{}, AxisList<3>{}, Mul,
                Add>(bQ, bK);
            scores *= inv_sqrt;

            // softmax over SeqLen_K
            const auto b_attn = Softmax<3>(scores);
            bcache_store(b_attn, battn_buf_);

            // for Snap(), expose first sample from first Batch
            attn_weights_ = TensorIndex<0, 0>(b_attn);

            // [Batch, Heads, SeqLen_Q, HeadDim]
            const auto b_attended = BatchContract<AxisList<0, 1>{}, AxisList<0, 2>{}, AxisList<3>{}, AxisList<1>{}, Mul,
                Add>(b_attn, bV);
            bcache_store(b_attended, battended_buf_);

            // out needs to be: [Batch, SeqLen, EmbDims...]
            // attended: [Batch, Heads, SeqLen_Q, HeadDim]
            // WO_: [EmbDims..., Heads, HeadDim]

            // so contract attended[1, 3] = ['Heads', 'HeadDim'] and WO_[N_emb, N_emb+1] = ['Heads', 'HeadDim']
            return Contract<AxisList<1, 3>{}, AxisList<N_emb, N_emb + 1>{}, Mul, Add>(b_attended, WO_.value);
        }

        // @doc: template<size_t Batch> Tensor<Batch, SeqLen, EmbDims...> MultiHeadAttentionBlock::BatchedBackward(const Tensor<Batch, SeqLen, EmbDims...> &delta_A, const Tensor<Batch, SeqLen, EmbDims...> &a, const Tensor<Batch, SeqLen, EmbDims...> &a_prev)
        /**
         * Echoes 'Backward', but incorporates 'Batch'
         * Batched backward pass goes from `[Batch, SeqLen, EmbDims...]` -> `[Batch, SeqLen, EmbDims...]`
         */
        template<size_t Batch>
        Tensor<Batch, SeqLen, EmbDims...> BatchedBackward(
            const Tensor<Batch, SeqLen, EmbDims...> &delta_A,
            const Tensor<Batch, SeqLen, EmbDims...> & /*a*/,
            const Tensor<Batch, SeqLen, EmbDims...> & /*a_prev*/) {
            // same gradient logic as Backward, using same Batch logic from BatchedForward
            const float inv_sqrt = 1.f / std::sqrt(static_cast<float>(HeadDim));
            const float inv_batch = 1.f / static_cast<float>(Batch);

            const auto bQ = bcache_load<Tensor<Batch, SeqLen, Heads, HeadDim> >(bQ_buf_);
            const auto bK = bcache_load<Tensor<Batch, SeqLen, Heads, HeadDim> >(bK_buf_);
            const auto bV = bcache_load<Tensor<Batch, SeqLen, Heads, HeadDim> >(bV_buf_);
            const auto battn = bcache_load<Tensor<Batch, Heads, SeqLen, SeqLen> >(battn_buf_);
            const auto battended = bcache_load<Tensor<Batch, Heads, SeqLen, HeadDim> >(battended_buf_);
            const auto bX = bcache_load<Tensor<Batch, SeqLen, EmbDims...> >(bX_buf_);


            WO_.grad += Contract<AxisList<0, 1>{}, AxisList<0, 2>{}, Mul, Add>(delta_A, battended);
            WO_.grad *= inv_batch;

            const auto d_attended = BatchedDAttended<Batch>(delta_A, WO_.value);

            const auto d_attn = BatchContract<AxisList<0, 2>{}, AxisList<0, 2>{}, AxisList<3>{}, AxisList<3>{}, Mul,
                Add>(d_attended, bV);

            const auto d_V = BatchContract<AxisList<0, 1>{}, AxisList<0, 2>{}, AxisList<2>{}, AxisList<1>{}, Mul, Add>(
                battn, d_attended);

            auto d_scores = SoftmaxPrime<3>(d_attn, battn);
            d_scores *= inv_sqrt;

            const auto d_Q = BatchContract<AxisList<0, 1>{}, AxisList<0, 2>{}, AxisList<3>{}, AxisList<1>{}, Mul, Add>(
                d_scores, bK);


            const auto d_K = BatchContract<AxisList<0, 1>{}, AxisList<0, 2>{}, AxisList<2>{}, AxisList<1>{}, Mul, Add>(
                d_scores, bQ);


            WQ_.grad += Contract<AxisList<0, 2>{}, AxisList<0, 1>{}, Mul, Add>(d_Q, bX);
            WQ_.grad *= inv_batch;
            WK_.grad += Contract<AxisList<0, 2>{}, AxisList<0, 1>{}, Mul, Add>(d_K, bX);
            WK_.grad *= inv_batch;
            WV_.grad += Contract<AxisList<0, 2>{}, AxisList<0, 1>{}, Mul, Add>(d_V, bX);
            WV_.grad *= inv_batch;


            auto result = Contract<AxisList<1, 3>{}, AxisList<0, 1>{}, Mul, Add>(d_Q, WQ_.value);
            result += Contract<AxisList<1, 3>{}, AxisList<0, 1>{}, Mul, Add>(d_K, WK_.value);
            result += Contract<AxisList<1, 3>{}, AxisList<0, 1>{}, Mul, Add>(d_V, WV_.value);
            return result;
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
