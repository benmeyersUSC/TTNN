#pragma once
#include <limits>
#include "TensorContract.hpp"
#include "NetworkUtil.hpp"

namespace TTTN {
    // @doc: template<size_t SeqLen, size_t Heads, bool Masked, size_t... EmbDims> class MultiHeadAttentionBlock
    /**
     * `Block` implementation for **mult-head self-attention** over sequences of arbitrary-rank token embeddings
     * Parameterized by `size_t SeqLen`, `size_t Heads`, `bool Masked` (causal mask), and `size_t...EmbDims`
     * When `Masked = true`, positions `k > q` in the attention score matrix are set to `-inf` before softmax
     */
    template<size_t SeqLen, size_t Heads, bool Masked, size_t... EmbDims>
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
        // @doc: static constexpr float MultiHeadAttentionBlock::inv_sqrt
        static constexpr float inv_sqrt = 1.f / std::sqrt(static_cast<float>(HeadDim));
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
        // @doc: mutable LearnedContraction<InputTensor, QKV_Type, 1> MultiHeadAttentionBlock::lc_Q_
        /** Query projection `LearnedContraction`; weight accessed via `lc_Q_.W_` */
        // @doc: mutable LearnedContraction<InputTensor, QKV_Type, 1> MultiHeadAttentionBlock::lc_K_
        /** Key projection `LearnedContraction`; weight accessed via `lc_K_.W_` */
        // @doc: mutable LearnedContraction<InputTensor, QKV_Type, 1> MultiHeadAttentionBlock::lc_V_
        /** Value projection `LearnedContraction`; weight accessed via `lc_V_.W_` */
        // QKV: [SeqLen,EmbDims...] x [Heads,HeadDim,EmbDims...] -> [SeqLen,Heads,HeadDim], NFree=1 (SeqLen passes through)
        mutable LearnedContraction<InputTensor, QKV_Type, 1> lc_Q_, lc_K_, lc_V_;
        // @doc: mutable LearnedContraction<QKV_Type, OutputTensor, 1> MultiHeadAttentionBlock::lc_O_
        /** Output projection `LearnedContraction`; weight accessed via `lc_O_.W_` */
        // Out: [SeqLen,Heads,HeadDim] x [EmbDims...,Heads,HeadDim] -> [SeqLen,EmbDims...], NFree=1 (SeqLen passes through)
        // attended_ must be permuted to [SeqLen,Heads,HeadDim] before >> and after << in backward
        mutable LearnedContraction<QKV_Type, OutputTensor, 1> lc_O_;

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

        // @doc: mutable std::vector<float> MultiHeadAttentionBlock::bQ_buf_
        /** Cached `std::vector<float>` for batched `Q`, used by `BatchedBackward` (not a `Tensor` because `Batch` is a function template parameter, not a class parameter) */
        // bQ/bK/bV are LC *outputs* — LC only caches its input, so we still need these for the attention score backward
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
        auto all_params() { return std::tie(lc_Q_.W_, lc_K_.W_, lc_V_.W_, lc_O_.W_); }
        // @doc: auto MultiHeadAttentionBlock::all_params() const
        /** Returns `std::tuple` of `const Param&` */
        auto all_params() const { return std::tie(lc_Q_.W_, lc_K_.W_, lc_V_.W_, lc_O_.W_); }

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
        MultiHeadAttentionBlock() = default;

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
            X_cache_ = X;

            // [SeqLen, EmbDims...] x [Heads, HeadDim, EmbDims...] -> [SeqLen, Heads, HeadDim]
            Q_ = X >> lc_Q_;
            K_ = X >> lc_K_;
            V_ = X >> lc_V_;

            // Q: [SeqLen, Heads, HeadDim]
            // K: [SeqLen, Heads, HeadDim]
            // ?(Q, K) -> [Heads, SeqLen, SeqLen]
            // ? :
            //      batch axes: Q/K[1] = 'Heads`
            //      contract axes: Q/K[2] = 'HeadDim'
            auto rawQK = BatchContract<AxisList<1>{}, AxisList<1>{}, AxisList<2>{}, AxisList<2>{}, Mul, Add>(Q_, K_);
            rawQK *= inv_sqrt;
            // rawQK: [Heads, SeqLen_Q, SeqLen_K]
            if constexpr (Masked) {
                constexpr float neg_inf = -std::numeric_limits<float>::infinity();
                for (size_t h = 0; h < Heads; ++h) {
                    for (size_t q = 0; q < SeqLen - 1; ++q) {
                        std::fill_n(&rawQK(h, q, q + 1), SeqLen - q - 1, neg_inf);
                    }
                }
            }
            // softmax needs to apply 'over the keys', so index 2
            attn_weights_ = Softmax<2>(rawQK);

            // attn_weights_: [Heads, SeqLen_Q, SeqLen_K]
            // ?(attn_weights_, V_) -> [Heads, SeqLen, HeadDim]
            // ? :
            //      batch axes: attn_weights_/V_[0] = 'Heads'
            //      contract axes: attn_weights_[2] = 'SeqLen_K', V_[0] = 'SeqLen'
            attended_ = BatchContract<AxisList<0>{}, AxisList<1>{}, AxisList<2>{}, AxisList<0>{}, Mul, Add>(
                attn_weights_, V_);

            // attended_: [Heads, SeqLen, HeadDim] — permute to [SeqLen, Heads, HeadDim] for minor-aligned >>
            const auto attended_perm = Permute<1, 0, 2>(attended_);
            return attended_perm >> lc_O_;
        }


        // @doc: InputTensor MultiHeadAttentionBlock::Backward(const OutputTensor &delta_A, const OutputTensor &a, const InputTensor &a_prev)
        /**
         * Backpropagates gradients through attended, attention pattern,`Q`, `K`, and `V` representations, passing same-shape gradient upstream
         * Extensive commenting below and in the code
         */
        InputTensor Backward(const OutputTensor &delta_A,
                             const OutputTensor & /*a*/,
                             const InputTensor & /*a_prev*/) {
            // now need dLoss/d_attended_
            // delta_A: [SeqLen, EmbDims...]
            // ?(delta_A, WO_) -> [SeqLen, Heads, HeadDim] — ΣΠ contracted over EmbDims...
            // NOTE: d_attended [SeqLen, Heads, HeadDim] differs from attended_ [Heads, SeqLen, HeadDim] (intermediary shape)
            // lc_O_ << delta_A returns [SeqLen, Heads, HeadDim] — permute to [Heads, SeqLen, HeadDim] for downstream
            const auto d_attended = Permute<1, 0, 2>(lc_O_ << delta_A);

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

            // d_Q/K/V: [Heads, SeqLen, HeadDim] — permute to [SeqLen, Heads, HeadDim] for lc_ <<
            // sum contributions from all three projections back to input tokens: [SeqLen, EmbDims...]
            auto result = lc_Q_ << Permute<1, 0, 2>(d_Q);
            result += lc_K_ << Permute<1, 0, 2>(d_K);
            result += lc_V_ << Permute<1, 0, 2>(d_V);
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
            // [Batch, SeqLen, EmbDims...] x [Heads, HeadDim, EmbDims...] -> [Batch, SeqLen, Heads, HeadDim]
            const auto bQ = X >> lc_Q_; // lc_Q_ caches X in its bX_buf_ for W grad in backward
            const auto bK = X >> lc_K_;
            const auto bV = X >> lc_V_;
            // bQ/bK/bV are LC *outputs* — must cache separately for attention score backward
            bcache_store(bQ, bQ_buf_);
            bcache_store(bK, bK_buf_);
            bcache_store(bV, bV_buf_);

            // [Batch, Heads, SeqLen_Q, SeqLen_K]
            auto scores = BatchContract<AxisList<0, 2>{}, AxisList<0, 2>{}, AxisList<3>{}, AxisList<3>{}, Mul,
                Add>(bQ, bK);
            scores *= inv_sqrt;
            if constexpr (Masked) {
                constexpr float neg_inf = -std::numeric_limits<float>::infinity();
                for (size_t b = 0; b < Batch; ++b) {
                    for (size_t h = 0; h < Heads; ++h) {
                        for (size_t q = 0; q < SeqLen - 1; ++q) {
                            std::fill_n(&scores(b, h, q, q + 1), SeqLen - q - 1, neg_inf);
                        }
                    }
                }
            }

            // softmax over SeqLen_K
            const auto b_attn = Softmax<3>(scores);
            bcache_store(b_attn, battn_buf_);

            // for Snap(), expose first sample from first Batch
            attn_weights_ = TensorIndex<0, 0>(b_attn);

            // [Batch, Heads, SeqLen_Q, HeadDim]
            const auto b_attended = BatchContract<AxisList<0, 1>{}, AxisList<0, 2>{}, AxisList<3>{}, AxisList<1>{}, Mul,
                Add>(b_attn, bV);

            // b_attended: [Batch, Heads, SeqLen, HeadDim] — permute to [Batch, SeqLen, Heads, HeadDim] for minor-aligned >>
            const auto b_attended_perm = Permute<0, 2, 1, 3>(b_attended);
            bcache_store(b_attended_perm, battended_buf_);
            return b_attended_perm >> lc_O_;
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
            // bQ/bK/bV cached as LC outputs in their own buffers (LC's bX_buf_ holds X, not the projected output)
            const auto bQ = bcache_load<Tensor<Batch, SeqLen, Heads, HeadDim> >(bQ_buf_);
            const auto bK = bcache_load<Tensor<Batch, SeqLen, Heads, HeadDim> >(bK_buf_);
            const auto bV = bcache_load<Tensor<Batch, SeqLen, Heads, HeadDim> >(bV_buf_);
            const auto battn = bcache_load<Tensor<Batch, Heads, SeqLen, SeqLen> >(battn_buf_);
            // battended_buf_ stores the permuted form [Batch, SeqLen, Heads, HeadDim] (set during BatchedForward)
            const auto b_attended_perm = bcache_load<Tensor<Batch, SeqLen, Heads, HeadDim> >(battended_buf_);
            // bX now lives in lc_Q_/K_/V_.bX_buf_ (each LC caches X during batched_forward, used for W grad in <<)

            // lc_O_ << delta_A returns [Batch, SeqLen, Heads, HeadDim] — downstream BatchContracts expect [B,S,H,D]
            const auto d_attended = lc_O_ << delta_A;

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

            // d_Q/K/V: [Batch, Heads, SeqLen, HeadDim] — permute to [Batch, SeqLen, Heads, HeadDim] for lc_ <<
            auto result = lc_Q_ << Permute<0, 2, 1, 3>(d_Q);
            result += lc_K_ << Permute<0, 2, 1, 3>(d_K);
            result += lc_V_ << Permute<0, 2, 1, 3>(d_V);
            return result;
        }
    };


    // @doc: template<size_t SeqLenQ, size_t SeqLenKV, size_t Heads, size_t EmbDim> class MultiHeadCrossAttentionBlock
    template<size_t SeqLenQ, size_t SeqLenKV, size_t Heads, size_t EmbDim>
    class MultiHeadCrossAttentionBlock {
    public:
        // @doc: using MultiHeadCrossAttentionBlock::InputTensor
        using InputTensor = Tensor<SeqLenQ + SeqLenKV, EmbDim>;
        // packed input [DecodedSize, EmbDim] + [EncodedSize, EmbDim]
        // @doc: using MultiHeadCrossAttentionBlock::OutputTensor
        using OutputTensor = Tensor<SeqLenQ, EmbDim>;

        // @doc: using MultiHeadCrossAttentionBlock::QSideTensor
        using QSideTensor = Tensor<SeqLenQ, EmbDim>;
        // @doc: using MultiHeadCrossAttentionBlock::KVSideTensor
        using KVSideTensor = Tensor<SeqLenKV, EmbDim>;

        // @doc: static constexpr size_t MultiHeadCrossAttentionBlock::EmbSize
        static constexpr size_t EmbSize = EmbDim;
        // @doc: static constexpr size_t MultiHeadCrossAttentionBlock::HeadDim
        static constexpr size_t HeadDim = EmbDim / Heads;
        // @doc: static constexpr float MultiHeadCrossAttentionBlock::inv_sqrt
        static constexpr float inv_sqrt = 1.f / std::sqrt(static_cast<float>(HeadDim));
        static_assert(EmbDim % Heads == 0,
                      "Heads must be a factor of EmbDim");

        // @doc: using MultiHeadCrossAttentionBlock::W_Q_Type
        using W_Q_Type = Tensor<Heads, HeadDim, EmbDim>;
        // @doc: using MultiHeadCrossAttentionBlock::W_KV_Type
        using W_KV_Type = Tensor<Heads, HeadDim, EmbDim>;
        // @doc: using MultiHeadCrossAttentionBlock::W_O_Type
        using W_O_Type = Tensor<EmbDim, Heads, HeadDim>;

        // @doc: using MultiHeadCrossAttentionBlock::QProj_Type
        using QProj_Type = Tensor<SeqLenQ, Heads, HeadDim>;
        // @doc: using MultiHeadCrossAttentionBlock::KVProj_Type
        using KVProj_Type = Tensor<SeqLenKV, Heads, HeadDim>;
        // @doc: using MultiHeadCrossAttentionBlock::Scores_Type
        using Scores_Type = Tensor<Heads, SeqLenQ, SeqLenKV>;
        // @doc: using MultiHeadCrossAttentionBlock::Attended_Type
        using Attended_Type = Tensor<Heads, SeqLenQ, HeadDim>;

    private:
        // @doc: mutable LearnedContraction<QSideTensor, QProj_Type, 1> MultiHeadCrossAttentionBlock::lc_Q_
        mutable LearnedContraction<QSideTensor, QProj_Type, 1> lc_Q_;
        // declare we want QSideTensor to map to QProj_Type
        // Tensor<SeqLenQ, EmbDim> -> Tensor<SeqLenQ, Heads, HeadDim>

        // @doc: mutable LearnedContraction<KVSideTensor, KVProj_Type, 1> MultiHeadCrossAttentionBlock::lc_K_
        mutable LearnedContraction<KVSideTensor, KVProj_Type, 1> lc_K_;
        // @doc: mutable LearnedContraction<KVSideTensor, KVProj_Type, 1> MultiHeadCrossAttentionBlock::lc_V_
        mutable LearnedContraction<KVSideTensor, KVProj_Type, 1> lc_V_;
        // want to map KVSideTensor -> KVProj_Type
        // Tensor<SeqLenKV, EmbDim> -> Tensor<SeqLenKV, Heads, HeadDim>


        // @doc: mutable LearnedContraction<QProj_Type, OutputTensor, 1> MultiHeadCrossAttentionBlock::lc_O_
        mutable LearnedContraction<QProj_Type, OutputTensor, 1> lc_O_;
        // map attended QProj_Type (or Attended_Type) to OutputTensor
        // Tensor<SeqLenQ, Heads, HeadDim> -> Tensor<SeqLenQ, EmbDim>
        // reason its from QProj is because of LearnedContraction's expectation that left axes are free


        // caches for backward pass
        // @doc: mutable QSideTensor MultiHeadCrossAttentionBlock::Q_in_cache_
        mutable QSideTensor Q_in_cache_{};
        // @doc: mutable KVSideTensor MultiHeadCrossAttentionBlock::KV_in_cache_
        mutable KVSideTensor KV_in_cache_{};
        // @doc: mutable QProj_Type MultiHeadCrossAttentionBlock::Q_
        mutable QProj_Type Q_{};
        // @doc: mutable KVProj_Type MultiHeadCrossAttentionBlock::K_
        mutable KVProj_Type K_{};
        // @doc: mutable KVProj_Type MultiHeadCrossAttentionBlock::V_
        mutable KVProj_Type V_{};
        // @doc: mutable Scores_Type MultiHeadCrossAttentionBlock::attn_weights_
        mutable Scores_Type attn_weights_{};
        // @doc: mutable Attended_Type MultiHeadCrossAttentionBlock::attended_
        mutable Attended_Type attended_{};

    public:
        // @doc: auto MultiHeadCrossAttentionBlock::all_params()
        auto all_params() { return std::tie(lc_Q_.W_, lc_K_.W_, lc_V_.W_, lc_O_.W_); }
        // @doc: auto MultiHeadCrossAttentionBlock::all_params() const
        auto all_params() const { return std::tie(lc_Q_.W_, lc_K_.W_, lc_V_.W_, lc_O_.W_); }

        // @doc: const Scores_Type &MultiHeadCrossAttention::attn_weights() const
        const Scores_Type &attn_weights() const { return attn_weights_; }

        // @doc: void MultiHeadCrossAttention::peek(SnapshotMap &out, const std::string &prefix) const
        void peek(SnapshotMap &out, const std::string &prefix) const {
            snap_add(out, prefix + "cross_attn_weights", attn_weights_);
        }

        // @doc: MultiHeadCrossAttention::MultiHeadCrossAttention()
        MultiHeadCrossAttentionBlock() = default;

        // @doc: OutputTensor MultiHeadCrossAttentionBlock::Forward(const InputTensor &x) const
        OutputTensor Forward(const InputTensor &x) const {
            // unpack input tensor into Q and KV seqs
            Unpack(x, Q_in_cache_, KV_in_cache_);
            // map to Q, K, V representations
            Q_ = Q_in_cache_ >> lc_Q_;
            K_ = KV_in_cache_ >> lc_K_;
            V_ = KV_in_cache_ >> lc_V_;
            // Q_: Tensor<SeqLenQ, Heads, HeadDim>
            // K_: Tensor<SeqLenKV, Heads, HeadDim>
            // scores: Tensor<Heads, SeqLenQ, SeqLenKV>
            //      Batch: Heads
            //      Contract: HeadDim
            auto scores = BatchContract<AxisList<1>{}, AxisList<1>{}, AxisList<2>{}, AxisList<2>{}>(Q_, K_);
            scores *= inv_sqrt;

            // softmax over K (for each Q, normalize the Ks attending)
            attn_weights_ = Softmax<2>(scores);

            // attn_weights_: Tensor<Heads, SeqLenQ, SeqLenKV>
            // V_: Tensor<SeqLenKV, Heads, HeadDim>
            // attended_: Tensor<Heads, SeqLenQ, HeadDim>
            //      Batch: Heads
            //      Contract: SeqLenKV
            attended_ = BatchContract<AxisList<0>{}, AxisList<1>{}, AxisList<2>{}, AxisList<0>{}>(attn_weights_, V_);

            // attended -> QProj_Type
            // Tensor<Heads, SeqLenQ, HeadDim> -> Tensor<SeqLenQ, Heads, HeadDim>
            return Permute<1, 0, 2>(attended_) >> lc_O_;
        }

        // @doc: InputTensor MultiHeadCrossAttentionBlock::Backward(const OutputTensor &delta_A, const OutputTensor &/*a*/, const InputTensor &/*a_prev*/)
        InputTensor Backward(const OutputTensor &delta_A, const OutputTensor &/*a*/,
                             const InputTensor &/*a_prev*/) {
            // backward through learned O contraction
            // de-permute into Attended_Type from QProj_Type
            const auto d_attended = Permute<1, 0, 2>(lc_O_ << delta_A);

            // attended -> V
            // attended -> scores

            // scores -> logitScores

            // scores -> Q
            // scores > K

            // Q -> inputQ
            // K, V -(+)> inputKV

            // return Pack(inputQ, inputKV)


            // TODO: mirror MultiHeadAttentionBlock::Backward, but Q and KV come from different sources:
            //       - dQ_side  flows through lc_Q_ << ... -> QSideTensor
            //       - dKV_side accumulates lc_K_ << ... and lc_V_ << ... -> KVSideTensor
            //       - return Pack(dQ_side, dKV_side)
            return InputTensor{};
        }

        // @doc: template<size_t Batch> Tensor<Batch, SeqLenQ, EmbDim> MultiHeadCrossAttentionBlock::BatchedForward(const Tensor<Batch, SeqLenQ + SeqLenKV, EmbDim> &/*X*/) const
        template<size_t Batch>
        Tensor<Batch, SeqLenQ, EmbDim> BatchedForward(const Tensor<Batch, SeqLenQ + SeqLenKV, EmbDim> &/*X*/) const {
            // TODO: batched analogue of Forward; split along axis 1 into [B, SeqLenQ, EmbDim] and [B, SeqLenKV, EmbDim]
            return Tensor<Batch, SeqLenQ, EmbDim>{};
        }

        // @doc: template<size_t Batch> Tensor<Batch, SeqLenQ + SeqLenKV, EmbDim> MultiHeadCrossAttentionBlock::BatchedBackward(const Tensor<Batch, SeqLenQ, EmbDim> &/*delta_A*/, const Tensor<Batch, SeqLenQ, EmbDim> &/*a*/, const Tensor<Batch, SeqLenQ + SeqLenKV, EmbDim> &/*a_prev*/)
        template<size_t Batch>
        Tensor<Batch, SeqLenQ + SeqLenKV, EmbDim> BatchedBackward(const Tensor<Batch, SeqLenQ, EmbDim> &/*delta_A*/,
                                                                  const Tensor<Batch, SeqLenQ, EmbDim> &/*a*/,
                                                                  const Tensor<Batch, SeqLenQ + SeqLenKV, EmbDim> &
                                                                  /*a_prev*/) {
            // TODO: batched analogue of Backward; stitch dQ and dKV halves back into a [B, SeqLenQ+SeqLenKV, EmbDim] tensor
            return Tensor<Batch, SeqLenQ + SeqLenKV, EmbDim>{};
        }
    };


    // @doc: template<size_t Heads, size_t... EmbDims> struct MHAttention
    /**
     * `BlockRecipe` for unmasked `MultiHeadAttentionBlock`
     * Takes in `size_t Heads` for head count and `size_t...EmbDims` indicating the dimensionality of the embeddings
     * `InputT` passed to `Resolve` should have its first axis be `SeqLen`
     * `Resolve = MultiHeadAttentionBlock<TensorFirstDim<InputT>::value, Heads, false, EmbDims...>`
     */
    template<size_t Heads, size_t... EmbDims>
    struct MHAttention {
        using OutputTensor = Tensor<1, EmbDims...>;

        template<typename InputT> requires IsTensor<InputT>
        using Resolve = MultiHeadAttentionBlock<TensorFirstDim<InputT>::value, Heads, false, EmbDims...>;
    };

    // @doc: template<size_t Heads, size_t... EmbDims> struct MHCausalAttention
    /**
     * `BlockRecipe` for causal (masked) `MultiHeadAttentionBlock`
     * Identical to `MHAttention` but passes `Masked = true` — positions `k > q` are masked to `-inf` before softmax
     * `Resolve = MultiHeadAttentionBlock<TensorFirstDim<InputT>::value, Heads, true, EmbDims...>`
     */
    template<size_t Heads, size_t... EmbDims>
    struct MHCausalAttention {
        using OutputTensor = Tensor<1, EmbDims...>;

        template<typename InputT> requires IsTensor<InputT>
        using Resolve = MultiHeadAttentionBlock<TensorFirstDim<InputT>::value, Heads, true, EmbDims...>;
    };
} // namespace TTTN
