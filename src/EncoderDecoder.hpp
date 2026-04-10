#pragma once
#include <array>
#include <tuple>
#include <utility>
#include "BlockSequence.hpp"
#include "TransformerBlock.hpp"
#include "TTTN_ML.hpp"
#include "NetworkUtil.hpp"

namespace TTTN {
    template<size_t SrcLen, size_t TgtLen, size_t VocabSize,
        size_t EmbDim, size_t Heads, size_t FFNHidden,
        size_t NEnc, size_t NDec,
        size_t PadId>
    class EncoderDecoderBlock {
        static_assert(EmbDim % Heads == 0, "EmbDim must be divisible by Heads");
        static_assert(PadId < VocabSize, "PadId must be a valid vocab index");
        static_assert(NEnc >= 1 && NDec >= 1, "Need at least one encoder and one decoder layer");

    public:
        // [src one-hot, tgt shifted one-hot] packed on axis 0
        using InputTensor = Tensor<SrcLen + TgtLen, VocabSize>;
        // logits over vocab for each target position
        using OutputTensor = Tensor<TgtLen, VocabSize>;

        // internal tensor types
        using SrcOneHot = Tensor<SrcLen, VocabSize>;
        using TgtOneHot = Tensor<TgtLen, VocabSize>;
        using EncHidden = Tensor<SrcLen, EmbDim>;
        using DecHidden = Tensor<TgtLen, EmbDim>;
        using EmbedTable = Tensor<VocabSize, EmbDim>;

        // NEnc self-attn transformer blocks over encoder hidden states
        using EncoderType = RepeatedBlockSequence<
            TransformerBlock<EncHidden, Heads, FFNHidden, /*PreNorm=*/true, /*Masked=*/false>,
            NEnc
        >;

    private:
        // shared embedding table (src embed, tgt embed, and tied output projection)
        Param<EmbedTable> embed_{};

        // encoder stack
        EncoderType enc_{};

        struct CrossDecoderStack {
            // type aliases used by both Layer and CrossDecoderStack
            using SelfAttn = MultiHeadAttentionBlock<TgtLen, Heads, /*Masked=*/true, EmbDim>;
            using CrossAttn = MultiHeadCrossAttentionBlock<TgtLen, SrcLen, Heads, EmbDim>;
            using FFN = BlockSequence<
                MapDenseMDBlock<Tensor<TgtLen, EmbDim>, Tensor<FFNHidden>, 1, ReLU>,
                MapDenseMDBlock<Tensor<TgtLen, FFNHidden>, Tensor<EmbDim>, 1>,
                LayerNormBlock<TgtLen, EmbDim>
            >;

            struct Layer {
                // mmha and ffn have matching Input==Output so ResidualBlock wraps them cleanly;
                // mhca takes packed [TgtLen+SrcLen, EmbDim] → [TgtLen, EmbDim], so residual is manual
                ResidualBlock<SelfAttn> rmmha_;
                CrossAttn mhca_;
                ResidualBlock<FFN> rffn_;

                DecHidden Forward(const DecHidden &x, const EncHidden &enc_out) const {
                    // masked self-attn + residual (handled by ResidualBlock)
                    auto selfed = rmmha_.Forward(x);
                    // cross-attn: pack decoder state with encoder output, residual on Q-side
                    auto crossed = mhca_.Forward(ConcatAxis<0>(selfed, enc_out)) + selfed;
                    // FFN + residual (handled by ResidualBlock)
                    return rffn_.Forward(crossed);
                }

                // returns (d_x, d_enc_out_layer)
                std::pair<DecHidden, EncHidden> Backward(const DecHidden &d_out) {
                    // backward through FFN residual
                    auto d_crossed = rffn_.Backward(d_out, {}, {});
                    // backward through cross-attn; Backward returns [TgtLen+SrcLen, EmbDim]
                    auto d_packed = mhca_.Backward(d_crossed, {}, {});
                    auto [d_selfed, d_enc] = SplitAxis<0, TgtLen>(d_packed);
                    d_selfed += d_crossed; // cross-attn residual
                    // backward through self-attn residual
                    auto d_x = rmmha_.Backward(d_selfed, {}, {});
                    return {d_x, d_enc};
                }

                template<size_t Batch>
                Tensor<Batch, TgtLen, EmbDim> BatchedForward(
                    const Tensor<Batch, TgtLen, EmbDim> &x,
                    const Tensor<Batch, SrcLen, EmbDim> &enc_out) const {
                    auto selfed  = rmmha_.template BatchedForward<Batch>(x);
                    auto crossed = mhca_.template BatchedForward<Batch>(ConcatAxis<1>(selfed, enc_out)) + selfed;
                    return rffn_.template BatchedForward<Batch>(crossed);
                }

                // returns (d_x, d_enc_out_layer)
                template<size_t Batch>
                std::pair<Tensor<Batch, TgtLen, EmbDim>, Tensor<Batch, SrcLen, EmbDim>>
                BatchedBackward(const Tensor<Batch, TgtLen, EmbDim> &d_out) {
                    auto d_crossed = rffn_.template BatchedBackward<Batch>(d_out, {}, {});
                    auto d_packed  = mhca_.template BatchedBackward<Batch>(d_crossed, {}, {});
                    auto [d_selfed, d_enc] = SplitAxis<1, TgtLen>(d_packed);
                    d_selfed += d_crossed;
                    auto d_x = rmmha_.template BatchedBackward<Batch>(d_selfed, {}, {});
                    return {d_x, d_enc};
                }

                auto all_params() {
                    return std::tuple_cat(rmmha_.all_params(), mhca_.all_params(), rffn_.all_params());
                }

                auto all_params() const {
                    return std::tuple_cat(rmmha_.all_params(), mhca_.all_params(), rffn_.all_params());
                }

                void zero_grad() { ZeroAllGrads(all_params()); }

                void peek(SnapshotMap &out, const std::string &prefix) const {
                    rmmha_.peek(out, prefix + "self_attn.");
                    mhca_.peek(out, prefix + "cross_attn.");
                    rffn_.peek(out, prefix + "ffn.");
                }
            };

            std::array<Layer, NDec> layers_{};

            // ---- forward: walk all NDec layers, threading enc_out into every cross-attn ----
            DecHidden Forward(const DecHidden &dec_in, const EncHidden &enc_out) const {
                return [&]<size_t... Is>(std::index_sequence<Is...>) {
                    DecHidden x = dec_in;
                    ((x = layers_[Is].Forward(x, enc_out)), ...);
                    return x;
                }(std::make_index_sequence<NDec>{});
            }


            // ---- backward: walk layers in reverse, accumulating d_enc_out across all layers ----
            std::pair<DecHidden, EncHidden> Backward(const DecHidden &d_out) {
                DecHidden d_x = d_out;
                EncHidden d_enc_accum{};
                for (int i = static_cast<int>(NDec) - 1; i >= 0; --i) {
                    auto [d_layer_in, d_enc_layer] = layers_[i].Backward(d_x);
                    d_x = d_layer_in;
                    d_enc_accum += d_enc_layer;
                }
                return {d_x, d_enc_accum};
            }

            // ---- batched forward: same fold as single-sample, threading batched enc_out ----
            template<size_t Batch>
            Tensor<Batch, TgtLen, EmbDim> BatchedForward(
                const Tensor<Batch, TgtLen, EmbDim> &dec_in,
                const Tensor<Batch, SrcLen, EmbDim> &enc_out) const {
                return [&]<size_t... Is>(std::index_sequence<Is...>) {
                    Tensor<Batch, TgtLen, EmbDim> x = dec_in;
                    ((x = layers_[Is].template BatchedForward<Batch>(x, enc_out)), ...);
                    return x;
                }(std::make_index_sequence<NDec>{});
            }

            // ---- batched backward: reverse accumulation of d_enc_out ----
            template<size_t Batch>
            std::pair<Tensor<Batch, TgtLen, EmbDim>, Tensor<Batch, SrcLen, EmbDim>>
            BatchedBackward(const Tensor<Batch, TgtLen, EmbDim> &d_out) {
                Tensor<Batch, TgtLen, EmbDim> d_x = d_out;
                Tensor<Batch, SrcLen, EmbDim> d_enc_accum{};
                for (int i = static_cast<int>(NDec) - 1; i >= 0; --i) {
                    auto [d_layer_in, d_enc_layer] = layers_[i].template BatchedBackward<Batch>(d_x);
                    d_x = d_layer_in;
                    d_enc_accum += d_enc_layer;
                }
                return {d_x, d_enc_accum};
            }

            auto all_params() {
                return [this]<size_t... Is>(std::index_sequence<Is...>) {
                    return std::tuple_cat(layers_[Is].all_params()...);
                }(std::make_index_sequence<NDec>{});
            }

            auto all_params() const {
                return [this]<size_t... Is>(std::index_sequence<Is...>) {
                    return std::tuple_cat(layers_[Is].all_params()...);
                }(std::make_index_sequence<NDec>{});
            }

            void zero_grad() { for (auto &L: layers_) L.zero_grad(); }

            void peek(SnapshotMap &out, const std::string &prefix) const {
                for (size_t i = 0; i < NDec; ++i)
                    layers_[i].peek(out, prefix + "layer_" + std::to_string(i) + ".");
            }
        };

        CrossDecoderStack dec_{};

        // single-sample caches (populated during Forward for Backward)
        mutable SrcOneHot src_oh_{};
        mutable TgtOneHot tgt_oh_{};
        mutable EncHidden src_emb_{};
        mutable EncHidden enc_out_{};
        mutable DecHidden tgt_emb_{};
        mutable DecHidden dec_out_{};

        // batched caches (std::vector<float> because Batch is a function template param)
        mutable std::vector<float> b_src_oh_{};   // Tensor<Batch, SrcLen, VocabSize>
        mutable std::vector<float> b_tgt_oh_{};   // Tensor<Batch, TgtLen, VocabSize>
        mutable std::vector<float> b_src_emb_{};  // Tensor<Batch, SrcLen, EmbDim> with PE (enc a_prev)
        mutable std::vector<float> b_dec_out_{};  // Tensor<Batch, TgtLen, EmbDim> (pre-projection)

        // embedding lookup: [SeqLen, VocabSize] x [VocabSize, EmbDim] -> [SeqLen, EmbDim]
        template<size_t SeqLen>
        static Tensor<SeqLen, EmbDim> Embed(const Tensor<SeqLen, VocabSize> &oh, const EmbedTable &E) {
            return Contract<AxisList<1>{}, AxisList<0>{}, Mul, Add>(oh, E);
        }

        // batched embedding lookup: [Batch, SeqLen, VocabSize] x [VocabSize, EmbDim] -> [Batch, SeqLen, EmbDim]
        // flatten to [Batch*SeqLen, Vocab], use single-sample Contract, reshape back
        template<size_t Batch, size_t SeqLen>
        static Tensor<Batch, SeqLen, EmbDim> BatchEmbed(
            const Tensor<Batch, SeqLen, VocabSize> &oh, const EmbedTable &E) {
            const auto flat = Contract<AxisList<1>{}, AxisList<0>{}, Mul, Add>(
                Reshape<Batch * SeqLen, VocabSize>(oh), E);
            return Reshape<Batch, SeqLen, EmbDim>(flat);
        }

        // batched weight-tied projection: [Batch, SeqLen, EmbDim] x [VocabSize, EmbDim] -> [Batch, SeqLen, VocabSize]
        template<size_t Batch, size_t SeqLen>
        static Tensor<Batch, SeqLen, VocabSize> BatchProject(
            const Tensor<Batch, SeqLen, EmbDim> &h, const EmbedTable &E) {
            const auto flat = Contract<AxisList<1>{}, AxisList<1>{}, Mul, Add>(
                Reshape<Batch * SeqLen, EmbDim>(h), E);
            return Reshape<Batch, SeqLen, VocabSize>(flat);
        }

        // batched embed grad: dE[v,e] += Σ_b Σ_t A[b,t,v] * B[b,t,e]
        // flatten both to [Batch*SeqLen, *], then Contract<0,0> -> [V, EmbDim] or [EmbDim, V]
        template<size_t Batch, size_t SeqLen>
        static EmbedTable BatchEmbedGrad(
            const Tensor<Batch, SeqLen, VocabSize> &oh,
            const Tensor<Batch, SeqLen, EmbDim>   &h) {
            return Contract<AxisList<0>{}, AxisList<0>{}, Mul, Add>(
                Reshape<Batch * SeqLen, VocabSize>(oh),
                Reshape<Batch * SeqLen, EmbDim>(h));
        }

        // add sinusoidal PE to each batch element independently
        // AddPositionalEncoding<SeqAxis=1> on [Batch,SeqLen,EmbDim] computes wrong EmbSize;
        // this helper applies the correct single-sample PE to every batch slice
        template<size_t Batch, size_t SeqLen>
        static void BatchAddPE(Tensor<Batch, SeqLen, EmbDim> &X) {
            Tensor<SeqLen, EmbDim> pe{};
            AddPositionalEncoding(pe);   // adds to zeros → pe holds raw PE values
            constexpr size_t SliceSize = SeqLen * EmbDim;
            for (size_t b = 0; b < Batch; ++b)
                for (size_t i = 0; i < SliceSize; ++i)
                    X.flat(b * SliceSize + i) += pe.flat(i);
        }

    public:
        EncoderDecoderBlock() {
            XavierInitMD(embed_.value, VocabSize, EmbDim);
        }

        // ===========================================================================================
        // ============================== Block concept surface ======================================
        // ===========================================================================================

        // ---- single-sample forward ----
        OutputTensor Forward(const InputTensor &x) const {
            // 1. split packed one-hot input
            std::tie(src_oh_, tgt_oh_) = SplitAxis<0, SrcLen>(x);
            // 2. embed src + positional encoding
            src_emb_ = Embed(src_oh_, embed_.value);
            AddPositionalEncoding(src_emb_);
            // 3. encode
            enc_out_ = enc_.Forward(src_emb_);
            // 4. embed tgt (shifted) + positional encoding
            tgt_emb_ = Embed(tgt_oh_, embed_.value);
            AddPositionalEncoding(tgt_emb_);
            // 5. decode (cross-attn stack threads enc_out_ through every layer)
            dec_out_ = dec_.Forward(tgt_emb_, enc_out_);
            // 6. weight-tied output projection: contract EmbDim, free axes -> [TgtLen, VocabSize]
            return Contract<AxisList<1>{}, AxisList<1>{}, Mul, Add>(dec_out_, embed_.value);
        }

        // ---- single-sample backward ----
        InputTensor Backward(const OutputTensor &delta_A,
                             const OutputTensor & /*a*/,
                             const InputTensor & /*a_prev*/) {
            // 1. grad through weight-tied projection
            //    logits[t,v] = Σ_e dec_out[t,e]*E[v,e]
            //    d_dec_out[t,e] = Σ_v delta[t,v]*E[v,e]
            auto d_dec_out = Contract<AxisList<1>{}, AxisList<0>{}, Mul, Add>(delta_A, embed_.value);
            //    dE[v,e] += Σ_t delta[t,v]*dec_out[t,e]
            embed_.grad += Contract<AxisList<0>{}, AxisList<0>{}, Mul, Add>(delta_A, dec_out_);

            // 2. grad through decoder (returns d_tgt_emb, accumulated d_enc_out)
            auto [d_tgt_emb, d_enc_out] = dec_.Backward(d_dec_out);

            // 3. grad through encoder (mActs populated during Forward; a/a_prev unused)
            auto d_src_emb = enc_.Backward(d_enc_out, {}, {});

            // 4. accumulate embed grad from tgt embedding
            //    tgt_emb[t,e] = Σ_v tgt_oh[t,v]*E[v,e]  ->  dE[v,e] += Σ_t tgt_oh[t,v]*d_tgt_emb[t,e]
            embed_.grad += Contract<AxisList<0>{}, AxisList<0>{}, Mul, Add>(tgt_oh_, d_tgt_emb);
            // 5. accumulate embed grad from src embedding
            embed_.grad += Contract<AxisList<0>{}, AxisList<0>{}, Mul, Add>(src_oh_, d_src_emb);

            // gradient w.r.t. one-hot token ids has no meaning; return zero
            return InputTensor{};
        }

        // ---- batched forward ----
        template<size_t Batch>
        Tensor<Batch, TgtLen, VocabSize> BatchedForward(
            const Tensor<Batch, SrcLen + TgtLen, VocabSize> &X) const {
            using BSrcOH  = Tensor<Batch, SrcLen, VocabSize>;
            using BTgtOH  = Tensor<Batch, TgtLen, VocabSize>;
            using BSrcEmb = Tensor<Batch, SrcLen, EmbDim>;
            // 1. split packed input
            auto [b_src_oh, b_tgt_oh] = SplitAxis<1, SrcLen>(X);
            batch_cache_store(b_src_oh, b_src_oh_);
            batch_cache_store(b_tgt_oh, b_tgt_oh_);
            // 2. embed src + PE
            auto b_src_emb = BatchEmbed<Batch, SrcLen>(b_src_oh, embed_.value);
            BatchAddPE<Batch, SrcLen>(b_src_emb);
            batch_cache_store(b_src_emb, b_src_emb_);  // needed for enc backward a_prev
            // 3. encode
            auto b_enc_out = enc_.template BatchedForward<Batch>(b_src_emb);
            // 4. embed tgt + PE
            auto b_tgt_emb = BatchEmbed<Batch, TgtLen>(b_tgt_oh, embed_.value);
            BatchAddPE<Batch, TgtLen>(b_tgt_emb);
            // 5. decode
            auto b_dec_out = dec_.template BatchedForward<Batch>(b_tgt_emb, b_enc_out);
            batch_cache_store(b_dec_out, b_dec_out_);
            // 6. weight-tied projection via flatten: [Batch,TgtLen,EmbDim] → [Batch,TgtLen,VocabSize]
            return BatchProject<Batch, TgtLen>(b_dec_out, embed_.value);
        }

        // ---- batched backward ----
        template<size_t Batch>
        Tensor<Batch, SrcLen + TgtLen, VocabSize> BatchedBackward(
            const Tensor<Batch, TgtLen, VocabSize> &delta_A,
            const Tensor<Batch, TgtLen, VocabSize> & /*a*/,
            const Tensor<Batch, SrcLen + TgtLen, VocabSize> & /*a_prev*/) {
            using BSrcOH  = Tensor<Batch, SrcLen, VocabSize>;
            using BTgtOH  = Tensor<Batch, TgtLen, VocabSize>;
            using BSrcEmb = Tensor<Batch, SrcLen, EmbDim>;
            using BDecOut = Tensor<Batch, TgtLen, EmbDim>;
            const auto b_src_oh  = batch_cache_load<BSrcOH>(b_src_oh_);
            const auto b_tgt_oh  = batch_cache_load<BTgtOH>(b_tgt_oh_);
            const auto b_src_emb = batch_cache_load<BSrcEmb>(b_src_emb_);
            const auto b_dec_out = batch_cache_load<BDecOut>(b_dec_out_);

            // 1. grad through projection (flatten → Contract → reshape)
            //    d_dec_out[b,t,e] = Σ_v delta[b,t,v]*E[v,e]
            auto d_dec_out = Reshape<Batch, TgtLen, EmbDim>(
                Contract<AxisList<1>{}, AxisList<0>{}, Mul, Add>(
                    Reshape<Batch * TgtLen, VocabSize>(delta_A), embed_.value));
            //    dE[v,e] += Σ_b Σ_t delta[b,t,v]*dec_out[b,t,e]
            embed_.grad += BatchEmbedGrad<Batch, TgtLen>(delta_A, b_dec_out);

            // 2. grad through decoder (internal caches from BatchedForward)
            auto [d_tgt_emb, d_enc_out] = dec_.template BatchedBackward<Batch>(d_dec_out);

            // 3. grad through encoder (re-runs BatchedForward from b_src_emb to repopulate caches)
            auto d_src_emb = enc_.template BatchedBackward<Batch>(d_enc_out, {}, b_src_emb);

            // 4. accumulate embed grad from tgt: dE[v,e] += Σ_b Σ_t tgt_oh[b,t,v]*d_tgt_emb[b,t,e]
            embed_.grad += BatchEmbedGrad<Batch, TgtLen>(b_tgt_oh, d_tgt_emb);
            // 5. accumulate embed grad from src
            embed_.grad += BatchEmbedGrad<Batch, SrcLen>(b_src_oh, d_src_emb);

            return Tensor<Batch, SrcLen + TgtLen, VocabSize>{};
        }

        // ---- aggregate params: shared embed + entire encoder + entire decoder ----
        auto all_params() {
            return std::tuple_cat(std::tie(embed_), enc_.all_params(), dec_.all_params());
        }

        auto all_params() const {
            return std::tuple_cat(std::tie(embed_), enc_.all_params(), dec_.all_params());
        }

        // ---- snapshot transparency ----
        void peek(SnapshotMap &out, const std::string &prefix) const {
            enc_.peek(out, prefix + "encoder.");
            dec_.peek(out, prefix + "decoder.");
        }

        // ---- inference helpers ----
        EncHidden EncodeOnly(const SrcOneHot &src) const {
            auto emb = Embed(src, embed_.value);
            AddPositionalEncoding(emb);
            return enc_.Forward(emb);
        }

        OutputTensor DecodeStep(const EncHidden &enc_out, const TgtOneHot &tgt_so_far) const {
            auto emb = Embed(tgt_so_far, embed_.value);
            AddPositionalEncoding(emb);
            auto dec_out = dec_.Forward(emb, enc_out);
            return Contract<AxisList<1>{}, AxisList<1>{}, Mul, Add>(dec_out, embed_.value);
        }
    };
} // namespace TTTN
