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

                // Single-sample forward. The primitives are batched-only now, so this
                // is a batch-of-1 wrapper rather than a second copy of the logic.
                DecHidden Forward(const DecHidden &x, const EncHidden &enc_out) const {
                    Tensor<1, TgtLen, EmbDim> bx{};
                    Tensor<1, SrcLen, EmbDim> benc{};
                    TensorSet<0>(bx, 0, x);
                    TensorSet<0>(benc, 0, enc_out);
                    return TensorGet<0>(Forward<1>(bx, benc), 0);
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

                // Cache holds each sub-block's cache plus the activations their
                // Backward needs as `a` / `a_prev`.
                template<size_t Batch>
                struct TrainingCacheData {
                    typename ResidualBlock<SelfAttn>::template TrainingCache<Batch> rmmha{};
                    typename CrossAttn::template TrainingCache<Batch>               mhca{};
                    typename ResidualBlock<FFN>::template TrainingCache<Batch>      rffn{};
                    Tensor<Batch, TgtLen, EmbDim>          x{};        // layer input
                    Tensor<Batch, TgtLen, EmbDim>          selfed{};   // after self-attn residual
                    Tensor<Batch, TgtLen + SrcLen, EmbDim> packed{};   // cross-attn input
                    Tensor<Batch, TgtLen, EmbDim>          mhca_out{}; // cross-attn output, pre-residual
                    Tensor<Batch, TgtLen, EmbDim>          crossed{};  // after cross-attn residual
                    Tensor<Batch, TgtLen, EmbDim>          out{};      // layer output
                };
                template<size_t Batch> using TrainingCache = TrainingCacheData<Batch>;

                // ---- batched forward: pure inference, no cache ----
                template<size_t Batch>
                Tensor<Batch, TgtLen, EmbDim> Forward(
                    const Tensor<Batch, TgtLen, EmbDim> &x,
                    const Tensor<Batch, SrcLen, EmbDim> &enc_out) const {
                    auto selfed = rmmha_.template Forward<Batch>(x);
                    auto crossed = mhca_.template Forward<Batch>(ConcatAxis<1>(selfed, enc_out)) + selfed;
                    return rffn_.template Forward<Batch>(crossed);
                }

                // ---- batched forward: training, populates cache ----
                template<size_t Batch>
                Tensor<Batch, TgtLen, EmbDim> Forward(
                    const Tensor<Batch, TgtLen, EmbDim> &x,
                    const Tensor<Batch, SrcLen, EmbDim> &enc_out,
                    TrainingCache<Batch> &cache) const {
                    cache.x = x;
                    cache.selfed = rmmha_.template Forward<Batch>(x, cache.rmmha);
                    cache.packed = ConcatAxis<1>(cache.selfed, enc_out);
                    cache.mhca_out = mhca_.template Forward<Batch>(cache.packed, cache.mhca);
                    cache.crossed = cache.mhca_out + cache.selfed;   // cross-attn residual
                    cache.out = rffn_.template Forward<Batch>(cache.crossed, cache.rffn);
                    return cache.out;
                }

                // returns (d_x, d_enc_out_layer)
                template<size_t Batch>
                std::pair<Tensor<Batch, TgtLen, EmbDim>, Tensor<Batch, SrcLen, EmbDim> >
                Backward(const Tensor<Batch, TgtLen, EmbDim> &d_out,
                         const TrainingCache<Batch> &cache) {
                    auto d_crossed = rffn_.template Backward<Batch>(
                        d_out, cache.out, cache.crossed, cache.rffn);
                    auto d_packed = mhca_.template Backward<Batch>(
                        d_crossed, cache.mhca_out, cache.packed, cache.mhca);
                    auto [d_selfed, d_enc] = SplitAxis<1, TgtLen>(d_packed);
                    d_selfed += d_crossed;                            // cross-attn residual
                    auto d_x = rmmha_.template Backward<Batch>(
                        d_selfed, cache.selfed, cache.x, cache.rmmha);
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
                Tensor<1, TgtLen, EmbDim> bx{};
                Tensor<1, SrcLen, EmbDim> benc{};
                TensorSet<0>(bx, 0, dec_in);
                TensorSet<0>(benc, 0, enc_out);
                return TensorGet<0>(Forward<1>(bx, benc), 0);
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

            template<size_t Batch>
            struct TrainingCacheData {
                std::array<typename Layer::template TrainingCache<Batch>, NDec> layers{};
            };
            template<size_t Batch> using TrainingCache = TrainingCacheData<Batch>;

            // ---- batched forward: same fold as single-sample, threading batched enc_out ----
            template<size_t Batch>
            Tensor<Batch, TgtLen, EmbDim> Forward(
                const Tensor<Batch, TgtLen, EmbDim> &dec_in,
                const Tensor<Batch, SrcLen, EmbDim> &enc_out) const {
                return [&]<size_t... Is>(std::index_sequence<Is...>) {
                    Tensor<Batch, TgtLen, EmbDim> x = dec_in;
                    ((x = layers_[Is].template Forward<Batch>(x, enc_out)), ...);
                    return x;
                }(std::make_index_sequence<NDec>{});
            }

            // ---- batched forward: training, populates per-layer caches ----
            template<size_t Batch>
            Tensor<Batch, TgtLen, EmbDim> Forward(
                const Tensor<Batch, TgtLen, EmbDim> &dec_in,
                const Tensor<Batch, SrcLen, EmbDim> &enc_out,
                TrainingCache<Batch> &cache) const {
                return [&]<size_t... Is>(std::index_sequence<Is...>) {
                    Tensor<Batch, TgtLen, EmbDim> x = dec_in;
                    ((x = layers_[Is].template Forward<Batch>(x, enc_out, cache.layers[Is])), ...);
                    return x;
                }(std::make_index_sequence<NDec>{});
            }

            // ---- batched backward: reverse accumulation of d_enc_out ----
            template<size_t Batch>
            std::pair<Tensor<Batch, TgtLen, EmbDim>, Tensor<Batch, SrcLen, EmbDim> >
            Backward(const Tensor<Batch, TgtLen, EmbDim> &d_out,
                     const TrainingCache<Batch> &cache) {
                Tensor<Batch, TgtLen, EmbDim> d_x = d_out;
                Tensor<Batch, SrcLen, EmbDim> d_enc_accum{};
                for (int i = static_cast<int>(NDec) - 1; i >= 0; --i) {
                    auto [d_layer_in, d_enc_layer] =
                        layers_[i].template Backward<Batch>(d_x, cache.layers[i]);
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
        mutable std::vector<float> b_src_oh_{}; // Tensor<Batch, SrcLen, VocabSize>
        mutable std::vector<float> b_tgt_oh_{}; // Tensor<Batch, TgtLen, VocabSize>
        mutable std::vector<float> b_src_emb_{}; // Tensor<Batch, SrcLen, EmbDim> with PE (enc a_prev)
        mutable std::vector<float> b_dec_out_{}; // Tensor<Batch, TgtLen, EmbDim> (pre-projection)

        


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
            const Tensor<Batch, SeqLen, EmbDim> &h) {
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
            AddPositionalEncoding(pe); // adds to zeros → pe holds raw PE values
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

        // ---- training cache: encoder + decoder caches, plus the activations
        //      Backward needs (the one-hots, the encoder input, the decoder output) ----
        template<size_t Batch>
        struct TrainingCacheData {
            typename EncoderType::template TrainingCache<Batch>      enc{};
            typename CrossDecoderStack::template TrainingCache<Batch> dec{};
            Tensor<Batch, SrcLen, VocabSize> src_oh{};
            Tensor<Batch, TgtLen, VocabSize> tgt_oh{};
            Tensor<Batch, SrcLen, EmbDim>    src_emb{};
            Tensor<Batch, TgtLen, EmbDim>    dec_out{};
        };
        template<size_t Batch> using TrainingCache = TrainingCacheData<Batch>;

        // ---- batched forward: pure inference, no cache ----
        template<size_t Batch>
        Tensor<Batch, TgtLen, VocabSize> Forward(
            const Tensor<Batch, SrcLen + TgtLen, VocabSize> &X) const {
            auto [b_src_oh, b_tgt_oh] = SplitAxis<1, SrcLen>(X);
            auto b_src_emb = BatchEmbed<Batch, SrcLen>(b_src_oh, embed_.value);
            BatchAddPE<Batch, SrcLen>(b_src_emb);
            auto b_enc_out = enc_.template Forward<Batch>(b_src_emb);
            auto b_tgt_emb = BatchEmbed<Batch, TgtLen>(b_tgt_oh, embed_.value);
            BatchAddPE<Batch, TgtLen>(b_tgt_emb);
            auto b_dec_out = dec_.template Forward<Batch>(b_tgt_emb, b_enc_out);
            return BatchProject<Batch, TgtLen>(b_dec_out, embed_.value);
        }

        // ---- batched forward: training, populates cache ----
        template<size_t Batch>
        Tensor<Batch, TgtLen, VocabSize> Forward(
            const Tensor<Batch, SrcLen + TgtLen, VocabSize> &X,
            TrainingCache<Batch> &cache) const {
            std::tie(cache.src_oh, cache.tgt_oh) = SplitAxis<1, SrcLen>(X);
            cache.src_emb = BatchEmbed<Batch, SrcLen>(cache.src_oh, embed_.value);
            BatchAddPE<Batch, SrcLen>(cache.src_emb);
            auto b_enc_out = enc_.template Forward<Batch>(cache.src_emb, cache.enc);
            auto b_tgt_emb = BatchEmbed<Batch, TgtLen>(cache.tgt_oh, embed_.value);
            BatchAddPE<Batch, TgtLen>(b_tgt_emb);
            cache.dec_out = dec_.template Forward<Batch>(b_tgt_emb, b_enc_out, cache.dec);
            return BatchProject<Batch, TgtLen>(cache.dec_out, embed_.value);
        }

        // ---- batched backward ----
        template<size_t Batch>
        Tensor<Batch, SrcLen + TgtLen, VocabSize> Backward(
            const Tensor<Batch, TgtLen, VocabSize> &delta_A,
            const Tensor<Batch, TgtLen, VocabSize> & /*a*/,
            const Tensor<Batch, SrcLen + TgtLen, VocabSize> & /*a_prev*/,
            const TrainingCache<Batch> &cache) {
            // 1. grad through projection (flatten -> Contract -> reshape)
            //    d_dec_out[b,t,e] = sum_v delta[b,t,v]*E[v,e]
            auto d_dec_out = Reshape<Batch, TgtLen, EmbDim>(
                Contract<AxisList<1>{}, AxisList<0>{}, Mul, Add>(
                    Reshape<Batch * TgtLen, VocabSize>(delta_A), embed_.value));
            //    dE[v,e] += sum_b sum_t delta[b,t,v]*dec_out[b,t,e]
            embed_.grad += BatchEmbedGrad<Batch, TgtLen>(delta_A, cache.dec_out);

            // 2. grad through decoder
            auto [d_tgt_emb, d_enc_out] = dec_.template Backward<Batch>(d_dec_out, cache.dec);

            // 3. grad through encoder -- the cache carries its activations, so
            //    unlike the old path this no longer re-runs the forward pass
            auto d_src_emb = enc_.template Backward<Batch>(
                d_enc_out, {}, cache.src_emb, cache.enc);

            // 4. embed grad from tgt: dE[v,e] += sum_b sum_t tgt_oh[b,t,v]*d_tgt_emb[b,t,e]
            embed_.grad += BatchEmbedGrad<Batch, TgtLen>(cache.tgt_oh, d_tgt_emb);
            // 5. embed grad from src
            embed_.grad += BatchEmbedGrad<Batch, SrcLen>(cache.src_oh, d_src_emb);

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

        // ---- J-lens support: interior activations and interior gradients ----

        struct InteriorActivations {
            EncHidden enc_out{};                        // encoder output (cross-attn memory)
            std::array<DecHidden, NDec> dec_layer_in{}; // input of decoder layer i (0 = tgt_emb + PE)
            DecHidden dec_out{};                        // decoder stack output (pre-projection)
            OutputTensor logits{};
        };

        InteriorActivations ForwardInterior(const InputTensor &x) const {
            InteriorActivations acts;
            std::tie(src_oh_, tgt_oh_) = SplitAxis<0, SrcLen>(x);
            src_emb_ = Embed(src_oh_, embed_.value);
            AddPositionalEncoding(src_emb_);
            enc_out_ = enc_.Forward(src_emb_);
            acts.enc_out = enc_out_;
            tgt_emb_ = Embed(tgt_oh_, embed_.value);
            AddPositionalEncoding(tgt_emb_);
            DecHidden h = tgt_emb_;
            [&]<size_t... Is>(std::index_sequence<Is...>) {
                (([&] {
                    acts.dec_layer_in[Is] = h;
                    h = dec_.layers_[Is].Forward(h, enc_out_);
                }()), ...);
            }(std::make_index_sequence<NDec>{});
            dec_out_ = h;
            acts.dec_out = dec_out_;
            acts.logits = Contract<AxisList<1>{}, AxisList<1>{}, Mul, Add>(dec_out_, embed_.value);
            return acts;
        }

        struct InteriorGradients {
            EncHidden d_enc_out{};                        // cotangent arriving at the encoder output
            std::array<DecHidden, NDec> d_dec_layer_in{}; // cotangent at decoder layer i's input
            DecHidden d_dec_out{};                        // cotangent at the decoder stack output
        };

        InteriorGradients BackwardInterior(const OutputTensor &delta) {
            InteriorGradients g;
            // through the weight-tied projection (linear part only; embed_.grad untouched)
            g.d_dec_out = Contract<AxisList<1>{}, AxisList<0>{}, Mul, Add>(delta, embed_.value);
            DecHidden d_x = g.d_dec_out;
            EncHidden d_enc_accum{};
            for (int i = static_cast<int>(NDec) - 1; i >= 0; --i) {
                auto [d_in, d_enc] = dec_.layers_[static_cast<size_t>(i)].Backward(d_x);
                g.d_dec_layer_in[static_cast<size_t>(i)] = d_in;
                d_x = d_in;
                d_enc_accum += d_enc;
            }
            g.d_enc_out = d_enc_accum;
            return g;
        }

        void ZeroGradAll() {
            ZeroAllGrads(all_params());
        }

        // ---- inference helpers ----
        EncHidden EncodeOnly(const SrcOneHot &src) const {
            auto emb = Embed(src, embed_.value);
            AddPositionalEncoding(emb);
            Tensor<1, SrcLen, EmbDim> bemb{};
            TensorSet<0>(bemb, 0, emb);
            return TensorGet<0>(enc_.template Forward<1>(bemb), 0);
        }

        OutputTensor DecodeStep(const EncHidden &enc_out, const TgtOneHot &tgt_so_far) const {
            auto emb = Embed(tgt_so_far, embed_.value);
            AddPositionalEncoding(emb);
            auto dec_out = dec_.Forward(emb, enc_out);
            return Contract<AxisList<1>{}, AxisList<1>{}, Mul, Add>(dec_out, embed_.value);
        }
    };
} // namespace TTTN
