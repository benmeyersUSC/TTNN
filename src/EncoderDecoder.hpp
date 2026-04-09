#pragma once
#include <array>
#include <tuple>
#include <utility>
#include "BlockSequence.hpp"
#include "TransformerBlock.hpp"
#include "TTTN_ML.hpp"
#include "NetworkUtil.hpp"

namespace TTTN {

    // ---- helper: build BlockSequence<B, B, ..., B> with N copies of B ----
    namespace detail {
        template<template<typename...> class Seq, typename B, typename Idx> struct RepeatSeqImpl;
        template<template<typename...> class Seq, typename B, size_t... Is>
        struct RepeatSeqImpl<Seq, B, std::index_sequence<Is...>> {
            template<size_t> using Same = B;
            using type = Seq<Same<Is>...>;
        };
    }
    template<typename B, size_t N>
    using RepeatedBlockSequence =
        typename detail::RepeatSeqImpl<BlockSequence, B, std::make_index_sequence<N>>::type;


    template<size_t SrcLen, size_t TgtLen, size_t Vocab,
             size_t EmbDim, size_t Heads, size_t FFNHidden,
             size_t NEnc,   size_t NDec,
             size_t PadId>
    class EncoderDecoderBlock {
        static_assert(EmbDim % Heads == 0, "EmbDim must be divisible by Heads");
        static_assert(PadId < Vocab,       "PadId must be a valid vocab index");
        static_assert(NEnc >= 1 && NDec >= 1, "Need at least one encoder and one decoder layer");

    public:
        // ---- Block concept surface ----
        using InputTensor  = Tensor<SrcLen + TgtLen, Vocab>;   // [src one-hot ; tgt_shifted one-hot]
        using OutputTensor = Tensor<TgtLen, Vocab>;            // logits

        // ---- internal hidden tensor types ----
        using SrcOneHot   = Tensor<SrcLen, Vocab>;
        using TgtOneHot   = Tensor<TgtLen, Vocab>;
        using EncHidden   = Tensor<SrcLen, EmbDim>;
        using DecHidden   = Tensor<TgtLen, EmbDim>;
        using EmbedTable  = Tensor<Vocab, EmbDim>;

        // ---- THE SEVERANCE POINT (publicly exposed) ----
        // Standalone encoder type — a real reusable Block. Anyone can instantiate it
        // directly in an encoder-only network. Same Heads/FFNHidden/PreNorm choices
        // as the encoder used inside this enc-dec block.
        using EncoderType = RepeatedBlockSequence<
            TransformerBlock<EncHidden, Heads, FFNHidden, /*PreNorm=*/true, /*Masked=*/false>,
            NEnc
        >;

        // ---- compile-time constants ----
        static constexpr size_t kSrcLen    = SrcLen;
        static constexpr size_t kTgtLen    = TgtLen;
        static constexpr size_t kVocab     = Vocab;
        static constexpr size_t kEmbDim    = EmbDim;
        static constexpr size_t kHeads     = Heads;
        static constexpr size_t kFFNHidden = FFNHidden;
        static constexpr size_t kNEnc      = NEnc;
        static constexpr size_t kNDec      = NDec;
        static constexpr size_t kPadId     = PadId;

    private:
        // ---- shared embedding (top-level owned; used by src embed, tgt embed, AND tied output projection) ----
        Param<EmbedTable> embed_{};

        // ---- encoder: standalone, reusable Block type ----
        EncoderType enc_{};

        // ---- decoder: PRIVATE nested helper, NOT a public Block.
        //      Holds NDec layers' worth of self-attn, cross-attn, FFN, and LayerNorm Params.
        //      Forward signature is intentionally non-Block: takes (dec_in, enc_out), returns DecHidden.
        //      Backward returns std::pair<DecHidden, EncHidden> so the d_enc_out contribution can be
        //      accumulated by the wrapping EncoderDecoderBlock.
        struct CrossDecoderStack {
            // ---- per-layer parameter bundle (TODO step 2: declare actual Param members) ----
            struct Layer {
                // self-attention (masked): Q, K, V, O projections + LayerNorm
                // cross-attention:         Q, K, V, O projections + LayerNorm (Q from dec, K/V from enc_out)
                // FFN:                     W1, W2 + LayerNorm
                // ALL as Param<...> members so all_params() can collect them via tuple_cat.

                auto all_params()       { return std::tuple<>{}; /* TODO: tuple_cat all members */ }
                auto all_params() const { return std::tuple<>{}; /* TODO: tuple_cat all members */ }

                void zero_grad() { /* TODO: zero_grad each Param member */ }
                void peek(SnapshotMap& /*out*/, const std::string& /*prefix*/) const { /* TODO */ }
            };

            std::array<Layer, NDec> layers_{};

            // ---- forward: walk all NDec layers, threading enc_out into every cross-attn ----
            DecHidden Forward(const DecHidden& dec_in, const EncHidden& /*enc_out*/) const {
                // TODO step 3: loop layers_, each does masked self-attn → cross-attn(enc_out) → FFN
                return dec_in;  // pass-through for skeleton
            }

            // ---- backward: returns (d_dec_in, accumulated d_enc_out across all layers) ----
            std::pair<DecHidden, EncHidden> Backward(
                const DecHidden& delta_out,
                const DecHidden& /*a_out*/,
                const DecHidden& /*a_in*/,
                const EncHidden& /*enc_out*/) {
                // TODO step 4: walk layers_ in reverse, accumulating d_enc_out from each layer's
                //              cross-attn backward into a single EncHidden, returning final dDec
                return {delta_out, EncHidden{}};
            }

            // ---- batched variants ----
            template<size_t Batch>
            Tensor<Batch, TgtLen, EmbDim> BatchedForward(
                const Tensor<Batch, TgtLen, EmbDim>& dec_in,
                const Tensor<Batch, SrcLen, EmbDim>& /*enc_out*/) const {
                return dec_in;
            }

            template<size_t Batch>
            std::pair<Tensor<Batch, TgtLen, EmbDim>, Tensor<Batch, SrcLen, EmbDim>>
            BatchedBackward(
                const Tensor<Batch, TgtLen, EmbDim>& delta_out,
                const Tensor<Batch, TgtLen, EmbDim>& /*a_out*/,
                const Tensor<Batch, TgtLen, EmbDim>& /*a_in*/,
                const Tensor<Batch, SrcLen, EmbDim>& /*enc_out*/) {
                return {delta_out, Tensor<Batch, SrcLen, EmbDim>{}};
            }

            auto all_params() {
                // TODO: std::tuple_cat over layers_[i].all_params() for all i
                return std::tuple<>{};
            }
            auto all_params() const {
                return std::tuple<>{};
            }

            void zero_grad() { for (auto& L : layers_) L.zero_grad(); }

            void peek(SnapshotMap& out, const std::string& prefix) const {
                for (size_t i = 0; i < NDec; ++i) {
                    layers_[i].peek(out, prefix + "layer_" + std::to_string(i) + ".");
                }
            }
        };

        CrossDecoderStack dec_{};

    public:
        EncoderDecoderBlock() {
            // TODO step 2: XavierInitMD(embed_.value, Vocab, EmbDim);
        }

        // ---- ergonomic helper: pack (src, tgt_shifted) one-hots into the Block-conformant InputTensor ----
        static InputTensor Pack(const SrcOneHot& /*src*/, const TgtOneHot& /*tgt_shifted*/) {
            InputTensor out{};
            // TODO step 3: copy src into rows [0, SrcLen), tgt_shifted into rows [SrcLen, SrcLen+TgtLen)
            return out;
        }

        // ---- inverse helper, useful for tests / introspection ----
        static std::pair<SrcOneHot, TgtOneHot> Unpack(const InputTensor& /*packed*/) {
            return {SrcOneHot{}, TgtOneHot{}};  // TODO step 3
        }

        // ===========================================================================================
        // ============================== Block concept surface ======================================
        // ===========================================================================================

        // ---- single-sample forward ----
        OutputTensor Forward(const InputTensor& /*x*/) const {
            // TODO step 3:
            //   1. (src, tgt_shifted) = Unpack(x)
            //   2. src_emb = embed(src) + sinusoidal PE     (lookup via embed_.value)
            //   3. enc_out = src_emb >> enc_                (uses standalone EncoderType)
            //   4. tgt_emb = embed(tgt_shifted) + sinusoidal PE
            //   5. dec_out = dec_.Forward(tgt_emb, enc_out)
            //   6. logits  = dec_out contracted with embed_.value^T   (weight tying)
            //   7. return logits
            return OutputTensor{};
        }

        // ---- single-sample backward ----
        InputTensor Backward(const OutputTensor& /*delta_A*/,
                             const OutputTensor& /*a*/,
                             const InputTensor&  /*a_prev*/) {
            // TODO step 4:
            //   1. d_dec_out = grad through tied projection (also accumulates into embed_.grad)
            //   2. (d_tgt_emb, d_enc_out) = dec_.Backward(d_dec_out, ..., enc_out_cache)
            //   3. d_src_emb = enc_ << BackwardArgs{d_enc_out, enc_out_cache, src_emb_cache}
            //   4. accumulate d_tgt_emb -> embed_.grad rows touched by tgt_shifted
            //   5. accumulate d_src_emb -> embed_.grad rows touched by src
            //   6. pack zero d-input (or repacked d-onehots) into InputTensor and return
            //      (input is one-hot ids; gradient w.r.t. one-hot input has no real meaning here,
            //       but Block::Backward must return InputTensor — return zero-init or the embedding
            //       grad reshaped, depending on what we settle on in step 4)
            return InputTensor{};
        }

        // ---- batched forward ----
        template<size_t Batch>
        Tensor<Batch, SrcLen + TgtLen, Vocab> /* unused — actual return is Tensor<Batch, TgtLen, Vocab> */
        BatchedForwardPlaceholder(const Tensor<Batch, SrcLen + TgtLen, Vocab>&) const = delete;

        template<size_t Batch>
        Tensor<Batch, TgtLen, Vocab> BatchedForward(
            const Tensor<Batch, SrcLen + TgtLen, Vocab>& /*X*/) const {
            // TODO step 3 (batched): same as Forward but threaded over the batch axis
            return Tensor<Batch, TgtLen, Vocab>{};
        }

        // ---- batched backward ----
        template<size_t Batch>
        Tensor<Batch, SrcLen + TgtLen, Vocab> BatchedBackward(
            const Tensor<Batch, TgtLen, Vocab>& /*delta_A*/,
            const Tensor<Batch, TgtLen, Vocab>& /*a*/,
            const Tensor<Batch, SrcLen + TgtLen, Vocab>& /*a_prev*/) {
            // TODO step 4 (batched)
            return Tensor<Batch, SrcLen + TgtLen, Vocab>{};
        }

        // ---- aggregate params: shared embed + entire encoder + entire decoder ----
        auto all_params() {
            return std::tuple_cat(std::tie(embed_), enc_.all_params(), dec_.all_params());
        }
        auto all_params() const {
            return std::tuple_cat(std::tie(embed_), enc_.all_params(), dec_.all_params());
        }

        // ---- snapshot transparency: severance visible in keys ----
        void peek(SnapshotMap& out, const std::string& prefix) const {
            enc_.peek(out, prefix + "encoder.");
            dec_.peek(out,  prefix + "decoder.");
        }

        // ---- public introspection / inference helpers (severance exposed) ----
        // TODO step 3: implement once Forward is wired
        // EncHidden EncodeOnly(const SrcOneHot& src) const;
        // OutputTensor DecodeStep(const EncHidden& enc_out, const TgtOneHot& tgt_so_far) const;
    };

} // namespace TTTN
