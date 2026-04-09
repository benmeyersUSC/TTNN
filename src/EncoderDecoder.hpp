#pragma once
#include <array>
#include <fstream>
#include <stdexcept>
#include <string>
#include "BlockSequence.hpp"
#include "TransformerBlock.hpp"
#include "TTTN_ML.hpp"
#include "NetworkUtil.hpp"

namespace TTTN {

    template<size_t SrcLen, size_t TgtLen, size_t Vocab,
             size_t EmbDim, size_t Heads, size_t FFNHidden,
             size_t NEnc,   size_t NDec,
             size_t PadId>
    class EncoderDecoder {
        static_assert(EmbDim % Heads == 0, "EmbDim must be divisible by Heads");
        static_assert(PadId < Vocab,       "PadId must be a valid vocab index");
        static_assert(NEnc >= 1 && NDec >= 1, "Need at least one encoder and one decoder layer");

    public:
        // ---- public type surface ----
        using SrcTensor    = Tensor<SrcLen, Vocab>;       // one-hot src tokens
        using TgtTensor    = Tensor<TgtLen, Vocab>;       // one-hot tgt tokens (also logit shape)
        using LogitsTensor = Tensor<TgtLen, Vocab>;
        using EncHidden    = Tensor<SrcLen, EmbDim>;
        using DecHidden    = Tensor<TgtLen, EmbDim>;

        static constexpr size_t kSrcLen   = SrcLen;
        static constexpr size_t kTgtLen   = TgtLen;
        static constexpr size_t kVocab    = Vocab;
        static constexpr size_t kEmbDim   = EmbDim;
        static constexpr size_t kHeads    = Heads;
        static constexpr size_t kFFNHidden= FFNHidden;
        static constexpr size_t kNEnc     = NEnc;
        static constexpr size_t kNDec     = NDec;
        static constexpr size_t kPadId    = PadId;

    private:
        // ---- shared embedding (used by src embed, tgt embed, AND tied output projection) ----
        Param<Tensor<Vocab, EmbDim>> embed_{};

        // ---- encoder: standard pre-norm unmasked transformer stack ----
        using EncBlock = TransformerBlock<EncHidden, Heads, FFNHidden, /*PreNorm=*/true, /*Masked=*/false>;
        using EncStack = BlockSequence<
            // expanded by helper below; for now just N copies via parameter pack trick
            EncBlock  // placeholder — see RepeatedSequence note in TODO
        >;
        // TODO(skeleton): replace single-block EncStack with `BlockSequence<EncBlock, EncBlock, ..., EncBlock>` of NEnc copies.
        //   options: (a) recursive type builder `MakeRepeated<EncBlock, NEnc>::type`,
        //            (b) std::array<EncBlock, NEnc> with manual forward/backward loops (simpler, no BlockSequence dance).
        //   Will resolve when we wire forward/backward in step #3.
        EncStack enc_{};

        // ---- decoder layers: PRIVATE nested type, not a Block, two-input forward, pair-returning backward ----
        struct DecoderLayer {
            // self-attention pieces (masked), cross-attention pieces, FFN pieces
            // TODO(skeleton): declare LayerNorms, masked self-attn projections (Q/K/V/O),
            //                 cross-attn projections (Q from dec, K/V from enc_out), FFN W1/W2.
            //                 All as `Param<...>` members so all_params() can collect them.

            // forward: dec_hidden + enc_out -> dec_hidden_next
            DecHidden Forward(const DecHidden& /*dec_hidden*/, const EncHidden& /*enc_out*/) const {
                // TODO: implement in step #3
                return DecHidden{};
            }

            // backward: returns (d_dec_hidden_in, d_enc_out_contribution)
            std::pair<DecHidden, EncHidden> Backward(
                const DecHidden& /*delta_out*/,
                const DecHidden& /*a_out*/,
                const DecHidden& /*a_in*/,
                const EncHidden& /*enc_out*/) {
                // TODO: implement in step #4
                return {DecHidden{}, EncHidden{}};
            }

            // batched variants
            template<size_t Batch>
            Tensor<Batch, TgtLen, EmbDim> BatchedForward(
                const Tensor<Batch, TgtLen, EmbDim>& /*dec_hidden*/,
                const Tensor<Batch, SrcLen, EmbDim>& /*enc_out*/) const {
                return Tensor<Batch, TgtLen, EmbDim>{};
            }

            template<size_t Batch>
            std::pair<Tensor<Batch, TgtLen, EmbDim>, Tensor<Batch, SrcLen, EmbDim>>
            BatchedBackward(
                const Tensor<Batch, TgtLen, EmbDim>& /*delta_out*/,
                const Tensor<Batch, TgtLen, EmbDim>& /*a_out*/,
                const Tensor<Batch, TgtLen, EmbDim>& /*a_in*/,
                const Tensor<Batch, SrcLen, EmbDim>& /*enc_out*/) {
                return {Tensor<Batch, TgtLen, EmbDim>{}, Tensor<Batch, SrcLen, EmbDim>{}};
            }

            auto all_params()       { return std::tuple<>{}; /* TODO: tuple_cat all members */ }
            auto all_params() const { return std::tuple<>{}; /* TODO: tuple_cat all members */ }
        };

        std::array<DecoderLayer, NDec> dec_{};

        // ---- optimizer state ----
        AdamState mAdam_{};

    public:
        EncoderDecoder() {
            // TODO(skeleton): XavierInit on embed_.value with fan_in=Vocab, fan_out=EmbDim
        }

        // ---- forward: src + teacher-forced shifted tgt -> logits ----
        LogitsTensor Forward(const SrcTensor& /*src_tokens*/, const TgtTensor& /*tgt_shifted*/) const {
            // TODO(step #3):
            //   1. enc_in  = embed(src_tokens) + sinusoidal PE
            //   2. enc_out = enc_in >> enc_
            //   3. dec_in  = embed(tgt_shifted) + sinusoidal PE
            //   4. h = dec_in; for each layer L in dec_: h = L.Forward(h, enc_out)
            //   5. logits  = h contracted with embed_.value^T   (weight tying)
            //   6. return logits  (raw — softmax happens inside SequenceCEL via Loss math, or add a Softmax step here if cleaner)
            return LogitsTensor{};
        }

        template<size_t Batch>
        Tensor<Batch, TgtLen, Vocab> BatchedForward(
            const Tensor<Batch, SrcLen, Vocab>& /*src_tokens*/,
            const Tensor<Batch, TgtLen, Vocab>& /*tgt_shifted*/) const {
            return Tensor<Batch, TgtLen, Vocab>{};
        }

        // ---- training entry points ----
        float Fit(const SrcTensor& /*src*/,
                  const TgtTensor& /*tgt_shifted*/,
                  const TgtTensor& /*tgt_labels*/,
                  float /*lr*/) {
            // TODO(step #5):
            //   forward -> softmax -> SequenceCEL<PadId>::Loss/Grad
            //   backward through projection (tied), through decoder stack with local d_enc_out accumulator,
            //   through encoder, into embed_ from BOTH src and tgt sites.
            //   ZeroGrad, accumulate, mAdam_.step(), UpdateAll(all_params(), mAdam_, lr).
            return 0.f;
        }

        template<size_t Batch>
        float BatchFit(const Tensor<Batch, SrcLen, Vocab>& /*src*/,
                       const Tensor<Batch, TgtLen, Vocab>& /*tgt_shifted*/,
                       const Tensor<Batch, TgtLen, Vocab>& /*tgt_labels*/,
                       float /*lr*/) {
            // TODO(step #5): per-sample SequenceCEL averaging (PAD counts vary per sample)
            return 0.f;
        }

        void ZeroGrad() {
            embed_.zero_grad();
            // enc_.ZeroGrad();  // TODO: enable once enc_ is properly NEnc-deep
            // for (auto& L : dec_) std::apply([](auto&... p){ (p.zero_grad(), ...); }, L.all_params());
        }

        // ---- aggregate all params for Adam update ----
        // TODO(skeleton): return std::tuple_cat of (embed_, enc_.all_params(), each dec_ layer's all_params())
        //   Currently returns just embed_ so the type compiles.
        auto all_params()       { return std::tie(embed_); }
        auto all_params() const { return std::tie(embed_); }

        // ---- serialization: mirrors TrainableTensorNetwork::Save/Load ----
        // Format: [embed][enc params][dec layer 0 params]...[dec layer NDec-1 params][AdamState]
        void Save(const std::string& path) const {
            std::ofstream f(path, std::ios::binary);
            if (!f) throw std::runtime_error("Cannot write: " + path);
            SaveAll(std::tie(embed_), f);
            // TODO: enc_ and dec_ params once they exist
            f.write(reinterpret_cast<const char*>(&mAdam_), sizeof(AdamState));
        }

        void Load(const std::string& path) {
            std::ifstream f(path, std::ios::binary);
            if (!f) throw std::runtime_error("Cannot read: " + path);
            LoadAll(std::tie(embed_), f);
            // TODO: enc_ and dec_ params once they exist
            f.read(reinterpret_cast<char*>(&mAdam_), sizeof(AdamState));
        }
    };

} // namespace TTTN
