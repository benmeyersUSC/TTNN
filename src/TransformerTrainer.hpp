#pragma once
#include "EncoderDecoder.hpp"
#include <concepts>
#include <type_traits>

#include "TrainableTensorNetwork.hpp"

template<typename T>
concept IsEnum = std::is_enum_v<T>;

template<typename T>
concept TokenEnum =
        // T must be an enum value
        std::is_enum_v<T> && requires
        {
            // and we must have a PAD and a COUNT value
            { T::PAD } -> std::same_as<T>;
            { T::COUNT } -> std::same_as<T>;
        };


namespace TTTN {
    template<
        // NETWORK SPEC
        size_t SrcLen,
        size_t TgtLen,
        TokenEnum Token,
        size_t EmbeddingDimension,
        size_t NumHeads,
        size_t FFNSize,
        size_t NEnc,
        size_t NDec,

        // DATA SPEC
        size_t PadId,
        size_t NSubsetsTrain,
        size_t SubsetSizeTrain,
        size_t ChunkSizeTrain,
        size_t Laps,

        // BATCH AND LOGGING CONSTANTS
        size_t ChunksPerGroup,
        size_t Batch,
        size_t LogEvery,
        size_t SentinelLogEvery,

        // TF CONSTANTS
        float TF_LR_MIN,
        float TF_LR_MAX,
        float TF_RAMP_SIZE,

        // RL CONSTANTS
        float RL_LR_MIN,
        float RL_LR_MAX,
        float RL_RAMP_SIZE,
        size_t RL_K,
        float RL_BaselineDecay,

        // SS
        float SS_RAMP_START,
        float SS_RAMP_END,
        float SS_MIN,
        float SS_MAX,

        // LEN CURRICULUM
        float TGT_LEN_RAMP,
        size_t TGT_LEN_MIN
    >
    class TransformerTrainer {
        using BlockT = EncoderDecoderBlock<SrcLen, TgtLen, Token::COUNT, EmbeddingDimension, NumHeads, FFNSize, NEnc,
            NDec,
            Token::PAD>;
        using NetworkT = TrainableTensorNetwork<BlockT>;
        using InputT = NetworkT::InputTensor;
        using OutputT = NetworkT::OutputTensor;

        static constexpr size_t TotalTrain = NSubsetsTrain * SubsetSizeTrain * Laps;
        static constexpr size_t ExamplesPerEpoch = NSubsetsTrain * SubsetSizeTrain;
        static constexpr size_t ExamplesPerGroup = ChunksPerGroup * ChunkSizeTrain;
        static constexpr size_t GroupsPerLap = (TotalTrain + ExamplesPerGroup - 1) / ExamplesPerGroup;

        static const std::string &CheckpointDir() {
            static const std::string dir =
                    "checkpoints_e" + std::to_string(EmbeddingDimension) +
                    "_h" + std::to_string(NumHeads) +
                    "_f" + std::to_string(FFNSize) +
                    "_" + std::to_string(NEnc) + "enc" + std::to_string(NDec) + "dec";

            return dir;
        }

        using Example = std::pair<std::vector<uint8_t>, std::vector<uint8_t> >;
        using TokenizedExample = std::pair<std::vector<Token>, std::vector<Token> >;

        static TokenizedExample TokenizeExample(const Example &ex) {
            std::vector<Token> src, asmb;
            src.reserve(SrcLen);
            asmb.reserve(TgtLen);
            for (const auto &n: ex.first) src.emplace_back(static_cast<Token>(n));
            for (const auto &n: ex.second) src.emplace_back(static_cast<Token>(n));
            return {src, asmb};
        }

        struct AccuracyStats {
            size_t correct = 0;
            size_t total = 0;
        };

        static AccuracyStats CalculateStrictAccuracy(const std::vector<Token> &predicted,
                                                     const std::vector<Token> &truth) {
            AccuracyStats stats;
            for (int i = 0; i < TgtLen; ++i) {
                const Token pred = i < static_cast<int>(predicted.size()) ? static_cast<Token>(truth[i]) : Token::PAD;
                const Token actual = i < static_cast<int>(truth.size()) ? static_cast<Token>(truth[i]) : Token::PAD;

                if (actual == Token::PAD) {
                    continue;
                }

                ++stats.total;
                if (pred == actual) {
                    ++stats.correct;
                }
            }
            return stats;
        }

        static float CELR(const long total_seen) {
            static constexpr float TF_Epoch_Inv = 1.f / (TF_RAMP_SIZE * ExamplesPerEpoch);
            const float t = std::clamp(static_cast<float>(total_seen) * TF_Epoch_Inv, 0.f, 1.f);
            return TF_LR_MAX + (TF_LR_MIN - TF_LR_MAX) * t;
        }

        static float RLLR(const long total_seen) {
            static constexpr float RL_Epoch_Inv = 1.f / (RL_RAMP_SIZE * ExamplesPerEpoch);
            const float t = std::clamp(static_cast<float>(total_seen) * RL_Epoch_Inv, 0.f, 1.f);
            return RL_LR_MAX + (RL_LR_MIN - RL_LR_MAX) * t;
        }

        static constexpr float EX_EP_INV = 1.f / ExamplesPerEpoch;

        static float ScheduledSamplingRate(const long total_seen) {
            const float epoch = static_cast<float>(total_seen) * EX_EP_INV;

            if (epoch < SS_RAMP_START) {
                return 0.f;
            }
            return std::clamp((epoch - SS_RAMP_START) / (SS_RAMP_END - SS_RAMP_START), std::max(0.f, SS_MIN),
                              std::min(1.f, SS_MAX));
        }

        static int MaxAsmLength(const long total_seen) {
            const float epoch = static_cast<float>(total_seen) * EX_EP_INV;
            if (epoch >= TGT_LEN_RAMP) {
                return TgtLen;
            }
            static constexpr float tlri = 1.f / TGT_LEN_MIN;
            return std::min(static_cast<int>(TgtLen),
                            static_cast<int>(TGT_LEN_MIN + (TgtLen - TGT_LEN_MIN) * epoch * tlri));
        }

        std::string EpochBar(const long total_seen, const int width = 64) {
            const long in_ep = total_seen % ExamplesPerEpoch;
            const float frac = static_cast<float>(in_ep) / ExamplesPerEpoch;
            const int fill = static_cast<int>(frac * width);
            std::string bar = "[";
            for (int i = 0; i < width; ++i) {
                bar += (i < fill ? '#' : ' ');
            }
            char tail[64];
            std::snprintf(tail, sizeof(tail), "] %5ld/%d", in_ep, EXAMPLES_PER_EPOCH);
            return bar + tail;
        }
    };
}
