#pragma once
#include "EncoderDecoder.hpp"
#include <concepts>
#include <iostream>
#include <type_traits>

#include "TrainableTensorNetwork.hpp"


struct DataCursor {
    int lap = 0;
    int step = 0;
    long total_seen = 0;

    struct ChunkRef {
        int subset;
        int row;
    };

    void Load(std::string_view path) {
        std::ifstream f(path.data());
        if (!f) return;
        f >> lap >> step >> total_seen; // read the file right in
    }

    void Save(std::string_view path) const {
        std::ofstream f(path.data());
        f << lap << " " << step << " " << total_seen << "\n";
    }

    static int ChunksPerLap(int n_subsets, int subset_size, int chunk_size) {
        return n_subsets * (subset_size / chunk_size);
    }

    static std::vector<int> LapPermutation(int lap_n, int n_chunks) {
        std::vector<int> perm(n_chunks);
        std::iota(perm.begin(), perm.end(), 0);
        std::mt19937 rng(static_cast<uint32_t>(0xC0FFEE ^ lap_n));
        std::ranges::shuffle(perm, rng);
        return perm;
    }

    static ChunkRef DecodeChunk(int idx, int n_subsets, int subset_size, int chunk_size) {
        const int chunks_per_subset = subset_size / chunk_size;
        return ChunkRef{
            idx / chunks_per_subset, // subset
            (idx % chunks_per_subset) * chunk_size // row
        };
    }

    ChunkRef CurrentChunk(int n_subsets, int subset_size, int chunk_size) const {
        const int n = ChunksPerLap(n_subsets, subset_size, chunk_size);
        const auto perm = LapPermutation(lap, n);
        return DecodeChunk(perm[step], n_subsets, subset_size, chunk_size);
    }

    void Advance(int chunk_size, int n_subsets, int subset_size) {
        int n = ChunksPerLap(n_subsets, subset_size, chunk_size);
        ++step;
        if (step >= n) {
            step = 0;
            ++lap;
        }
    }

    void PrintCursor(const int n_subsets,
                     int subset_size, int chunk_size) const {
        const int n = ChunksPerLap(n_subsets, subset_size, chunk_size);
        auto [subset, row] = CurrentChunk(n_subsets, subset_size, chunk_size);
        double pct = n ? (100.0 * step) / n : 0.0;
        std::cout << "[cursor] lap=" << lap
                << "  step=" << step << "/" << n
                << "  (" << pct << "% of lap)"
                << "  total_seen=" << total_seen
                << "  next: subset" << subset << " row " << row
                << "\n";
    }
};


template<typename T>
concept IsEnum = std::is_enum_v<T>;

template<typename T>
concept TokenEnum =
        // T must be an enum value
        std::is_enum_v<T> && requires
        {
            // and we must have PAD, BOS tokens and a COUNT value (always last!)
            { T::PAD } -> std::same_as<T>;
            { T::BOS } -> std::same_as<T>;
            { T::EOS } -> std::same_as<T>;
            { T::COUNT } -> std::same_as<T>;
        };


namespace
TTTN {
    template
    <
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
        size_t N_AR_TEST_SUBSETS,
        size_t AR_TEST_N_SAMPLES,

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
        static constexpr size_t VocabSize = Token::COUNT;
        using BlockT = EncoderDecoderBlock<SrcLen, TgtLen, VocabSize, EmbeddingDimension, NumHeads, FFNSize, NEnc
            ,
            NDec,
            Token::PAD>;
        using NetworkT = TrainableTensorNetwork<BlockT>;
        using InputT = NetworkT::InputTensor;
        using OutputT = NetworkT::OutputTensor;

        NetworkT Network;

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
                const Token pred = i < static_cast<int>(predicted.size())
                                       ? static_cast<Token>(truth[i])
                                       : Token::PAD;
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

        static std::string EpochBar(const long total_seen, const int width = 64) {
            const long in_ep = total_seen % ExamplesPerEpoch;
            const float frac = static_cast<float>(in_ep) / ExamplesPerEpoch;
            const int fill = static_cast<int>(frac * width);
            std::string bar = "[";
            for (int i = 0; i < width; ++i) {
                bar += (i < fill ? '#' : ' ');
            }
            char tail[64];
            std::snprintf(tail, sizeof(tail), "] %5ld/%lu", in_ep, ExamplesPerEpoch);
            return bar + tail;
        }

        struct RLState {
            float baseline = 0.f;
            bool init = false;

            void LoadRLState(const std::string &path) {
                std::ifstream f(path);
                if (!f)return;
                int init_flag = 0;
                f >> baseline >> init_flag;
                init = init_flag != 0;
            }

            void SaveRLState(std::string_view path) const {
                std::ofstream f(path.data());
                f << baseline << " " << (init ? 1 : 0) << "\n";
            }
        };

        static std::vector<uint8_t> ReadVec(std::string_view path, const long byte_offset, size_t n_bytes) {
            std::ifstream f(path.data(), std::ios::binary);
            if (!f) throw std::runtime_error("can't open file");
            f.seekg(byte_offset);
            std::vector<uint8_t> buf(n_bytes);
            f.read(reinterpret_cast<char *>(buf.data()), n_bytes);
            // if we just read in fewer bytes than n_bytes, throw
            if (static_cast<size_t>(f.gcount()) != n_bytes) {
                throw std::runtime_error("short read on " + path);
            }
            return buf;
        }

        static std::vector<Example> GetNextChunk(std::string_view cursor_path, int max_tgt_len) {
            DataCursor cur;
            cur.Load(cursor_path);
            DataCursor::ChunkRef ref = cur.CurrentChunk(NSubsetsTrain, SubsetSizeTrain, ChunkSizeTrain);

            int n_chunks = cur.ChunksPerLap(NSubsetsTrain, SubsetSizeTrain, ChunkSizeTrain);
            std::cout << "\033[2m  · lap" << cur.lap << " step " << cur.step << "/" << n_chunks
                    << "  s" << ref.subset << " r" << ref.row << "\033[0m\n";

            // get subset file name from ChunkRef
            std::ostringstream sd;
            sd << "data/subset" << ref.subset << "/train";
            const std::string base = sd.str();

            const auto src_off = static_cast<long>(ref.row) * SrcLen;
            const auto tgt_off = static_cast<long>(ref.row) * TgtLen;

            // read in source and target
            auto src_bytes = ReadVec(base + ".src.bin", src_off, ChunkSizeTrain * SrcLen);
            auto tgt_bytes = ReadVec(base + ".tgt.bin", tgt_off, ChunkSizeTrain * TgtLen);

            // fill vector of Examples
            std::vector<Example> chunk;
            chunk.reserve(ChunkSizeTrain);

            for (int i = 0; i < ChunkSizeTrain; ++i) {
                const uint8_t *xs = src_bytes.data() + i * SrcLen;
                const uint8_t *ys = tgt_bytes.data() + i * TgtLen;

                if (max_tgt_len < TgtLen) {
                    int non_pad = 0;
                    for (int j = 0; j < TgtLen; ++j) {
                        if (ys[j] != Token::PAD) {
                            ++non_pad;
                        }
                    }
                    if (non_pad > max_tgt_len) {
                        continue;
                    }
                }
                chunk.emplace_back(
                    std::vector<uint8_t>(xs, xs + SrcLen), // (start_ptr, end_ptr) constructor
                    std::vector<uint8_t>(ys, ys + TgtLen)
                );
            }
            return chunk;
        }

        static std::pair<InputT, OutputT> EncodeExample(const Example &ex) {
            Tensor<SrcLen, VocabSize> src_oh;
            Tensor<TgtLen, VocabSize> tgt_oh;
            Tensor<TgtLen, VocabSize> tgt_shifted_oh;

            for (size_t i = 0; i < SrcLen; ++i) {
                src_oh(i/*-th row/token in sequence*/, ex.first[i]/*-th token in vocab*/) = 1.f;
            }
            for (size_t i = 0; i < TgtLen; ++i) {
                tgt_oh(i, ex.second[i]) = 1.f;
            }

            tgt_shifted_oh(0, static_cast<size_t>(Token::BOS)) = 1.f;
            for (size_t i = 0; i < TgtLen; ++i) {
                tgt_shifted_oh(i, ex.second[i - 1]) = 1.f;
            }

            return {ConcatAxis<0>(src_oh, tgt_shifted_oh), tgt_oh};
        }

        template<bool NLL = false>
        std::pair<std::vector<Token>, float> AutoregressiveDecode(const Example &ex) {
            typename BlockT::SrcOneHot src_oh{};
            for (size_t i = 0; i < SrcLen; ++i) {
                src_oh(i, ex.first[i]) = 1.0f;
            }

            const auto &block = Network.template block<0>();
            const auto enc_out = block.EncodeOnly(src_oh);

            typename BlockT::TgtOneHot seed{};
            seed(0, static_cast<size_t>(Token::BOS)) = 1.f;

            std::vector<Token> out;
            out.reserve(TgtLen);

            double total_nll = 0.0;
            long n_tokens = 0;


            for (size_t step = 0; step < TgtLen; ++step) {
                const auto logits = block.DecodeStep(enc_out, seed);

                size_t best = 0;
                float best_val = logits(step, 0);
                for (size_t v = 1; v < VocabSize; ++v) {
                    if (logits(step, v) > best_val) {
                        best_val = logits(step, v);
                        best = v;
                    }
                }

                if constexpr (NLL) {
                    const uint8_t true_tok = ex.second[step];
                    if (true_tok == static_cast<uint8_t>(Token::PAD))break;
                    double sum_exp = 0.0;
                    for (size_t v = 0; v < VocabSize; ++v) {
                        sum_exp += std::exp(static_cast<double>(logits(step, v) - best_val));
                    }
                    total_nll -= logits(step, true_tok) - best_val - std::log(sum_exp);
                    ++n_tokens;
                }

                const Token tok = static_cast<Token>(best);
                out.push_back(tok);
                if (tok == Token::EOS) break;

                if (step + 1 < TgtLen) {
                    seed(step + 1, best) = 1.f;
                }
            }

            if (out.empty() || out.back() != Token::EOS) {
                out.push_back(Token::EOS);
            }
            return {out, n_tokens > 0 ? static_cast<float>(total_nll / n_tokens) : 0.f};
        }

        static std::vector<Token> ArgmaxDecode(const OutputT &logits) {
            std::vector<Token> out;
            for (size_t t = 0; t < TgtLen; ++t) {
                size_t best = 0;
                float best_val = logits(t, 0);
                for (size_t v = 1; v < VocabSize; ++v) {
                    if (logits(t, v) > best_val) {
                        best_val = logits(t, v);
                        best = v;
                    }
                }
                out.push_back(static_cast<Token>(best));
            }
            return out;
        }

        InputT EncodeInpWithSS(const Example &ex, float p_sample, std::mt19937 &rng) {
            auto [tf_input, _tgt] = EncodeExample(ex);
            if (p_sample <= 0.f) return tf_input;

            const auto logits = Network.Forward(tf_input);
            const auto predicted = ArgmaxDecode(logits);

            // real number coin
            std::uniform_real_distribution<float> coin(0.f, 1.f);
            // for each token
            for (size_t t = 1; t < TgtLen; ++t) {
                // if the coin flips within p,
                if (coin(rng) < p_sample) {
                    // 0 out prev one-hot (true label)
                    tf_input(SrcLen + t, ex.second[t - 1]) = 0.0f;
                    // and 1 in model's prediction at this index
                    tf_input(SrcLen + t, static_cast<size_t>(predicted[t - 1])) = 1.0f;
                }
            }
            return tf_input;
        }

        struct DistResult {
            std::vector<double> true_dist; // VOCAB normalized frequencies
            std::vector<double> pred_dist; // VOCAB average softmax probs
            std::vector<double> conf_hist; // 50 bins [0,1]: fraction at each P(correct_tok) bucket
            long n_tokens = 0;
        };

        float TFEvalTestSet(DistResult &dist_out, unsigned bins = 50) {
            float total_loss = 0.f;
            int total_examples = 0;

            std::vector<double> true_counts(VocabSize, 0.0);
            std::vector<double> pred_soft_sum(VocabSize, 0.0);
            std::vector<double> conf_hist_counts(bins, 0.0);

            long n_tokens = 0;

            for (int s = 0; s < NSubsetsTrain; ++s) {
                std::ostringstream sd;
                sd << "data/subset" << s << "/test";
                const std::string base = sd.str();

                std::ifstream probe(base + ".src.bin", std::ios::binary | std::ios::ate);
                if (!probe) continue;
                size_t n_rows = static_cast<size_t>(probe.tellg()) / SrcLen;
                if (n_rows == 0) continue;

                std::ifstream src_f(base + ".src.bin", std::ios::binary);
                std::ifstream tgt_f(base + ".tgt.bin", std::ios::binary);
                std::vector<uint8_t> src_buf(n_rows * SrcLen);
                std::vector<uint8_t> tgt_buf(n_rows * TgtLen);
                src_f.read(reinterpret_cast<char *>(src_buf.data()), src_buf.size());
                tgt_f.read(reinterpret_cast<char *>(tgt_buf.data()), tgt_buf.size());

                for (size_t r = 0; r < n_rows; ++r) {
                    Example ex{
                        std::vector<uint8_t>(src_buf.data() + r * SrcLen,
                                             src_buf.data() + (r + 1) * SrcLen),
                        std::vector<uint8_t>(tgt_buf.data() + r * TgtLen,
                                             tgt_buf.data() + (r + 1) * TgtLen)
                    };

                    const auto [inp, tgt] = EncodeExample(ex);
                    // straight up TF forward pass
                    const auto logits = Network.Forward(inp);

                    total_loss += SequenceSoftmaxCEL<Token::PAD>::Loss(logits, tgt).flat(0);
                    ++total_examples;

                    // now build distribution for viz
                    for (size_t t = 0; t < TgtLen; ++t) {
                        size_t true_tok = 0;
                        for (size_t v = 0; v < VocabSize; ++v) {
                            if (tgt(t, v) > 0.5f) {
                                true_tok = v;
                                break;
                            }
                        }
                        if (true_tok == static_cast<size_t>(Token::PAD)) continue;

                        true_counts[true_tok] += 1.0;
                        ++n_tokens;

                        float max_l = logits(t, 0);
                        for (size_t v = 1; v < VocabSize; ++v) {
                            if (logits(t, v) > max_l) {
                                max_l = logits(t, v);
                            }
                        }
                        double sum_exp = 0.0;
                        for (size_t v = 0; v < VocabSize; ++v) {
                            sum_exp += std::exp(logits(t, v) - max_l);
                        }
                        for (size_t v = 0; v < VocabSize; ++v) {
                            pred_soft_sum[v] += std::exp(logits(t, v) - max_l) / sum_exp;
                        }
                        double p_correct = std::exp(logits(t, true_tok) - max_l) / sum_exp;
                        int bin = std::min(bins - 1, static_cast<unsigned>(p_correct * bins));
                        ++conf_hist_counts[bin];
                    }
                }
            }

            dist_out.n_tokens = n_tokens;
            dist_out.true_dist.resize(VocabSize);
            dist_out.pred_dist.resize(VocabSize);
            for (size_t v = 0; v < VocabSize; ++v) {
                dist_out.true_dist[v] = n_tokens > 0 ? true_counts[v] / n_tokens : 0.0;
                dist_out.pred_dist[v] = n_tokens > 0 ? pred_soft_sum[v] / n_tokens : 0.0;
            }
            dist_out.conf_hist.resize(bins);
            double hist_total = 0.0;
            for (auto &c: conf_hist_counts) hist_total += c;
            for (int i = 0; i < bins; ++i) {
                dist_out.conf_hist[i] = hist_total > 0.0 ? conf_hist_counts[i] / hist_total : 0.0;
            }
            return total_examples > 0 ? total_loss / total_examples : 0.f;
        }
    };
}
