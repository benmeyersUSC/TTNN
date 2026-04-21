#pragma once
#include "EncoderDecoder.hpp"
#include "TrainableTensorNetwork.hpp"
#include <concepts>
#include <iostream>
#include <type_traits>

namespace
TTTN {
    template<typename F>
    auto timed(const char *label, F &&fn) {
        using Clock = std::chrono::high_resolution_clock;
        const auto t0 = Clock::now();
        if constexpr (std::is_void_v<std::invoke_result_t<F> >) {
            fn();
            const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            std::cout << "[" << label << "]  " << ms << " ms\n";
        } else {
            auto result = fn();
            const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            std::cout << "[" << label << "]  " << ms << " ms\n";
            return result;
        }
    }


    struct DataCursor {
        int lap = 0;
        int step = 0;
        long total_seen = 0;
        std::string path;

        explicit DataCursor(std::string_view p) : path(p) {
        }

        struct ChunkRef {
            int subset;
            int row;
        };

        void Load() {
            std::ifstream f(path);
            if (!f) return;
            f >> lap >> step >> total_seen;
        }

        void Save() const {
            std::ofstream f(path);
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

        [[nodiscard]] ChunkRef CurrentChunk(int n_subsets, int subset_size, int chunk_size) const {
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

    // Contract: define an enum with PAD, BOS, EOS, COUNT (COUNT must be last — sets VocabSize).
    // Also provide a free function in the same namespace: std::string TokenName(T t)
    template<typename T>
    concept TokenEnum =
            std::is_enum_v<T>
            && requires {
                { T::PAD }   -> std::same_as<T>;
                { T::BOS }   -> std::same_as<T>;
                { T::EOS }   -> std::same_as<T>;
                { T::COUNT } -> std::same_as<T>;
            }
            && requires(T t) {
                { TokenName(t) } -> std::convertible_to<std::string>;
            };


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
        using BatchInpT = PrependBatch<Batch, InputT>::type;
        using BatchTgtT = PrependBatch<Batch, OutputT>::type;


        static constexpr size_t TotalTrain = NSubsetsTrain * SubsetSizeTrain * Laps;
        static constexpr size_t ExamplesPerEpoch = NSubsetsTrain * SubsetSizeTrain;
        static constexpr size_t ExamplesPerGroup = ChunksPerGroup * ChunkSizeTrain;
        static constexpr size_t Groups = (TotalTrain + ExamplesPerGroup - 1) / ExamplesPerGroup;

        struct RLState {
            float baseline = 0.f;
            bool init = false;
            std::string path;

            // LR schedule: ramps from lr_max -> lr_min over ramp_size epochs
            float lr_min, lr_max, ramp_size;

            explicit RLState(std::string_view rl_state_path)
                : path(rl_state_path.data()),
                  lr_min(RL_LR_MIN), lr_max(RL_LR_MAX), ramp_size(RL_RAMP_SIZE) {
            }

            [[nodiscard]] float LR(long total_seen) const {
                const float t = std::clamp(
                    static_cast<float>(total_seen) / (ramp_size * ExamplesPerEpoch), 0.f, 1.f);
                return lr_max + (lr_min - lr_max) * t;
            }

            void Load() {
                std::ifstream f(path);
                if (!f) return;
                int init_flag = 0;
                f >> baseline >> init_flag;
                init = init_flag != 0;
            }

            void Save() const {
                std::ofstream f(path.data());
                f << baseline << " " << (init ? 1 : 0) << "\n";
            }
        };

        NetworkT Network;
        RLState RL_State;
        DataCursor Cursor;
        std::string dashboard_script_;   // path to transformer_trainer_dashboard.py; empty = disabled

    public:
        // dashboard_script: path to tools/transformer_trainer_dashboard.py (or "" to disable).
        explicit TransformerTrainer(std::string_view dashboard_script = "")
            : RL_State(CheckpointDir() + "/rl_state.txt"),
              Cursor(CheckpointDir() + "/cursor.txt"),
              dashboard_script_(dashboard_script) {
        }

    private:
        static const std::string &ModelString() {
            static const std::string str = std::to_string(EmbeddingDimension) +
                                           "_h" + std::to_string(NumHeads) +
                                           "_f" + std::to_string(FFNSize) +
                                           "_" + std::to_string(NEnc) + "enc" + std::to_string(NDec) + "dec";
            return str;
        }

        static const std::string &CheckpointDir() {
            static const std::string dir = "checkpoints_e" + ModelString();
            return dir;
        }

        using Example = std::pair<std::vector<uint8_t>, std::vector<uint8_t> >;
        using TokenizedExample = std::pair<std::vector<Token>, std::vector<Token> >;

        static TokenizedExample TokenizeExample(const Example &ex) {
            std::vector<Token> src, tgt;
            src.reserve(SrcLen);
            tgt.reserve(TgtLen);
            for (const auto &n: ex.first) src.emplace_back(static_cast<Token>(n));
            for (const auto &n: ex.second) tgt.emplace_back(static_cast<Token>(n));
            return {src, tgt};
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
                                       ? static_cast<Token>(predicted[i])
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


        static std::vector<uint8_t> ReadVec(std::string_view path, const long byte_offset, size_t n_bytes) {
            std::ifstream f(path.data(), std::ios::binary);
            if (!f) throw std::runtime_error("can't open file");
            f.seekg(byte_offset);
            std::vector<uint8_t> buf(n_bytes);
            f.read(reinterpret_cast<char *>(buf.data()), static_cast<long>(n_bytes));
            // if we just read in fewer bytes than n_bytes, throw
            if (static_cast<size_t>(f.gcount()) != n_bytes) {
                throw std::runtime_error("short read on data file");
            }
            return buf;
        }

        std::vector<Example> GetNextChunk(int max_tgt_len) {
            auto [subset, row] = Cursor.CurrentChunk(NSubsetsTrain, SubsetSizeTrain, ChunkSizeTrain);

            int n_chunks = DataCursor::ChunksPerLap(NSubsetsTrain, SubsetSizeTrain, ChunkSizeTrain);
            std::cout << "\033[2m  · lap" << Cursor.lap << " step " << Cursor.step << "/" << n_chunks
                    << "  s" << subset << " r" << row << "\033[0m\n";

            // get subset file name from ChunkRef
            std::ostringstream sd;
            sd << "data/subset" << subset << "/train";
            const std::string base = sd.str();

            const auto src_off = static_cast<long>(row) * SrcLen;
            const auto tgt_off = static_cast<long>(row) * TgtLen;

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
            for (size_t i = 1; i < TgtLen; ++i) {
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

                const size_t best = ArgmaxAt(logits, step);

                if constexpr (NLL) {
                    const uint8_t true_tok = ex.second[step];
                    if (true_tok == static_cast<uint8_t>(Token::PAD)) break;
                    const auto [_, sum_exp] = SoftmaxStatsAt(logits, step);
                    total_nll -= (logits(step, true_tok) - logits(step, best)) - std::log(sum_exp);
                    ++n_tokens;
                }

                const auto tok = static_cast<Token>(best);
                out.push_back(tok);
                if (tok == Token::EOS) break;

                if (step + 1 < TgtLen) {
                    seed(step + 1, best) = 1.f;
                }
            }

            if (out.empty() || out.back() != Token::EOS) {
                out.push_back(Token::EOS);
            }
            return {out, n_tokens > 0 ? static_cast<float>(total_nll / static_cast<double>(n_tokens)) : 0.f};
        }

        static std::vector<Token> ArgmaxDecode(const OutputT &logits) {
            std::vector<Token> out;
            for (size_t t = 0; t < TgtLen; ++t)
                out.push_back(static_cast<Token>(ArgmaxAt(logits, t)));
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

        static void FillDist(DistResult &dist, long n_tokens,
                             const std::vector<double> &true_counts,
                             const std::vector<double> &pred_soft_sum,
                             const std::vector<double> &conf_hist_counts,
                             int bins) {
            dist.n_tokens = n_tokens;
            dist.true_dist.assign(VocabSize, 0.0);
            dist.pred_dist.assign(VocabSize, 0.0);
            for (size_t v = 0; v < VocabSize; ++v) {
                dist.true_dist[v] = n_tokens > 0 ? true_counts[v] / static_cast<double>(n_tokens) : 0.0;
                dist.pred_dist[v] = n_tokens > 0 ? pred_soft_sum[v] / static_cast<double>(n_tokens) : 0.0;
            }
            dist.conf_hist.assign(bins, 0.0);
            double hist_total = 0.0;
            for (const auto &c: conf_hist_counts) hist_total += c;
            for (int i = 0; i < bins; ++i)
                dist.conf_hist[i] = hist_total > 0.0 ? conf_hist_counts[i] / hist_total : 0.0;
        }

        template<typename T>
        static std::vector<T> ShuffledIota(T n, std::mt19937 &rng) {
            std::vector<T> v(static_cast<size_t>(n));
            std::iota(v.begin(), v.end(), T{0});
            std::ranges::shuffle(v, rng);
            return v;
        }

        void TFTestPass(const size_t n_rows, const int bins, const std::vector<uint8_t> &src_buf,
                        const std::vector<uint8_t> &tgt_buf,
                        double &tf_loss,
                        unsigned &tf_examples,
                        unsigned &tf_tokens, std::vector<double> &tf_true, std::vector<double> &tf_pred_soft,
                        std::vector<double> &tf_conf_hist) {
            for (size_t r = 0; r < n_rows; ++r) {
                const Example ex{
                    std::vector<uint8_t>(src_buf.data() + r * SrcLen, src_buf.data() + (r + 1) * SrcLen),
                    std::vector<uint8_t>(tgt_buf.data() + r * TgtLen, tgt_buf.data() + (r + 1) * TgtLen)
                };
                const auto [inp, tgt] = EncodeExample(ex);
                const auto logits = Network.Forward(inp);
                tf_loss += SequenceSoftmaxCEL<Token::PAD>::Loss(logits, tgt).flat(0);
                ++tf_examples;

                for (size_t t = 0; t < TgtLen; ++t) {
                    size_t true_tok = VocabSize;
                    for (size_t v = 0; v < VocabSize; ++v) {
                        if (tgt(t, v) > 0.5f) {
                            true_tok = v;
                            break;
                        }
                    }
                    if (true_tok == VocabSize || true_tok == static_cast<size_t>(Token::PAD)) continue;

                    tf_true[true_tok] += 1.0;
                    ++tf_tokens;

                    const auto [max_idx, sum_exp] = SoftmaxStatsAt(logits, t);
                    const float max_l = logits(t, max_idx);
                    for (size_t v = 0; v < VocabSize; ++v)
                        tf_pred_soft[v] += std::exp(logits(t, v) - max_l) / sum_exp;
                    double p_correct = std::exp(logits(t, true_tok) - max_l) / sum_exp;
                    tf_conf_hist[std::min(bins - 1, static_cast<int>(p_correct * bins))] += 1.0;
                }
            }
        }


        void ARTestPass(const long ar_this_subset, const int bins, const std::vector<size_t> &row_order,
                        const std::vector<uint8_t> &src_buf,
                        const std::vector<uint8_t> &tgt_buf, double &ar_nll, long &ar_tokens, long &ar_done,
                        std::vector<double> &ar_true, std::vector<double> &ar_pred_soft,
                        std::vector<double> &ar_conf_hist) {
            const auto &block = Network.template block<0>();
            for (long i = 0; i < ar_this_subset; ++i) {
                const size_t r = row_order[static_cast<size_t>(i)];
                const uint8_t *xs = src_buf.data() + r * SrcLen;
                const uint8_t *ys = tgt_buf.data() + r * TgtLen;

                typename BlockT::SrcOneHot src_oh{};
                for (size_t t = 0; t < SrcLen; ++t)
                    src_oh(t, xs[t]) = 1.0f;
                const auto enc_out = block.EncodeOnly(src_oh);

                typename BlockT::TgtOneHot seed{};
                seed(0, static_cast<size_t>(Token::BOS)) = 1.0f;

                for (size_t step = 0; step < TgtLen; ++step) {
                    const uint8_t true_tok = ys[step];
                    if (true_tok == static_cast<uint8_t>(Token::PAD)) break;

                    const auto logits = block.DecodeStep(enc_out, seed);

                    const auto [best, sum_exp] = SoftmaxStatsAt(logits, step);
                    const float max_l = logits(step, best);

                    ar_nll -= (logits(step, true_tok) - max_l) - std::log(sum_exp);
                    ++ar_tokens;

                    ar_true[true_tok] += 1.0;
                    for (size_t v = 0; v < VocabSize; ++v)
                        ar_pred_soft[v] += std::exp(static_cast<double>(logits(step, v) - max_l)) / sum_exp;
                    double p_correct_ar = std::exp(static_cast<double>(logits(step, true_tok) - max_l)) / sum_exp;
                    ar_conf_hist[std::min(bins - 1, static_cast<int>(p_correct_ar * bins))] += 1.0;

                    if (static_cast<Token>(best) == Token::EOS) break;
                    if (step + 1 < TgtLen)
                        seed(step + 1, best) = 1.0f;
                }
                ++ar_done;
            }
        }

        std::pair<float, float> EvalTestSet(DistResult &tf_dist, DistResult &ar_dist, std::mt19937 &rng,
                                            int bins = 50) {
            // TF accumulators
            double tf_loss = 0.0;
            long tf_examples = 0;
            long tf_tokens = 0;
            std::vector<double> tf_true(VocabSize, 0.0);
            std::vector<double> tf_pred_soft(VocabSize, 0.0);
            std::vector<double> tf_conf_hist(bins, 0.0);

            // AR accumulators
            double ar_nll = 0.0;
            long ar_tokens = 0;
            long ar_done = 0;
            const long ar_quota = static_cast<long>(N_AR_TEST_SUBSETS) * AR_TEST_N_SAMPLES;
            std::vector<double> ar_true(VocabSize, 0.0);
            std::vector<double> ar_pred_soft(VocabSize, 0.0);
            std::vector<double> ar_conf_hist(bins, 0.0);

            std::cout << "[test_eval] ar_quota=" << ar_quota << "  tf=all subsets\n";

            // shuffle subset order so AR samples aren't always front-loaded on subset 0

            for (auto order = ShuffledIota<int>(NSubsetsTrain, rng); int s: order) {
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
                // read in once
                src_f.read(reinterpret_cast<char *>(src_buf.data()), static_cast<long>(src_buf.size()));
                tgt_f.read(reinterpret_cast<char *>(tgt_buf.data()), static_cast<long>(tgt_buf.size()));

                // shuffle row order for AR sampling within this subset
                auto row_order = ShuffledIota<size_t>(n_rows, rng);

                // if we have more AR to do, then this subset we want the min of sample-param and subset size
                const long ar_this_subset = ar_done < ar_quota
                                                ? std::min(static_cast<long>(AR_TEST_N_SAMPLES), ar_quota - ar_done)
                                                : 0;


                // TF pass: all rows
                TFTestPass(n_rows, bins, src_buf, tgt_buf, tf_loss, tf_examples, tf_tokens, tf_true, tf_pred_soft,
                           tf_conf_hist);

                // AR pass: first ar_this_subset rows in shuffled order
                ARTestPass(ar_this_subset, bins, row_order, src_buf, tgt_buf, ar_nll, ar_tokens, ar_done, ar_true,
                           ar_pred_soft, ar_conf_hist);
            }

            FillDist(tf_dist, tf_tokens, tf_true, tf_pred_soft, tf_conf_hist, bins);
            FillDist(ar_dist, ar_tokens, ar_true, ar_pred_soft, ar_conf_hist, bins);

            return {
                tf_examples > 0 ? static_cast<float>(tf_loss / static_cast<double>(tf_examples)) : 0.f,
                ar_tokens > 0 ? static_cast<float>(ar_nll / static_cast<double>(ar_tokens)) : 0.f
            };
        }

        // Writes dist fields prefixed by `prefix`: {prefix}_n_tokens, {prefix}_pred_dist,
        // {prefix}_conf_hist. For prefix=="tf" also writes token_names and true_dist (once).
        static void DistToJSON(const DistResult *dist, std::ofstream &f,
                               std::string_view prefix, const int bins) {
            if (!dist || dist->n_tokens <= 0) return;
            const std::string p(prefix);
            f << std::setprecision(8);
            f << ",\n  \"" << p << "_n_tokens\": " << dist->n_tokens;

            if (prefix == "tf") {
                f << ",\n  \"token_names\": [";
                for (size_t v = 0; v < VocabSize; ++v) {
                    f << "\"" << TokenName(static_cast<Token>(v)) << "\"";
                    if (v + 1 < VocabSize) f << ", ";
                }
                f << "],\n  \"true_dist\": [";
                for (size_t v = 0; v < VocabSize; ++v) {
                    f << dist->true_dist[v];
                    if (v + 1 < VocabSize) f << ", ";
                }
                f << "]";
            }

            f << ",\n  \"" << p << "_pred_dist\": [";
            for (size_t v = 0; v < VocabSize; ++v) {
                f << dist->pred_dist[v];
                if (v + 1 < VocabSize) f << ", ";
            }
            f << "]";
            if (!dist->conf_hist.empty()) {
                f << ",\n  \"" << p << "_conf_hist\": [";
                for (int i = 0; i < bins; ++i) {
                    f << dist->conf_hist[i];
                    if (i + 1 < bins) f << ", ";
                }
                f << "]";
            }
        }

        static void WriteProgressJSON(const std::string &path, const long total_seen, const int lap,
                                      const float train_loss, const float test_loss, const float avg_tf_pct,
                                      const float avg_autoreg_pct,
                                      const float avg_rl_reward = -1.f,
                                      const float avg_rl_accuracy = -1.f, const float ar_test_loss = -1.f,
                                      const DistResult *tf_dist = nullptr, const DistResult *ar_dist = nullptr,
                                      const int dist_bins = 50) {
            std::ofstream f(path);
            f << std::fixed << std::setprecision(6);
            f << "{\n";

            // Static config block — lets the Python dashboard be dataset-agnostic
            f << "  \"config\": {\n";
            f << "    \"examples_per_epoch\": " << ExamplesPerEpoch << ",\n";
            f << "    \"src_len\": "            << SrcLen           << ",\n";
            f << "    \"tgt_len\": "            << TgtLen           << ",\n";
            f << "    \"vocab_size\": "         << VocabSize        << ",\n";
            f << "    \"pad_id\": "             << static_cast<size_t>(Token::PAD) << ",\n";
            f << "    \"tf_lr_min\": "          << TF_LR_MIN        << ",\n";
            f << "    \"tf_lr_max\": "          << TF_LR_MAX        << ",\n";
            f << "    \"tf_ramp_size\": "       << TF_RAMP_SIZE     << ",\n";
            f << "    \"ss_ramp_start\": "      << SS_RAMP_START    << ",\n";
            f << "    \"ss_ramp_end\": "        << SS_RAMP_END      << ",\n";
            f << "    \"ss_min\": "             << SS_MIN           << ",\n";
            f << "    \"ss_max\": "             << SS_MAX           << ",\n";
            f << "    \"tgt_len_min\": "        << TGT_LEN_MIN      << ",\n";
            f << "    \"tgt_len_ramp\": "       << TGT_LEN_RAMP     << ",\n";
            f << "    \"rl_lr_min\": "          << RL_LR_MIN        << ",\n";
            f << "    \"rl_lr_max\": "          << RL_LR_MAX        << ",\n";
            f << "    \"rl_ramp_size\": "       << RL_RAMP_SIZE     << "\n";
            f << "  },\n";

            f << "  \"total_seen\": "      << total_seen      << ",\n";
            f << "  \"lap\": "             << lap             << ",\n";
            f << "  \"train_loss\": "      << train_loss      << ",\n";
            f << "  \"test_loss\": "       << test_loss       << ",\n";
            f << "  \"avg_tf_pct\": "      << avg_tf_pct      << ",\n";
            f << "  \"avg_autoreg_pct\": " << avg_autoreg_pct << ",\n";
            f << "  \"avg_rl_reward\": "   << avg_rl_reward   << ",\n";
            f << "  \"avg_rl_accuracy\": " << avg_rl_accuracy << ",\n";
            f << "  \"ar_test_loss\": "    << ar_test_loss    << ",\n";
            f << std::setprecision(4);
            f << "  \"ss_rate\": "         << ScheduledSamplingRate(total_seen) << ",\n";
            f << "  \"max_tgt_len\": "     << MaxAsmLength(total_seen);
            DistToJSON(tf_dist, f, "tf", dist_bins);
            DistToJSON(ar_dist, f, "ar", dist_bins);
            f << "\n}\n";
        }


        std::pair<float, float> RL_Update(const int n_examples, const std::vector<Example> &chunk,
                                          std::mt19937 &ss_rng) {
            using RLBatchInp = PrependBatch<RL_K, InputT>::type;
            using RLBatchTgt = PrependBatch<RL_K, OutputT>::type;

            std::uniform_int_distribution<size_t> pick_ex(0, n_examples - 1);
            float reward_sum = 0.f;
            float accuracy_sum = 0.f;
            RLBatchInp rl_x{};
            RLBatchTgt rl_y{};

            for (int k = 0; k < RL_K; ++k) {
                const auto &ex = chunk[pick_ex(ss_rng)];

                // AR decode + NLL against GT
                const auto [decoded, nll] = AutoregressiveDecode<true>(ex);

                // reward is NNLL, negative-negative-log likelihood == log likelihood
                // we want the negative of NLL to be as high as possible!
                // LL(1), which is the best possible value here, = 0 and everything else is negative
                reward_sum += -nll;

                const auto acc = CalculateStrictAccuracy(decoded, ex.second);
                accuracy_sum += acc.total > 0 ? static_cast<float>(acc.correct) / acc.total : 0.f;

                const auto [rl_inp, rl_tgt] = EncodeExample({ex.first, decoded});
                TensorSet<0>(rl_x, k, rl_inp);
                TensorSet<0>(rl_y, k, rl_tgt);
            }
            const float avg_R = reward_sum / RL_K;
            const float avg_acc = accuracy_sum / RL_K;

            if (!RL_State.init) {
                RL_State.baseline = avg_R;
                RL_State.init = true;
            } else {
                RL_State.baseline = RL_BaselineDecay * RL_State.baseline + (1.f - RL_BaselineDecay) * avg_R;
            }
            RL_State.Save();

            const float advantage = avg_R - RL_State.baseline;
            const float rl_lr = RL_State.LR(Cursor.total_seen) * advantage;

            Network.template BatchFit<SequenceSoftmaxCEL<Token::PAD>, RL_K>(rl_x, rl_y, rl_lr);

            std::cout << "\033[2m  [rl] nll_R=" << avg_R
                    << "  acc=" << avg_acc
                    << "  base=" << RL_State.baseline
                    << "  A=" << advantage
                    << "  rl_lr=" << rl_lr << "\033[0m\n";
            return {avg_R, avg_acc};
        }

    public:
        void Print(std::ostream &os) {
            std::filesystem::create_directories(CheckpointDir());
            std::cout << "[config] checkpoint dir: " << CheckpointDir() << "\n";

            std::cout << "[model] EMB=" << EmbeddingDimension << " HEADS=" << NumHeads
                    << " FFN=" << FFNSize << " ENC=" << NEnc << " DEC=" << NDec
                    << " VOCAB=" << VocabSize << "\n";
            std::cout << "[model] params: " << NetworkT::TotalParamCount
                    << " (" << NetworkT::TotalParamCount * 4 / 1024 / 1024 << " MB)\n";
            std::cout << "[config] chunks/group=" << ChunksPerGroup
                    << " groups/lap=" << Groups
                    << " batch=" << Batch << "\n";
            std::cout << "[config] ce_lr=" << CELR(0) << "→" << CELR(100 * ExamplesPerEpoch)
                    << "  rl_lr=" << RL_State.LR(0) << "→" << RL_State.LR(100 * ExamplesPerEpoch)
                    << "  rl_k=" << RL_K << "\n";
        }

        void Train() {
            timed("FULL TRAINING SESSION", [this] { train_(); });
        }

        // AR-decode a raw source token sequence and return predicted tokens up to EOS.
        std::vector<Token> Infer(const std::vector<uint8_t> &src_ids) {
            Example ex{src_ids, std::vector<uint8_t>(TgtLen, static_cast<uint8_t>(Token::PAD))};
            auto [tokens, _nll] = AutoregressiveDecode(ex);
            return tokens;
        }

        void RunDashboard() const {
            if (dashboard_script_.empty()) return;
            const std::string output = CheckpointDir() + "/training_dashboard.html";
            const std::string cmd    = "python3 " + dashboard_script_
                                     + " " + CheckpointDir()
                                     + " " + output
                                     + " && open " + output;
            std::cout << "[dashboard] " << cmd << "\n";
            std::system(cmd.c_str());
        }

        void train_() {
            const std::string checkpoint_model = CheckpointDir() + "/model.bin";
            const std::string progress_base = CheckpointDir() + "/progress";
            RL_State.Load();
            std::cout << "[rl] loaded state: baseline=" << RL_State.baseline
                    << "  init=" << RL_State.init << "\n";
            Cursor.Load();
            std::cout << "[cursor] lap=" << Cursor.lap
                    << "  step=" << Cursor.step
                    << "  total_seen=" << Cursor.total_seen << "\n";

            std::filesystem::create_directories(CheckpointDir());
            Print(std::cout);

            if (std::filesystem::exists(checkpoint_model)) {
                Network.Load(checkpoint_model);
                std::cout << "[model] loaded from " << checkpoint_model << "\n";
            } else {
                std::cout << "[model] fresh init\n";
            }

            std::mt19937 ss_rng(42);

            float loss_acc = 0.f;
            int batches = 0;

            float rl_reward_acc = 0.f;
            float rl_acc_acc = 0.f;
            int rl_reward_count = 0;

            for (auto g = 0; g < Groups; ++g) {
                const DataCursor precursor(Cursor);

                const int cur_max_len = MaxAsmLength(precursor.total_seen);
                const float p_ss = ScheduledSamplingRate(precursor.total_seen);
                const float epoch_approx = static_cast<float>(precursor.total_seen) / ExamplesPerEpoch;

                std::vector<Example> chunk;
                chunk.reserve(ExamplesPerGroup);
                for (auto i = 0; i < ChunksPerGroup; ++i) {
                    auto next = GetNextChunk(cur_max_len);
                    chunk.insert(chunk.end(), next.begin(), next.end());
                    Cursor.Advance(ChunkSizeTrain, NSubsetsTrain, SubsetSizeTrain);
                    Cursor.Save();
                }

                if (Cursor.lap > precursor.lap) {
                    std::cout << "\033[2m  ── dataset lap wrap → lap " << Cursor.lap << " ──\033[0m\n";
                }

                const size_t n_examples = chunk.size();
                const size_t n_batches = n_examples / Batch;
                if (n_batches == 0) {
                    std::cout << "  [g" << g + 1 << "/" << Groups
                            << "] skipped (0 examples after length filter, max_len="
                            << cur_max_len << ")\n";
                    continue;
                }


                float group_loss = 0.f;
                bool training_ok = false;
                try {
                    for (auto b = 0; b < n_batches; ++b) {
                        timed("Batch", [this, &chunk, &b, &p_ss, &ss_rng, &group_loss, &loss_acc, &batches]() {
                            BatchInpT batch_x;
                            BatchTgtT batch_y;
                            for (auto i = 0; i < Batch; ++i) {
                                const auto &ex = chunk[b * Batch + i];
                                const auto [_tf_inp, tgt] = EncodeExample(ex);
                                const auto inp = EncodeInpWithSS(ex, p_ss, ss_rng);
                                TensorSet<0>(batch_x, i, inp);
                                TensorSet<0>(batch_y, i, tgt);
                            }
                            const float loss = Network.template BatchFit<SequenceSoftmaxCEL<Token::PAD>, Batch>(
                                batch_x, batch_y, CELR(Cursor.total_seen));
                            group_loss += loss;
                            loss_acc += loss;
                            batches++;
                        });
                        const long prev_total = Cursor.total_seen;
                        Cursor.total_seen += Batch;
                        Cursor.Save();

                        if ((b + 1) % LogEvery == 0) {
                            char line[256];
                            std::snprintf(line, sizeof(line),
                                          "%s  g%d/%lu b%d/%zu  loss=%.3f  ss=%.2f  len=%d  ep=%.2f",
                                          EpochBar(Cursor.total_seen).c_str(),
                                          g + 1, Groups, b + 1, n_batches
                                          , group_loss / static_cast<float>(b + 1), p_ss,
                                          cur_max_len,
                                          static_cast<float>(Cursor.total_seen) / ExamplesPerEpoch);
                            std::cout << line << "\n";
                        }

                        if (Cursor.total_seen / ExamplesPerEpoch > prev_total / ExamplesPerEpoch) {
                            const float running_training_loss = batches > 0
                                                                    ? loss_acc / static_cast<float>(batches)
                                                                    : group_loss / (b + 1);
                            const float cur_rl_reward = rl_reward_count > 0 ? rl_reward_acc / rl_reward_count : -1.f;
                            const float cur_rl_acc = rl_reward_count > 0
                                                         ? rl_acc_acc / rl_reward_count
                                                         : -1.f;

                            const long ep = Cursor.total_seen / ExamplesPerEpoch;
                            std::cout << "\n══════════ " << ep << " × "
                                    << ExamplesPerEpoch << " TRAINED ══════════\n";

                            DistResult tf_dist, ar_dist;
                            auto [tf_loss, ar_loss] = timed("[test]", [this, &tf_dist, &ar_dist]() {
                                return EvalTestSet(tf_dist, ar_dist);
                            });
                            std::cout << "[test] tf_loss=" << tf_loss << "\n";
                            std::cout << "[test] ar_loss=" << ar_loss << "\n";

                            WriteProgressJSON(progress_base + "_" + std::to_string(Cursor.total_seen) + ".json",
                                              Cursor.total_seen, Cursor.lap, running_training_loss, tf_loss, 0.f, 0.f,
                                              cur_rl_reward,
                                              cur_rl_acc, ar_loss, &tf_dist, &ar_dist);
                            Network.Save(progress_base + "_save_" + std::to_string(Cursor.total_seen) + ".bin");
                            Network.Save(checkpoint_model);

                            std::cout << "[save] model + test-JSON @ trained=" << Cursor.total_seen << "\n";
                            RunDashboard();

                            loss_acc = 0.f;
                            batches = 0;
                            rl_reward_acc = 0.f;
                            rl_acc_acc = 0.f;
                            rl_reward_count = 0;
                        }
                    }

                    training_ok = true;
                } catch (const std::exception &e) {
                    std::cerr << "\n[ERROR] training failed at group " << g << ": " << e.what() << "\n";
                }

                if (!training_ok) {
                    Cursor = precursor;
                    std::cerr << "[cursor] rolled back to trained=" << precursor.total_seen << "\n";
                }

                if (n_examples >= RL_K) {
                    auto [rl_nll_r, rl_acc_r] = RL_Update(n_examples, chunk, ss_rng);
                    rl_reward_acc += rl_nll_r;
                    rl_acc_acc += rl_acc_r;
                    rl_reward_count++;
                }

                std::cout << "\033[2m  [g" << g + 1 << "/" << Groups << "] done  "
                        << "loss=" << group_loss / n_batches
                        << "  trained=" << Cursor.total_seen << "\033[0m\n";
            }
            Network.Save(checkpoint_model);
            RL_State.Save();
            Cursor.Save();
            std::cout << "[save] session end: model + rl_state written\n";
        }
    };
}
