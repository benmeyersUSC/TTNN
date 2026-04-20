#pragma once
#include "EncoderDecoder.hpp"
#include <concepts>
#include <iostream>
#include <type_traits>

#include "TrainableTensorNetwork.hpp"

namespace
TTTN {
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

        struct RLState {
            float baseline = 0.f;
            bool init = false;
            std::string path;

            // LR schedule: ramps from lr_max -> lr_min over ramp_size epochs
            float lr_min, lr_max, ramp_size;

            RLState(std::string_view rl_state_path)
                : path(rl_state_path.data()),
                  lr_min(RL_LR_MIN), lr_max(RL_LR_MAX), ramp_size(RL_RAMP_SIZE) {
            }

            float LR(long total_seen) const {
                const float t = std::clamp(
                    static_cast<float>(total_seen) / (ramp_size * ExamplesPerEpoch), 0.f, 1.f);
                return lr_max + (lr_min - lr_max) * t;
            }

            void LoadRLState() {
                std::ifstream f(path);
                if (!f) return;
                int init_flag = 0;
                f >> baseline >> init_flag;
                init = init_flag != 0;
            }

            void SaveRLState() const {
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

                const size_t best = ArgmaxAt(logits, step);

                if constexpr (NLL) {
                    const uint8_t true_tok = ex.second[step];
                    if (true_tok == static_cast<uint8_t>(Token::PAD)) break;
                    const auto [_, sum_exp] = SoftmaxStatsAt(logits, step);
                    total_nll -= (logits(step, true_tok) - logits(step, best)) - std::log(sum_exp);
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
                dist.true_dist[v] = n_tokens > 0 ? true_counts[v] / n_tokens : 0.0;
                dist.pred_dist[v] = n_tokens > 0 ? pred_soft_sum[v] / n_tokens : 0.0;
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
                src_f.read(reinterpret_cast<char *>(src_buf.data()), src_buf.size());
                tgt_f.read(reinterpret_cast<char *>(tgt_buf.data()), tgt_buf.size());

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
                tf_examples > 0 ? static_cast<float>(tf_loss / tf_examples) : 0.f,
                ar_tokens > 0 ? static_cast<float>(ar_nll / ar_tokens) : 0.f
            };
        }

        static void DistToJSON(const DistResult *dist, std::ofstream &f, std::string_view prefix, const int bins) {
            if (dist && dist->n_tokens > 0) {
                f << std::setprecision(8);
                f << ",\n  \"n_tokens\": " << dist->n_tokens << ",\n";

                f << "  \"token_names\": [";
                for (size_t v = 0; v < VocabSize; ++v) {
                    f << "\"" << TokenName(static_cast<Token>(v)) << "\"";
                    if (v + 1 < VocabSize) f << ", ";
                }
                f << "],\n";

                f << "  \"true_dist\": [";
                for (size_t v = 0; v < VocabSize; ++v) {
                    f << dist->true_dist[v];
                    if (v + 1 < VocabSize) f << ", ";
                }
                f << "],\n";

                f << "  \"pred_dist\": [";
                for (size_t v = 0; v < VocabSize; ++v) {
                    f << dist->pred_dist[v];
                    if (v + 1 < VocabSize) f << ", ";
                }
                f << "]";
                if (!dist->conf_hist.empty()) {
                    f << ",\n  \"" << prefix.data() << "_conf_hist\": [";
                    for (int i = 0; i < bins; ++i) {
                        f << dist->conf_hist[i];
                        if (i + 1 < bins) f << ", ";
                    }
                    f << "]";
                }
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
            f << "  \"total_seen\": " << total_seen << ",\n";
            f << "  \"lap\": " << lap << ",\n";
            f << "  \"train_loss\": " << train_loss << ",\n";
            f << "  \"test_loss\": " << test_loss << ",\n";
            f << "  \"avg_tf_pct\": " << avg_tf_pct << ",\n";
            f << "  \"avg_autoreg_pct\": " << avg_autoreg_pct << ",\n";
            f << "  \"avg_rl_reward\": " << avg_rl_reward << ",\n";
            f << "  \"avg_rl_accuracy\": " << avg_rl_accuracy << ",\n";
            f << "  \"ar_test_loss\": " << ar_test_loss << ",\n";
            f << std::setprecision(4);
            f << "  \"ss_rate\": " << ScheduledSamplingRate(total_seen) << ",\n";
            f << "  \"max_tgt_len\": " << MaxAsmLength(total_seen);
            DistToJSON(tf_dist, f, "tf", dist_bins);
            DistToJSON(ar_dist, f, "ar", dist_bins);
            f << "\n}\n";
        }


        std::pair<float, float> RL_Update(const int n_examples, const std::vector<Example> &chunk,
                                          RLState &rl_s, DataCursor &cur, std::mt19937 &ss_rng
        ) {
            using RLBatchInp = typename PrependBatch<RL_K, InputT>::type;
            using RLBatchTgt = typename PrependBatch<RL_K, OutputT>::type;

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

                std::vector<uint8_t> tgt_bytes(TgtLen, Token::PAD);
                for (size_t t = 0; t < decoded.size() && t < TgtLen; ++t) {
                    tgt_bytes[t] = static_cast<uint8_t>(decoded[t]);
                }
                const auto [rl_inp, rl_tgt] = EncodeExample({ex.first, tgt_bytes});
                TensorSet<0>(rl_x, k, rl_inp);
                TensorSet<0>(rl_y, k, rl_tgt);
            }
            const float avg_R = reward_sum / RL_K;
            const float avg_acc = accuracy_sum / RL_K;

            if (!rl_s.init) {
                rl_s.baseline = avg_R;
                rl_s.init = true;
            } else {
                rl_s.baseline = RL_BaselineDecay * rl_s.baseline + (1.f - RL_BaselineDecay) * avg_R;
            }
            rl_s.SaveRLState();

            const float advantage = avg_R - rl_s.baseline;
            const float rl_lr = rl_s.LR(cur.total_seen) * advantage;

            Network.template BatchFit<SequenceSoftmaxCEL<Token::PAD>, RL_K>(rl_x, rl_y, rl_lr);

            std::cout << "\033[2m  [rl] nll_R=" << avg_R
                    << "  acc=" << avg_acc
                    << "  base=" << rl_s.baseline
                    << "  A=" << advantage
                    << "  rl_lr=" << rl_lr << "\033[0m\n";
            return {avg_R, avg_acc};
        }
    };
}
