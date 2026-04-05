#include "src/TTTN.hpp"
#include <iostream>
#include <iomanip>
#include <array>
#include <algorithm>
#include <cmath>
#include <filesystem>

using namespace TTTN;

// ── Image primitives ──────────────────────────────────────────────────────────

struct RGB {
    uint8_t r, g, b;
};

static RGB grayRGB(float v) {
    const uint8_t g = static_cast<uint8_t>(std::clamp(v, 0.f, 1.f) * 255.f);
    return {g, g, g};
}

static RGB hotRGB(float v) {
    v = std::clamp(v, 0.f, 1.f);
    return {
        static_cast<uint8_t>(std::min(v * 3.f, 1.f) * 255),
        static_cast<uint8_t>(std::max(v * 3.f - 1.f, 0.f) * 255),
        static_cast<uint8_t>(std::max(v * 3.f - 2.f, 0.f) * 255)
    };
}

// ── Generic SnapshotMap visualizer ───────────────────────────────────────────
//
// Renders every rank-2 or rank-3 entry in a SnapshotMap as a hot-colormap PPM.
// Rank-3 (e.g. Tensor<H,S,S>): H slices placed side by side.
// Output: viz/<key_with_dots_replaced_by_underscores>.ppm

static void viz_entry(const std::string &path, const SnapshotEntry &e, size_t scale) {
    if (e.shape.size() < 2) return;
    const size_t n_sl = e.shape.size() == 2 ? 1 : e.shape[0];
    const size_t rows = e.shape[e.shape.size() - 2];
    const size_t cols = e.shape[e.shape.size() - 1];
    const size_t sl_sz = rows * cols;

    float mx = 0.f;
    for (float v: e.data) mx = std::max(mx, v);

    const size_t gap = std::max(scale / 2, size_t(1));
    const size_t img_w = n_sl * cols * scale + (n_sl > 1 ? (n_sl - 1) * gap : 0);
    const size_t img_h = rows * scale;

    std::ofstream f(path, std::ios::binary);
    f << "P6\n" << img_w << " " << img_h << "\n255\n";
    auto put = [&](RGB c) {
        f.put(c.r);
        f.put(c.g);
        f.put(c.b);
    };
    const RGB dark{20, 20, 20};

    for (size_t r = 0; r < rows; ++r)
        for (size_t py = 0; py < scale; ++py)
            for (size_t sl = 0; sl < n_sl; ++sl) {
                for (size_t c = 0; c < cols; ++c) {
                    float v = mx > 1e-6f ? e.data[sl * sl_sz + r * cols + c] / mx : 0.f;
                    for (size_t px = 0; px < scale; ++px) put(hotRGB(v));
                }
                if (sl + 1 < n_sl)
                    for (size_t px = 0; px < gap; ++px) put(dark);
            }
}

void viz_snapshot(const std::string &dir, const SnapshotMap &snaps, size_t scale = 8) {
    std::filesystem::create_directories(dir);
    for (const auto &[key, entry]: snaps) {
        if (entry.shape.size() < 2) continue;
        std::string fname = key;
        for (char &c: fname) if (c == '.') c = '_';
        const std::string path = dir + "/" + fname + "ppm";
        viz_entry(path, entry, scale);
        std::cout << "  viz -> " << path << "\n";
        std::system(("open " + path).c_str());
    }
}

// ── MNIST attention visualizer ────────────────────────────────────────────────
//
// Checkerboard row+col attention on 28×28 images.
// Terminal: digit | row-attn | col-attn  (unicode shading, 28 rows).
// Image:    viz/attn_viz.ppm  (grayscale digit + hot attn panels, stacked).

template<size_t H, size_t S>
static Tensor<S, S> headAvgNorm(const Tensor<H, S, S> &t) {
    Tensor<S, S> out;
    const float inv = 1.f / static_cast<float>(H);
    for (size_t h = 0; h < H; ++h)
        for (size_t q = 0; q < S; ++q)
            for (size_t k = 0; k < S; ++k)
                out(q, k) += t(h, q, k) * inv;
    float mx = 0.f;
    for (size_t i = 0; i < S * S; ++i) mx = std::max(mx, out.flat(i));
    if (mx > 1e-6f) for (size_t i = 0; i < S * S; ++i) out.flat(i) /= mx;
    return out;
}

static void printAttnViz(const Tensor<28, 28> &digit,
                         const Tensor<28, 28> &row_attn,
                         const Tensor<28, 28> &col_attn) {
    static constexpr const char *shade[] = {" ", "░", "▒", "▓", "█"};
    auto cell = [](float v) { return shade[static_cast<size_t>(std::clamp(v, 0.f, 1.f) * 4.f)]; };
    std::cout << "  digit                        "
            << "row-attn (row→row)           "
            << "col-attn (col→col)\n";
    for (size_t r = 0; r < 28; ++r) {
        std::cout << "  ";
        for (size_t c = 0; c < 28; ++c) std::cout << cell(digit(r, c));
        std::cout << "   ";
        for (size_t c = 0; c < 28; ++c) std::cout << cell(row_attn(r, c));
        std::cout << "   ";
        for (size_t c = 0; c < 28; ++c) std::cout << cell(col_attn(r, c));
        std::cout << "\n";
    }
}

struct AttnSample {
    size_t label, pred;
    Tensor<28, 28> digit, row_attn, col_attn;
};

static void writeAndOpenAttnPPM(const std::string &path,
                                const std::vector<AttnSample> &samples,
                                size_t scale = 8) {
    const size_t cell = 28 * scale;
    const size_t gap = scale;
    const size_t sep = scale / 2;
    const size_t img_w = cell * 3 + gap * 2;
    const size_t n = samples.size();
    const size_t img_h = n * cell + (n > 1 ? (n - 1) * sep : 0);

    std::ofstream f(path, std::ios::binary);
    f << "P6\n" << img_w << " " << img_h << "\n255\n";
    auto put = [&](RGB c) {
        f.put(c.r);
        f.put(c.g);
        f.put(c.b);
    };
    const RGB dark{20, 20, 20};

    for (size_t si = 0; si < n; ++si) {
        const auto &s = samples[si];
        for (size_t py = 0; py < cell; ++py) {
            const size_t r = py / scale;
            for (size_t px = 0; px < cell; ++px) put(grayRGB(s.digit(r, px / scale)));
            for (size_t px = 0; px < gap; ++px) put(dark);
            for (size_t px = 0; px < cell; ++px) put(hotRGB(s.row_attn(r, px / scale)));
            for (size_t px = 0; px < gap; ++px) put(dark);
            for (size_t px = 0; px < cell; ++px) put(hotRGB(s.col_attn(r, px / scale)));
        }
        if (si + 1 < n)
            for (size_t py = 0; py < sep; ++py)
                for (size_t px = 0; px < img_w; ++px) put(dark);
    }
    std::cout << "  image -> " << path << "  (" << img_w << "x" << img_h << " px)\n";
    std::system(("open " + path).c_str());
}

template<typename Net>
static AttnSample extractAttnSample(const Net &net, const Tensor<28, 28> &img, size_t label) {
    const auto pred_out = net.Forward(img);
    size_t pred = 0;
    for (size_t c = 1; c < 10; ++c)
        if (pred_out.flat(c) > pred_out.flat(pred)) pred = c;

    const auto &inner_par = net.template block<0>().block_a();
    const auto &row_mhattn = inner_par.block_a().template block<0>();
    const auto &col_mhattn = inner_par.block_b().inner().template block<0>();

    return {
        label, pred, img,
        headAvgNorm(row_mhattn.attn_weights()),
        headAvgNorm(col_mhattn.attn_weights())
    };
}

void runMNISTAttnViz() {
    std::cout << "\n=== MNIST Attention Visualizer (checkerboard row+col) ===\n";

    auto train_data = LoadCSV<20000, 785>("data/mnist_train.csv", true);

    using RowAttn = ComposeBlocks<
        MHAttention<7, 28>,
        MapDense<1, Tensor<28>, ReLU>,
        MapDense<1, Tensor<28> >
    >;
    using ColAttn = Transposed<ComposeBlocks<
        MHAttention<7, 28>,
        MapDense<1, Tensor<28>, ReLU>,
        MapDense<1, Tensor<28> >
    > >;

    typename NetworkBuilder<
        Input<28, 28>,
        Residual<Parallel<RowAttn, ColAttn> >,
        Dense<10>,
        SoftmaxLayer<0>
    >::type net;
    std::cout << "    params: " << net.TotalParamCount << "\n\n";

    std::mt19937 rng{42};
    constexpr size_t Batch = 64;

    Tensor<20000, 28, 28> X_train;
    Tensor<20000, 10> Y_train;
    for (size_t i = 0; i < 20000; ++i) {
        for (size_t p = 0; p < 784; ++p)
            X_train.flat(i * 784 + p) = train_data(i, p + 1) / 255.f;
        const auto label = static_cast<size_t>(train_data(i, 0));
        for (size_t c = 0; c < 10; ++c)
            Y_train(i, c) = (c == label) ? 1.f : 0.f;
    }

    for (int epoch = 0; epoch < 10; ++epoch) {
        const float loss = net.template RunEpoch<CEL, Batch>(X_train, Y_train, rng, 0.001f);
        std::cout << "  epoch " << epoch << "  CEL=" << std::fixed << std::setprecision(4) << loss << "\n";
    }

    std::cout << "\n  -- attention patterns (one sample per digit) --\n";
    std::vector<AttnSample> ppm_samples;
    std::array<int, 10> shown{};
    shown.fill(0);

    for (size_t i = 0; i < 2000 && *std::min_element(shown.begin(), shown.end()) == 0; ++i) {
        const auto label = static_cast<size_t>(train_data(i, 0));
        if (shown[label]) continue;
        shown[label] = 1;

        Tensor<28, 28> img;
        for (size_t p = 0; p < 784; ++p)
            img.flat(p) = train_data(i, p + 1) / 255.f;

        const auto s = extractAttnSample(net, img, label);
        std::cout << "\n  [digit " << s.label << "  pred=" << s.pred << "]\n\n";
        printAttnViz(s.digit, s.row_attn, s.col_attn);
        ppm_samples.push_back(s);
    }

    std::sort(ppm_samples.begin(), ppm_samples.end(),
              [](const AttnSample &a, const AttnSample &b) { return a.label < b.label; });

    std::filesystem::create_directories("viz");
    writeAndOpenAttnPPM("viz/attn_viz.ppm", ppm_samples);
}

// ── Bracket attention visualizer ──────────────────────────────────────────────
//
// Token vocab: 0=(  1=)  2=[  3=]  4={  5=}  6=blank
// Network: Input<32,7> → embed → MHAttn(4 heads) → FFN → Dense<2> → Softmax
// snap() collects "block_1.attn_weights" (shape [4,32,32]) after each Forward.
// PPM: 4 files (one per head) with token-color labeled query/key axes.

static constexpr const char *BRACKET_CHARS = "()[]{} ";

static const RGB BRACKET_COLORS[7] = {
    {100, 149, 237}, // ( cornflower blue
    {135, 206, 250}, // ) sky blue
    {60, 179, 113}, // [ medium sea green
    {144, 238, 144}, // ] light green
    {255, 165, 0}, // { orange
    {220, 20, 60}, // } crimson
    {25, 25, 25}, // blank
};

// 5-wide × 7-tall pixel bitmaps for ()[]{} and blank.
// Each row is 5 bits; bit 4 = leftmost pixel (col 0).
static const uint8_t BRACKET_FONT[7][7] = {
    {0x04, 0x08, 0x10, 0x10, 0x10, 0x08, 0x04}, // (
    {0x04, 0x02, 0x01, 0x01, 0x01, 0x02, 0x04}, // )
    {0x0C, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0C}, // [
    {0x06, 0x02, 0x02, 0x02, 0x02, 0x02, 0x06}, // ]
    {0x06, 0x08, 0x08, 0x10, 0x08, 0x08, 0x06}, // {
    {0x0C, 0x02, 0x02, 0x01, 0x02, 0x02, 0x0C}, // }
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, // blank
};

struct BracketSample {
    size_t label, pred;
    std::array<int, 32> tokens;
    SnapshotMap snaps; // from net.snap() — contains "block_1.attn_weights"
};

static void printBracketsViz(const BracketSample &s) {
    static constexpr const char *shade[] = {" ", "░", "▒", "▓", "█"};
    auto cell = [](float v) { return shade[static_cast<size_t>(std::clamp(v, 0.f, 1.f) * 4.f)]; };

    std::cout << "  seq: ";
    for (int tok: s.tokens) std::cout << BRACKET_CHARS[tok];
    std::cout << "\n  label=" << s.label << "  pred=" << s.pred
            << "  (" << (s.label == s.pred ? "correct" : "WRONG") << ")\n\n";

    const auto &e = s.snaps.at("block_1.attn_weights");
    // shape [4, 32, 32] — normalize each head, show all 4 side by side
    std::array<std::array<float, 32 * 32>, 4> norm;
    for (size_t h = 0; h < 4; ++h) {
        float mx = 0.f;
        for (size_t i = 0; i < 32 * 32; ++i) mx = std::max(mx, e.data[h * 32 * 32 + i]);
        for (size_t i = 0; i < 32 * 32; ++i)
            norm[h][i] = mx > 1e-6f ? e.data[h * 32 * 32 + i] / mx : 0.f;
    }

    std::cout << "  head-0                          "
            << "head-1                          "
            << "head-2                          "
            << "head-3\n";
    for (size_t q = 0; q < 32; ++q) {
        std::cout << "  ";
        for (size_t h = 0; h < 4; ++h) {
            for (size_t k = 0; k < 32; ++k) std::cout << cell(norm[h][q * 32 + k]);
            if (h < 3) std::cout << "  ";
        }
        std::cout << "\n";
    }
}

// PPM: 4 separate files (one per head) in `dir/`.
// Layout: [corner | key-token labels →] / [query-token labels ↓ | heatmap]
// Samples stacked vertically with a dark separator.
static void writeBracketsAttnPPM(const std::string &dir,
                                 const std::vector<BracketSample> &samples,
                                 size_t scale = 16) {
    std::filesystem::create_directories(dir);

    static constexpr size_t FW = 5, FH = 7;
    const RGB dark{20, 20, 20};

    auto draw_cell = [&](std::ofstream &f, int tok, size_t py_in_cell) {
        auto put = [&](RGB c) {
            f.put(c.r);
            f.put(c.g);
            f.put(c.b);
        };
        RGB bg = BRACKET_COLORS[tok];
        int cy = (int) py_in_cell - (int) (scale - FH) / 2;
        for (size_t px = 0; px < scale; ++px) {
            int cx = (int) px - (int) (scale - FW) / 2;
            bool lit = cx >= 0 && cx < (int) FW && cy >= 0 && cy < (int) FH
                       && ((BRACKET_FONT[tok][cy] >> (FW - 1 - cx)) & 1);
            put(lit ? RGB{255, 255, 255} : bg);
        }
    };

    const size_t n = samples.size();
    const size_t sep = scale / 2;
    const size_t img_w = scale + 32 * scale;
    const size_t block_h = scale + 32 * scale;
    const size_t img_h = n * block_h + (n > 1 ? (n - 1) * sep : 0);

    for (size_t head = 0; head < 4; ++head) {
        const std::string path = dir + "/head_" + std::to_string(head) + ".ppm";
        std::ofstream f(path, std::ios::binary);
        f << "P6\n" << img_w << " " << img_h << "\n255\n";
        auto put = [&](RGB c) {
            f.put(c.r);
            f.put(c.g);
            f.put(c.b);
        };

        for (size_t si = 0; si < n; ++si) {
            const auto &s = samples[si];
            const auto &e = s.snaps.at("block_1.attn_weights");
            const size_t off = head * 32 * 32;

            // normalize this head
            float mx = 0.f;
            for (size_t i = 0; i < 32 * 32; ++i) mx = std::max(mx, e.data[off + i]);
            auto val = [&](size_t q, size_t k) {
                return mx > 1e-6f ? e.data[off + q * 32 + k] / mx : 0.f;
            };

            // top margin: corner + key token labels
            for (size_t py = 0; py < scale; ++py) {
                for (size_t px = 0; px < scale; ++px) put(dark);
                for (size_t k = 0; k < 32; ++k) draw_cell(f, s.tokens[k], py);
            }

            // 32 query rows
            for (size_t q = 0; q < 32; ++q)
                for (size_t py = 0; py < scale; ++py) {
                    draw_cell(f, s.tokens[q], py);
                    for (size_t k = 0; k < 32; ++k)
                        for (size_t px = 0; px < scale; ++px)
                            put(hotRGB(val(q, k)));
                }

            if (si + 1 < n)
                for (size_t py = 0; py < sep; ++py)
                    for (size_t px = 0; px < img_w; ++px) put(dark);
        }

        std::cout << "  image -> " << path << "  (" << img_w << "x" << img_h << " px)\n";
        std::system(("open " + path).c_str());
    }
}

template<typename Net>
static BracketSample extractBracketSample(const Net &net,
                                          const Tensor<32, 7> &x,
                                          size_t label,
                                          const std::array<int, 32> &tokens) {
    const auto pred_out = net.Forward(x);
    const size_t pred = pred_out.flat(1) > pred_out.flat(0) ? 1u : 0u;
    return {label, pred, tokens, net.snap()};
}

// ── Numeric sequence attention visualizer ─────────────────────────────────────
//
// Shared transformer for 4 sequence tasks (sorted / palindrome / modular / parity).
// Token vocab: 0-6 = integer values, 7 = blank.
// Network: Input<32,8> → embed → MHAttn(4 heads) → FFN → Dense<2> → Softmax
// Same CSV shape as brackets (label, t0..t31); blank=7 for all tasks.

static const RGB NUM_COLORS[8] = {
    {70, 130, 200}, // 0 blue
    {60, 179, 113}, // 1 green
    {210, 80, 80}, // 2 red
    {220, 180, 50}, // 3 yellow
    {160, 60, 200}, // 4 purple
    {50, 200, 200}, // 5 cyan
    {230, 140, 50}, // 6 orange
    {25, 25, 25}, // 7 blank (dark)
};

// 5-wide × 7-tall pixel bitmaps for digits 0-6 and blank.
// Each row is 5 bits; bit 4 = leftmost pixel (col 0).
static const uint8_t NUM_FONT[8][7] = {
    {0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E}, // 0
    {0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E}, // 1
    {0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F}, // 2
    {0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E}, // 3
    {0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02}, // 4
    {0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E}, // 5
    {0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E}, // 6
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, // 7 blank
};

// PPM: 4 files (one per head). Token label row/col = solid color square per value.
static void writeSeqAttnPPM(const std::string &dir,
                            const std::vector<BracketSample> &samples,
                            size_t scale = 16) {
    std::filesystem::create_directories(dir);
    const RGB dark{20, 20, 20};
    const size_t n = samples.size();
    const size_t sep = scale / 2;
    const size_t img_w = scale + 32 * scale;
    const size_t block_h = scale + 32 * scale;
    const size_t img_h = n * block_h + (n > 1 ? (n - 1) * sep : 0);

    const size_t img_sq = scale + 32 * scale;

    // digit font helper: is pixel (px, py) within cell lit for this token?
    auto font_pixel = [&](int tok, size_t px, size_t py) -> bool {
        const int cx = static_cast<int>(px) - (static_cast<int>(scale) - 5) / 2;
        const int cy = static_cast<int>(py) - (static_cast<int>(scale) - 7) / 2;
        if (cx < 0 || cx >= 5 || cy < 0 || cy >= 7) return false;
        return (NUM_FONT[tok][cy] >> (4 - cx)) & 1;
    };

    for (size_t si = 0; si < n; ++si) {
        const auto &s = samples[si];
        const auto &e = s.snaps.at("block_1.attn_weights");

        const std::string path = dir + "/attn_" + std::to_string(si) + "_lbl" +
                                 std::to_string(s.label) + ".ppm";
        std::ofstream f(path, std::ios::binary);
        f << "P6\n" << img_sq << " " << img_sq << "\n255\n";
        auto put = [&](RGB c) {
            f.put(c.r);
            f.put(c.g);
            f.put(c.b);
        };
        auto put_tok = [&](int tok, size_t px, size_t py) {
            put(font_pixel(tok, px, py) ? RGB{255, 255, 255} : NUM_COLORS[tok]);
        };

        // average attention weights across all 4 heads, then renorm
        std::array<float, 32 * 32> avg{};
        for (size_t h = 0; h < 4; ++h)
            for (size_t i = 0; i < 32 * 32; ++i)
                avg[i] += e.data[h * 32 * 32 + i];
        float mx = *std::max_element(avg.begin(), avg.end());
        auto val = [&](size_t q, size_t k) {
            return mx > 1e-6f ? avg[q * 32 + k] / mx : 0.f;
        };

        // top row: dark corner + key-token color squares with digit overlay
        for (size_t py = 0; py < scale; ++py) {
            for (size_t px = 0; px < scale; ++px) put(dark);
            for (size_t k = 0; k < 32; ++k)
                for (size_t px = 0; px < scale; ++px) put_tok(s.tokens[k], px, py);
        }
        // 32 query rows: query token cell + heatmap
        for (size_t q = 0; q < 32; ++q)
            for (size_t py = 0; py < scale; ++py) {
                for (size_t px = 0; px < scale; ++px) put_tok(s.tokens[q], px, py);
                for (size_t k = 0; k < 32; ++k)
                    for (size_t px = 0; px < scale; ++px)
                        put(hotRGB(val(q, k)));
            }

        std::cout << "  -> " << path << "\n";
        std::system(("open " + path).c_str());
    }
}

void runSeqTasksViz() {
    std::cout << "\n=== Sequence Tasks (sorted / palindrome / modular / parity) ===\n";

    // One architecture for all 4 tasks — vocab size 8 (0-6 = values, 7 = blank)
    using SeqNet = typename NetworkBuilder<
        Input<32, 8>,
        MapDense<1, Tensor<32> >,
        MHAttention<4, 32>,
        MapDense<1, Tensor<64>, ReLU>,
        MapDense<1, Tensor<32> >,
        Dense<2>,
        SoftmaxLayer<0>
    >::type;

    constexpr size_t TrainN = 10000, TestN = 1000, Cols = 33;
    constexpr size_t Batch = 32;
    constexpr size_t EvalN = 500;
    constexpr int Epochs = 18;
    constexpr float LR = 0.0005f;

    auto make_input = [](const auto &ds, size_t row) {
        Tensor<32, 8> x;
        for (size_t t = 0; t < 32; ++t) {
            const auto tok = static_cast<size_t>(ds(row, t + 1));
            for (size_t c = 0; c < 8; ++c) x(t, c) = (c == tok) ? 1.f : 0.f;
        }
        return x;
    };


    struct Task {
        std::string name, train_csv, test_csv, viz_dir;
    };
    const std::array<Task, 4> tasks = {
        {
            {"Sorted", "data/sorted_train.csv", "data/sorted_test.csv", "viz/sorted"},
            {"Palindrome", "data/reverse_train.csv", "data/reverse_test.csv", "viz/reverse"},
            {"Modular-7", "data/modular_train.csv", "data/modular_test.csv", "viz/modular"},
            {"Parity", "data/parity_train.csv", "data/parity_test.csv", "viz/parity"},
        }
    };

    for (const auto &task: tasks) {
        std::cout << "\n--- " << task.name << " ---\n";
        auto train_data = LoadCSV<TrainN, Cols>(task.train_csv, true);
        auto test_data = LoadCSV<TestN, Cols>(task.test_csv, true);

        Tensor<TrainN, 32, 8> X_train;
        Tensor<TrainN, 2> Y_train;
        for (size_t i = 0; i < TrainN; ++i) {
            const auto label = static_cast<size_t>(train_data(i, 0));
            for (size_t t = 0; t < 32; ++t) {
                const auto tok = static_cast<size_t>(train_data(i, t + 1));
                for (size_t c = 0; c < 8; ++c) X_train(i, t, c) = (c == tok) ? 1.f : 0.f;
            }
            for (size_t c = 0; c < 2; ++c) Y_train(i, c) = (c == label) ? 1.f : 0.f;
        }

        SeqNet net;
        std::mt19937 rng{42};
        std::cout << "    params: " << net.TotalParamCount << "\n\n";

        auto sample_acc = [&](const auto &dataset) {
            auto eval = RandomBatch<EvalN>(dataset, rng);
            Tensor<EvalN, 32, 8> Xe;
            Tensor<EvalN, 2> Ye;
            for (size_t b = 0; b < EvalN; ++b) {
                const auto label = static_cast<size_t>(eval(b, 0));
                for (size_t t = 0; t < 32; ++t) {
                    const auto tok = static_cast<size_t>(eval(b, t + 1));
                    for (size_t c = 0; c < 8; ++c) Xe(b, t, c) = (c == tok) ? 1.f : 0.f;
                }
                for (size_t c = 0; c < 2; ++c) Ye(b, c) = (c == label) ? 1.f : 0.f;
            }
            const auto A = net.template BatchedForwardAll<EvalN>(Xe);
            return BatchAccuracy(A.template get<6>(), Ye);
        };

        for (int epoch = 0; epoch < Epochs; ++epoch) {
            const float bef = sample_acc(train_data);
            const float loss = net.template RunEpoch<CEL, Batch>(X_train, Y_train, rng, LR);
            const float aft = sample_acc(train_data);
            std::cout << "  epoch " << std::setw(2) << epoch
                    << "  CEL=" << std::fixed << std::setprecision(4) << loss
                    << "  train: " << std::setprecision(1) << bef << "% -> " << aft << "%\n";
        }

        {
            auto raw = RandomBatch<1000>(test_data, rng);
            Tensor<1000, 32, 8> Xt;
            Tensor<1000, 2> Yt;
            for (size_t b = 0; b < 1000; ++b) {
                const auto label = static_cast<size_t>(raw(b, 0));
                for (size_t t = 0; t < 32; ++t) {
                    const auto tok = static_cast<size_t>(raw(b, t + 1));
                    for (size_t c = 0; c < 8; ++c) Xt(b, t, c) = (c == tok) ? 1.f : 0.f;
                }
                for (size_t c = 0; c < 2; ++c) Yt(b, c) = (c == label) ? 1.f : 0.f;
            }
            const auto At = net.template BatchedForwardAll<1000>(Xt);
            std::cout << "\n  test accuracy: " << std::setprecision(1)
                    << BatchAccuracy(At.template get<6>(), Yt) << "%\n";
        }

        // pick top-3 longest sequences per class for richer attention patterns
        constexpr int NPick = 3;
        struct Candidate {
            size_t row;
            int nonblank;
        };
        std::vector<Candidate> cand0, cand1;
        for (size_t i = 0; i < TrainN; ++i) {
            const auto lbl = static_cast<size_t>(train_data(i, 0));
            int nb = 0;
            for (size_t t = 0; t < 32; ++t)
                if (static_cast<int>(train_data(i, t + 1)) != 7) ++nb;
            if (lbl == 1) cand1.push_back({i, nb});
            else cand0.push_back({i, nb});
        }
        auto by_len = [](const Candidate &a, const Candidate &b) { return a.nonblank > b.nonblank; };
        std::sort(cand0.begin(), cand0.end(), by_len);
        std::sort(cand1.begin(), cand1.end(), by_len);

        std::vector<BracketSample> ppm_samples;
        auto pick = [&](const std::vector<Candidate> &cands, size_t lbl) {
            const int n = std::min(NPick, (int) cands.size());
            for (int k = 0; k < n; ++k) {
                const size_t i = cands[k].row;
                std::array<int, 32> tokens;
                for (size_t t = 0; t < 32; ++t)
                    tokens[t] = static_cast<int>(train_data(i, t + 1));
                const auto pred_out = net.Forward(make_input(train_data, i));
                const size_t pred = pred_out.flat(1) > pred_out.flat(0) ? 1u : 0u;
                ppm_samples.push_back({lbl, pred, tokens, net.Snap()});
            }
        };
        pick(cand1, 1);
        pick(cand0, 0);

        std::cout << "\n  -- attention patterns --\n";
        writeSeqAttnPPM(task.viz_dir, ppm_samples);
    }
}

// ── Generic CSV classifier ────────────────────────────────────────────────────
//
// col 0 = integer class label, cols 1..Cols-1 = features.
// Network: Input<Features> → Dense<128,ReLU> → Dense<64,ReLU> → Dense<NumClasses> → Softmax

template<size_t TrainRows, size_t TestRows, size_t Cols, size_t NumClasses,
    size_t Batch = 32, size_t EvalN = 1000, size_t TestBatch = 1000>
void RunCSVClassifier(const std::string &name,
                      const std::string &train_csv,
                      const std::string &test_csv,
                      float lr = 0.001f,
                      int epochs = 5,
                      float norm = 255.f,
                      bool skip_hdr = true) {
    constexpr size_t Features = Cols - 1;

    std::cout << "\n=== " << name << " ("
            << Features << " -> 128 -> 64 -> " << NumClasses << ", Softmax+CEL) ===\n";

    auto train_data = LoadCSV<TrainRows, Cols>(train_csv, skip_hdr);
    auto test_data = LoadCSV<TestRows, Cols>(test_csv, skip_hdr);

    typename NetworkBuilder<
        Input<Features>,
        Dense<128, ReLU>,
        Dense<64, ReLU>,
        Dense<NumClasses>,
        SoftmaxLayer<0>
    >::type net;
    std::cout << "    params: " << net.TotalParamCount << "\n\n";

    std::mt19937 rng{42};

    Tensor<TrainRows, Features> X_train;
    Tensor<TrainRows, NumClasses> Y_train;
    for (size_t i = 0; i < TrainRows; ++i) {
        const auto label = static_cast<size_t>(train_data(i, 0));
        for (size_t p = 0; p < Features; ++p) X_train(i, p) = train_data(i, p + 1) / norm;
        for (size_t c = 0; c < NumClasses; ++c) Y_train(i, c) = (c == label) ? 1.f : 0.f;
    }

    auto sample_acc = [&](const auto &dataset) {
        auto eval = RandomBatch<EvalN>(dataset, rng);
        Tensor<EvalN, Features> Xe;
        Tensor<EvalN, NumClasses> Ye;
        for (size_t b = 0; b < EvalN; ++b) {
            const auto label = static_cast<size_t>(eval(b, 0));
            for (size_t p = 0; p < Features; ++p) Xe(b, p) = eval(b, p + 1) / norm;
            for (size_t c = 0; c < NumClasses; ++c) Ye(b, c) = (c == label) ? 1.f : 0.f;
        }
        return BatchAccuracy(net.template BatchedForwardAll<EvalN>(Xe).template get<4>(), Ye);
    };

    for (int epoch = 0; epoch < epochs; ++epoch) {
        const float bef = sample_acc(train_data);
        const float loss = net.template RunEpoch<CEL, Batch>(X_train, Y_train, rng, lr);
        const float aft = sample_acc(train_data);
        std::cout << "  epoch " << std::setw(2) << epoch
                << "  CEL=" << std::fixed << std::setprecision(4) << loss
                << "  train: " << std::setprecision(1) << bef << "% -> " << aft << "%\n";
    }

    auto raw = RandomBatch<TestBatch>(test_data, rng);
    Tensor<TestBatch, Features> Xt;
    Tensor<TestBatch, NumClasses> Yt;
    for (size_t b = 0; b < TestBatch; ++b) {
        const auto label = static_cast<size_t>(raw(b, 0));
        for (size_t p = 0; p < Features; ++p) Xt(b, p) = raw(b, p + 1) / norm;
        for (size_t c = 0; c < NumClasses; ++c) Yt(b, c) = (c == label) ? 1.f : 0.f;
    }
    std::cout << "\n  test accuracy (" << TestBatch << " held-out): "
            << std::fixed << std::setprecision(1)
            << BatchAccuracy(net.template BatchedForwardAll<TestBatch>(Xt).template get<4>(), Yt) << "%\n";
}

// ─────────────────────────────────────────────────────────────────────────────

void runLCSmoke() {
    std::cout << "\n=== LearnedContraction smoke test ===\n";

    using Net = typename NetworkBuilder<
        Input<128>,
        Dense<256, ReLU>,
        Dense<256, ReLU>,
        Dense<256, ReLU>,
        Dense<256, ReLU>,
        Dense<256, ReLU>,
        Dense<256, ReLU>,
        Dense<256, ReLU>,
        Dense<256, ReLU>,
        Dense<4>
    >::type;

    Net net;
    constexpr size_t Batch = 4;

    Tensor<Batch, 128> X;
    for (size_t i = 0; i < Tensor<Batch, 128>::Size; ++i) X.flat(i) = static_cast<float>(i % 5) * 0.1f;
    Tensor<Batch, 4> target;
    target.fill(0.f);

    for (int step = 0; step < 100; ++step) {
        const float loss = net.template BatchFit<MSE, Batch>(X, target, 0.001f);
        std::cout << "  step " << step << "  loss=" << loss << "\n";
    }
}

void runMNISTDense() {
    std::cout << "\n=== MNIST Dense (LearnedContraction) ===\n";

    auto train_data = LoadCSV<20000, 785>("data/mnist_train.csv", true);
    auto test_data = LoadCSV<2000, 785>("data/mnist_test.csv", true);

    typename NetworkBuilder<
        Input<784>,
        Dense<256, ReLU>,
        Dense<64, ReLU>,
        Dense<10>,
        SoftmaxLayer<0>
    >::type net;
    std::cout << "    params: " << net.TotalParamCount << "\n\n";

    constexpr size_t TrainN = 20000, TestN = 2000;
    Tensor<TrainN, 784> X_train;
    Tensor<TrainN, 10> Y_train;
    for (size_t i = 0; i < TrainN; ++i) {
        for (size_t p = 0; p < 784; ++p)
            X_train.flat(i * 784 + p) = train_data(i, p + 1) / 255.f;
        const auto label = static_cast<size_t>(train_data(i, 0));
        for (size_t c = 0; c < 10; ++c)
            Y_train(i, c) = (c == label) ? 1.f : 0.f;
    }
    Tensor<TestN, 784> X_test;
    Tensor<TestN, 10> Y_test;
    for (size_t i = 0; i < TestN; ++i) {
        for (size_t p = 0; p < 784; ++p)
            X_test.flat(i * 784 + p) = test_data(i, p + 1) / 255.f;
        const auto label = static_cast<size_t>(test_data(i, 0));
        for (size_t c = 0; c < 10; ++c)
            Y_test(i, c) = (c == label) ? 1.f : 0.f;
    }

    std::mt19937 rng{42};
    constexpr size_t Batch = 64;

    for (int epoch = 0; epoch < 10; ++epoch) {
        const float loss = net.template RunEpoch<CEL, Batch>(X_train, Y_train, rng, 0.001f);

        // accuracy on test set
        size_t correct = 0;
        for (size_t i = 0; i < TestN; ++i) {
            Tensor<784> x;
            for (size_t p = 0; p < 784; ++p) x.flat(p) = X_test.flat(i * 784 + p);
            const auto out = net.Forward(x);
            size_t pred = 0;
            for (size_t c = 1; c < 10; ++c) if (out.flat(c) > out.flat(pred)) pred = c;
            const auto label = static_cast<size_t>(test_data(i, 0));
            if (pred == label) ++correct;
        }
        std::cout << "  epoch " << epoch
                << "  CEL=" << std::fixed << std::setprecision(4) << loss
                << "  acc=" << std::fixed << std::setprecision(1)
                << (100.f * correct / TestN) << "%\n";
    }
}

int main() {
    // runMNISTAttnViz();
    // runBracketsAttnViz();
    runSeqTasksViz();
    // runLCSmoke();
    runMNISTDense();
    return 0;
}
