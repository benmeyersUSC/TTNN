#include "src/TTTN.hpp"
#include <iostream>
#include <iomanip>
#include <array>
#include <cmath>

using namespace TTTN;

// XOR: 2 inputs, 4 hidden (ReLU), 1 output (Sigmoid)
void runXOR() {
    std::cout << "=== XOR ===\n";

    NetworkBuilder<
        Input<2>,
        Dense<4, ReLU>,
        Dense<1, Sigmoid>
    >::type net;
    std::cout << "    params: " << net.TotalParamCount << "\n\n";

    std::array inputs{
        Tensor<2>{0.f, 0.f},
        Tensor<2>{0.f, 1.f},
        Tensor<2>{1.f, 0.f},
        Tensor<2>{1.f, 1.f},
    };
    std::array targets{
        Tensor<1>{0.f},
        Tensor<1>{1.f},
        Tensor<1>{1.f},
        Tensor<1>{0.f},
    };

    for (int e = 0; e < 1000; ++e) {
        float total_loss = 0.f;
        for (int i = 0; i < 4; ++i)
            total_loss += net.Fit<BinaryCEL>(inputs[i], targets[i], 0.01f);
        if (e % 100 == 0)
            std::cout << "  epoch " << e << "  loss=" << total_loss / 4.f << "\n";
    }

    std::cout << "  predictions:\n";
    for (int i = 0; i < 4; ++i) {
        const auto out = net.Forward(inputs[i]);
        std::cout << "    [" << inputs[i].flat(0) << "," << inputs[i].flat(1) << "]"
                << " -> " << out.flat(0)
                << "  (target=" << targets[i].flat(0) << ")\n";
    }
}

// 3-bit parity: output is 1 iff an odd number of inputs are 1.
// All 8 corners of the unit cube must be classified — strictly harder than XOR.
// Architecture: 3 -> 8 (ReLU) -> 4 (ReLU) -> 1 (Sigmoid)
void runParity() {
    std::cout << "\n=== 3-bit Parity ===\n";

    NetworkBuilder<
        Input<3>,
        Dense<8, ReLU>,
        Dense<4, ReLU>,
        Dense<1, Sigmoid>
    >::type net;
    std::cout << "    params: " << net.TotalParamCount << "\n\n";

    // all 8 corners of {0,1}^3
    std::array inputs{
        Tensor<3>{0.f, 0.f, 0.f}, // 0 ones  -> 0
        Tensor<3>{0.f, 0.f, 1.f}, // 1 one   -> 1
        Tensor<3>{0.f, 1.f, 0.f}, // 1 one   -> 1
        Tensor<3>{0.f, 1.f, 1.f}, // 2 ones  -> 0
        Tensor<3>{1.f, 0.f, 0.f}, // 1 one   -> 1
        Tensor<3>{1.f, 0.f, 1.f}, // 2 ones  -> 0
        Tensor<3>{1.f, 1.f, 0.f}, // 2 ones  -> 0
        Tensor<3>{1.f, 1.f, 1.f}, // 3 ones  -> 1
    };
    std::array targets{
        Tensor<1>{0.f},
        Tensor<1>{1.f},
        Tensor<1>{1.f},
        Tensor<1>{0.f},
        Tensor<1>{1.f},
        Tensor<1>{0.f},
        Tensor<1>{0.f},
        Tensor<1>{1.f},
    };

    for (int e = 0; e < 1000; ++e) {
        float total_loss = 0.f;
        for (int i = 0; i < 8; ++i)
            total_loss += net.Fit<BinaryCEL>(inputs[i], targets[i], 0.005f);
        if (e % 100 == 0)
            std::cout << "  epoch " << e << "  loss=" << total_loss / 8.f << "\n";
    }

    std::cout << "  predictions:\n";
    for (int i = 0; i < 8; ++i) {
        const auto out = net.Forward(inputs[i]);
        std::cout << "    [" << inputs[i].flat(0) << "," << inputs[i].flat(1) << "," << inputs[i].flat(2) << "]"
                << " -> " << std::round(out.flat(0))
                << "  (target=" << targets[i].flat(0) << ")\n";
    }
}

// XOR full-batch B=4: demonstrates gradient cancellation.
// XOR's +/- targets are perfectly symmetric, so all 4 gradients sum to zero — stuck at loss=ln(2).
void runXORFullBatch() {
    std::cout << "\n=== XOR (full-batch, B=4) — expect gradient cancellation ===\n";

    NetworkBuilder<
        Input<2>,
        Dense<4, ReLU>,
        Dense<1, Sigmoid>
    >::type net;
    std::cout << "    params: " << net.TotalParamCount << "\n\n";

    Tensor<4, 2> X{0.f, 0.f, 0.f, 1.f, 1.f, 0.f, 1.f, 1.f};
    Tensor<4, 1> T{0.f, 1.f, 1.f, 0.f};

    for (int e = 0; e < 1000; ++e) {
        const float loss = net.BatchFit<BinaryCEL, 4>(X, T, 0.01f);
        if (e % 100 == 0)
            std::cout << "  epoch " << e << "  loss=" << loss << "\n";
    }

    std::cout << "  predictions:\n";
    const auto A_final = net.BatchedForwardAll<4>(X);
    const auto &Y = A_final.template get<2>();
    for (int i = 0; i < 4; ++i)
        std::cout << "    [" << X(i, 0) << "," << X(i, 1) << "]"
                << " -> " << Y(i, 0) << "  (target=" << T(i, 0) << ")\n";
}

// CombineNetworks: encoder + decoder declared separately, combined into an autoencoder.
// Encoder:     Input<16> -> Dense<8, ReLU> -> Dense<4>   (bottleneck)
// Decoder:     Input<4>  -> Dense<8, ReLU> -> Dense<16>
// Autoencoder: CombineNetworks<Encoder, Decoder>::type    (all 6 blocks, end-to-end)
//
// Demonstrates:
//   - type-level network composition with compile-time shape check
//   - training the combined network end-to-end
//   - using the encoder standalone (same type, independent weights)
void runCombineNetworks() {
    std::cout << "\n=== CombineNetworks: Encoder + Decoder -> Autoencoder ===\n";

    // --- declare sub-network types independently ---
    using Encoder = NetworkBuilder<
        Input<16>,
        Dense<8, ReLU>,
        Dense<4> // linear bottleneck
    >::type;

    using Decoder = NetworkBuilder<
        Input<4>,
        Dense<8, ReLU>,
        Dense<16> // linear reconstruction
    >::type;

    // compile-time check: Encoder::OutputTensor == Decoder::InputTensor (both Tensor<4>)
    using Autoencoder = CombineNetworks<Encoder, Decoder>::type;

    Encoder enc; // standalone encoder — own weights
    Autoencoder ae; // full autoencoder   — own weights

    std::cout << "    encoder params:     " << enc.TotalParamCount << "\n";
    std::cout << "    autoencoder params: " << ae.TotalParamCount << "\n\n";

    // four 16-dim training vectors spread across [0,1]
    std::array<Tensor<16>, 4> xs = {
        {
            {0.1f, 0.9f, 0.4f, 0.7f, 0.2f, 0.8f, 0.3f, 0.6f, 0.5f, 0.1f, 0.95f, 0.05f, 0.65f, 0.35f, 0.75f, 0.25f},
            {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f},
            {0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 0.9f, 0.8f, 0.7f, 0.6f, 0.5f},
            {1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f},
        }
    };

    // train the autoencoder end-to-end (all 6 blocks update together)
    for (int e = 0; e < 500; ++e) {
        float total = 0.f;
        for (auto &x: xs)
            total += ae.Fit<MSE>(x, x, 0.005f);
        if (e % 100 == 0)
            std::cout << "  epoch " << std::setw(4) << e
                    << "  MSE = " << std::fixed << std::setprecision(6) << total / 4.f << "\n";
    }

    // show reconstruction through the combined network
    std::cout << "\n  autoencoder reconstruction (first sample):\n";
    std::cout << std::fixed << std::setprecision(2);
    const auto recon = ae.Forward(xs[0]);
    std::cout << "    input  : ";
    for (size_t i = 0; i < 16; ++i) std::cout << xs[0].flat(i) << " ";
    std::cout << "\n";
    std::cout << "    output : ";
    for (size_t i = 0; i < 16; ++i) std::cout << recon.flat(i) << " ";
    std::cout << "\n";

    // use the standalone encoder independently — its weights are separate from ae
    // train it briefly on the same inputs (identity target = embedding of itself)
    std::cout << "\n  encoder standalone: bottleneck embeddings (4-dim) after short solo training:\n";
    for (int e = 0; e < 200; ++e)
        for (auto &x: xs)
            enc.Fit<MSE>(x, {0.25f, 0.5f, 0.75f, 1.0f}, 0.005f); // arbitrary target to show it trains
    for (size_t s = 0; s < 4; ++s) {
        const auto emb = enc.Forward(xs[s]);
        std::cout << "    sample " << s << " -> [";
        for (size_t i = 0; i < 4; ++i)
            std::cout << std::setprecision(3) << emb.flat(i) << (i < 3 ? ", " : "");
        std::cout << "]\n";
    }
}

// Rank-5 tensor autoencoder.
// Input lives in Tensor<2,3,2,2,2> (48 elements). The network compresses it through
// rank-2 bottleneck tensors, then reconstructs the original rank-5 shape.
// Shape-as-type: every layer boundary is a distinct tensor type, checked at compile time.
// Rank-9 tensor autoencoder.
// Input lives in Tensor<2,2,2,2,2,2,2,2,2> = 2^9 = 512 elements.
// Compressed through rank-3 then rank-2 intermediates, reconstructed back to rank-9.
void runRankNineAutoencoder() {
    std::cout << "\n=== Rank-9 Tensor Autoencoder ===\n";
    std::cout <<
            "    Tensor<2^9>(512) -> Tensor<4,4,2>(32) -> Tensor<4,2>(8) -> Tensor<4,4,2>(32) -> Tensor<2^9>(512)\n";

    NetworkBuilder<
        Input<2, 2, 2, 2, 2, 2, 2, 2, 2>,
        DenseMD<Tensor<4, 4, 2>, ReLU>, // 512 -> 32, rank-3
        DenseMD<Tensor<4, 2>, ReLU>, // 32  ->  8, rank-2 bottleneck
        DenseMD<Tensor<1, 2, 2, 1>, ReLU>,
        DenseMD<Tensor<4, 2>, ReLU>,
        DenseMD<Tensor<4, 4, 2>, ReLU>, // 8   -> 32, rank-3
        DenseMD<Tensor<2, 2, 2, 2, 2, 2, 2, 2, 2> > // 32  -> 512, back to rank-9
    >::type net;
    std::cout << "    params: " << net.TotalParamCount << "\n";
    std::cout << "    one sample, MSE loss, watch backprop trickle through\n\n";

    // 512 values in [0,1] generated from a sine wave — varied but deterministic
    Tensor<2, 2, 2, 2, 2, 2, 2, 2, 2> x;
    for (size_t i = 0; i < 512; ++i)
        x.flat(i) = 0.5f + 0.45f * std::sin(static_cast<float>(i) * 3.14159f / 32.f);


    for (int e = 0; e < 300; ++e) {
        const float loss = net.Fit<MSE>(x, x, 0.0001f);
        if (e % 10 == 0)
            std::cout << "  epoch " << std::setw(4) << e
                    << "  MSE = " << std::fixed << std::setprecision(6) << loss << "\n";
    }

    std::cout << "\n  final reconstruction (flat):\n";
    const auto out = net.Forward(x);
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "    target : ";
    for (size_t i = 0; i < 27; ++i) std::cout << x.flat(i) << " ";
    std::cout << "...\n    output : ";
    for (size_t i = 0; i < 27; ++i) std::cout << out.flat(i) << " ";
    std::cout << "...\n";
}

// Transformer autoencoder: attention + per-token FFN (MapDense), the real transformer block.
// ComposeBlocks vanishes at build time — the network is a flat chain of concrete blocks.
// With MapDense providing per-token nonlinear capacity, the model can actually learn identity.
void runAttentionAutoencoder() {
    std::cout << "\n=== Transformer Autoencoder (seq=8, emb=6x6, 2 blocks: attn+FFN) ===\n";

    // Define a transformer block: attention + expand/project FFN
    using TBlock = ComposeBlocks<
        MHAttention<4, 6, 6>,
        MapDense<1, Tensor<72>, ReLU>, // per-token FFN: 36 → 72
        MapDense<1, Tensor<6, 6> > // per-token FFN: 72 → 36
    >;

    NetworkBuilder<
        Input<8, 6, 6>,
        TBlock,
        TBlock
    >::type net;
    std::cout << "    params: " << net.TotalParamCount << "\n\n";

    // 8 tokens × 6×6 embedding, values from a sine wave
    Tensor<8, 6, 6> x;
    for (size_t i = 0; i < 288; ++i)
        x.flat(i) = 0.5f + 0.4f * std::sin(static_cast<float>(i) * 3.14159f / 8.f);

    for (int e = 0; e < 3000; ++e) {
        const float loss = net.Fit<MSE>(x, x, 0.003f);
        if (e % 100 == 0)
            std::cout << "  epoch " << std::setw(4) << e
                    << "  MSE = " << std::fixed << std::setprecision(6) << loss << "\n";
    }

    const auto out = net.Forward(x);
    std::cout << "\n  target : ";
    for (size_t i = 0; i < 36; ++i)
        std::cout << std::fixed << std::setprecision(2) << x.flat(i) << " ";
    std::cout << "...\n  output : ";
    for (size_t i = 0; i < 36; ++i)
        std::cout << std::fixed << std::setprecision(2) << out.flat(i) << " ";
    std::cout << "...\n";
}

// MNIST with transformer: attention + per-token FFN + Dense classifier.
// ComposeBlocks defines a reusable transformer block that flattens at build time.
void runMNISTAttention() {
    std::cout << "\n=== MNIST Transformer (28 tokens x 28 dims, attn+FFN -> Dense -> Softmax+CEL) ===\n";

    auto train_data = LoadCSV<60000, 785>("mnist_train.csv", true);
    auto test_data = LoadCSV<10000, 785>("mnist_test.csv", true);

    using TBlock = ComposeBlocks<
        MHAttention<4, 28>,
        MapDense<1, Tensor<28>, ReLU>, // per-row FFN: 28 → 28
        MapDense<1, Tensor<28> > // per-row FFN: 28 → 28
    >;

    // TBlock flattens to 3 concrete blocks; total = 6 blocks + Dense + Softmax = 8
    typename NetworkBuilder<
        Input<28, 28>,
        TBlock,
        Dense<10>,
        SoftmaxLayer<0>
    >::type net;
    std::cout << "    params: " << net.TotalParamCount << "\n\n";

    std::mt19937 rng{42};
    constexpr size_t Batch = 32;
    constexpr size_t EvalN = 500;
    // NumBlocks after flattening: MHAttention, MapDense, MapDense, Dense, Softmax = 5
    constexpr size_t FinalIdx = 5;

    auto prep = [](const auto &batch, Tensor<Batch, 28, 28> &X, Tensor<Batch, 10> &Y) {
        for (size_t b = 0; b < Batch; ++b) {
            const auto label = static_cast<size_t>(batch(b, 0));
            for (size_t p = 0; p < 784; ++p)
                X.flat(b * 784 + p) = batch(b, p + 1) / 255.f;
            for (size_t c = 0; c < 10; ++c)
                Y(b, c) = (c == label) ? 1.f : 0.f;
        }
    };

    auto sample_acc = [&](const auto &dataset) -> float {
        auto eval = RandomBatch<EvalN>(dataset, rng);
        Tensor<EvalN, 28, 28> X_eval;
        Tensor<EvalN, 10> Y_eval;
        for (size_t b = 0; b < EvalN; ++b) {
            const auto label = static_cast<size_t>(eval(b, 0));
            for (size_t p = 0; p < 784; ++p)
                X_eval.flat(b * 784 + p) = eval(b, p + 1) / 255.f;
            for (size_t c = 0; c < 10; ++c)
                Y_eval(b, c) = (c == label) ? 1.f : 0.f;
        }
        const auto A = net.template BatchedForwardAll<EvalN>(X_eval);
        const auto &pred = A.template get<FinalIdx>();
        return BatchAccuracy(pred, Y_eval);
    };

    for (int epoch = 0; epoch < 3; ++epoch) {
        const float acc_before = sample_acc(train_data);
        const float avg_loss = RunEpoch<CEL, Batch>(net, train_data, rng, 0.001f, prep);
        const float acc_after = sample_acc(train_data);
        std::cout << "  after epoch " << std::setw(2) << epoch
                << "  CEL=" << std::fixed << std::setprecision(4) << avg_loss
                << "  train: " << std::setprecision(1)
                << acc_before << "% -> " << acc_after
                << "%\n";
    }

    // test accuracy
    auto raw = RandomBatch<EvalN>(test_data, rng);
    Tensor<EvalN, 28, 28> X_test;
    Tensor<EvalN, 10> Y_test;
    for (size_t b = 0; b < EvalN; ++b) {
        const auto label = static_cast<size_t>(raw(b, 0));
        for (size_t p = 0; p < 784; ++p)
            X_test.flat(b * 784 + p) = raw(b, p + 1) / 255.f;
        for (size_t c = 0; c < 10; ++c)
            Y_test(b, c) = (c == label) ? 1.f : 0.f;
    }
    const auto A_test = net.template BatchedForwardAll<EvalN>(X_test);
    const auto &pred_test = A_test.template get<FinalIdx>();
    std::cout << "\n  test accuracy (" << EvalN << " held-out): "
            // << std::fixed << std::setprecision(1) << BatchAccuracy(pred_test, Y_test) << "%\n"
            ;
}

// Generic CSV classifier: col 0 = integer class label, cols 1..Cols-1 = features.
// Network is hardcoded: Input<Cols-1> -> Dense<128,ReLU> -> Dense<64,ReLU> -> Dense<NumClasses> -> Softmax
// Features are normalized by `norm` (default 255 for pixel data).
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

    auto prep = [norm](const auto &batch, Tensor<Batch, Features> &X, Tensor<Batch, NumClasses> &Y) {
        for (size_t b = 0; b < Batch; ++b) {
            const auto label = static_cast<size_t>(batch(b, 0));
            for (size_t p = 0; p < Features; ++p) X(b, p) = batch(b, p + 1) / norm;
            for (size_t c = 0; c < NumClasses; ++c) Y(b, c) = (c == label) ? 1.f : 0.f;
        }
    };

    auto sample_acc = [&](const auto &dataset) -> float {
        auto eval = RandomBatch<EvalN>(dataset, rng);
        Tensor<EvalN, Features> X_eval;
        Tensor<EvalN, NumClasses> Y_eval;
        for (size_t b = 0; b < EvalN; ++b) {
            const auto label = static_cast<size_t>(eval(b, 0));
            for (size_t p = 0; p < Features; ++p) X_eval(b, p) = eval(b, p + 1) / norm;
            for (size_t c = 0; c < NumClasses; ++c) Y_eval(b, c) = (c == label) ? 1.f : 0.f;
        }
        const auto A = net.template BatchedForwardAll<EvalN>(X_eval);
        const auto &pred = A.template get<4>();
        return BatchAccuracy(pred, Y_eval);
    };

    for (int epoch = 0; epoch < epochs; ++epoch) {
        const float acc_before = sample_acc(train_data);
        const float avg_loss = RunEpoch<CEL, Batch>(net, train_data, rng, lr, prep);
        const float acc_after = sample_acc(train_data);
        std::cout << "  after epoch " << std::setw(2) << epoch
                << "  CEL=" << std::fixed << std::setprecision(4) << avg_loss
                << "  train: " << std::setprecision(1) << acc_before << "% -> " << acc_after << "%\n";
    }

    // test accuracy via BatchAccuracy (ReduceSum<1>(pred ⊙ Y) vs ReduceMax<1>(pred))
    auto raw = RandomBatch<TestBatch>(test_data, rng);
    Tensor<TestBatch, Features> X_test;
    Tensor<TestBatch, NumClasses> Y_test;
    for (size_t b = 0; b < TestBatch; ++b) {
        const auto label = static_cast<size_t>(raw(b, 0));
        for (size_t p = 0; p < Features; ++p) X_test(b, p) = raw(b, p + 1) / norm;
        for (size_t c = 0; c < NumClasses; ++c) Y_test(b, c) = (c == label) ? 1.f : 0.f;
    }
    const auto A_test = net.template BatchedForwardAll<TestBatch>(X_test);
    const auto &pred_batch = A_test.template get<4>();
    std::cout << "\n  test accuracy (" << TestBatch << " held-out): "
            << std::fixed << std::setprecision(1) << BatchAccuracy(pred_batch, Y_test) << "%\n";
}

int main() {
    // runXOR();
    // runParity();
    // runXORFullBatch();
    // runCombineNetworks();
    // runRankNineAutoencoder();
    // runAttentionAutoencoder();
    // ── MNIST-family: all 785-col CSVs, col 0 = label, cols 1-784 = pixels 0-255 ──────────────


    RunCSVClassifier<60000, 10000, 785, 10>("MNIST", "mnist_train.csv", "mnist_test.csv", 0.0001f);
    RunCSVClassifier<60000, 10000, 785, 10>("Fashion-MNIST", "fashion_mnist_train.csv", "fashion_mnist_test.csv",
                                            0.0001f);
    runMNISTAttention();

    // RunCSVClassifier<60000,  10000,  785, 10>("KMNIST",        "kmnist_train.csv",        "kmnist_test.csv",        0.0001f); // Kuzushiji (Japanese cursive), ~93%
    // RunCSVClassifier<27455,   7172,  785, 24>("Sign MNIST",    "sign_mnist_train.csv",    "sign_mnist_test.csv",    0.0001f); // ASL A-Z (no J/Z), ~90%
    // RunCSVClassifier<112800, 18800,  785, 47>("EMNIST",        "emnist_train.csv",        "emnist_test.csv",        0.0001f); // letters+digits, 47 classes, ~85%
    return 0;
}
