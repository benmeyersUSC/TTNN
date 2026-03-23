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
        Dense<4, ActivationFunction::ReLU>,
        Dense<1, ActivationFunction::Sigmoid>
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
        Dense<8, ActivationFunction::ReLU>,
        Dense<4, ActivationFunction::ReLU>,
        Dense<1, ActivationFunction::Sigmoid>
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
        Dense<4, ActivationFunction::ReLU>,
        Dense<1, ActivationFunction::Sigmoid>
    >::type net;
    std::cout << "    params: " << net.TotalParamCount << "\n\n";

    Tensor<4, 2> X{0.f,0.f, 0.f,1.f, 1.f,0.f, 1.f,1.f};
    Tensor<4, 1> T{0.f, 1.f, 1.f, 0.f};

    for (int e = 0; e < 1000; ++e) {
        const float loss = net.BatchFit<BinaryCEL, 4>(X, T, 0.01f);
        if (e % 100 == 0)
            std::cout << "  epoch " << e << "  loss=" << loss << "\n";
    }

    std::cout << "  predictions:\n";
    const auto A_final = net.BatchedForwardAll<4>(X);
    const auto& Y = A_final.template get<2>();
    for (int i = 0; i < 4; ++i)
        std::cout << "    [" << X(i,0) << "," << X(i,1) << "]"
                  << " -> " << Y(i,0) << "  (target=" << T(i,0) << ")\n";
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
        Dense<8, ActivationFunction::ReLU>,
        Dense<4>          // linear bottleneck
    >::type;

    using Decoder = NetworkBuilder<
        Input<4>,
        Dense<8, ActivationFunction::ReLU>,
        Dense<16>         // linear reconstruction
    >::type;

    // compile-time check: Encoder::OutputTensor == Decoder::InputTensor (both Tensor<4>)
    using Autoencoder = CombineNetworks<Encoder, Decoder>::type;

    Encoder     enc;   // standalone encoder — own weights
    Autoencoder ae;    // full autoencoder   — own weights

    std::cout << "    encoder params:     " << enc.TotalParamCount << "\n";
    std::cout << "    autoencoder params: " << ae.TotalParamCount  << "\n\n";

    // four 16-dim training vectors spread across [0,1]
    std::array<Tensor<16>, 4> xs = {{
        {0.1f,0.9f,0.4f,0.7f,0.2f,0.8f,0.3f,0.6f,0.5f,0.1f,0.95f,0.05f,0.65f,0.35f,0.75f,0.25f},
        {0.5f,0.5f,0.5f,0.5f,0.5f,0.5f,0.5f,0.5f,0.5f,0.5f,0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f},
        {0.0f,0.1f,0.2f,0.3f,0.4f,0.5f,0.6f,0.7f,0.8f,0.9f,1.0f, 0.9f, 0.8f, 0.7f, 0.6f, 0.5f},
        {1.0f,0.0f,1.0f,0.0f,1.0f,0.0f,1.0f,0.0f,1.0f,0.0f,1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f},
    }};

    // train the autoencoder end-to-end (all 6 blocks update together)
    for (int e = 0; e < 500; ++e) {
        float total = 0.f;
        for (auto& x : xs)
            total += ae.Fit<MSE>(x, x, 0.005f);
        if (e % 100 == 0)
            std::cout << "  epoch " << std::setw(4) << e
                      << "  MSE = " << std::fixed << std::setprecision(6) << total / 4.f << "\n";
    }

    // show reconstruction through the combined network
    std::cout << "\n  autoencoder reconstruction (first sample):\n";
    std::cout << std::fixed << std::setprecision(2);
    const auto recon = ae.Forward(xs[0]);
    std::cout << "    input  : "; for (size_t i = 0; i < 16; ++i) std::cout << xs[0].flat(i) << " "; std::cout << "\n";
    std::cout << "    output : "; for (size_t i = 0; i < 16; ++i) std::cout << recon.flat(i)  << " "; std::cout << "\n";

    // use the standalone encoder independently — its weights are separate from ae
    // train it briefly on the same inputs (identity target = embedding of itself)
    std::cout << "\n  encoder standalone: bottleneck embeddings (4-dim) after short solo training:\n";
    for (int e = 0; e < 200; ++e)
        for (auto& x : xs)
            enc.Fit<MSE>(x, {0.25f, 0.5f, 0.75f, 1.0f}, 0.005f);  // arbitrary target to show it trains
    for (size_t s = 0; s < 4; ++s) {
        const auto emb = enc.Forward(xs[s]);
        std::cout << "    sample " << s << " -> [";
        for (size_t i = 0; i < 4; ++i)
            std::cout << std::setprecision(3) << emb.flat(i) << (i<3?", ":"");
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
    std::cout << "    Tensor<2^9>(512) -> Tensor<4,4,2>(32) -> Tensor<4,2>(8) -> Tensor<4,4,2>(32) -> Tensor<2^9>(512)\n";

    NetworkBuilder<
        Input<2,2,2,2,2,2,2,2,2>,
        DenseMD<Tensor<4,4,2>, ActivationFunction::ReLU>,   // 512 -> 32, rank-3
        DenseMD<Tensor<4,2>,   ActivationFunction::ReLU>,   // 32  ->  8, rank-2 bottleneck
        DenseMD<Tensor<1, 2, 2, 1>,   ActivationFunction::ReLU>,
        DenseMD<Tensor<4,2>,   ActivationFunction::ReLU>,
        DenseMD<Tensor<4,4,2>, ActivationFunction::ReLU>,   // 8   -> 32, rank-3
        DenseMD<Tensor<2,2,2,2,2,2,2,2,2>>                  // 32  -> 512, back to rank-9
    >::type net;
    std::cout << "    params: " << net.TotalParamCount << "\n";
    std::cout << "    one sample, MSE loss, watch backprop trickle through\n\n";

    // 512 values in [0,1] generated from a sine wave — varied but deterministic
    Tensor<2,2,2,2,2,2,2,2,2> x;
    for (size_t i = 0; i < 512; ++i)
        x.flat(i) = 0.5f + 0.45f * std::sin(static_cast<float>(i) * 3.14159f / 32.f);

    // constexpr size_t N = decltype(x)::Size;  // 512

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

// Attention autoencoder: 6 tokens, embedding dim 8, 2 heads of dim 4.
// Single MHAttention layer learns to reconstruct a sine-wave sequence via MSE.
// Cross-token mixing is the only tool available — no dense projection, no skip.
void runAttentionAutoencoder() {
    std::cout << "\n=== Attention Autoencoder (seq=6, emb=8, heads=2) ===\n";

    NetworkBuilder<
        Input<6, 8>,
        MHAttention<2, 8>
    >::type net;
    std::cout << "    params: " << net.TotalParamCount << "\n\n";

    // 6 tokens × 8 dims, values from a sine wave
    Tensor<6, 8> x;
    for (size_t i = 0; i < 48; ++i)
        x.flat(i) = 0.5f + 0.4f * std::sin(static_cast<float>(i) * 3.14159f / 8.f);

    for (int e = 0; e < 500; ++e) {
        const float loss = net.Fit<MSE>(x, x, 0.001f);
        if (e % 50 == 0)
            std::cout << "  epoch " << std::setw(4) << e
                      << "  MSE = " << std::fixed << std::setprecision(6) << loss << "\n";
    }

    const auto out = net.Forward(x);
    std::cout << "\n  target : ";
    for (size_t i = 0; i < 8; ++i)
        std::cout << std::fixed << std::setprecision(2) << x.flat(i) << " ";
    std::cout << "...\n  output : ";
    for (size_t i = 0; i < 8; ++i)
        std::cout << std::fixed << std::setprecision(2) << out.flat(i) << " ";
    std::cout << "...\n";
}

// MNIST: 784 inputs -> 128 (ReLU) -> 64 (ReLU) -> 10 (Softmax), trained with CEL.
// Expects mnist_train.csv in the working directory:
//   format: label,pixel0,pixel1,...,pixel783  (header row, 60000 data rows)
//   download from: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
void runMNIST() {
    std::cout << "\n=== MNIST (784 -> 128 -> 64 -> 10 -> Softmax, CEL) ===\n";

    // Shape stated at compile time — the type IS the schema.
    auto train_data = LoadCSV<60000, 785>("mnist_train.csv", /*skip_header=*/true);

    NetworkBuilder<
        Input<784>,
        Dense<128, ActivationFunction::ReLU>,
        Dense<64, ActivationFunction::ReLU>,
        Dense<10>,                  // linear logits
        SoftmaxLayer<0>             // axis-0 softmax as its own block
    >::type net;
    std::cout << "    params: " << net.TotalParamCount << "\n\n";

    std::mt19937 rng{42};
    constexpr size_t Batch = 64;
    constexpr int    Steps = 200;

    for (int epoch = 0; epoch < 10; ++epoch) {
        std::cout << "  epoch " << epoch;
        float total_loss = 0.f;
        int   correct    = 0;

        for (int step = 0; step < Steps; ++step) {
            auto batch = RandomBatch<Batch>(train_data, rng);

            Tensor<Batch, 784> X;
            Tensor<Batch, 10>  Y;
            for (size_t b = 0; b < Batch; ++b) {
                const auto label = static_cast<size_t>(batch(b, 0));
                for (size_t p = 0; p < 784; ++p)
                    X(b, p) = batch(b, p + 1) / 255.f;
                for (size_t c = 0; c < 10; ++c)
                    Y(b, c) = (c == label) ? 1.f : 0.f;
            }

            // BatchFit returns pre-update avg loss — no manual CEL call needed
            total_loss += net.BatchFit<CEL, Batch>(X, Y, 0.001f);

            // accuracy from post-update predictions
            // const auto A = net.BatchedForwardAll<Batch>(X);
            const auto preds = net.BatchedForward<Batch>(X);
            // const auto& preds = A.template get<>();   // after SoftmaxLayer
            for (size_t b = 0; b < Batch; ++b) {
                size_t pred_label = 0; float best = preds(b, 0);
                for (size_t i = 1; i < 10; ++i)
                    if (preds(b, i) > best) { best = preds(b, i); pred_label = i; }
                if (pred_label == static_cast<size_t>(batch(b, 0))) ++correct;
            }
        }

        const float n = static_cast<float>(Steps * Batch);
        std::cout << "  avg CEL = " << std::fixed << std::setprecision(4) << total_loss / static_cast<float>(Steps)
                  << "  acc = " << std::setprecision(1) << 100.f * static_cast<float>(correct) / n << "%\n";
    }
}

int main() {
    runXOR();
    runParity();
    runXORFullBatch();
    runCombineNetworks();
    runRankNineAutoencoder();
    runAttentionAutoencoder();
    runMNIST();   // uncomment after placing mnist_train.csv in the working directory
    return 0;
}
