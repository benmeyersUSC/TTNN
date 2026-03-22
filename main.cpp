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
        for (int i = 0; i < 4; ++i) {
            const auto out = net.Forward(inputs[i]);
            Tensor<1> grad;
            const float o = out.flat(0), t = targets[i].flat(0);
            grad.flat(0) = (o - t) / (o * (1.f - o) + EPS);
            net.TrainStep(inputs[i], grad, 0.01f);
            total_loss += CrossEntropyLoss(out, targets[i]);
        }
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
        for (int i = 0; i < 8; ++i) {
            const auto out = net.Forward(inputs[i]);
            Tensor<1> grad;
            const float o = out.flat(0), t = targets[i].flat(0);
            grad.flat(0) = (o - t) / (o * (1.f - o) + EPS);
            net.TrainStep(inputs[i], grad, 0.005f);
            total_loss += CrossEntropyLoss(out, targets[i]);
        }
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
        const auto& Y = std::get<2>(net.BatchedForwardAll<4>(X));
        Tensor<4, 1> grad;
        float total_loss = 0.f;
        for (int i = 0; i < 4; ++i) {
            const float o = Y(i,0), t = T(i,0);
            grad(i,0) = (o - t) / (o * (1.f - o) + EPS);
            total_loss -= t * std::log(std::max(o, EPS)) + (1.f-t) * std::log(std::max(1.f-o, EPS));
        }
        if (e % 100 == 0)
            std::cout << "  epoch " << e << "  loss=" << total_loss / 4.f << "\n";
        net.BatchTrainStep<4>(X, grad, 0.01f);
    }

    std::cout << "  predictions:\n";
    const auto& Y = std::get<2>(net.BatchedForwardAll<4>(X));
    for (int i = 0; i < 4; ++i)
        std::cout << "    [" << X(i,0) << "," << X(i,1) << "]"
                  << " -> " << Y(i,0) << "  (target=" << T(i,0) << ")\n";
}
// Autoencoder: 20 -> 8 -> 4 (bottleneck) -> 8 -> 20, Linear output, MSE loss.
// One training example. Watch the reconstruction loss fall epoch by epoch.
void runAutoencoder() {
    std::cout << "\n=== Autoencoder (20 -> 8 -> 4 -> 8 -> 20) ===\n";
    std::cout << "    one sample, MSE loss, watch backprop work\n\n";

    NetworkBuilder<
        Input<20>,
        Dense<8,  ActivationFunction::ReLU>,
        Dense<4,  ActivationFunction::ReLU>,   // bottleneck
        Dense<8,  ActivationFunction::ReLU>,
        Dense<20>                               // Linear output
    >::type net;
    std::cout << "    params: " << net.TotalParamCount << "\n\n";

    // one example: 20 values spread across [0,1]
    Tensor<20> x{
        0.10f, 0.90f, 0.40f, 0.70f, 0.20f,
        0.80f, 0.30f, 0.60f, 0.50f, 0.10f,
        0.95f, 0.05f, 0.65f, 0.35f, 0.75f,
        0.25f, 0.85f, 0.45f, 0.55f, 0.15f,
    };

    for (int e = 0; e < 279; ++e) {
        const auto out = net.Forward(x);

        // MSE gradient: dL/d(out_i) = 2*(out_i - x_i) / 20
        Tensor<20> grad;
        float loss = 0.f;
        for (size_t i = 0; i < 20; ++i) {
            const float diff = out.flat(i) - x.flat(i);
            loss += diff * diff;
            grad.flat(i) = 2.f * diff / 20.f;
        }
        loss /= 20.f;

        net.TrainStep(x, grad, 0.001f);

        if (e % 10 == 0)
            std::cout << "  epoch " << std::setw(4) << e
                      << "  MSE = " << std::fixed << std::setprecision(6) << loss << "\n";
    }

    std::cout << "\n  final reconstruction:\n";
    const auto out = net.Forward(x);
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "    target : ";
    for (size_t i = 0; i < 20; ++i) std::cout << x.flat(i) << " ";
    std::cout << "\n    output : ";
    for (size_t i = 0; i < 20; ++i) std::cout << out.flat(i) << " ";
    std::cout << "\n";
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

    constexpr size_t N = decltype(x)::Size;  // 512

    for (int e = 0; e < 300; ++e) {
        const auto out = net.Forward(x);

        decltype(x) grad;
        float loss = 0.f;
        for (size_t i = 0; i < N; ++i) {
            const float diff = out.flat(i) - x.flat(i);
            loss += diff * diff;
            grad.flat(i) = 2.f * diff / static_cast<float>(N);
        }
        loss /= static_cast<float>(N);

        net.TrainStep(x, grad, 0.0001f);

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
        const auto out = net.Forward(x);

        Tensor<6, 8> grad;
        float loss = 0.f;
        for (size_t i = 0; i < 48; ++i) {
            const float diff = out.flat(i) - x.flat(i);
            loss += diff * diff;
            grad.flat(i) = 2.f * diff / 48.f;
        }
        loss /= 48.f;

        net.TrainStep(x, grad, 0.001f);

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

int main() {
    runXOR();
    runParity();
    runXORFullBatch();
    runAutoencoder();
    runRankNineAutoencoder();
    runAttentionAutoencoder();
    return 0;
}
