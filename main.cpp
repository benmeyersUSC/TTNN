#include "src/TTTN.hpp"
#include <iostream>
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
int main() {
    runXOR();
    runParity();
    runXORFullBatch();
    return 0;
}
