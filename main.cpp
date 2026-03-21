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

int main() {
    runXOR();
    runParity();
    return 0;
}
