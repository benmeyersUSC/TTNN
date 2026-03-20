#include "src/TTTN.hpp"
#include <iostream>
#include <array>

using namespace TTTN;

int main() {
    // XOR problem: 2 inputs -> 4 hidden (ReLU) -> 1 output (Sigmoid)
    TrainableTensorNetwork<
        DenseBlock<2, 4, ActivationFunction::ReLU>,
        DenseBlock<4, 1, ActivationFunction::Sigmoid>
    > net;

    // XOR dataset
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
    auto x = Tensor<1, 1, 1, 1, 1>{2.0f};
    auto y = Tensor<1, 1, 1, 1, 1>{4.0f};
    auto z = Einsum(x, y);
    constexpr int epochs = 5100;

    for (int e = 0; e < epochs; ++e) {
        float total_loss = 0.f;
        for (int i = 0; i < 4; ++i) {
            const float lr = 0.01f;
            const auto out = net.Forward(inputs[i]);

            // dL/dA for BCE: (out - target) / (out * (1 - out) + EPS)
            Tensor<1> grad;
            const float o = out.flat(0), t = targets[i].flat(0);
            grad.flat(0) = (o - t) / (o * (1.f - o) + EPS);

            net.TrainStep(inputs[i], grad, lr);
            total_loss += CrossEntropyLoss(out, targets[i]);
        }
        if (e % 1000 == 0)
            std::cout << "Epoch " << e << "  loss=" << total_loss / 4.f << "\n";
    }

    std::cout << "\nFinal predictions:\n";
    for (int i = 0; i < 4; ++i) {
        const auto out = net.Forward(inputs[i]);
        std::cout << "  [" << inputs[i].flat(0) << "," << inputs[i].flat(1) << "]"
                << " -> " << out.flat(0)
                << "  (target=" << targets[i].flat(0) << ")\n";
    }

    return 0;
}
