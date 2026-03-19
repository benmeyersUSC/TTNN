#include "src/nn.hpp"
#include <iostream>
#include <array>

int main() {
    // XOR problem: 2 inputs -> 4 hidden (ReLU) -> 1 output (Sigmoid)
    NeuralNetwork<
        DenseBlock<2, 4, ActivationFunction::ReLU>,
        DenseBlock<4, 1, ActivationFunction::Sigmoid>
    > net;

    // XOR dataset
    std::array<Tensor<2>, 4> inputs = {
        Tensor<2>{0.f, 0.f},
        Tensor<2>{0.f, 1.f},
        Tensor<2>{1.f, 0.f},
        Tensor<2>{1.f, 1.f},
    };
    std::array<Tensor<1>, 4> targets = {
        Tensor<1>{0.f},
        Tensor<1>{1.f},
        Tensor<1>{1.f},
        Tensor<1>{0.f},
    };

    const float lr = 0.01f;
    const int epochs = 5000;

    for (int e = 0; e < epochs; ++e) {
        float total_loss = 0.f;
        for (int i = 0; i < 4; ++i) {
            auto out = net.Forward(inputs[i]);
            // dL/da for BCE: (out - target) / (out * (1 - out) + EPS)
            Tensor<1> grad;
            float o = out.flat(0), t = targets[i].flat(0);
            grad.flat(0) = (o - t) / (o * (1.f - o) + 1e-8f);
            net.TrainStep(inputs[i], grad, lr);
            total_loss += CrossEntropyLoss(out, targets[i]);
        }
        if (e % 1000 == 0) {
            std::cout << "Epoch " << e << "  loss=" << total_loss / 4.f << "\n";
        }
    }

    std::cout << "\nFinal predictions:\n";
    for (int i = 0; i < 4; ++i) {
        auto out = net.Forward(inputs[i]);
        std::cout << "  [" << inputs[i].flat(0) << "," << inputs[i].flat(1) << "]"
                  << " -> " << out.flat(0)
                  << "  (target=" << targets[i].flat(0) << ")\n";
    }

    return 0;
}
