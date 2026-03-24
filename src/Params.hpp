#pragma once
#include <cmath>
#include <fstream>
#include <tuple>

namespace TTTN {
    // Param<TensorT>: a single trainable tensor bundled with its gradient and Adam moments.
    // Blocks declare named Param members (e.g. Param<W_Type> W_) and access W_.value / W_.grad.
    // Adam update, zero-grad, save/load are handled here — no block ever writes its own optimizer loop.
    template<typename TensorT>
    struct Param {
        TensorT value{};
        TensorT grad{};
        TensorT m{};
        TensorT v{};

        static constexpr size_t Size = TensorT::Size;

        void zero_grad() { grad.fill(0.f); }

        void update(float b1, float b2, float lr, float mCorr, float vCorr, float eps) {
            ParForEach(Size, [&](const size_t i) {
                const float g = grad.flat(i);
                m.flat(i) = b1 * m.flat(i) + (1.f - b1) * g;
                v.flat(i) = b2 * v.flat(i) + (1.f - b2) * g * g;
                value.flat(i) -= lr * (m.flat(i) * mCorr) / (std::sqrt(v.flat(i) * vCorr) + eps);
            });
        }

        void save(std::ofstream &f) const { value.Save(f); }
        void load(std::ifstream &f) { value.Load(f); }
    };

    // Bulk operations over a std::tie(...) of Param references.
    // Blocks define:  auto all_params() { return std::tie(W_, b_); }
    // Then delegate:  ZeroAllGrads(all_params());

    template<typename Tuple>
    void ZeroAllGrads(Tuple &&params) {
        std::apply([](auto &... p) { (p.zero_grad(), ...); }, params);
    }

    template<typename Tuple>
    void UpdateAll(Tuple &&params, float b1, float b2, float lr,
                   float mCorr, float vCorr, float eps = 1e-8f) {
        std::apply([&](auto &... p) { (p.update(b1, b2, lr, mCorr, vCorr, eps), ...); }, params);
    }

    template<typename Tuple>
    void SaveAll(Tuple &&params, std::ofstream &f) {
        std::apply([&](const auto &... p) { (p.save(f), ...); }, params);
    }

    template<typename Tuple>
    void LoadAll(Tuple &&params, std::ifstream &f) {
        std::apply([&](auto &... p) { (p.load(f), ...); }, params);
    }

    // Total parameter count from a pack of Param types (for static constexpr ParamCount).
    template<typename... Params>
    constexpr size_t TotalParamSize = (Params::Size + ...);
} // namespace TTTN
