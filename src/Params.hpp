#pragma once
#include <cmath>
#include <fstream>
#include <tuple>

namespace TTTN {
    // AdamState: all Adam hyperparameters + per-network bias-correction state in one place.
    // TTN owns one instance; passes it (const ref) to UpdateAll each step.
    struct AdamState {
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float eps   = 1e-8f;
        float mCorr = 1.f;    // 1 / (1 - β1^t)
        float vCorr = 1.f;    // 1 / (1 - β2^t)
        int   t     = 0;

        // Advance timestep and recompute bias corrections. Call once per Update().
        // @doc: void step()
        /** Increments `t`, recomputes `mCorr = 1/(1-β1^t)` and `vCorr = 1/(1-β2^t)`. Call once per `Update()`. */
        void step() {
            ++t;
            mCorr = 1.f / (1.f - std::pow(beta1, static_cast<float>(t)));
            vCorr = 1.f / (1.f - std::pow(beta2, static_cast<float>(t)));
        }
    };

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

        // @doc: void zero_grad()
        /** Zeroes `grad` — called by `ZeroAllGrads` at the start of each training step */
        void zero_grad() { grad.fill(0.f); }

        // @doc: void update(const AdamState& adam, float lr)
        /** One Adam step: updates `m`, `v`, then applies bias-corrected weight update to `value` */
        void update(const AdamState &adam, float lr) {
            ParForEach(Size, [&](const size_t i) {
                const float g = grad.flat(i);
                m.flat(i) = adam.beta1 * m.flat(i) + (1.f - adam.beta1) * g;
                v.flat(i) = adam.beta2 * v.flat(i) + (1.f - adam.beta2) * g * g;
                value.flat(i) -= lr * (m.flat(i) * adam.mCorr) / (std::sqrt(v.flat(i) * adam.vCorr) + adam.eps);
            });
        }

        // @doc: void save(std::ofstream& f) const
        /** Serializes `value` to binary file */
        void save(std::ofstream &f) const { value.Save(f); }
        // @doc: void load(std::ifstream& f)
        /** Deserializes `value` from binary file */
        void load(std::ifstream &f) { value.Load(f); }
    };

    // Bulk operations over a std::tie(...) of Param references.
    // Blocks define:  auto all_params() { return std::tie(W_, b_); }
    // Then delegate:  ZeroAllGrads(all_params());

    // @doc: template<typename Tuple> void ZeroAllGrads(Tuple&& params)
    /** Calls `zero_grad()` on every `Param` in the tuple */
    template<typename Tuple>
    void ZeroAllGrads(Tuple &&params) {
        std::apply([](auto &... p) { (p.zero_grad(), ...); }, params);
    }

    // @doc: template<typename Tuple> void UpdateAll(Tuple&& params, const AdamState& adam, float lr)
    /** Calls `update(adam, lr)` on every `Param` in the tuple */
    template<typename Tuple>
    void UpdateAll(Tuple &&params, const AdamState &adam, float lr) {
        std::apply([&](auto &... p) { (p.update(adam, lr), ...); }, params);
    }

    // @doc: template<typename Tuple> void SaveAll(Tuple&& params, std::ofstream& f)
    /** Calls `save(f)` on every `Param` in the tuple */
    template<typename Tuple>
    void SaveAll(Tuple &&params, std::ofstream &f) {
        std::apply([&](const auto &... p) { (p.save(f), ...); }, params);
    }

    // @doc: template<typename Tuple> void LoadAll(Tuple&& params, std::ifstream& f)
    /** Calls `load(f)` on every `Param` in the tuple */
    template<typename Tuple>
    void LoadAll(Tuple &&params, std::ifstream &f) {
        std::apply([&](auto &... p) { (p.load(f), ...); }, params);
    }

    // Total parameter count from a pack of Param types (for static constexpr ParamCount).
    template<typename... Params>
    constexpr size_t TotalParamSize = (Params::Size + ...);

    // TupleParamCount: sum ::Size across every Param in an all_params() tuple.
    // Handles reference element types (std::tie returns tuple<Param<T>&, ...>).
    // Returns 0 for std::tuple<> — parameter-free blocks.
    template<typename Tuple, size_t... Is>
    constexpr size_t tuple_param_count_impl(std::index_sequence<Is...>) {
        return (size_t(0) + ... + std::remove_reference_t<std::tuple_element_t<Is, Tuple>>::Size);
    }
    template<typename Tuple>
    constexpr size_t TupleParamCount =
        tuple_param_count_impl<std::remove_cvref_t<Tuple>>(
            std::make_index_sequence<std::tuple_size_v<std::remove_cvref_t<Tuple>>>{});
} // namespace TTTN
