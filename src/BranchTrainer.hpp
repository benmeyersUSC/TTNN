#pragma once
#include <array>
#include <tuple>
#include "TrainableTensorNetwork.hpp"
#include "TTTN_ML.hpp"

namespace TTTN {

    // @doc: struct NoLoss
    struct NoLoss {};

    // @doc: template<size_t TapIndex, typename HeadNet> struct HeadBranch
    template<size_t TapIndex, typename HeadNet>
    struct HeadBranch {
        // @doc: static constexpr size_t HeadBranch::Tap
        static constexpr size_t Tap = TapIndex;
        // @doc: using HeadBranch::Net
        using Net = HeadNet;
        // @doc: HeadNet HeadBranch::net
        HeadNet net;
    };

    // @doc: template<typename TrunkNet, typename... Heads> class BranchTrainer
    template<typename TrunkNet, typename... Heads>
    class BranchTrainer {
        static constexpr size_t NumHeads = sizeof...(Heads);

        using InputTensor       = typename TrunkNet::InputTensor;
        using TrunkOutputTensor = typename TrunkNet::OutputTensor;
        using TrunkActivations  = typename TrunkNet::Activations;

        // Taps must be non-decreasing
        static constexpr bool check_taps() {
            if constexpr (NumHeads <= 1) return true;
            std::array<size_t, NumHeads> t = {Heads::Tap...};
            for (size_t i = 1; i < NumHeads; ++i)
                if (t[i] < t[i - 1]) return false;
            return true;
        }
        static_assert(check_taps(), "HeadBranch tap indices must be non-decreasing");

        // Each head's InputTensor must match the trunk activation at its tap
        static constexpr bool check_head_types() {
            return []<size_t... Is>(std::index_sequence<Is...>) {
                return (std::is_same_v<
                    typename std::tuple_element_t<Is, std::tuple<Heads...>>::Net::InputTensor,
                    std::remove_cvref_t<decltype(
                        std::declval<TrunkActivations>().template get<
                            std::tuple_element_t<Is, std::tuple<Heads...>>::Tap>()
                    )>
                > && ...);
            }(std::index_sequence_for<Heads...>{});
        }
        static_assert(check_head_types(),
                      "Each head's InputTensor must match the trunk activation at its tap index");

        // @doc: TrunkNet BranchTrainer::trunk_
        TrunkNet trunk_;
        // @doc: std::tuple<Heads...> BranchTrainer::heads_
        std::tuple<Heads...> heads_;

        // Forward all heads from their tap activations
        template<size_t... Is>
        auto forward_heads(const TrunkActivations &A, std::index_sequence<Is...>) const {
            return std::make_tuple(
                std::get<Is>(heads_).net.ForwardAll(
                    A.template get<std::tuple_element_t<Is, std::tuple<Heads...>>::Tap>()
                )...
            );
        }

        // Recursive backward: process heads from I=NumHeads-1 down to 0,
        // chaining BackwardRange calls through trunk segments.
        // PrevTap: activation index we're currently at in the trunk.
        template<typename HeadLossesTuple, size_t I, size_t PrevTap,
                 typename GradT, typename HeadActsTuple, typename HeadTargetsTuple>
        void backward_heads(const TrunkActivations &A,
                            const GradT &grad,
                            HeadActsTuple &head_acts,
                            const HeadTargetsTuple &head_targets) {
            using HeadI    = std::tuple_element_t<I, std::tuple<Heads...>>;
            using HeadLossI = std::tuple_element_t<I, HeadLossesTuple>;
            constexpr size_t tapI = HeadI::Tap;

            // Backward trunk segment [tapI, PrevTap]
            auto grad_at_tap = trunk_.template BackwardRange<tapI, PrevTap>(A, grad);

            // Sum in head I's input gradient if it has a loss
            if constexpr (!std::is_same_v<HeadLossI, NoLoss>) {
                const auto &head_A  = std::get<I>(head_acts);
                const auto &head_out = head_A.template get<HeadI::Net::NumBlocks>();
                const auto head_grad = HeadLossI::Grad(head_out, std::get<I>(head_targets));
                grad_at_tap += std::get<I>(heads_).net.template BackwardRange<0, HeadI::Net::NumBlocks>(
                    head_A, head_grad);
            }

            if constexpr (I > 0) {
                backward_heads<HeadLossesTuple, I - 1, tapI>(A, grad_at_tap, head_acts, head_targets);
            } else {
                // Final trunk prefix segment [0, tapI]
                trunk_.template BackwardRange<0, tapI>(A, grad_at_tap);
            }
        }

    public:
        // @doc: TrunkNet &BranchTrainer::trunk()
        TrunkNet       &trunk()       { return trunk_; }
        const TrunkNet &trunk() const { return trunk_; }

        // @doc: template<size_t I> auto &BranchTrainer::head()
        template<size_t I> auto       &head()       { return std::get<I>(heads_); }
        template<size_t I> const auto &head() const { return std::get<I>(heads_); }

        // @doc: template<typename TrunkLoss, typename... HeadLosses> float BranchTrainer::Fit(const InputTensor &x, const TrunkOutputTensor &trunk_target, float trunk_lr, const std::tuple<typename Heads::Net::OutputTensor...> &head_targets, const std::array<float, NumHeads> &head_lrs)
        template<typename TrunkLoss, typename... HeadLosses>
            requires (sizeof...(HeadLosses) == NumHeads)
        float Fit(const InputTensor &x,
                  const TrunkOutputTensor &trunk_target,
                  float trunk_lr,
                  const std::tuple<typename Heads::Net::OutputTensor...> &head_targets,
                  const std::array<float, NumHeads> &head_lrs) {
            // Forward
            const auto A    = trunk_.ForwardAll(x);
            auto head_As    = forward_heads(A, std::index_sequence_for<Heads...>{});

            // ZeroGrad
            trunk_.ZeroGrad();
            [&]<size_t... Is>(std::index_sequence<Is...>) {
                (std::get<Is>(heads_).net.ZeroGrad(), ...);
            }(std::index_sequence_for<Heads...>{});

            // Compute total loss
            float total_loss = 0.f;
            if constexpr (!std::is_same_v<TrunkLoss, NoLoss>)
                total_loss += TrunkLoss::Loss(A.template get<TrunkNet::NumBlocks>(), trunk_target);
            [&]<size_t... Is>(std::index_sequence<Is...>) {
                ([&] {
                    using HL = std::tuple_element_t<Is, std::tuple<HeadLosses...>>;
                    if constexpr (!std::is_same_v<HL, NoLoss>) {
                        using HeadI = std::tuple_element_t<Is, std::tuple<Heads...>>;
                        total_loss += HL::Loss(
                            std::get<Is>(head_As).template get<HeadI::Net::NumBlocks>(),
                            std::get<Is>(head_targets));
                    }
                }(), ...);
            }(std::index_sequence_for<Heads...>{});

            // Backward: initial grad at trunk output (zero if no trunk loss)
            TrunkOutputTensor trunk_grad{};
            if constexpr (!std::is_same_v<TrunkLoss, NoLoss>)
                trunk_grad = TrunkLoss::Grad(A.template get<TrunkNet::NumBlocks>(), trunk_target);

            backward_heads<std::tuple<HeadLosses...>, NumHeads - 1, TrunkNet::NumBlocks>(
                A, trunk_grad, head_As, head_targets);

            // Update — each network steps its own Adam
            trunk_.Update(trunk_lr);
            [&]<size_t... Is>(std::index_sequence<Is...>) {
                (std::get<Is>(heads_).net.Update(head_lrs[Is]), ...);
            }(std::index_sequence_for<Heads...>{});

            return total_loss;
        }
    };

} // namespace TTTN
