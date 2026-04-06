#pragma once
#include <array>
#include <tuple>
#include "TrainableTensorNetwork.hpp"
#include "TTTN_ML.hpp"

namespace TTTN {
    // @doc: struct NoLoss
    /** Sentinel type. Pass as `TrunkLoss` or any `HeadLosses...` entry to suppress loss computation and gradient contribution for that branch. */
    struct NoLoss {
    };

    // forward declaration for is_head_branch_v specialization below
    template<size_t TapIndex, typename HeadNet, typename HeadLoss>
        requires IsTrainableNetwork<HeadNet> &&
                 (std::is_same_v<HeadLoss, NoLoss> || LossFunction<HeadLoss, typename HeadNet::OutputTensor>)
    struct HeadBranch;

    // @doc: template<typename T> inline constexpr bool is_head_branch_v
    /** Type trait to SFINAE-match only `HeadBranch` objects */
    template<typename T>
    inline constexpr bool is_head_branch_v = false;

    template<size_t TapIndex, typename HeadNet, typename HeadLoss>
    inline constexpr bool is_head_branch_v<HeadBranch<TapIndex, HeadNet, HeadLoss> > = true;

    // @doc: template<typename T> concept IsHeadBranch
    /** `concept` wrapper around `is_head_branch_v` */
    template<typename T>
    concept IsHeadBranch = is_head_branch_v<T>;

    // @doc: template<size_t TapIndex, typename HeadNet, typename HeadLoss = NoLoss> requires IsTrainableNetwork<HeadNet> && (std::is_same_v<HeadLoss, NoLoss> || LossFunction<HeadLoss, typename HeadNet::OutputTensor>) struct HeadBranch
    /** Wrapper pairing a head network with its trunk tap index and loss function */
    template<size_t TapIndex, typename HeadNet, typename HeadLoss = NoLoss>
        requires IsTrainableNetwork<HeadNet> &&
                 (std::is_same_v<HeadLoss, NoLoss> || LossFunction<HeadLoss, typename HeadNet::OutputTensor>)
    struct HeadBranch {
        // @doc: static constexpr size_t HeadBranch::Tap
        /** Index of trunk network activations to take as input to this `HeadBranch` */
        static constexpr size_t Tap = TapIndex;
        // @doc: using HeadBranch::Net
        /** Type alias for underlying `TrainableTensorNetwork` type in this `HeadBranch` */
        using Net = HeadNet;
        // @doc: using HeadBranch::Loss
        /** Loss function for this head; `NoLoss` suppresses gradient contribution */
        using Loss = HeadLoss;
        // @doc: HeadNet HeadBranch::net
        /** Member instance of `Net` for this `HeadBranch` */
        HeadNet net;
    };

    // @doc: template<typename TrunkNet, typename... Heads> requires IsTrainableNetwork<TrunkNet> && (IsHeadBranch<Heads> && ...) class BranchTrainer
    /**
     * Owns one `TrunkNet` and a `std::tuple<Heads...>` of `HeadBranch` instances
     * Tap indices must be non-decreasing (enforced by `static_assert`)
     * Each head's `InputTensor` must match the trunk activation tensor at its tap index (enforced by `static_assert`)
     */
    template<typename TrunkNet, typename... Heads>
        requires IsTrainableNetwork<TrunkNet> && (IsHeadBranch<Heads> && ...)
    class BranchTrainer {
        static constexpr size_t NumHeads = sizeof...(Heads);

        // @doc: using BranchTrainer::InputTensor
        /** Alias for `TrunkNet::InputTensor` */
        using InputTensor = TrunkNet::InputTensor;
        // @doc: using BranchTrainer::TrunkOutputTensor
        /** Alias for `TrunkNet::OutputTensor` */
        using TrunkOutputTensor = TrunkNet::OutputTensor;
        // @doc: using BranchTrainer::TrunkActivations
        /** Alias for `TrunkNet::Activations` */
        using TrunkActivations = TrunkNet::Activations;

        // @doc: using BranchTrainer::HeadTapGrads
        /** Tuple of per-head input-tap gradient tensors */
        using HeadTapGrads = std::tuple<typename Heads::Net::InputTensor...>;
        // @doc: using BranchTrainer::HeadOutputs
        /** Tuple of per-head output target tensors */
        using HeadOutputs = std::tuple<typename Heads::Net::OutputTensor...>;

        static constexpr bool check_taps() {
            if constexpr (NumHeads <= 1) return true;
            std::array<size_t, NumHeads> t = {Heads::Tap...};
            for (size_t i = 1; i < NumHeads; ++i)
                if (t[i] < t[i - 1]) return false;
            return true;
        }

        static_assert(check_taps(), "HeadBranch tap indices must be non-decreasing");

        static constexpr bool check_head_types() {
            return []<size_t... I>(std::index_sequence<I...>) {
                return (std::is_same_v<
                    typename std::tuple_element_t<I, std::tuple<Heads...> >::Net::InputTensor,
                    std::remove_cvref_t<decltype(
                        std::declval<TrunkActivations>().template get<
                            std::tuple_element_t<I, std::tuple<Heads...> >::Tap>()
                    )>
                > && ...);
            }(std::index_sequence_for<Heads...>{});
        }

        static_assert(check_head_types(),
                      "Each head's InputTensor must match the trunk activation at its tap index");

        // @doc: TrunkNet BranchTrainer::trunk_
        /** Member instance of `TrunkNet` type */
        TrunkNet trunk_;
        // @doc: std::tuple<Heads...> BranchTrainer::heads_
        /** `std::tuple` of `HeadBranch` objects */
        std::tuple<Heads...> heads_;

        // @doc: template<size_t... I> auto BranchTrainer::forward_heads(const TrunkActivations &A, std::index_sequence<I...>) const
        /** Forward pass through all heads, returning `std::tuple` of heads' activation tuples */
        template<size_t... I>
        auto forward_heads(const TrunkActivations &A, std::index_sequence<I...>) const {
            return std::make_tuple(
                std::get<I>(heads_).net.ForwardAll(
                    A.template get<std::tuple_element_t<I, std::tuple<Heads...> >::Tap>()
                )...
            );
        }

        // @doc: template<size_t Batch, size_t... I> auto BranchTrainer::forward_heads_batched(const TrunkNet::BatchedActivations<Batch> &A, std::index_sequence<I...>) const
        /** Batched forward pass through all heads, returning `std::tuple` of heads' batched activation tuples */
        template<size_t Batch, size_t... I>
        auto forward_heads_batched(const TrunkNet::template BatchedActivations<Batch> &A,
                                   std::index_sequence<I...>) const {
            return std::make_tuple(
                std::get<I>(heads_).net.template BatchedForwardAll<Batch>(
                    A.template get<std::tuple_element_t<I, std::tuple<Heads...> >::Tap>()
                )...
            );
        }


        // @doc: template<typename TrunkLoss, size_t... Is> requires (std::is_same_v<TrunkLoss, NoLoss> || LossFunction<TrunkLoss, typename TrunkNet::OutputTensor>) float BranchTrainer::compute_losses(const TrunkActivations &A, const TrunkOutputTensor &trunk_target, const HeadOutputs &head_targets, std::index_sequence<Is...>) const
        /** Accumulates scalar loss contributions from trunk and all heads */
        template<typename TrunkLoss, size_t... Is> requires (
            std::is_same_v<TrunkLoss, NoLoss> || LossFunction<TrunkLoss, typename TrunkNet::OutputTensor>)
        float compute_losses(const TrunkActivations &A,
                             const TrunkOutputTensor &trunk_target,
                             const HeadOutputs &head_targets,
                             std::index_sequence<Is...>) const {
            float total = 0.f;
            // if not NoLoss, get trunk loss from its Loss
            if constexpr (!std::is_same_v<TrunkLoss, NoLoss>) {
                total += TrunkLoss::Loss(A.template get<TrunkNet::NumBlocks>(), trunk_target);
            }
            // now accumulate each head's loss
            ([&] {
                // get the head
                using HeadI = std::tuple_element_t<Is, std::tuple<Heads...> >;
                // head loss
                using HL = HeadI::Loss;
                // if HL is not NoLoss
                if constexpr (!std::is_same_v<HL, NoLoss>) {
                    // add loss at head
                    total += HL::Loss(
                        // get output
                        std::get<Is>(forward_heads(A, std::index_sequence_for<Heads...>{})).template get<
                            HeadI::Net::NumBlocks>(),
                        // get target
                        std::get<Is>(head_targets));
                }
            }(), ...);
            return total;
        }

        // @doc: template<typename TrunkLoss, size_t Batch, size_t... Is> requires (std::is_same_v<TrunkLoss, NoLoss> || LossFunction<TrunkLoss, typename TrunkNet::OutputTensor>) float BranchTrainer::compute_losses_batched(const TrunkNet::BatchedActivations<Batch> &A, const PrependBatch<Batch, TrunkOutputTensor>::type &trunk_target, const std::tuple<typename PrependBatch<Batch, typename Heads::Net::OutputTensor>::type...> &head_targets, std::index_sequence<Is...>) const
        /** Batched: accumulates scalar loss contributions from trunk and all heads across the batch */
        template<typename TrunkLoss, size_t Batch, size_t... Is> requires (
            std::is_same_v<TrunkLoss, NoLoss> || LossFunction<TrunkLoss, typename TrunkNet::OutputTensor>)
        float compute_losses_batched(const TrunkNet::template BatchedActivations<Batch> &A,
                                     const typename PrependBatch<Batch, TrunkOutputTensor>::type &trunk_target,
                                     const std::tuple<typename PrependBatch<Batch, typename
                                         Heads::Net::OutputTensor>::type...> &head_targets,
                                     std::index_sequence<Is...>) const {
            float total = 0.f;
            if constexpr (!std::is_same_v<TrunkLoss, NoLoss>) {
                // get batched output
                const auto &trunk_out = A.template get<TrunkNet::NumBlocks>();
                // for each batched output
                for (size_t b = 0; b < Batch; ++b) {
                    // define and fill Tensors for prediction and target (this should use TensorIndex)
                    TrunkOutputTensor pred_b, target_b;
                    for (size_t i = 0; i < TrunkOutputTensor::Size; ++i) {
                        pred_b.flat(i) = trunk_out.flat(b * TrunkOutputTensor::Size + i);
                        target_b.flat(i) = trunk_target.flat(b * TrunkOutputTensor::Size + i);
                    }
                    // compute loss for this batched answer
                    total += TrunkLoss::Loss(pred_b, target_b);
                }
            }
            ([&] {
                // get the head
                using HeadI = std::tuple_element_t<Is, std::tuple<Heads...> >;
                // head loss
                using HL = HeadI::Loss;
                // if HL is not NoLoss
                if constexpr (!std::is_same_v<HL, NoLoss>) {
                    // get batched activations from head
                    const auto head_As = forward_heads_batched<Batch>(A, std::index_sequence_for<Heads...>{});
                    // get last activation = output
                    const auto &head_out = std::get<Is>(head_As).template get<HeadI::Net::NumBlocks>();
                    // for each batched output
                    for (size_t b = 0; b < Batch; ++b) {
                        // define and fill Tensors for prediction and target (this should use TensorIndex)
                        typename HeadI::Net::OutputTensor pred_b, target_b;
                        for (size_t i = 0; i < HeadI::Net::OutputTensor::Size; ++i) {
                            pred_b.flat(i) = head_out.flat(b * HeadI::Net::OutputTensor::Size + i);
                            target_b.flat(i) = std::get<Is>(head_targets).flat(b * HeadI::Net::OutputTensor::Size + i);
                        }
                        total += HL::Loss(pred_b, target_b);
                    }
                }
            }(), ...);
            return total;
        }


        // @doc: template<size_t I, size_t PrevTap> void BranchTrainer::backward_heads(const TrunkActivations &A, const TrunkOutputTensor &grad, const HeadTapGrads &head_tap_grads)
        /** Pure gradient propagation: chains `BackwardRange` through trunk segments, summing in precomputed per-head tap gradients */
        template<size_t I, size_t PrevTap>
        void backward_heads(const TrunkActivations &A,
                            const TrunkOutputTensor &grad,
                            const HeadTapGrads &head_tap_grads) {
            // I-th tap index
            constexpr size_t tapI = std::tuple_element_t<I, std::tuple<Heads...> >::Tap;
            // trunk gradient at I-th tap site
            TrunkOutputTensor grad_at_tap = trunk_.template BackwardRange<tapI, PrevTap>(A, grad);
            // increment gradient at tap by this head's computed gradient wrt its input
            grad_at_tap += std::get<I>(head_tap_grads);

            // recurse back to previous tap
            if constexpr (I > 0) {
                backward_heads<I - 1, tapI>(A, grad_at_tap, head_tap_grads);
            }
            // when done with taps, pass the accumulated gradient back up the full trunk
            else {
                trunk_.template BackwardRange<0, tapI>(A, grad_at_tap);
            }
        }

        // @doc: template<size_t Batch, size_t I, size_t PrevTap> void BranchTrainer::backward_heads_batched(const TrunkNet::BatchedActivations<Batch> &A, const PrependBatch<Batch, TrunkOutputTensor>::type &grad, const std::tuple<typename PrependBatch<Batch, typename Heads::Net::InputTensor>::type...> &head_tap_grads)
        /** Batched recursive backward: same structure as `backward_heads`, operating on `Batch`-prepended tensors */
        template<size_t Batch, size_t I, size_t PrevTap>
        void backward_heads_batched(const TrunkNet::template BatchedActivations<Batch> &A,
                                    const typename PrependBatch<Batch, TrunkOutputTensor>::type &grad,
                                    const std::tuple<typename PrependBatch<Batch, typename
                                        Heads::Net::InputTensor>::type...> &head_tap_grads) {
            // I-th tap index
            constexpr size_t tapI = std::tuple_element_t<I, std::tuple<Heads...> >::Tap;
            // batched trunk gradient at I-th tap site
            typename PrependBatch<Batch, TrunkOutputTensor>::type grad_at_tap =
                    trunk_.template BatchedBackwardRange<Batch, tapI, PrevTap>(A, grad);
            // add I-th head's batched gradient wrt its input
            grad_at_tap += std::get<I>(head_tap_grads);

            // recurse back to previous tap
            if constexpr (I > 0) {
                backward_heads_batched<Batch, I - 1, tapI>(A, grad_at_tap, head_tap_grads);
            }
            // when done with taps, pass the accumulated gradient back up the full trunk
            else {
                trunk_.template BatchedBackwardRange<Batch, 0, tapI>(A, grad_at_tap);
            }
        }

    public:
        // @doc: TrunkNet &BranchTrainer::trunk()
        /** Getter for `TrunkNet&` */
        TrunkNet &trunk() { return trunk_; }

        // @doc: const TrunkNet &BranchTrainer::trunk() const
        /** Getter for `const TrunkNet&` */
        const TrunkNet &trunk() const { return trunk_; }

        // @doc: template<size_t I> auto &BranchTrainer::head()
        /** Getter for `&` to `I`-th `HeadBranch` from `heads_` */
        template<size_t I>
        auto &head() { return std::get<I>(heads_); }

        // @doc: template<size_t I> const auto &BranchTrainer::head() const
        /** Getter for `const &` to `I`-th `HeadBranch` from `heads_` */
        template<size_t I>
        const auto &head() const { return std::get<I>(heads_); }

        // @doc: template<typename TrunkLoss = NoLoss> float BranchTrainer::Fit(const InputTensor &x, const TrunkOutputTensor &trunk_target, float trunk_lr, const HeadOutputs &head_targets, const std::array<float, NumHeads> &head_lrs)
        /** ###### */
        template<typename TrunkLoss = NoLoss>
        float Fit(const InputTensor &x,
                  const TrunkOutputTensor &trunk_target,
                  float trunk_lr,
                  const HeadOutputs &head_targets,
                  const std::array<float, NumHeads> &head_lrs) {
            // get trunk full activations
            const auto A = trunk_.ForwardAll(x);
            // get tuple of head activation tuples
            auto head_As = forward_heads(A, std::index_sequence_for<Heads...>{});

            // clear trunk grad
            trunk_.ZeroGrad();
            // clear branch net grads
            [&]<size_t... Is>(std::index_sequence<Is...>) {
                (std::get<Is>(heads_).net.ZeroGrad(), ...);
            }(std::index_sequence_for<Heads...>{});

            // get total loss via helper
            const float total_loss =
                    compute_losses<TrunkLoss>(A, trunk_target, head_targets, std::index_sequence_for<Heads...>{});

            // trunk output grad
            TrunkOutputTensor trunk_grad{};
            if constexpr (!std::is_same_v<TrunkLoss, NoLoss>) {
                // compute gradient of trunk loss with respect to trunk output
                trunk_grad = TrunkLoss::Grad(A.template get<TrunkNet::NumBlocks>(), trunk_target);
            }

            // per-head tap grads: run each head backward, or zero if NoLoss
            HeadTapGrads head_tap_grads{};
            // for I in Heads
            [&]<size_t... I>(std::index_sequence<I...>) {
                ([&] {
                    // get I-th head
                    using HeadI = std::tuple_element_t<I, std::tuple<Heads...> >;
                    // I-th head loss
                    using HL = HeadI::Loss;
                    if constexpr (!std::is_same_v<HL, NoLoss>) {
                        // head's activations tuple
                        const auto &head_A = std::get<I>(head_As);
                        // head's final activation (output)
                        const auto &head_out = head_A.template get<HeadI::Net::NumBlocks>();
                        // gradient of head loss with respect to head output
                        const auto head_grad = HL::Grad(head_out, std::get<I>(head_targets));
                        // assign gradient of I-th head with respect to trunk output at tap site with BackwardRange
                        std::get<I>(head_tap_grads) =
                                std::get<I>(heads_).net.template BackwardRange<0, HeadI::Net::NumBlocks>(
                                    head_A, head_grad);
                        // update I-th head weights, calling Update, passing I-th lr
                        std::get<I>(heads_).net.Update(head_lrs[I]);
                    }
                }(), ...);
            }(std::index_sequence_for<Heads...>{});

            // backward_heads plugs each head's gradient wrt trunk into the trunk
            backward_heads<NumHeads - 1, TrunkNet::NumBlocks>(A, trunk_grad, head_tap_grads);
            // update gradient-juiced trunk
            trunk_.Update(trunk_lr);
            return total_loss;
        }

        // @doc: template<size_t Batch, typename TrunkLoss = NoLoss> float BranchTrainer::BatchFit(const PrependBatch<Batch, InputTensor>::type &X, const PrependBatch<Batch, TrunkOutputTensor>::type &trunk_target, float trunk_lr, const std::tuple<typename PrependBatch<Batch, typename Heads::Net::OutputTensor>::type...> &head_targets, const std::array<float, NumHeads> &head_lrs)
        /** ###### */
        template<size_t Batch, typename TrunkLoss = NoLoss>
        float BatchFit(const typename PrependBatch<Batch, InputTensor>::type &X,
                       const typename PrependBatch<Batch, TrunkOutputTensor>::type &trunk_target,
                       float trunk_lr,
                       const std::tuple<typename PrependBatch<Batch, typename Heads::Net::OutputTensor>::type...> &
                       head_targets,
                       const std::array<float, NumHeads> &head_lrs) {
            const auto A = trunk_.template BatchedForwardAll<Batch>(X);
            auto head_As = forward_heads_batched<Batch>(A, std::index_sequence_for<Heads...>{});

            trunk_.ZeroGrad();
            [&]<size_t... Is>(std::index_sequence<Is...>) {
                (std::get<Is>(heads_).net.ZeroGrad(), ...);
            }(std::index_sequence_for<Heads...>{});

            const float total_loss =
                    compute_losses_batched<TrunkLoss, Batch>(A, trunk_target, head_targets,
                                                             std::index_sequence_for<Heads...>{});

            // batched trunk output grad (averaged over batch)
            typename PrependBatch<Batch, TrunkOutputTensor>::type trunk_grad{};
            if constexpr (!std::is_same_v<TrunkLoss, NoLoss>) {
                const auto &trunk_out = A.template get<TrunkNet::NumBlocks>();
                for (size_t b = 0; b < Batch; ++b) {
                    TrunkOutputTensor pred_b, target_b;
                    for (size_t i = 0; i < TrunkOutputTensor::Size; ++i) {
                        pred_b.flat(i) = trunk_out.flat(b * TrunkOutputTensor::Size + i);
                        target_b.flat(i) = trunk_target.flat(b * TrunkOutputTensor::Size + i);
                    }
                    const auto g = TrunkLoss::Grad(pred_b, target_b);
                    for (size_t i = 0; i < TrunkOutputTensor::Size; ++i)
                        trunk_grad.flat(b * TrunkOutputTensor::Size + i) =
                                g.flat(i) / static_cast<float>(Batch);
                }
            }

            // per-head batched tap grads
            std::tuple<typename PrependBatch<Batch, typename Heads::Net::InputTensor>::type...> head_tap_grads{};
            [&]<size_t... Is>(std::index_sequence<Is...>) {
                ([&] {
                    using HeadI = std::tuple_element_t<Is, std::tuple<Heads...> >;
                    using HL = typename HeadI::Loss;
                    if constexpr (!std::is_same_v<HL, NoLoss>) {
                        const auto &head_A = std::get<Is>(head_As);
                        const auto &head_out = head_A.template get<HeadI::Net::NumBlocks>();
                        typename PrependBatch<Batch, typename HeadI::Net::OutputTensor>::type head_grad{};
                        for (size_t b = 0; b < Batch; ++b) {
                            typename HeadI::Net::OutputTensor pred_b, target_b;
                            for (size_t i = 0; i < HeadI::Net::OutputTensor::Size; ++i) {
                                pred_b.flat(i) = head_out.flat(b * HeadI::Net::OutputTensor::Size + i);
                                target_b.flat(i) = std::get<Is>(head_targets).flat(
                                    b * HeadI::Net::OutputTensor::Size + i);
                            }
                            const auto g = HL::Grad(pred_b, target_b);
                            for (size_t i = 0; i < HeadI::Net::OutputTensor::Size; ++i)
                                head_grad.flat(b * HeadI::Net::OutputTensor::Size + i) =
                                        g.flat(i) / static_cast<float>(Batch);
                        }
                        std::get<Is>(head_tap_grads) =
                                std::get<Is>(heads_).net.template BatchedBackwardRange<Batch, 0, HeadI::Net::NumBlocks>(
                                    head_A, head_grad);
                        std::get<Is>(heads_).net.Update(head_lrs[Is]);
                    }
                }(), ...);
            }(std::index_sequence_for<Heads...>{});

            backward_heads_batched<Batch, NumHeads - 1, TrunkNet::NumBlocks>(A, trunk_grad, head_tap_grads);
            trunk_.Update(trunk_lr);

            return total_loss;
        }
    };
} // namespace TTTN
