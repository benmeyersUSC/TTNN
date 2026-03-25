#pragma once
#include "TensorOps.hpp"
// #include "NetworkUtil.hpp"

namespace TTTN {
    // =========================================================================
    // ChainBlock
    // =========================================================================
    //
    // A compound block that chains N sub-blocks sequentially.
    // Satisfies ConcreteBlock so it can be used inside ParallelBlock, ResidualBlock, etc.
    //
    // Unlike TrainableTensorNetwork, this exposes the simple Forward/Backward interface
    // rather than the ActivationsWrap pattern. It manages its own internal activations.
    //
    // =========================================================================

    template<ConcreteBlock... Blocks>
    class ChainBlock {
        static_assert(sizeof...(Blocks) >= 1, "ChainBlock needs at least one block");
        static constexpr size_t N = sizeof...(Blocks);
        using BlockTuple = std::tuple<Blocks...>;

        // Connectivity check: each block's output must match the next block's input
        static constexpr bool check_connected() {
            if constexpr (N == 1) return true;
            else
                return []<size_t... Is>(std::index_sequence<Is...>) {
                    return (std::is_same_v<
                        typename std::tuple_element_t<Is, BlockTuple>::OutputTensor,
                        typename std::tuple_element_t<Is + 1, BlockTuple>::InputTensor> && ...);
                }(std::make_index_sequence<N - 1>{});
        }

        static_assert(check_connected(), "ChainBlock: block output/input types don't chain");

    public:
        using InputTensor = typename std::tuple_element_t<0, BlockTuple>::InputTensor;
        using OutputTensor = typename std::tuple_element_t<N - 1, BlockTuple>::OutputTensor;

    private:
        BlockTuple blocks_;

        // Internal activations tuple: (InputTensor, Block0::Output, Block1::Output, ..., OutputTensor)
        template<typename... Bs>
        struct ActivationTypes;

        template<typename Last>
        struct ActivationTypes<Last> {
            using type = std::tuple<typename Last::InputTensor, typename Last::OutputTensor>;
        };

        template<typename First, typename... Rest>
        struct ActivationTypes<First, Rest...> {
            using type = decltype(std::tuple_cat(
                std::declval<std::tuple<typename First::InputTensor> >(),
                std::declval<typename ActivationTypes<Rest...>::type>()));
        };

        using ActTuple = typename ActivationTypes<Blocks...>::type;
        mutable ActTuple acts_{};

        // Forward: walk blocks, fill activations
        template<size_t I = 0>
        void forward_impl() const {
            if constexpr (I < N) {
                std::get<I + 1>(acts_) = std::get<I>(blocks_).Forward(std::get<I>(acts_));
                forward_impl<I + 1>();
            }
        }

        // Backward: walk blocks in reverse
        template<size_t I, typename Delta>
        auto backward_impl(const Delta &delta) -> typename std::tuple_element_t<I - 1, BlockTuple>::InputTensor {
            auto grad = std::get<I - 1>(blocks_).Backward(delta, std::get<I>(acts_), std::get<I - 1>(acts_));
            if constexpr (I > 1) {
                return backward_impl<I - 1>(grad);
            } else {
                return grad;
            }
        }

        // Batched forward
        template<size_t Batch, size_t I, typename X>
        auto batched_forward_impl(const X &x) const {
            auto out = std::get<I>(blocks_).template BatchedForward<Batch>(x);
            if constexpr (I + 1 < N) {
                return batched_forward_impl<Batch, I + 1>(out);
            } else {
                return out;
            }
        }

        // Batched backward — needs cached batched activations
        // For simplicity, re-forward to get them
        template<size_t Batch>
        struct BatchedActs {
            template<typename... Bs>
            struct Types;

            template<typename Last>
            struct Types<Last> {
                using type = std::tuple<
                    typename PrependBatch<Batch, typename Last::InputTensor>::type,
                    typename PrependBatch<Batch, typename Last::OutputTensor>::type>;
            };

            template<typename First, typename... Rest>
            struct Types<First, Rest...> {
                using type = decltype(std::tuple_cat(
                    std::declval<std::tuple<typename PrependBatch<Batch, typename First::InputTensor>::type> >(),
                    std::declval<typename Types<Rest...>::type>()));
            };

            using type = typename Types<Blocks...>::type;
        };

        template<size_t Batch, size_t I = 0, typename ActsT>
        void batched_forward_fill(ActsT &acts) const {
            if constexpr (I < N) {
                std::get<I + 1>(acts) = std::get<I>(blocks_).template BatchedForward<Batch>(std::get<I>(acts));
                batched_forward_fill<Batch, I + 1>(acts);
            }
        }

        template<size_t Batch, size_t I, typename ActsT, typename Delta>
        auto batched_backward_impl(const ActsT &acts, const Delta &delta) {
            auto grad = std::get<I - 1>(blocks_).template BatchedBackward<Batch>(
                delta, std::get<I>(acts), std::get<I - 1>(acts));
            if constexpr (I > 1) {
                return batched_backward_impl<Batch, I - 1>(acts, grad);
            } else {
                return grad;
            }
        }

    public:
        auto all_params() {
            return [&]<size_t... Is>(std::index_sequence<Is...>) {
                return std::tuple_cat(std::get<Is>(blocks_).all_params()...);
            }(std::make_index_sequence<N>{});
        }

        auto all_params() const {
            return [&]<size_t... Is>(std::index_sequence<Is...>) {
                return std::tuple_cat(std::get<Is>(blocks_).all_params()...);
            }(std::make_index_sequence<N>{});
        }

        OutputTensor Forward(const InputTensor &x) const {
            std::get<0>(acts_) = x;
            forward_impl();
            return std::get<N>(acts_);
        }

        InputTensor Backward(const OutputTensor &delta_A,
                             const OutputTensor & /*a*/,
                             const InputTensor & /*a_prev*/) {
            return backward_impl<N>(delta_A);
        }

        template<size_t Batch>
        auto BatchedForward(const typename PrependBatch<Batch, InputTensor>::type &X) const
            -> typename PrependBatch<Batch, OutputTensor>::type {
            return batched_forward_impl<Batch, 0>(X);
        }

        template<size_t Batch>
        auto BatchedBackward(
            const typename PrependBatch<Batch, OutputTensor>::type &delta_A,
            const typename PrependBatch<Batch, OutputTensor>::type & /*a*/,
            const typename PrependBatch<Batch, InputTensor>::type &a_prev)
            -> typename PrependBatch<Batch, InputTensor>::type {
            // Re-forward to fill batched activations
            typename BatchedActs<Batch>::type acts;
            std::get<0>(acts) = a_prev;
            batched_forward_fill<Batch>(acts);
            return batched_backward_impl<Batch, N>(acts, delta_A);
        }
    };
}
