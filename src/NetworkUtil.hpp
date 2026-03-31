#pragma once
#include <concepts>
#include "Tensor.hpp"
#include "Params.hpp"
#include "Snapshot.hpp"

namespace TTTN {
    // @doc: template<typename T> concept PeekableBlock
    /**
     * Opt-in `concept` for `ConcreteBlock`s to be able to expose their internal activations to an owning `TrainableTensorNetwork`
     * Compliant `ConcreteBlock`s must implement `void peek(SnapshotMap& m, const std::string& s)`
     */
    template<typename T>
    concept PeekableBlock = requires(const T &t, SnapshotMap &m, const std::string &s)
    {
        { t.peek(m, s) } -> std::same_as<void>;
    };


    // @doc: template<typename T> concept ConcreteBlock
    /**
     * Any block in a `TrainableTensorNetwork` must satisfy `ConcreteBlock`:
     * Defined `InputTensor` and `OutputTensor` types which are `Tensor` objects
     * `OutputTensor Forward(InputTensor)`
     * `InputTensor Backward(OutputTensor, OutputTensor, InputTensor)`
     * `auto all_params()` and `auto all_params() const`
     * `TrainableTensorNetwork` blocks need not belong to a specific hierarchy; just satisfy this `concept`
     */
    template<typename T> concept ConcreteBlock =
            requires { typename T::InputTensor; } &&
            requires { typename T::OutputTensor; } &&
            IsTensor<typename T::InputTensor> &&
            IsTensor<typename T::OutputTensor> &&
            requires(T t, const T ct,
                     typename T::InputTensor in,
                     typename T::OutputTensor out)
            {
                { ct.Forward(in) } -> std::same_as<typename T::OutputTensor>;
                { t.Backward(out, out, in) } -> std::same_as<typename T::InputTensor>;
                { t.all_params() }; // non-const: ZeroGrad + Update
                { ct.all_params() }; // const: Save
            };

    // @doc: template<typename B> concept Block
    /**
     * Declarable recipe to define a `ConcreteBlock` in a `TrainableTensorNetwork` template argument list
     * `Block`s must define an `OutputTensor` type and alias a `ConcreteBlock` as `Resolve`
     * `Block` argument lists passed to `NetworkBuilder` will be resolved into full `ConcreteBlock`s with chained `InputTensor` attributes
     */
    template<typename B>
    concept Block =
            requires { typename B::OutputTensor; } &&
            IsTensor<typename B::OutputTensor> &&
            ConcreteBlock<typename B::template Resolve<typename B::OutputTensor> >;


    template<typename In, Block... Recipes>
    struct BuildChain;

    // @doc: template<typename Prev, Block Last> struct BuildChain<Prev, Last>
    /**
     * Build `std::tuple` of `ConcreteBlock`s from a variadic argument list of `Block`s
     * Base case for recursive `BuildChain`
     */
    template<typename Prev, Block Last>
    struct BuildChain<Prev, Last> {
        using type = std::tuple<typename Last::template Resolve<typename Prev::OutputTensor> >;
    };

    // @doc: template<typename Prev, Block Next, Block... Rest> struct BuildChain<Prev, Next, Rest...>
    /**
     * Build `std::tuple` of `ConcreteBlock`s from a variadic argument list of `Block`s
     * Recursive case: `std::tuple_cat` of
     * first `Block`'s `ConcreteBlock` as given by its `Resolve` member
     * next `Block`s' `ConcreteBlock`s
     * Used by `ApplyBuildChain`
     */
    template<typename Prev, Block Next, Block... Rest>
    struct BuildChain<Prev, Next, Rest...> {
        // get next's ConcreteBlock
        using Resolved = typename Next::template Resolve<typename Prev::OutputTensor>;

        // recurse down, grabbing next's ConcreteBlock
        using type = decltype(std::tuple_cat(
            std::declval<std::tuple<Resolved> >(),
            std::declval<typename BuildChain<Resolved, Rest...>::type>()
        ));
    };

    // @doc: template<size_t... Dims> struct Input
    /**
     * `Block` type which begins and allows a variadic argument list of `Block`s to be processed by `BuildChain` via `ApplyBuildChain`
     * Defines `OutputTensor = Tensor<Dims...>` to begin chain
     */
    template<size_t... Dims>
    struct Input {
        using OutputTensor = Tensor<Dims...>;
    };


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
        auto backward_impl(const Delta &delta) -> InputTensor {
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
        template<size_t I>
        const auto &block() const { return std::get<I>(blocks_); }

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


    // FlattenRecipes: recursively expand ComposeBlocks → std::tuple<Block...>
    template<typename... Rs>
    struct FlattenRecipes;

    template<>
    struct FlattenRecipes<> {
        using type = std::tuple<>;
    };


    template<typename... Recipes>
    struct ComposeBlocks {
        // For use inside Parallel/Residual/Transposed:
        // Resolve<InputT> builds the chain of concrete blocks and wraps in ChainBlock.

        // We need OutputTensor. Walk the recipe chain to find the final output type.
        // ResolveChain: given an input type, resolve each recipe in sequence, return tuple of blocks.
        template<typename In, typename... Rs>
        struct ResolveChain;

        template<typename In, typename Last>
        struct ResolveChain<In, Last> {
            using Resolved = typename Last::template Resolve<In>;
            using type = std::tuple<Resolved>;
            using OutputT = typename Resolved::OutputTensor;
        };

        template<typename In, typename First, typename... Rest>
        struct ResolveChain<In, First, Rest...> {
            using Resolved = typename First::template Resolve<In>;
            using Tail = ResolveChain<typename Resolved::OutputTensor, Rest...>;
            using type = decltype(std::tuple_cat(
                std::declval<std::tuple<Resolved> >(),
                std::declval<typename Tail::type>()));
            using OutputT = typename Tail::OutputT;
        };

        // OutputTensor: resolve with a dummy to find the output type.
        // Use the last recipe's OutputTensor as a placeholder input to walk the chain.
        // This is a rough heuristic — the real output depends on InputT.
        // For the Block concept check (which resolves with OutputTensor), this works.

        // Helper: extract the last recipe's OutputTensor
        template<typename... Rs>
        struct LastOutputTensor;

        template<typename Last>
        struct LastOutputTensor<Last> {
            using type = typename Last::OutputTensor;
        };

        template<typename First, typename... Rest>
        struct LastOutputTensor<First, Rest...> {
            using type = typename LastOutputTensor<Rest...>::type;
        };

        using OutputTensor = typename LastOutputTensor<Recipes...>::type;

        // Resolve<InputT>: resolve the full chain into a ChainBlock
        template<typename InputT>
        struct ResolveImpl {
            using Chain = ResolveChain<InputT, Recipes...>;
            using BlockTuple = typename Chain::type;

            // Unpack tuple into ChainBlock template args
            template<typename Tuple>
            struct Apply;

            template<typename... Bs>
            struct Apply<std::tuple<Bs...> > {
                using type = ChainBlock<Bs...>;
            };

            using type = typename Apply<BlockTuple>::type;
        };

        template<typename InputT>
        using Resolve = typename ResolveImpl<InputT>::type;
    };


    // anything else: keep it and continue
    template<typename First, typename... Rest>
    struct FlattenRecipes<First, Rest...> {
        using type = decltype(std::tuple_cat(
            std::declval<std::tuple<First> >(),
            std::declval<typename FlattenRecipes<Rest...>::type>()));
    };

    // ApplyBuildChain: unpack a tuple of Block recipes into BuildChain
    template<typename In, typename Tuple>
    struct ApplyBuildChain;

    template<typename In, Block... Rs>
    struct ApplyBuildChain<In, std::tuple<Rs...> > {
        using type = typename BuildChain<In, Rs...>::type;
    };


    // ActivationsWrap: thin owning wrapper around an activations tuple.
    //
    // Calling get<N>() on an rvalue (temporary) wrapper is deleted at compile time,
    // so `auto& y = net.ForwardAll(x).get<1>()` is a hard error — not a runtime dangle.
    // The user must bind to a named variable: `auto A = net.ForwardAll(x); A.get<1>();`
    template<typename TupleT>
    class ActivationsWrap {
        TupleT data_;

    public:
        explicit ActivationsWrap(TupleT t) : data_(std::move(t)) {
        }

        // Safe: reference into owned tuple; valid as long as this wrapper is alive.
        template<size_t N>
        auto get() const & -> const std::tuple_element_t<N, TupleT> & {
            return std::get<N>(data_);
        }

        template<size_t N>
        auto get() & -> std::tuple_element_t<N, TupleT> & {
            return std::get<N>(data_);
        }

        // Deleted: `temporary_wrap.get<N>()` is a compile error, not a dangle.
        template<size_t N>
        auto get() && -> std::tuple_element_t<N, TupleT> && = delete;

        // Raw tuple access for internal backward pass machinery.
        const TupleT &tuple() const { return data_; }
    };
};
