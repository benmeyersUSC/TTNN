#pragma once
#include "TrainableTensorNetwork.hpp"

namespace TTTN {
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

    // @doc: template<typename Tuple> struct ConcretesToSequence
    /**
     * Specialization: `template<ConcreteBlock... Bs> struct ConcretesToSequence<std::tuple<Bs...>>`
     * Unpacks a `std::tuple<ConcreteBlock...>` into `BlockSequence<Bs...>`
     * Used by `ComposeBlocks::ResolveImpl`
     */
    template<typename Tuple>
    struct ConcretesToSequence;

    template<ConcreteBlock... Bs>
    struct ConcretesToSequence<std::tuple<Bs...>> {
        using type = BlockSequence<Bs...>;
    };

    // @doc: template<typename Tuple> struct ConcretesToNetwork
    /**
     * Specialization: `template<ConcreteBlock... Bs> struct ConcretesToNetwork<std::tuple<Bs...>>`
     * Unpacks a `std::tuple<ConcreteBlock...>` into `TrainableTensorNetwork<Bs...>`
     * Used by `NetworkBuilder`
     */
    template<typename Tuple>
    struct ConcretesToNetwork;

    template<ConcreteBlock... Bs>
    struct ConcretesToNetwork<std::tuple<Bs...>> {
        using type = TrainableTensorNetwork<Bs...>;
    };

    // @doc: template<typename... Recipes> requires (Block<Recipes> && ...) struct ComposeBlocks
    /** Struct containing `ResolveChain` methods (several specializations) which unpack variadic lists of `Block`s into a `BlockSequence` type, saved in `ComposeBlocks::type` */
    template<typename... Recipes>
        requires (Block<Recipes> && ...)
    struct ComposeBlocks {
        template<typename In, typename... Rs>
        struct ResolveChain;


        // @doc: template<typename In, typename Last> struct ComposeBlocks::ResolveChain<In, Last>
        /**
         * Wraps a variadic list of `Block`s into a `std::tuple` of `ConcreteBlock`s
         * Used exclusively in `ResolveImpl` to turn unpack `Block` recipes into a `BlockSequence`
         * Base case, resolving the last `Block`'s `OutputTensor` by passing in the penultimate `Block`'s `OutputTensor` and wrapping `Resolved` in a `std::tuple`
         */
        template<typename In, typename Last>
        struct ResolveChain<In, Last> {
            using Resolved = Last::template Resolve<In>;
            using type = std::tuple<Resolved>;
            using OutputT = Resolved::OutputTensor;
        };

        // @doc: template<typename In, typename First, typename... Rest> struct ComposeBlocks::ResolveChain<In, First, Rest...>
        /**
         * Wraps a variadic list of `Block`s into a `std::tuple` of `ConcreteBlock`s
         * Used exclusively in `ResolveImpl` to turn unpack `Block` recipes into a `BlockSequence`
         * Recursive case, resolving `First` by passing in `In` (starts as `IsTensor<InputT>` in `ResolveImpl`), then defines `Tail` by recursing on `Resolve`, finally wrapping `Resolved` and `Tail` in `std::tuple_cat`
         */
        template<typename In, typename First, typename... Rest>
        struct ResolveChain<In, First, Rest...> {
            using Resolved = First::template Resolve<In>;
            using Tail = ResolveChain<typename Resolved::OutputTensor, Rest...>;
            using type = decltype(std::tuple_cat(
                std::declval<std::tuple<Resolved> >(),
                std::declval<typename Tail::type>()));
            using OutputT = Tail::OutputT;
        };


        template<typename... Rs>
        struct LastOutputTensor;

        // @doc: template<typename Last> struct ComposeBlocks::LastOutputTensor<Last>
        /**
         * Custom last-in-variadic getter, assuming that template args are `Block` recipes, recursing until the `Last` is reached and finally grabbing `Last::OutputTensor`
         * Base case: `type = Last::OutputTensor`
         */
        template<typename Last>
        struct LastOutputTensor<Last> {
            using type = Last::OutputTensor;
        };

        // @doc: template<typename First, typename... Rest> struct ComposeBlocks::LastOutputTensor<First, Rest...>
        /**
         * Custom last-in-variadic getter, assuming that template args are `Block` recipes, recursing until the `Last` is reached and finally grabbing `Last::OutputTensor`
         * Recursive case: `type = LastOutputTensor<Rest...>::type`
         */
        template<typename First, typename... Rest>
        struct LastOutputTensor<First, Rest...> {
            using type = LastOutputTensor<Rest...>::type;
        };

        // @doc: using ComposeBlocks::OutputTensor
        /** Type alias for `OutputTensor` of last `Block` in `Recipes...` */
        using OutputTensor = LastOutputTensor<Recipes...>::type;

        // @doc: template<typename InputT> requires IsTensor<InputT> struct ComposeBlocks::ResolveImpl
        /**
         * Implementation helper to take in `IsTensor<InputT>` and `Recipes...`
         * Runs `ResolveChain` to get a `std::tuple` of `ConcreteBlock`s, then calls `ConcretesToSequence` to produce `BlockSequence`, stored in `type`
         */
        template<typename InputT> requires IsTensor<InputT>
        struct ResolveImpl {
            using Chain = ResolveChain<InputT, Recipes...>;
            using BlockTuple = Chain::type;
            using type = ConcretesToSequence<BlockTuple>::type;
        };

        // @doc: template<typename InputT> requires IsTensor<InputT> using ComposeBlocks::Resolve
        /** Culmination: in compliance with `Block`, `ComposeBlocks::Resolve` takes in `IsTensor<InputT>` and resolves to a `ConcreteBlock<BlockSequence>` */
        template<typename InputT> requires IsTensor<InputT>
        using Resolve = ResolveImpl<InputT>::type;
    };


    template<typename NetA, typename NetB>
    struct ComposeNetworks;

    // @doc: template<ConcreteBlock... BlocksA, ConcreteBlock... BlocksB> struct ComposeNetworks<TrainableTensorNetwork<BlocksA...>, TrainableTensorNetwork<BlocksB...> >
    /**
     * `static_assert` that `std::is_same_v< typename TrainableTensorNetwork<BlocksA...>::OutputTensor, typename TrainableTensorNetwork<BlocksB...>::InputTensor>`
     * Simple unpack: `type = TrainableTensorNetwork<BlocksA..., BlocksB...>`
     */
    template<ConcreteBlock... BlocksA, ConcreteBlock... BlocksB>
    struct ComposeNetworks<TrainableTensorNetwork<BlocksA...>, TrainableTensorNetwork<BlocksB...> > {
        static_assert(
            std::is_same_v<
                typename TrainableTensorNetwork<BlocksA...>::OutputTensor,
                typename TrainableTensorNetwork<BlocksB...>::InputTensor>,
            "CombineNetworks: OutputTensor of first network must equal InputTensor of second");
        using type = TrainableTensorNetwork<BlocksA..., BlocksB...>;
    };


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
        using Resolved = Next::template Resolve<typename Prev::OutputTensor>;

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


    template<typename In, typename Tuple>
    struct ApplyBuildChain;

    // @doc: template<typename In, Block... Rs> struct ApplyBuildChain<In, std::tuple<Rs...> >
    /**
     * Expects an `Block<Input>` first and a trailing variadic list of `Block`s passes them to `BuildChain`
     * Used in `NetworkBuilder` to define `BlockTuple`, a `TrainableTensorNetwork`'s tuple of `ConcreteBlock`s
     */
    template<typename In, Block... Rs>
    struct ApplyBuildChain<In, std::tuple<Rs...> > {
        using type = BuildChain<In, Rs...>::type;
    };


    // @doc: template<typename In, typename... Recipes> requires requires { typename In::OutputTensor; } && IsTensor<typename In::OutputTensor> && (Block<Recipes> && ...) struct NetworkBuilder
    /**
     * Takes variadic list of `Block...` recipes and create a `TrainableTensorNetwork`
     * Calls `ApplyBuildChain` to get a `std::tuple` of `ConcreteBlock`s, then calls `ConcretesToNetwork`
     */
    template<typename In, typename... Recipes> requires
        requires { typename In::OutputTensor; } && IsTensor<typename In::OutputTensor> && (Block<Recipes> && ...)
    struct NetworkBuilder {
        using BlockTuple = ApplyBuildChain<In, std::tuple<Recipes...> >::type;
        using type = ConcretesToNetwork<BlockTuple>::type;
    };
}
