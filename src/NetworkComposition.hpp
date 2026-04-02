#pragma once
#include "TrainableTensorNetwork.hpp"

namespace TTTN {
    // @doc: template<typename B> concept BlockRecipe
    /**
     * Declarable recipe to define a `Block` in a `TrainableTensorNetwork` template argument list
     * `BlockRecipe`s must define an `OutputTensor` type and alias a `Block` as `Resolve`
     * `BlockRecipe` argument lists passed to `NetworkBuilder` will be resolved into full `Block`s with chained `InputTensor` attributes
     */
    template<typename B>
    concept BlockRecipe =
            requires { typename B::OutputTensor; } &&
            IsTensor<typename B::OutputTensor> &&
            Block<typename B::template Resolve<typename B::OutputTensor> >;

    // @doc: template<typename Tuple> struct ConcretesToSequence
    /**
     * Specialization: `template<Block... Bs> struct ConcretesToSequence<std::tuple<Bs...>>`
     * Unpacks a `std::tuple<Block...>` into `BlockSequence<Bs...>`
     * Used by `ComposeBlocks::ResolveImpl`
     */
    template<typename Tuple>
    struct ConcretesToSequence;

    template<Block... Bs>
    struct ConcretesToSequence<std::tuple<Bs...> > {
        using type = BlockSequence<Bs...>;
    };

    // @doc: template<typename Tuple> struct ConcretesToNetwork
    /**
     * Specialization: `template<Block... Bs> struct ConcretesToNetwork<std::tuple<Bs...>>`
     * Unpacks a `std::tuple<Block...>` into `TrainableTensorNetwork<Bs...>`
     * Used by `NetworkBuilder`
     */
    template<typename Tuple>
    struct ConcretesToNetwork;

    template<Block... Bs>
    struct ConcretesToNetwork<std::tuple<Bs...> > {
        using type = TrainableTensorNetwork<Bs...>;
    };

    // @doc: template<typename... Recipes> requires (BlockRecipe<Recipes> && ...) struct ComposeBlocks
    /** Struct containing `ResolveChain` methods (several specializations) which unpack variadic lists of `BlockRecipe`s into a `BlockSequence` type, saved in `ComposeBlocks::type` */
    template<typename... Recipes>
        requires (BlockRecipe<Recipes> && ...)
    struct ComposeBlocks {
        template<typename In, typename... Rs>
        struct ResolveChain;


        // @doc: template<typename In, typename Last> struct ComposeBlocks::ResolveChain<In, Last>
        /**
         * Wraps a variadic list of `BlockRecipe`s into a `std::tuple` of `Block`s
         * Used exclusively in `ResolveImpl` to unpack `BlockRecipe`s into a `BlockSequence`
         * Base case, resolving the last `BlockRecipe`'s `OutputTensor` by passing in the penultimate `BlockRecipe`'s `OutputTensor` and wrapping `Resolved` in a `std::tuple`
         */
        template<typename In, typename Last>
        struct ResolveChain<In, Last> {
            using Resolved = Last::template Resolve<In>;
            using type = std::tuple<Resolved>;
            using OutputT = Resolved::OutputTensor;
        };

        // @doc: template<typename In, typename First, typename... Rest> struct ComposeBlocks::ResolveChain<In, First, Rest...>
        /**
         * Wraps a variadic list of `BlockRecipe`s into a `std::tuple` of `Block`s
         * Used exclusively in `ResolveImpl` to unpack `BlockRecipe`s into a `BlockSequence`
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
         * Custom last-in-variadic getter, assuming that template args are `BlockRecipe`s, recursing until the `Last` is reached and finally grabbing `Last::OutputTensor`
         * Base case: `type = Last::OutputTensor`
         */
        template<typename Last>
        struct LastOutputTensor<Last> {
            using type = Last::OutputTensor;
        };

        // @doc: template<typename First, typename... Rest> struct ComposeBlocks::LastOutputTensor<First, Rest...>
        /**
         * Custom last-in-variadic getter, assuming that template args are `BlockRecipe`s, recursing until the `Last` is reached and finally grabbing `Last::OutputTensor`
         * Recursive case: `type = LastOutputTensor<Rest...>::type`
         */
        template<typename First, typename... Rest>
        struct LastOutputTensor<First, Rest...> {
            using type = LastOutputTensor<Rest...>::type;
        };

        // @doc: using ComposeBlocks::OutputTensor
        /** Type alias for `OutputTensor` of last `BlockRecipe` in `Recipes...` */
        using OutputTensor = LastOutputTensor<Recipes...>::type;

        // @doc: template<typename InputT> requires IsTensor<InputT> struct ComposeBlocks::ResolveImpl
        /**
         * Implementation helper to take in `IsTensor<InputT>` and `Recipes...`
         * Runs `ResolveChain` to get a `std::tuple` of `Block`s, then calls `ConcretesToSequence` to produce `BlockSequence`, stored in `type`
         */
        template<typename InputT> requires IsTensor<InputT>
        struct ResolveImpl {
            using Chain = ResolveChain<InputT, Recipes...>;
            using BlockTuple = Chain::type;
            using type = ConcretesToSequence<BlockTuple>::type;
        };

        // @doc: template<typename InputT> requires IsTensor<InputT> using ComposeBlocks::Resolve
        /** Culmination: in compliance with `BlockRecipe`, `ComposeBlocks::Resolve` takes in `IsTensor<InputT>` and resolves to a `Block<BlockSequence>` */
        template<typename InputT> requires IsTensor<InputT>
        using Resolve = ResolveImpl<InputT>::type;
    };


    template<typename NetA, typename NetB>
    struct ComposeNetworks;

    // @doc: template<Block... BlocksA, Block... BlocksB> struct ComposeNetworks<TrainableTensorNetwork<BlocksA...>, TrainableTensorNetwork<BlocksB...> >
    /**
     * `static_assert` that `std::is_same_v< typename TrainableTensorNetwork<BlocksA...>::OutputTensor, typename TrainableTensorNetwork<BlocksB...>::InputTensor>`
     * Simple unpack: `type = TrainableTensorNetwork<BlocksA..., BlocksB...>`
     */
    template<Block... BlocksA, Block... BlocksB>
    struct ComposeNetworks<TrainableTensorNetwork<BlocksA...>, TrainableTensorNetwork<BlocksB...> > {
        static_assert(
            std::is_same_v<
                typename TrainableTensorNetwork<BlocksA...>::OutputTensor,
                typename TrainableTensorNetwork<BlocksB...>::InputTensor>,
            "CombineNetworks: OutputTensor of first network must equal InputTensor of second");
        using type = TrainableTensorNetwork<BlocksA..., BlocksB...>;
    };


    template<typename In, BlockRecipe... Recipes>
    struct BuildChain;

    // @doc: template<typename Prev, BlockRecipe Last> struct BuildChain<Prev, Last>
    /**
     * Build `std::tuple` of `Block`s from a variadic argument list of `BlockRecipe`s
     * Base case for recursive `BuildChain`
     */
    template<typename Prev, BlockRecipe Last>
    struct BuildChain<Prev, Last> {
        using type = std::tuple<typename Last::template Resolve<typename Prev::OutputTensor> >;
    };

    // @doc: template<typename Prev, BlockRecipe Next, BlockRecipe... Rest> struct BuildChain<Prev, Next, Rest...>
    /**
     * Build `std::tuple` of `Block`s from a variadic argument list of `BlockRecipe`s
     * Recursive case: `std::tuple_cat` of
     * first `BlockRecipe`'s `Block` as given by its `Resolve` member
     * next `BlockRecipe`s' `Block`s
     * Used by `ApplyBuildChain`
     */
    template<typename Prev, BlockRecipe Next, BlockRecipe... Rest>
    struct BuildChain<Prev, Next, Rest...> {
        // get next's Block
        using Resolved = Next::template Resolve<typename Prev::OutputTensor>;

        // recurse down, grabbing next's Block
        using type = decltype(std::tuple_cat(
            std::declval<std::tuple<Resolved> >(),
            std::declval<typename BuildChain<Resolved, Rest...>::type>()
        ));
    };

    // @doc: template<size_t... Dims> struct Input
    /**
     * `BlockRecipe` type which begins and allows a variadic argument list of `BlockRecipe`s to be processed by `BuildChain` via `ApplyBuildChain`
     * Defines `OutputTensor = Tensor<Dims...>` to begin chain
     */
    template<size_t... Dims>
    struct Input {
        using OutputTensor = Tensor<Dims...>;
    };


    template<typename In, typename Tuple>
    struct ApplyBuildChain;

    // @doc: template<typename In, BlockRecipe... Rs> struct ApplyBuildChain<In, std::tuple<Rs...> >
    /**
     * Expects an `BlockRecipe<Input>` first and a trailing variadic list of `BlockRecipe`s passes them to `BuildChain`
     * Used in `NetworkBuilder` to define `BlockTuple`, a `TrainableTensorNetwork`'s tuple of `Block`s
     */
    template<typename In, BlockRecipe... Rs>
    struct ApplyBuildChain<In, std::tuple<Rs...> > {
        using type = BuildChain<In, Rs...>::type;
    };


    // @doc: template<typename In, typename... Recipes> requires requires { typename In::OutputTensor; } && IsTensor<typename In::OutputTensor> && (BlockRecipe<Recipes> && ...) struct NetworkBuilder
    /**
     * Takes variadic list of `BlockRecipe`s and creates a `TrainableTensorNetwork`
     * Calls `ApplyBuildChain` to get a `std::tuple` of `Block`s, then calls `ConcretesToNetwork`
     */
    template<typename In, typename... Recipes> requires
        requires { typename In::OutputTensor; } && IsTensor<typename In::OutputTensor> && (BlockRecipe<Recipes> && ...)
    struct NetworkBuilder {
        using BlockTuple = ApplyBuildChain<In, std::tuple<Recipes...> >::type;
        using type = ConcretesToNetwork<BlockTuple>::type;
    };
}
