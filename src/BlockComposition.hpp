#pragma once
#include "TrainableTensorNetwork.hpp"

namespace TTTN {
    template<typename... Recipes>
    struct ComposeBlocks {
        template<typename In, typename... Rs>
        struct ResolveChain;

        template<typename In, typename Last>
        struct ResolveChain<In, Last> {
            using Resolved = Last::template Resolve<In>;
            using type = std::tuple<Resolved>;
            using OutputT = Resolved::OutputTensor;
        };

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

        template<typename Last>
        struct LastOutputTensor<Last> {
            using type = Last::OutputTensor;
        };

        template<typename First, typename... Rest>
        struct LastOutputTensor<First, Rest...> {
            using type = LastOutputTensor<Rest...>::type;
        };

        using OutputTensor = LastOutputTensor<Recipes...>::type;

        template<typename InputT>
        struct ResolveImpl {
            using Chain = ResolveChain<InputT, Recipes...>;
            using BlockTuple = Chain::type;

            template<typename Tuple>
            struct Apply;

            template<typename... Bs>
            struct Apply<std::tuple<Bs...> > {
                using type = BlockSequence<Bs...>;
            };

            using type = Apply<BlockTuple>::type;
        };

        template<typename InputT>
        using Resolve = ResolveImpl<InputT>::type;
    };


    template<typename NetA, typename NetB>
    struct CombineNetworks;

    // @doc: template<ConcreteBlock... BlocksA, ConcreteBlock... BlocksB> struct CombineNetworks<TrainableTensorNetwork<BlocksA...>, TrainableTensorNetwork<BlocksB...> >
    /**
     * Unpacks two `ConcreteBlock...` arg lists into a new `TrainableTensorNetwork` composed of both sets
     * Asserts `std::is_same_v<OutputTensor of A, InputTensor of B>`
     */
    template<ConcreteBlock... BlocksA, ConcreteBlock... BlocksB>
    struct CombineNetworks<TrainableTensorNetwork<BlocksA...>, TrainableTensorNetwork<BlocksB...> > {
        static_assert(
            std::is_same_v<
                typename TrainableTensorNetwork<BlocksA...>::OutputTensor,
                typename TrainableTensorNetwork<BlocksB...>::InputTensor>,
            "CombineNetworks: OutputTensor of first network must equal InputTensor of second");
        using type = TrainableTensorNetwork<BlocksA..., BlocksB...>;
    };


    // @doc: template<typename In, typename... Recipes> struct NetworkBuilder
    /** Takes an `Input<Dims...>` and a variadic list of `Block` recipes; resolves them via `BuildChain` into a `TrainableTensorNetwork` type alias at `NetworkBuilder::type` */
    template<typename In, typename... Recipes>
    struct NetworkBuilder {
        using BlockTuple = ApplyBuildChain<In, std::tuple<Recipes...> >::type;

        template<typename Tuple>
        struct Apply;

        template<typename... Bs>
        struct Apply<std::tuple<Bs...> > {
            using type = TrainableTensorNetwork<Bs...>;
        };

        // now just unpack tuple directly into TrainableTensorNetwork
        using type = Apply<BlockTuple>::type;
    };
}
