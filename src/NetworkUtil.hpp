#pragma once
#include <concepts>

namespace TTTN {
    // ConcreteBlock (hidden) concept
    // *any object* which satisfies these criteria should be referenced in a Block::Resolve to be in a TTN
    template<typename T>
    concept ConcreteBlock = requires(T t, const T ct,
                                     Tensor<T::InSize> in,
                                     Tensor<T::OutSize> out,
                                     std::ofstream &of, std::ifstream &inf)
    {
        { T::InSize } -> std::convertible_to<size_t>;
        { T::OutSize } -> std::convertible_to<size_t>;
        { ct.Forward(in) } -> std::same_as<Tensor<T::OutSize> >;
        { t.Backward(out, out, in) } -> std::same_as<Tensor<T::InSize> >;
        { t.Update(0.f, 0.f, 0.f, 0.f, 0.f, 0.f) };
        { ct.Save(of) };
        { t.Load(inf) };
    };

    // Block concept must be a recipe to create a ConcreteBlock
    // takes in OutSize (and other args, like ActivationFuncton for Dense)
    template<typename T>
    concept Block = requires(T t, Tensor<T::OutSize> out)
    {
        { T::OutSize } -> std::convertible_to<size_t>;
        // 'template' keyword tells compiler that we're not saying "(Resolve < 1)..." as an expr
    } && ConcreteBlock<typename T::template Resolve<1> >;

    template<ConcreteBlock... Blocks>
    class TrainableTensorNetwork;


    template<typename In, Block... Recipes>
    struct BuildChain;

    // base case, one last Block to parse
    template<typename Prev, Block Last>
    struct BuildChain<Prev, Last> {
        // type is a tuple of: the type of the Last Resolve (full concrete block) when its InSize is Prev's OutSize

        // if our final layer is a Dense which returns 10 items, and the block before it returns 15,
        // then, here, type is std::tuple<DenseBlock<15,10>>
        using type = std::tuple<typename Last::template Resolve<Prev::OutSize> >;
    };

    // typename (not Block) because Input is not a Block, though has OutSize
    template<typename Prev, Block Next, Block... Rest>
    struct BuildChain<Prev, Next, Rest...> {
        // make the type for the Next block
        using Resolved = typename Next::template Resolve<Prev::OutSize>;

        // type is tuple cat of:
        using type = decltype(std::tuple_cat(
            // the Resolved Block 'here' +
            std::declval<std::tuple<Resolved> >(),
            // recurse on the rest of the Blocks, with Prev now being Resolved
            std::declval<typename BuildChain<Resolved, Rest...>::type>()
        ));
    };

    // thin input struct which has an OutSize (== InSize)
    // all it needs is OutSize (not Resolve) because the recursive kickoff of BuildChain only grabs Prev's OutSize
    template<size_t N>
    struct Input {
        static constexpr size_t OutSize = N;
    };

    // typename (not Block) because Input is not a Block
    template<typename In, Block... Blocks>
    struct NetworkBuilder {
        using BlockTuple = typename BuildChain<In, Blocks...>::type;

        template<typename Tuple>
        struct Apply;

        template<typename... Bs>
        struct Apply<std::tuple<Bs...> > {
            using type = TrainableTensorNetwork<Bs...>;
        };

        using type = typename Apply<BlockTuple>::type;
    };
};
