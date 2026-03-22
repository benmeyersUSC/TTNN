#pragma once
#include <concepts>

namespace TTTN {
    // ConcreteBlock (hidden) concept
    // any block in a TTN must define InputTensor and OutputTensor type aliases (must satisfy IsTensor),
    // and implement Forward/Backward/Update/ZeroGrad/Save/Load with matching signatures
    template<typename T>
    concept ConcreteBlock =
        requires { typename T::InputTensor; } &&
        requires { typename T::OutputTensor; } &&
        IsTensor<typename T::InputTensor> &&      
        IsTensor<typename T::OutputTensor> &&     
        requires(T t, const T ct,
                 typename T::InputTensor  in,
                 typename T::OutputTensor out,
                 std::ofstream &of, std::ifstream &inf)
        {
            { ct.Forward(in)           } -> std::same_as<typename T::OutputTensor>;
            { t.Backward(out, out, in) } -> std::same_as<typename T::InputTensor>;
            { t.Update(0.f, 0.f, 0.f, 0.f, 0.f, 0.f) };
            { t.ZeroGrad() };
            { ct.Save(of) };
            { t.Load(inf) };
        };

    // Block concept: a recipe type that advertises its OutputTensor and can be Resolved
    // with any input tensor type into a ConcreteBlock.
    // Self-composition check: Resolve<OutputTensor> must itself satisfy ConcreteBlock --
    // proves the recipe mechanism is valid for at least one instantiation (pure SFINAE).
    template<typename B>
    concept Block =
        requires { typename B::OutputTensor; } &&
        IsTensor<typename B::OutputTensor> &&
        ConcreteBlock<typename B::template Resolve<typename B::OutputTensor>>;

    


    template<typename In, Block... Recipes>
    struct BuildChain;

    // base case: one last Block to parse
    template<typename Prev, Block Last>
    struct BuildChain<Prev, Last> {
        // Resolve the last recipe using the previous block's OutputTensor as its InputTensor
        using type = std::tuple<typename Last::template Resolve<typename Prev::OutputTensor>>;
    };

    // typename (not Block) because Input is not a Block, though it has OutputTensor
    template<typename Prev, Block Next, Block... Rest>
    struct BuildChain<Prev, Next, Rest...> {
        // resolve Next with the previous OutputTensor
        using Resolved = typename Next::template Resolve<typename Prev::OutputTensor>;

        using type = decltype(std::tuple_cat(
            std::declval<std::tuple<Resolved>>(),
            std::declval<typename BuildChain<Resolved, Rest...>::type>()
        ));
    };

    // Input<Dims...>: the entry point of a NetworkBuilder chain.
    // Exposes OutputTensor = Tensor<Dims...> so BuildChain can thread it into the first Block's Resolve.
    template<size_t... Dims>
    struct Input {
        using OutputTensor = Tensor<Dims...>;
    };


    // prepends a Batch dimension to any Tensor<Dims...> type
    template<size_t Batch, typename T>
    struct PrependBatch;

    template<size_t Batch, size_t... Dims>
    struct PrependBatch<Batch, Tensor<Dims...>> {
        using type = Tensor<Batch, Dims...>;
    };
};