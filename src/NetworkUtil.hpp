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


    // ── ComposeBlocks: recipe combiner that vanishes at NetworkBuilder time ──────
    //
    // Groups sub-recipes into a single type alias.  NetworkBuilder flattens it
    // into its components before BuildChain runs — no runtime wrapper, no extra
    // block in the network.  Nesting works: ComposeBlocks<ComposeBlocks<A,B>, C>
    // flattens to A, B, C.
    //
    // Usage:
    //   using TransformerFFN = ComposeBlocks<
    //       MapDense<1, Tensor<FFN>, ReLU>,
    //       MapDense<1, Tensor<Emb>>
    //   >;
    //   using TBlock = ComposeBlocks<MHAttention<4, Emb>, TransformerFFN>;
    //
    //   NetworkBuilder<Input<Seq, Emb>, TBlock, TBlock, Dense<10>>::type net;
    //   // ≡  MHAttention, MapDense, MapDense, MHAttention, MapDense, MapDense, Dense
    template<typename... Recipes>
    struct ComposeBlocks {};

    // FlattenRecipes: recursively expand ComposeBlocks → std::tuple<Block...>
    template<typename... Rs> struct FlattenRecipes;

    template<>
    struct FlattenRecipes<> { using type = std::tuple<>; };

    // ComposeBlocks: splice its sub-recipes into the flat list
    template<typename... SubRs, typename... Rest>
    struct FlattenRecipes<ComposeBlocks<SubRs...>, Rest...> {
        using type = decltype(std::tuple_cat(
            std::declval<typename FlattenRecipes<SubRs...>::type>(),
            std::declval<typename FlattenRecipes<Rest...>::type>()));
    };

    // anything else: keep it and continue
    template<typename First, typename... Rest>
    struct FlattenRecipes<First, Rest...> {
        using type = decltype(std::tuple_cat(
            std::declval<std::tuple<First>>(),
            std::declval<typename FlattenRecipes<Rest...>::type>()));
    };

    // ApplyBuildChain: unpack a tuple of Block recipes into BuildChain
    template<typename In, typename Tuple> struct ApplyBuildChain;

    template<typename In, Block... Rs>
    struct ApplyBuildChain<In, std::tuple<Rs...>> {
        using type = typename BuildChain<In, Rs...>::type;
    };


    // prepends a Batch dimension to any Tensor<Dims...> type
    template<size_t Batch, typename T>
    struct PrependBatch;

    template<size_t Batch, size_t... Dims>
    struct PrependBatch<Batch, Tensor<Dims...>> {
        using type = Tensor<Batch, Dims...>;
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
        explicit ActivationsWrap(TupleT t) : data_(std::move(t)) {}

        // Safe: reference into owned tuple; valid as long as this wrapper is alive.
        template<size_t N>
        auto get() const& -> const std::tuple_element_t<N, TupleT>& {
            return std::get<N>(data_);
        }
        template<size_t N>
        auto get() & -> std::tuple_element_t<N, TupleT>& {
            return std::get<N>(data_);
        }

        // Deleted: `temporary_wrap.get<N>()` is a compile error, not a dangle.
        template<size_t N>
        auto get() && -> std::tuple_element_t<N, TupleT>&& = delete;

        // Raw tuple access for internal backward pass machinery.
        const TupleT& tuple() const { return data_; }
    };
};