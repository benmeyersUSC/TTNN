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


    // @doc: template<typename TupleT> class ActivationsWrap
    /**
     * Wrapper around a `std::tuple` of `Tensor`s representing intermediate activations of a `TrainableTensorNetwork`
     * Internally stores `std::tuple` and provides safe access to elements and entire `std::tuple` via overloaded `get` methods
     */
    template<typename TupleT>
    class ActivationsWrap {
        TupleT data_;

    public:
        // @doc: explicit ActivationsWrap::ActivationsWrap(TupleT t)
        /** `explicit` constructor which `move`s incoming `std::tuple` into member `data_` */
        explicit ActivationsWrap(TupleT t) : data_(std::move(t)) {
        }

        // @doc: template<size_t N> auto ActivationsWrap::get() const & -> const std::tuple_element_t<N, TupleT> &
        /** `const &` getter, valid as long as `ActivationsWrap` object exists */
        template<size_t N>
        auto get() const & -> const std::tuple_element_t<N, TupleT> & {
            return std::get<N>(data_);
        }

        // @doc: template<size_t N> auto ActivationsWrap::get() & -> std::tuple_element_t<N, TupleT> &
        /** `&` getter, valid as long as `ActivationsWrap` object exists */
        template<size_t N>
        auto get() & -> std::tuple_element_t<N, TupleT> & {
            return std::get<N>(data_);
        }

        // @doc: template<size_t N> auto ActivationsWrap::get() && -> std::tuple_element_t<N, TupleT> &&
        /**
         * Explicitly `delete`d function!
         * Getting temporary activation `Tensor` from temporary `ActivationsWrap` is a compile error, you must bind the `ActivationsWrap` object to a variable and get a reference
         * This is because it would return a dangling reference to a soon-deleted `ActivationsWrap` object
         * Instead of:
         * `auto& act = net.BatchedForwardAll(X).get<2>();`
         * You must do:
         * `auto& wrap = net.BatchedForwardAll(X);`
         * `auto& act = wrap.get<2>();`
         */
        template<size_t N>
        auto get() && -> std::tuple_element_t<N, TupleT> && = delete;

        // @doc: const TupleT &ActivationsWrap::tuple() const
        /** `const &` to raw `std::tuple` */
        const TupleT &tuple() const { return data_; }
    };


    template<typename... Bs>
    struct TensorTupleBuilder;

    // @doc: template<typename Last> struct TensorTupleBuilder<Last>
    /**
     * Recursively build `std::tuple` of `Tensor` objects representing intermediate activations of the network, wrapped by `ActivationsWrap`
     * Base case: one single `ConcreteBlock` left, whose `InputTensor` and `OutputTensor` are wrapped in a `std::tuple`
     */
    template<typename Last>
    struct TensorTupleBuilder<Last> {
        using type = std::tuple<typename Last::InputTensor, typename Last::OutputTensor>;
    };

    // @doc: template<typename First, typename... Rest> struct TensorTupleBuilder<First, Rest...>
    /**
     * Recursively build `std::tuple` of `Tensor` objects representing intermediate activations of the network, wrapped by `ActivationsWrap`
     * Recursive case: `std::tuple_cat` of `First` `InputTensor` object and `TensorTupleBuilder<Rest...>`
     */
    template<typename First, typename... Rest>
    struct TensorTupleBuilder<First, Rest...> {
        using type = decltype(std::tuple_cat(
            std::declval<std::tuple<typename First::InputTensor> >(),
            std::declval<typename TensorTupleBuilder<Rest...>::type>()
        ));
    };


    template<size_t Batch, typename... Bs>
    struct BatchedTensorTupleBuilder;

    // @doc: template<size_t Batch, typename Last> struct BatchedTensorTupleBuilder<Batch, Last>
    /**
     * For `Batched` functions and use-cases, create a `Batched` version of a `std::tuple` of activations by passing `PrependBatch<Batch, ...>` on all `Tensor`s that `TensorTupleBuilder` adds raw
     * Base case: one single `ConcreteBlock` left, whose `InputTensor` and `OutputTensor` are wrapped in `PrependBatch<Batch, ...>` and then in a `std::tuple`
     */
    template<size_t Batch, typename Last>
    struct BatchedTensorTupleBuilder<Batch, Last> {
        using type = std::tuple<
            typename PrependBatch<Batch, typename Last::InputTensor>::type,
            typename PrependBatch<Batch, typename Last::OutputTensor>::type>;
    };

    // @doc: template<size_t Batch, typename First, typename... Rest> struct BatchedTensorTupleBuilder<Batch, First, Rest...>
    /**
     * For `Batched` functions and use-cases, create a `Batched` version of a `std::tuple` of activations by passing `PrependBatch<Batch, ...>` on all `Tensor`s that `TensorTupleBuilder` adds raw
     * Recursive case: `std::tuple_cat` of `First` `PrependBatch<Batch, InputTensor>` object and `BatchedTensorTupleBuilder<Rest...>`
     */
    template<size_t Batch, typename First, typename... Rest>
    struct BatchedTensorTupleBuilder<Batch, First, Rest...> {
        using type = decltype(std::tuple_cat(
            std::declval<std::tuple<typename PrependBatch<Batch, typename First::InputTensor>::type> >(),
            std::declval<typename BatchedTensorTupleBuilder<Batch, Rest...>::type>()
        ));
    };
};
