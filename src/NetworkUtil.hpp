#pragma once
#include <concepts>
#include "Tensor.hpp"
#include "Snapshot.hpp"
#include <cmath>
#include <fstream>
#include <tuple>


namespace TTTN {
    struct AdamState {
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float eps = 1e-8f;
        float mCorr = 1.f; // 1 / (1 - b1^t)
        float vCorr = 1.f; // 1 / (1 - b2^t)
        int t = 0;

        // @doc: void AdamState::step()
        /**
         * Increment `t`
         * Recompute `mCorr`, `vCorr`
         */
        void step() {
            ++t;
            mCorr = 1.f / (1.f - std::pow(beta1, static_cast<float>(t)));
            vCorr = 1.f / (1.f - std::pow(beta2, static_cast<float>(t)));
        }
    };


    // @doc: template<typename TensorT> struct Param
    /** `struct` layer around a `ConcreteBlock` to abstract away management, Adam updates */
    template<typename TensorT>
    struct Param {
        TensorT value{};
        TensorT grad{};
        TensorT m{};
        TensorT v{};

        // @doc: static constexpr size_t Param::Size
        /** Size of parameter `Tensor` */
        static constexpr size_t Size = TensorT::Size;


        // @doc: void Param::zero_grad()
        /** Fill `grad` with `0.f` */
        void zero_grad() { grad.fill(0.f); }

        // @doc: void Param::update(const AdamState &adam, float lr)
        /** For each `float` parameter in `value`, use Adam moments and gradient to update */
        void update(const AdamState &adam, float lr) {
            ParForEach(Size, [&](const size_t i) {
                const float g = grad.flat(i);
                m.flat(i) = adam.beta1 * m.flat(i) + (1.f - adam.beta1) * g;
                v.flat(i) = adam.beta2 * v.flat(i) + (1.f - adam.beta2) * g * g;
                value.flat(i) -= lr * (m.flat(i) * adam.mCorr) / (std::sqrt(v.flat(i) * adam.vCorr) + adam.eps);
            });
        }

        // @doc: void Param::save(std::ofstream &f) const
        /** Call `Tensor::Save` on `value` */
        void save(std::ofstream &f) const { value.Save(f); }

        // @doc: void Param::save(std::ifstream &f)
        /** Call `Tensor::Load` on `value` */
        void load(std::ifstream &f) { value.Load(f); }
    };

    template<typename T>
    struct is_param : std::false_type {
    };

    template<typename TensorT>
    struct is_param<Param<TensorT> > : std::true_type {
    };

    // @doc: template<typename T> concept IsParam
    /** Concept to verify that a type `T` is a `Param` */
    template<typename T>
    concept IsParam = is_param<std::remove_cvref_t<T> >::value;

    template<typename Tuple>
    struct all_params_check : std::false_type {
    };

    template<typename... Ts>
    struct all_params_check<std::tuple<Ts...> > : std::bool_constant<(IsParam<Ts> && ...)> {
    };

    // @doc: template<typename Tuple> concept IsParamTuple
    /** Concept to verify that a type `Tuple` is a `std::tuple` of `Param` objects */
    template<typename Tuple>
    concept IsParamTuple = all_params_check<std::remove_cvref_t<Tuple> >::value;


    // @doc: template<IsParamTuple Tuple> void ZeroAllGrads(Tuple &&params)
    /** Calls `Param::zero_grad` on each `Param` in the `std::tuple` of `Param`s */
    template<IsParamTuple Tuple>
    void ZeroAllGrads(Tuple &&params) {
        std::apply([](auto &... p) { (p.zero_grad(), ...); }, params);
    }

    // @doc: template<IsParamTuple Tuple> void UpdateAll(Tuple &&params)
    /** Calls `Param::update` on each `Param` in the `std::tuple` of `Param`s */
    template<IsParamTuple Tuple>
    void UpdateAll(Tuple &&params, const AdamState &adam, float lr) {
        std::apply([&](auto &... p) { (p.update(adam, lr), ...); }, params);
    }

    // @doc: template<IsParamTuple Tuple> void SaveAll(Tuple &&params)
    /** Calls `Param::save` on each `Param` in the `std::tuple` of `Param`s */
    template<IsParamTuple Tuple>
    void SaveAll(Tuple &&params, std::ofstream &f) {
        std::apply([&](const auto &... p) { (p.save(f), ...); }, params);
    }

    // @doc: template<IsParamTuple Tuple> void LoadAll(Tuple &&params)
    /** Calls `Param::load` on each `Param` in the `std::tuple` of `Param`s */
    template<IsParamTuple Tuple>
    void LoadAll(Tuple &&params, std::ifstream &f) {
        std::apply([&](auto &... p) { (p.load(f), ...); }, params);
    }


    // @doc: template<IsParam... Params> constexpr size_t TotalParamSize
    /**
     * Sum of all `Param` sizes in variadic list of `Param`s
     * `(Params::Size + ...)`
     */
    template<IsParam... Params>
    constexpr size_t TotalParamSize = (Params::Size + ...);


    // @doc: template<IsParamTuple Tuple, size_t... Is> constexpr size_t tuple_param_count_impl(std::index_sequence<Is...>)
    /** Unpacks `IsParamTuple` and sums each `Param::Size`, giving the net size of a `std::tuple` of `Param`s */
    template<IsParamTuple Tuple, size_t... Is>
    constexpr size_t tuple_param_count_impl(std::index_sequence<Is...>) {
        return (static_cast<size_t>(0) + ... + std::remove_reference_t<std::tuple_element_t<Is, Tuple> >::Size);
    }


    // @doc: template<IsParamTuple Tuple> constexpr size_t TupleParamCount
    /**
     * Sum of all `Param` sizes in a `std::tuple` of `Param`s
     * Calls `tuple_param_count_impl`
     */
    template<IsParamTuple Tuple>
    constexpr size_t TupleParamCount =
            tuple_param_count_impl<std::remove_cvref_t<Tuple> >(
                std::make_index_sequence<std::tuple_size_v<std::remove_cvref_t<Tuple> > >{});


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

    // // @doc: template<typename B> concept Block
    // /**
    //  * Declarable recipe to define a `ConcreteBlock` in a `TrainableTensorNetwork` template argument list
    //  * `Block`s must define an `OutputTensor` type and alias a `ConcreteBlock` as `Resolve`
    //  * `Block` argument lists passed to `NetworkBuilder` will be resolved into full `ConcreteBlock`s with chained `InputTensor` attributes
    //  */
    // template<typename B>
    // concept Block =
    //         requires { typename B::OutputTensor; } &&
    //         IsTensor<typename B::OutputTensor> &&
    //         ConcreteBlock<typename B::template Resolve<typename B::OutputTensor> >;
    //

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
