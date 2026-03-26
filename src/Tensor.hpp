#pragma once
#include <array>
#include <fstream>
#include "TensorStorage.hpp"
#include "TensorPrereqs.hpp"


namespace TTTN {
    template<size_t... Dims>
    class Tensor {
    public:
        // @doc: static constexpr size_t Rank
        /** Number of dimensions */
        static constexpr size_t Rank = sizeof...(Dims);
        static constexpr size_t GetRank() { return Rank; }
        // @doc: static constexpr size_t Size
        /** Product of all dimensions = total distinct values in `Tensor` */
        static constexpr size_t Size = TensorDimsProduct<Dims...>::value;
        static constexpr size_t GetSize() { return Size; }
        // @doc: static constexpr std::array<size_t, Rank> Shape
        /** `<size_t... Dims>` captured into an array */
        static constexpr std::array<size_t, Rank> Shape = {Dims...};
        static constexpr std::array<size_t, Rank> GetShape() { return Shape; }
        // @doc: static constexpr std::array<size_t, Rank> Strides
        /**
         * Uses `ComputeStrides` to create array
         * The `Tensor::Strides` array is vital to mapping from indices into `Tensor::Shape` to flat indices for the backing array
         * In general, for a `Tensor` with `Tensor::Shape = [A, B, ..., N]`, its `Tensor::Strides = [A * B * ... * N, B * ... * N, ..., 1]`
         */
        static constexpr std::array<size_t, Rank> Strides = ComputeStrides<Dims...>::value;
        static constexpr std::array<size_t, Rank> GetStrides() { return Strides; }

        // @doc: float flat(size_t idx) const
        /** Returns `const float&` reference to item at `idx` in underlying array */
        // @doc: float& flat(size_t idx)
        /** Returns `float&` reference to item at `idx` in underlying array */
        // @doc: static constexpr std::array<size_t, Rank> FlatToMulti(size_t flat)
        /**
         * Inverse of `MultiToFlat`; map a flat index `[0, Size)` to its `Rank`-term index
         * Pattern matches a `std::index_sequence` parameterized by `<size_t... Rank>` (= `[0, ..., Rank]`) and unpacks into an array: `[(flat / Strides[0]) % Shape[0], ..., (flat / Strides[Rank]) % Shape[Rank]]`
         */
        // raw conversion, no mapping
        static constexpr std::array<size_t, Rank> flat_to_multi(const size_t flat) {
            std::array<size_t, Rank> multi{};
            size_t remainder = flat;
            for (size_t i = 0; i < Rank; ++i) {
                multi[i] = remainder / Strides[i];
                remainder %= Strides[i];
            }
            return multi;
        }

        // public accessor, applies the current axis_map_
        constexpr std::array<size_t, Rank> FlatToMulti(const size_t flat) const {
            return apply_index_map(flat_to_multi(flat), inverse_axis_map_);
        }


        // @doc: static constexpr size_t MultiToFlat(const std::array<size_t, Rank>& multi)
        /**
         * Inverse of `FlatToMulti`; map a `Rank`-term index to its flat index `[0, Size)`
         * Pattern matches a `std::index_sequence` parameterized by `<size_t... Rank>` (= `[0, ..., Rank]`) and `+`-folds terms `multi[0] * Strides[0] + ... + multi[Rank] * Strides[Rank]`
         * Dot product of given `multi` index and `Strides`
         */
        static constexpr size_t multi_to_flat(const std::array<size_t, Rank> &multi) {
            return [&]<size_t... Is>(std::index_sequence<Is...>) {
                return (... + (multi[Is] * Strides[Is]));
            }(std::make_index_sequence<Rank>{});
        }

        constexpr size_t MultiToFlat(const std::array<size_t, Rank> &multi) {
            return multi_to_flat(apply_index_map(multi, axis_map_));
        }

    private:
        TensorStorage<Size> storage_;

        std::array<size_t, Rank> axis_map_{};
        std::array<size_t, Rank> inverse_axis_map_{};

        void init_axis_map() {
            for (size_t i = 0; i < Rank; ++i) axis_map_[i] = i;
        }

        template<size_t Rank>
        static std::array<size_t, Rank> apply_index_map(
            const std::array<size_t, Rank> &idx,
            const std::array<size_t, Rank> &map) {
            std::array<size_t, Rank> out{};
            for (size_t i = 0; i < Rank; ++i)
                out[i] = idx[map[i]];
            return out;
        }

        static constexpr std::array<size_t, Rank> make_inverse_map(const std::array<size_t, Rank> &map) {
            std::array<size_t, Rank> inv{};
            for (size_t i = 0; i < Rank; ++i) inv[map[i]] = i;
            return inv;
        }

    public:
        Tensor() {
            init_axis_map();
        }

        void permute_inplace(const std::array<size_t, Rank> &perm) {
            std::array<size_t, Rank> new_map{};
            for (size_t i = 0; i < Rank; ++i) {
                new_map[i] = axis_map_[perm[i]];
            }
            axis_map_ = new_map;
            inverse_axis_map_ = make_inverse_map(axis_map_);
        }

        // Remap multi-indices according to current permutation
        template<typename... Indices>
        float &operator()(Indices... idxs) {
            static_assert(sizeof...(idxs) == Rank, "Number of indices must match tensor rank");
            const size_t idx_arr[] = {static_cast<size_t>(idxs)...};
            size_t flat_index = 0;
            for (size_t i = 0; i < Rank; ++i)
                flat_index += idx_arr[axis_map_[i]] * Strides[i];
            return storage_.ptr()[flat_index];
        }

        template<typename... Indices>
        float operator()(Indices... idxs) const {
            static_assert(sizeof...(idxs) == Rank, "Number of indices must match tensor rank");
            const size_t idx_arr[] = {static_cast<size_t>(idxs)...};
            size_t flat_index = 0;
            for (size_t i = 0; i < Rank; ++i)
                flat_index += idx_arr[axis_map_[i]] * Strides[i];
            return storage_.ptr()[flat_index];
        }

        // Optional: remap array-based access too
        float &operator()(const std::array<size_t, Rank> &multi) {
            std::array<size_t, Rank> mapped{};
            for (size_t i = 0; i < Rank; ++i) mapped[i] = multi[axis_map_[i]];
            return storage_.ptr()[MultiToFlat(mapped)];
        }

        float operator()(const std::array<size_t, Rank> &multi) const {
            std::array<size_t, Rank> mapped{};
            for (size_t i = 0; i < Rank; ++i) mapped[i] = multi[axis_map_[i]];
            return storage_.ptr()[MultiToFlat(mapped)];
        }


        // @doc: ~Tensor() = default
        /**
         * Default destructor
         * RAII: destructs `std::unique_ptr` to `float[Size]` on heap
         */
        ~Tensor() = default;

        // @doc: Tensor(const Tensor& other)
        /**
         * Deep copy constructor
         * Allocate new `float[Size]` on heap and `std::memcpy` from `other.data()`
         */
        Tensor(const Tensor &other) : storage_(other.storage_) {
        }

        // @doc: Tensor& operator=(const Tensor& other)
        /**
         * Deep copy assignment operator
         * `std::memcpy` from `other.data()`
         */
        Tensor &operator=(const Tensor &other) {
            if (this != &other) storage_ = other.storage_;
            return *this;
        }

        // @doc: Tensor(Tensor&&) noexcept = default
        /**
         * Default move constructor
         * `std::unique_ptr` to data handles this already
         */
        Tensor(Tensor &&) noexcept = default;

        // @doc: Tensor &operator=(Tensor&&) noexcept = default
        /**
         * Default move assigment operator
         * `std::unique_ptr` to data handles this already
         */
        Tensor &operator=(Tensor &&) noexcept = default;

        // @doc: const float* data() const
        /** Returns `const` pointer to `Tensor`'s underlying array */
        // @doc: float* data()
        /** Returns pointer to `Tensor`'s underlying array */
        float *data() { return storage_.ptr(); }
        [[nodiscard]] const float *data() const { return storage_.ptr(); }

        float &flat(size_t idx) { return storage_.ptr()[idx]; }
        [[nodiscard]] float flat(size_t idx) const { return storage_.ptr()[idx]; }

        // implicit conversion from Rank-0 Tensor to float
        operator float() const requires (Rank == 0) { return storage_.ptr()[0]; }

        // @doc: void Save(std::ofstream& f) const
        /** Writes entirety of flat backing array (`float[Size]`) to binary file */
        void Save(std::ofstream &f) const {
            f.write(reinterpret_cast<const char *>(storage_.ptr()), Size * sizeof(float));
        }

        // @doc: void Load(std::ifstream& f)
        /** Reads binary file into flat backing array (`float[Size]`) */
        void Load(std::ifstream &f) {
            f.read(reinterpret_cast<char *>(storage_.ptr()), Size * sizeof(float));
        }


        // INPLACE FUNCTIONAL
        // in TensorFunctions.hpp, there are Copy/Move versions of these functions, who all use these internally!

        // @doc: void fill(const float v) const
        /** Fill a `Tensor`'s underlying array with some `float v` */
        void fill(const float v) { std::fill_n(storage_.ptr(), Size, v); }

        // @doc: template<typename F> void apply(F f)
        /** Use `std::execution::par_unseq` + `std::for_each` to apply `float -> float` map `f` to `Tensor`'s underlying data in-place */
        template<typename F>
            requires requires(F f)
            {
                { f(*storage_.ptr()) } -> std::same_as<float>;
            }
        void apply(F f) {
            std::for_each(std::execution::par_unseq, storage_.ptr(), storage_.ptr() + Size, f);
        }

        // @doc: template<FloatBinaryOp F> void zip_apply(const Tensor& other, F f)
        /** In-place binary transform: `self[i] = f(self[i], other[i])` for all `i`. Mutating counterpart to `zip`. Accepts any `FloatBinaryOp` including op tags: `zip_apply(b, Add{})` */
        template<FloatBinaryOp F>
        void zip_apply(const Tensor &other, F f) {
            std::transform(std::execution::par_unseq,
                           storage_.ptr(), storage_.ptr() + Size, other.storage_.ptr(), storage_.ptr(), f);
        }

        // @doc: template<FloatUnaryOp F> Tensor map(F f) const
        /** Use `std::execution::par_unseq` to `std::transform` `Tensor`'s underlying data by `float -> float` map `f`, returning a new `Tensor` */
        template<FloatUnaryOp F>
        void map(F f) {
            std::transform(std::execution::par_unseq,
                           storage_.ptr(), storage_.ptr() + Size, storage_.ptr(), f);
        }

        // @doc: template<FloatBinaryOp F> Tensor zip(const Tensor& other, F f) const
        /** Use `std::execution::par_unseq` to `std::transform` two `Tensor`s' underlying data by `float -> float -> float` map `f`, returning a new `Tensor` */
        template<FloatBinaryOp F>
        void zip(const Tensor &other, F f) {
            std::transform(std::execution::par_unseq,
                           storage_.ptr(), storage_.ptr() + Size, other.storage_.ptr(), f);
        }
    };

    // @doc: struct is_tensor<T>
    /**
     * SFINAE type traits for verifying that a type is a `Tensor`
     * Specialize `<size_t...Dims>`: matches into `<Tensor<Dims...>`, inherits from `std::true_type`
     * Specialize <>: inherits from `std::false_type`, backup when former fails
     */
    template<typename T>
    struct is_tensor : std::false_type {
    };

    template<size_t... Dims>
    struct is_tensor<Tensor<Dims...> > : std::true_type {
    };

    // @doc: concept IsTensor<T>
    /** Wrapper `concept` around `is_tensor` type trait, satisfied if `T` is a `Tensor` */
    template<typename T>
    concept IsTensor = is_tensor<T>::value;
}
