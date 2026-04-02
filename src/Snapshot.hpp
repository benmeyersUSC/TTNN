#pragma once
#include "Tensor.hpp"
#include <string>
#include <vector>
#include <unordered_map>

namespace TTTN {
    struct SnapshotEntry {
        std::vector<size_t> shape; // compile-time dims captured at runtime
        std::vector<float> data; // flat copy of the tensor's values

        [[nodiscard]] size_t total() const { return data.size(); }

        // Interpret as a 2-D slice: shape[outer_axis] rows × inner columns.
        // For rank-3 tensors (e.g. Tensor<H,S,S>) index the outer axis first.
        [[nodiscard]] size_t rows(const size_t outer_axis = 0) const {
            return outer_axis < shape.size() ? shape[outer_axis] : 1;
        }

        [[nodiscard]] size_t cols() const {
            size_t c = 1;
            for (size_t i = 1; i < shape.size(); ++i) c *= shape[i];
            return c;
        }
    };

    using SnapshotMap = std::unordered_map<std::string, SnapshotEntry>;

    template<size_t... Dims>
    void snap_add(SnapshotMap &out, const std::string &key, const Tensor<Dims...> &t) {
        SnapshotEntry entry;
        entry.shape = {Dims...};
        entry.data.resize(Tensor<Dims...>::Size);
        for (size_t i = 0; i < Tensor<Dims...>::Size; ++i)
            entry.data[i] = t.flat(i);
        out.emplace(key, std::move(entry));
    }
} // namespace TTTN
