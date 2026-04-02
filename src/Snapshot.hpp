#pragma once
#include "Tensor.hpp"
#include <string>
#include <vector>
#include <unordered_map>

namespace TTTN {
    // @doc: struct SnapshotEntry
    /** `struct` to hold `data` and `shape` from `PeekableBlock`s' activation snapshots */
    struct SnapshotEntry {
        // @doc: std::vector<size_t> SnapshotEntry::shape
        /** `std::vector<size_t>` to hold an activation `Tensor`'s shape */
        std::vector<size_t> shape; 
        // @doc: std::vector<float> SnapshotEntry::data
        /** `std::vector<float>` to hold an activation `Tensor`'s data */
        std::vector<float> data;

        // @doc: [[nodiscard]] size_t SnapshotEntry::total() const      
        /** Getter for total size */
        [[nodiscard]] size_t total() const { return data.size(); }

        // @doc: [[nodiscard]] size_t SnapshotEntry::rows(const size_t outer_axis = 0) const
        /** Getter for number of rows */
        [[nodiscard]] size_t rows(const size_t outer_axis = 0) const {
            return outer_axis < shape.size() ? shape[outer_axis] : 1;
        }
        // @doc: [[nodiscard]] size_t SnapshotEntry::cols() const
        /** Getter for number of columns */
        [[nodiscard]] size_t cols() const {
            size_t c = 1;
            for (size_t i = 1; i < shape.size(); ++i) {
                c *= shape[i];
            }
            return c;
        }
    };

    // @doc: using SnapshotMap
    /** Type alias for `std::unordered_map<std::string, SnapshotEntry>` */
    using SnapshotMap = std::unordered_map<std::string, SnapshotEntry>;


    // @doc: template<size_t... Dims> void snap_add(SnapshotMap &out, const std::string &key, const Tensor<Dims...> &t)
    /** Take a `Tensor`, a `std::string` key, and an existing `SnapshotMap` and add the `Tensor` as a `SnapshotEntry` */
    template<size_t... Dims> void snap_add(SnapshotMap &out, const std::string &key, const Tensor<Dims...> &t) {
        SnapshotEntry entry;
        entry.shape = {Dims...};
        entry.data.resize(Tensor<Dims...>::Size);
        for (size_t i = 0; i < Tensor<Dims...>::Size; ++i)
            entry.data[i] = t.flat(i);
        out.emplace(key, std::move(entry));
    }
} // namespace TTTN
