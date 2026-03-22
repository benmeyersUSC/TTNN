#pragma once
#include <fstream>
#include <sstream>
#include <string>
#include <random>
#include <stdexcept>
#include <iostream>
#include "Tensor.hpp"

namespace TTTN {

    // ProgressBar: lightweight terminal progress bar.
    // Construct with total tick count and an optional label, then call tick() each step.
    //
    // Example — training loop:
    //   ProgressBar bar(epochs * steps, "training");
    //   for (int e = 0; e < epochs; ++e) {
    //       bar.set_label("epoch " + std::to_string(e));
    //       for (int s = 0; s < steps; ++s) {
    //           float loss = net.BatchFit<CEL, B>(X, Y, lr);
    //           bar.tick("loss=" + std::to_string(loss));
    //       }
    //   }
    class ProgressBar {
        size_t total_;
        size_t current_ = 0;
        std::string label_;
        static constexpr size_t width_ = 40;

        void render(const std::string& suffix) const {
            const size_t filled = current_ < total_ ? width_ * current_ / total_ : width_;
            std::cout << "\r";
            if (!label_.empty()) std::cout << label_ << "  ";
            std::cout << "[";
            for (size_t i = 0; i < width_; ++i)
                std::cout << (i < filled ? '#' : '.');
            std::cout << "] " << current_ << "/" << total_;
            if (!suffix.empty()) std::cout << "  " << suffix;
            if (current_ >= total_) std::cout << "\n";
            std::cout << std::flush;
        }

    public:
        explicit ProgressBar(size_t total, std::string label = "")
            : total_(total), label_(std::move(label)) {}

        // Advance by n steps. Optional suffix printed to the right (e.g. "loss=0.312").
        /** Advances by `n` steps and redraws. `suffix` is printed to the right of the bar (e.g. `"loss=0.312"`). */
        void tick(const std::string& suffix = "", size_t n = 1) {
            current_ = std::min(current_ + n, total_);
            render(suffix);
        }

        void set_label(const std::string& label) { label_ = label; }
        void reset() { current_ = 0; }
    };

    // Load a CSV file into a compile-time-typed Tensor<Rows, Cols>.
    // The caller states the dataset dimensions at compile time.
    // skip_header: if true, discards the first line before reading data.
    //
    // Binary cache: on first load the parsed tensor is written to
    //   <path>.<Rows>x<Cols>.bin
    // Subsequent calls load that file directly (pure fread, no CSV parsing).
    // Delete the .bin file if the underlying CSV changes.
    /** Parses a CSV into a `Tensor<Rows, Cols>`. On first call shows a progress bar, then writes a binary cache at `<path>.<Rows>x<Cols>.bin`; subsequent calls load that file directly (pure binary read, no CSV parsing). Delete the `.bin` file if the underlying CSV changes. */
    template<size_t Rows, size_t Cols>
    Tensor<Rows, Cols> LoadCSV(const std::string& path, bool skip_header = false) {
        const std::string cache_path =
            path + "." + std::to_string(Rows) + "x" + std::to_string(Cols) + ".bin";

        // fast path: binary cache exists
        {
            std::ifstream cache(cache_path, std::ios::binary);
            if (cache) {
                Tensor<Rows, Cols> result;
                result.Load(cache);
                std::cout << "LoadCSV: loaded cache " << cache_path << "\n";
                return result;
            }
        }

        // slow path: parse CSV
        std::ifstream f(path);
        if (!f) throw std::runtime_error("LoadCSV: cannot open: " + path);
        Tensor<Rows, Cols> result;
        std::string line;
        if (skip_header) std::getline(f, line);
        std::cout << "LoadCSV: parsing " << path << " [" << Rows << "x" << Cols << "]\n";
        ProgressBar bar(Rows);
        size_t row = 0, last_ticked = 0;
        while (std::getline(f, line) && row < Rows) {
            std::stringstream ss(line);
            std::string val;
            size_t col = 0;
            while (std::getline(ss, val, ',') && col < Cols)
                result(row, col++) = std::stof(val);
            ++row;
            // tick once per 1% boundary (or on the final row)
            if (row * 100 / Rows > last_ticked * 100 / Rows || row == Rows) {
                bar.tick("", row - last_ticked);
                last_ticked = row;
            }
        }
        std::cout << "  cached -> " << cache_path << "\n";

        // write cache for next time (best-effort — don't fail if write fails)
        if (std::ofstream cache(cache_path, std::ios::binary); cache)
            result.Save(cache);

        return result;
    }

    // Sample Batch random rows (with replacement) from Tensor<N, RestDims...>.
    // Returns Tensor<Batch, RestDims...>.
    template<size_t Batch, size_t N, size_t... RestDims>
    Tensor<Batch, RestDims...> RandomBatch(const Tensor<N, RestDims...>& ds, std::mt19937& rng) {
        static_assert(Batch <= N, "Batch cannot exceed dataset size");
        using RowType = Tensor<RestDims...>;
        Tensor<Batch, RestDims...> result;
        std::uniform_int_distribution<size_t> dist{0, N - 1};
        for (size_t b = 0; b < Batch; ++b) {
            const size_t idx = dist(rng);
            for (size_t i = 0; i < RowType::Size; ++i)
                result.flat(b * RowType::Size + i) = ds.flat(idx * RowType::Size + i);
        }
        return result;
    }

} // namespace TTTN
