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
        // @doc: void tick(const std::string& suffix = "", size_t n = 1)
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
    // Binary cache: one file per CSV, named <path>.bin, with an 8-byte header
    // storing (R_stored, C_stored) as uint32_t[2].  The cache always holds the
    // largest version ever parsed.
    //
    //   Request <= stored (same cols): result.Load reads exactly Rows*Cols floats — zero waste.
    //   Request <= stored (fewer cols): row-by-row read, skipping trailing columns.
    //   Request > stored: re-parse CSV and overwrite the cache with the new larger slice.
    //
    // Delete <path>.bin to force a re-parse (e.g. after the CSV changes).
    // @doc: template<size_t Rows, size_t Cols> Tensor<Rows, Cols> LoadCSV(const std::string& path, bool skip_header = false)
    /** Parses a CSV into a `Tensor<Rows, Cols>`. On first call shows a progress bar, then writes a binary cache at `<path>.<Rows>x<Cols>.bin`; subsequent calls load that file directly (pure binary read, no CSV parsing). Delete the `.bin` file if the underlying CSV changes. */
    template<size_t Rows, size_t Cols>
    Tensor<Rows, Cols> LoadCSV(const std::string& path, bool skip_header = false) {
        const std::string cache_path = path + ".bin";

        // fast path: cache exists and covers the request
        {
            std::ifstream cache(cache_path, std::ios::binary);
            if (cache) {
                uint32_t R_stored, C_stored;
                cache.read(reinterpret_cast<char*>(&R_stored), sizeof(uint32_t));
                cache.read(reinterpret_cast<char*>(&C_stored), sizeof(uint32_t));

                if (Rows <= R_stored && Cols <= C_stored) {
                    Tensor<Rows, Cols> result;
                    if (Cols == C_stored) {
                        // Rows are contiguous: read exactly Rows*Cols floats
                        result.Load(cache);
                    } else {
                        // Fewer columns requested: read row-by-row, skip trailing cols
                        std::vector<float> row_buf(C_stored);
                        for (size_t r = 0; r < Rows; ++r) {
                            cache.read(reinterpret_cast<char*>(row_buf.data()), C_stored * sizeof(float));
                            for (size_t c = 0; c < Cols; ++c)
                                result(r, c) = row_buf[c];
                        }
                    }
                    std::cout << "LoadCSV: cache hit " << cache_path
                              << " [" << R_stored << "x" << C_stored << " -> " << Rows << "x" << Cols << "]\n";
                    return result;
                }

                std::cout << "LoadCSV: cache too small (" << R_stored << "x" << C_stored
                          << "), re-parsing for " << Rows << "x" << Cols << "\n";
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
            if (row * 100 / Rows > last_ticked * 100 / Rows || row == Rows) {
                bar.tick("", row - last_ticked);
                last_ticked = row;
            }
        }

        // write/overwrite cache
        std::cout << "  cached -> " << cache_path << "\n";
        if (std::ofstream cache(cache_path, std::ios::binary); cache) {
            const uint32_t R = static_cast<uint32_t>(Rows);
            const uint32_t C = static_cast<uint32_t>(Cols);
            cache.write(reinterpret_cast<const char*>(&R), sizeof(uint32_t));
            cache.write(reinterpret_cast<const char*>(&C), sizeof(uint32_t));
            result.Save(cache);
        }

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
