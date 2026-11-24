#include "core/objective.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "util/threading.hpp"

namespace miter {

double sigmoid(double z) noexcept {
    if (z >= 0.0) {
        const double ez = std::exp(-z);
        return 1.0 / (1.0 + ez);
    }
    const double ez = std::exp(z);
    return ez / (1.0 + ez);
}

void validate_targets_and_weights(Objective obj, std::span<const double> y,
                                  std::span<const double> sample_weight) {
    if (y.empty()) {
        throw std::invalid_argument("Target array y must be non-empty");
    }
    if (!sample_weight.empty() && sample_weight.size() != y.size()) {
        throw std::invalid_argument("sample_weight length must match y length");
    }

    for (std::size_t i = 0; i < y.size(); ++i) {
        const double yi = y[i];
        if (!std::isfinite(yi)) {
            throw std::invalid_argument("Target array y contains NaN or Inf");
        }
        if (obj == Objective::Binary && (yi < 0.0 || yi > 1.0)) {
            throw std::invalid_argument("Binary objective requires targets in [0, 1]");
        }
    }

    if (!sample_weight.empty()) {
        double weight_sum = 0.0;
        for (const double w : sample_weight) {
            if (!std::isfinite(w) || w < 0.0) {
                throw std::invalid_argument("sample_weight must be finite and non-negative");
            }
            weight_sum += w;
        }
        if (weight_sum <= 0.0) {
            throw std::invalid_argument("Sum of sample_weight must be positive");
        }
    }
}

double compute_init_score(Objective obj, std::span<const double> y,
                          std::span<const double> sample_weight, int n_jobs) {
    const std::size_t n = y.size();
    const bool has_w = !sample_weight.empty();
    const std::size_t n_chunks = (n + kDetChunkRows - 1U) / kDetChunkRows;

    double sum_wy = 0.0;
    double sum_w = 0.0;

    if (n_chunks <= 1U) {
        if (!has_w) {
            for (const double yi : y) {
                sum_wy += yi;
            }
            sum_w = static_cast<double>(n);
        } else {
            for (std::size_t i = 0; i < n; ++i) {
                sum_wy += sample_weight[i] * y[i];
                sum_w += sample_weight[i];
            }
        }
    } else {
        std::vector<double> c_wy(n_chunks, 0.0);
        std::vector<double> c_w(n_chunks, 0.0);
        const int n_threads = resolve_num_threads(n_jobs);
        const bool use_omp = (n_threads > 1) && (n >= 32768U);
        if (use_omp) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
            for (std::ptrdiff_t c = 0; c < static_cast<std::ptrdiff_t>(n_chunks); ++c) {
                const auto cu = static_cast<std::size_t>(c);
                const std::size_t i0 = cu * kDetChunkRows;
                const std::size_t i1 = std::min(i0 + kDetChunkRows, n);
                double l_wy = 0.0;
                double l_w = 0.0;
                if (!has_w) {
                    for (std::size_t i = i0; i < i1; ++i) {
                        l_wy += y[i];
                    }
                    l_w = static_cast<double>(i1 - i0);
                } else {
                    for (std::size_t i = i0; i < i1; ++i) {
                        l_wy += sample_weight[i] * y[i];
                        l_w += sample_weight[i];
                    }
                }
                c_wy[cu] = l_wy;
                c_w[cu] = l_w;
            }
        } else {
            for (std::size_t c = 0; c < n_chunks; ++c) {
                const std::size_t i0 = c * kDetChunkRows;
                const std::size_t i1 = std::min(i0 + kDetChunkRows, n);
                double l_wy = 0.0;
                double l_w = 0.0;
                if (!has_w) {
                    for (std::size_t i = i0; i < i1; ++i) {
                        l_wy += y[i];
                    }
                    l_w = static_cast<double>(i1 - i0);
                } else {
                    for (std::size_t i = i0; i < i1; ++i) {
                        l_wy += sample_weight[i] * y[i];
                        l_w += sample_weight[i];
                    }
                }
                c_wy[c] = l_wy;
                c_w[c] = l_w;
            }
        }
        for (std::size_t c = 0; c < n_chunks; ++c) {
            sum_wy += c_wy[c];
            sum_w += c_w[c];
        }
    }

    const double mean = sum_wy / sum_w;
    if (obj == Objective::Regression) {
        return mean;
    }
    const double p0 = std::clamp(mean, 1e-15, 1.0 - 1e-15);
    return std::log(p0 / (1.0 - p0));
}

void compute_gradients(Objective obj, std::span<const double> scores, std::span<const double> y,
                       std::span<const double> sample_weight, float* g, float* h,
                       int n_jobs) noexcept {
    const std::size_t n = y.size();
    const bool has_w = !sample_weight.empty();
    const int n_threads = resolve_num_threads(n_jobs);
    const bool use_omp = (n_threads > 1) && (n >= 16384U);

    if (obj == Objective::Regression) {
        if (!has_w) {
            if (use_omp) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
                for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
                    g[i] = static_cast<float>(scores[static_cast<std::size_t>(i)] -
                                              y[static_cast<std::size_t>(i)]);
                    h[i] = 1.0F;
                }
            } else {
                for (std::size_t i = 0; i < n; ++i) {
                    g[i] = static_cast<float>(scores[i] - y[i]);
                    h[i] = 1.0F;
                }
            }
        } else {
            if (use_omp) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
                for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
                    const auto iu = static_cast<std::size_t>(i);
                    const double w = sample_weight[iu];
                    g[iu] = static_cast<float>(w * (scores[iu] - y[iu]));
                    h[iu] = static_cast<float>(w);
                }
            } else {
                for (std::size_t i = 0; i < n; ++i) {
                    const double w = sample_weight[i];
                    g[i] = static_cast<float>(w * (scores[i] - y[i]));
                    h[i] = static_cast<float>(w);
                }
            }
        }
        return;
    }

    if (!has_w) {
        if (use_omp) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
                const auto iu = static_cast<std::size_t>(i);
                const double p = sigmoid(scores[iu]);
                g[iu] = static_cast<float>(p - y[iu]);
                h[iu] = static_cast<float>(std::max(p * (1.0 - p), 1e-16));
            }
        } else {
            for (std::size_t i = 0; i < n; ++i) {
                const double p = sigmoid(scores[i]);
                g[i] = static_cast<float>(p - y[i]);
                h[i] = static_cast<float>(std::max(p * (1.0 - p), 1e-16));
            }
        }
    } else {
        if (use_omp) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
                const auto iu = static_cast<std::size_t>(i);
                const double p = sigmoid(scores[iu]);
                const double w = sample_weight[iu];
                g[iu] = static_cast<float>(w * (p - y[iu]));
                h[iu] = static_cast<float>(w * std::max(p * (1.0 - p), 1e-16));
            }
        } else {
            for (std::size_t i = 0; i < n; ++i) {
                const double p = sigmoid(scores[i]);
                const double w = sample_weight[i];
                g[i] = static_cast<float>(w * (p - y[i]));
                h[i] = static_cast<float>(w * std::max(p * (1.0 - p), 1e-16));
            }
        }
    }
}

namespace {

void compute_loss_chunk(Objective obj, const double* scores, const double* y,
                        const double* sample_weight, bool has_w, std::size_t i0, std::size_t i1,
                        double& loss_out, double& w_out) noexcept {
    double tl = 0.0;
    double tw = 0.0;
    if (obj == Objective::Regression) {
        if (!has_w) {
            for (std::size_t i = i0; i < i1; ++i) {
                const double diff = scores[i] - y[i];
                tl += diff * diff;
            }
            loss_out = tl;
            w_out = static_cast<double>(i1 - i0);
            return;
        }
        for (std::size_t i = i0; i < i1; ++i) {
            const double diff = scores[i] - y[i];
            const double w = sample_weight[i];
            tl += w * diff * diff;
            tw += w;
        }
        loss_out = tl;
        w_out = tw;
        return;
    }

    if (!has_w) {
        for (std::size_t i = i0; i < i1; ++i) {
            const double z = scores[i];
            const double max_z = (z > 0.0) ? z : 0.0;
            const double abs_z = (z >= 0.0) ? z : -z;
            tl += max_z - y[i] * z + std::log1p(std::exp(-abs_z));
        }
        loss_out = tl;
        w_out = static_cast<double>(i1 - i0);
        return;
    }
    for (std::size_t i = i0; i < i1; ++i) {
        const double z = scores[i];
        const double max_z = (z > 0.0) ? z : 0.0;
        const double abs_z = (z >= 0.0) ? z : -z;
        const double w = sample_weight[i];
        tl += w * (max_z - y[i] * z + std::log1p(std::exp(-abs_z)));
        tw += w;
    }
    loss_out = tl;
    w_out = tw;
}

} // namespace

double compute_loss(Objective obj, std::span<const double> scores, std::span<const double> y,
                    std::span<const double> sample_weight, int n_jobs) noexcept {
    const std::size_t n = y.size();
    const bool has_w = !sample_weight.empty();
    const double* sw_ptr = has_w ? sample_weight.data() : nullptr;
    const std::size_t n_chunks = (n + kDetChunkRows - 1U) / kDetChunkRows;

    if (n_chunks <= 1U) {
        double total_loss = 0.0;
        double total_w = 0.0;
        compute_loss_chunk(obj, scores.data(), y.data(), sw_ptr, has_w, 0U, n, total_loss, total_w);
        return total_loss / total_w;
    }

    constexpr std::size_t kStackChunks = 512U;
    double stack_loss[kStackChunks];
    double stack_w[kStackChunks];

    const int n_threads = resolve_num_threads(n_jobs);
    const bool use_omp = (n_threads > 1) && (n >= 32768U);
    double total_loss = 0.0;
    double total_w = 0.0;

    for (std::size_t c_base = 0; c_base < n_chunks; c_base += kStackChunks) {
        const std::size_t w_chunks = std::min(kStackChunks, n_chunks - c_base);
        if (use_omp) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
            for (std::ptrdiff_t wc = 0; wc < static_cast<std::ptrdiff_t>(w_chunks); ++wc) {
                const auto cu = c_base + static_cast<std::size_t>(wc);
                const std::size_t i0 = cu * kDetChunkRows;
                const std::size_t i1 = std::min(i0 + kDetChunkRows, n);
                compute_loss_chunk(obj, scores.data(), y.data(), sw_ptr, has_w, i0, i1,
                                   stack_loss[wc], stack_w[wc]);
            }
        } else {
            for (std::size_t wc = 0; wc < w_chunks; ++wc) {
                const std::size_t cu = c_base + wc;
                const std::size_t i0 = cu * kDetChunkRows;
                const std::size_t i1 = std::min(i0 + kDetChunkRows, n);
                compute_loss_chunk(obj, scores.data(), y.data(), sw_ptr, has_w, i0, i1,
                                   stack_loss[wc], stack_w[wc]);
            }
        }
        for (std::size_t wc = 0; wc < w_chunks; ++wc) {
            total_loss += stack_loss[wc];
            total_w += stack_w[wc];
        }
    }
    return total_loss / total_w;
}

} // namespace miter
