#include "core/objective.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

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
                          std::span<const double> sample_weight) {
    double sum_wy = 0.0;
    double sum_w = 0.0;
    if (sample_weight.empty()) {
        for (const double yi : y) {
            sum_wy += yi;
        }
        sum_w = static_cast<double>(y.size());
    } else {
        for (std::size_t i = 0; i < y.size(); ++i) {
            sum_wy += sample_weight[i] * y[i];
            sum_w += sample_weight[i];
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
                       std::span<const double> sample_weight, float* g, float* h) noexcept {
    const std::size_t n = y.size();
    const bool has_w = !sample_weight.empty();
    if (obj == Objective::Regression) {
        if (!has_w) {
            for (std::size_t i = 0; i < n; ++i) {
                g[i] = static_cast<float>(scores[i] - y[i]);
                h[i] = 1.0F;
            }
        } else {
            for (std::size_t i = 0; i < n; ++i) {
                const double w = sample_weight[i];
                g[i] = static_cast<float>(w * (scores[i] - y[i]));
                h[i] = static_cast<float>(w);
            }
        }
        return;
    }

    if (!has_w) {
        for (std::size_t i = 0; i < n; ++i) {
            const double p = sigmoid(scores[i]);
            g[i] = static_cast<float>(p - y[i]);
            h[i] = static_cast<float>(std::max(p * (1.0 - p), 1e-16));
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

double compute_loss(Objective obj, std::span<const double> scores, std::span<const double> y,
                    std::span<const double> sample_weight) noexcept {
    const std::size_t n = y.size();
    const bool has_w = !sample_weight.empty();
    double total_loss = 0.0;
    double total_w = 0.0;

    if (obj == Objective::Regression) {
        if (!has_w) {
            for (std::size_t i = 0; i < n; ++i) {
                const double diff = scores[i] - y[i];
                total_loss += diff * diff;
            }
            return total_loss / static_cast<double>(n);
        }
        for (std::size_t i = 0; i < n; ++i) {
            const double diff = scores[i] - y[i];
            const double w = sample_weight[i];
            total_loss += w * diff * diff;
            total_w += w;
        }
        return total_loss / total_w;
    }

    if (!has_w) {
        for (std::size_t i = 0; i < n; ++i) {
            const double z = scores[i];
            const double max_z = (z > 0.0) ? z : 0.0;
            const double abs_z = (z >= 0.0) ? z : -z;
            total_loss += max_z - y[i] * z + std::log1p(std::exp(-abs_z));
        }
        return total_loss / static_cast<double>(n);
    }
    for (std::size_t i = 0; i < n; ++i) {
        const double z = scores[i];
        const double max_z = (z > 0.0) ? z : 0.0;
        const double abs_z = (z >= 0.0) ? z : -z;
        const double w = sample_weight[i];
        total_loss += w * (max_z - y[i] * z + std::log1p(std::exp(-abs_z)));
        total_w += w;
    }
    return total_loss / total_w;
}

} // namespace miter
