#pragma once

#include <miter/model.hpp>
#include <miter/params.hpp>

#include <span>
#include <vector>

#include "data/dataset.hpp"

namespace miter {

struct TrainResult {
    Model model;
    std::vector<double> train_loss;
    std::vector<double> valid_loss;
    std::vector<double> final_train_scores;
};

TrainResult fit_axis_booster(const Dataset& ds, Objective objective, std::span<const double> y,
                             std::span<const double> sample_weight, const TransformedData* valid_ds,
                             std::span<const double> y_valid,
                             std::span<const double> valid_sample_weight, const Params& params);

} // namespace miter
