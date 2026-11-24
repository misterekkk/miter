#pragma once

#include <miter/model.hpp>

#include <string>
#include <string_view>

namespace miter {

std::string model_to_json(const Model& model);
Model model_from_json(std::string_view json_text);

void save_model_json(const Model& model, const std::string& path);
Model load_model_json(const std::string& path);

} // namespace miter
