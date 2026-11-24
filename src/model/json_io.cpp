#include "model/json_io.hpp"

#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <locale>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace miter {

namespace {

constexpr int kJsonFormatVersion = 1;

void write_escaped_string(std::ostringstream& os, std::string_view sv) {
    os << '"';
    for (const unsigned char ch : sv) {
        switch (ch) {
        case '"':
            os << "\\\"";
            break;
        case '\\':
            os << "\\\\";
            break;
        case '\b':
            os << "\\b";
            break;
        case '\f':
            os << "\\f";
            break;
        case '\n':
            os << "\\n";
            break;
        case '\r':
            os << "\\r";
            break;
        case '\t':
            os << "\\t";
            break;
        default:
            if (ch < 0x20U) {
                os << "\\u00" << std::hex << std::setw(2) << std::setfill('0')
                   << static_cast<unsigned int>(ch) << std::dec;
            } else {
                os << static_cast<char>(ch);
            }
            break;
        }
    }
    os << '"';
}

struct JsonValue;
using JsonArray = std::vector<JsonValue>;
using JsonObject = std::map<std::string, JsonValue, std::less<>>;

struct JsonValue {
    std::variant<std::nullptr_t, bool, double, std::string, JsonArray, JsonObject> data;

    bool is_bool() const noexcept { return std::holds_alternative<bool>(data); }
    bool is_number() const noexcept { return std::holds_alternative<double>(data); }
    bool is_string() const noexcept { return std::holds_alternative<std::string>(data); }
    bool is_array() const noexcept { return std::holds_alternative<JsonArray>(data); }
    bool is_object() const noexcept { return std::holds_alternative<JsonObject>(data); }

    bool as_bool() const {
        if (!is_bool()) {
            throw std::runtime_error("Expected JSON boolean");
        }
        return std::get<bool>(data);
    }

    double as_double() const {
        if (!is_number()) {
            throw std::runtime_error("Expected JSON number");
        }
        return std::get<double>(data);
    }

    std::int64_t as_i64() const {
        const double v = as_double();
        if (!std::isfinite(v) || std::floor(v) != v) {
            throw std::runtime_error("Expected JSON integer");
        }
        return static_cast<std::int64_t>(v);
    }

    const std::string& as_string() const {
        if (!is_string()) {
            throw std::runtime_error("Expected JSON string");
        }
        return std::get<std::string>(data);
    }

    const JsonArray& as_array() const {
        if (!is_array()) {
            throw std::runtime_error("Expected JSON array");
        }
        return std::get<JsonArray>(data);
    }

    const JsonObject& as_object() const {
        if (!is_object()) {
            throw std::runtime_error("Expected JSON object");
        }
        return std::get<JsonObject>(data);
    }
};

class JsonParser {
public:
    explicit JsonParser(std::string_view input) : input_(input) {}

    JsonValue parse() {
        skip_ws();
        JsonValue v = parse_value();
        skip_ws();
        if (pos_ != input_.size()) {
            throw std::runtime_error("Unexpected trailing characters in JSON");
        }
        return v;
    }

private:
    void skip_ws() noexcept {
        while (pos_ < input_.size() && (input_[pos_] == ' ' || input_[pos_] == '\t' ||
                                        input_[pos_] == '\n' || input_[pos_] == '\r')) {
            ++pos_;
        }
    }

    char peek() const {
        if (pos_ >= input_.size()) {
            throw std::runtime_error("Unexpected end of JSON");
        }
        return input_[pos_];
    }

    char get() {
        const char ch = peek();
        ++pos_;
        return ch;
    }

    JsonValue parse_value() {
        skip_ws();
        const char ch = peek();
        if (ch == '{') {
            return parse_object();
        }
        if (ch == '[') {
            return parse_array();
        }
        if (ch == '"') {
            return JsonValue{parse_string()};
        }
        if (ch == 't') {
            expect_literal("true");
            return JsonValue{true};
        }
        if (ch == 'f') {
            expect_literal("false");
            return JsonValue{false};
        }
        if (ch == 'n') {
            expect_literal("null");
            return JsonValue{nullptr};
        }
        if (ch == '-' || std::isdigit(static_cast<unsigned char>(ch)) != 0) {
            return JsonValue{parse_number()};
        }
        throw std::runtime_error("Invalid JSON token");
    }

    void expect_literal(std::string_view lit) {
        if (input_.substr(pos_, lit.size()) != lit) {
            throw std::runtime_error("Invalid JSON literal");
        }
        pos_ += lit.size();
    }

    std::string parse_string() {
        if (get() != '"') {
            throw std::runtime_error("Expected '\"'");
        }
        std::string out;
        while (pos_ < input_.size()) {
            const char ch = get();
            if (ch == '"') {
                return out;
            }
            if (ch == '\\') {
                const char esc = get();
                switch (esc) {
                case '"':
                    out.push_back('"');
                    break;
                case '\\':
                    out.push_back('\\');
                    break;
                case '/':
                    out.push_back('/');
                    break;
                case 'b':
                    out.push_back('\b');
                    break;
                case 'f':
                    out.push_back('\f');
                    break;
                case 'n':
                    out.push_back('\n');
                    break;
                case 'r':
                    out.push_back('\r');
                    break;
                case 't':
                    out.push_back('\t');
                    break;
                case 'u': {
                    if (pos_ + 4U > input_.size()) {
                        throw std::runtime_error("Invalid \\u escape");
                    }
                    unsigned int cp = 0U;
                    for (int k = 0; k < 4; ++k) {
                        const char h = get();
                        cp <<= 4U;
                        if (h >= '0' && h <= '9') {
                            cp |= static_cast<unsigned int>(h - '0');
                        } else if (h >= 'a' && h <= 'f') {
                            cp |= static_cast<unsigned int>(10 + (h - 'a'));
                        } else if (h >= 'A' && h <= 'F') {
                            cp |= static_cast<unsigned int>(10 + (h - 'A'));
                        } else {
                            throw std::runtime_error("Invalid hex digit in \\u escape");
                        }
                    }
                    out.push_back(static_cast<char>(cp & 0xFFU));
                    break;
                }
                default:
                    throw std::runtime_error("Invalid JSON escape sequence");
                }
            } else {
                out.push_back(ch);
            }
        }
        throw std::runtime_error("Unterminated JSON string");
    }

    double parse_number() {
        const std::size_t start = pos_;
        if (input_[pos_] == '-') {
            ++pos_;
        }
        if (pos_ >= input_.size() || std::isdigit(static_cast<unsigned char>(input_[pos_])) == 0) {
            throw std::runtime_error("Invalid JSON number");
        }
        while (pos_ < input_.size() &&
               std::isdigit(static_cast<unsigned char>(input_[pos_])) != 0) {
            ++pos_;
        }
        if (pos_ < input_.size() && input_[pos_] == '.') {
            ++pos_;
            if (pos_ >= input_.size() ||
                std::isdigit(static_cast<unsigned char>(input_[pos_])) == 0) {
                throw std::runtime_error("Invalid JSON fraction");
            }
            while (pos_ < input_.size() &&
                   std::isdigit(static_cast<unsigned char>(input_[pos_])) != 0) {
                ++pos_;
            }
        }
        if (pos_ < input_.size() && (input_[pos_] == 'e' || input_[pos_] == 'E')) {
            ++pos_;
            if (pos_ < input_.size() && (input_[pos_] == '+' || input_[pos_] == '-')) {
                ++pos_;
            }
            if (pos_ >= input_.size() ||
                std::isdigit(static_cast<unsigned char>(input_[pos_])) == 0) {
                throw std::runtime_error("Invalid JSON exponent");
            }
            while (pos_ < input_.size() &&
                   std::isdigit(static_cast<unsigned char>(input_[pos_])) != 0) {
                ++pos_;
            }
        }

        std::istringstream iss(std::string(input_.substr(start, pos_ - start)));
        iss.imbue(std::locale::classic());
        double val = 0.0;
        iss >> val;
        if (iss.fail() || !iss.eof() || !std::isfinite(val)) {
            throw std::runtime_error("Failed to parse finite JSON number");
        }
        return val;
    }

    JsonValue parse_array() {
        get(); // '['
        JsonArray arr;
        skip_ws();
        if (peek() == ']') {
            get();
            return JsonValue{std::move(arr)};
        }
        for (;;) {
            arr.push_back(parse_value());
            skip_ws();
            const char ch = get();
            if (ch == ']') {
                break;
            }
            if (ch != ',') {
                throw std::runtime_error("Expected ',' or ']' in JSON array");
            }
        }
        return JsonValue{std::move(arr)};
    }

    JsonValue parse_object() {
        get(); // '{'
        JsonObject obj;
        skip_ws();
        if (peek() == '}') {
            get();
            return JsonValue{std::move(obj)};
        }
        for (;;) {
            skip_ws();
            std::string key = parse_string();
            skip_ws();
            if (get() != ':') {
                throw std::runtime_error("Expected ':' in JSON object");
            }
            obj.emplace(std::move(key), parse_value());
            skip_ws();
            const char ch = get();
            if (ch == '}') {
                break;
            }
            if (ch != ',') {
                throw std::runtime_error("Expected ',' or '}' in JSON object");
            }
        }
        return JsonValue{std::move(obj)};
    }

    std::string_view input_;
    std::size_t pos_ = 0U;
};

const JsonValue& require_field(const JsonObject& obj, std::string_view key) {
    const auto it = obj.find(key);
    if (it == obj.end()) {
        throw std::runtime_error("Missing JSON field: " + std::string(key));
    }
    return it->second;
}

std::string_view objective_to_str(Objective obj) {
    return (obj == Objective::Regression) ? "regression" : "binary";
}

Objective objective_from_str(std::string_view s) {
    if (s == "regression") {
        return Objective::Regression;
    }
    if (s == "binary") {
        return Objective::Binary;
    }
    throw std::runtime_error("Unsupported objective in model JSON");
}

std::string_view node_kind_to_str(NodeKind k) {
    switch (k) {
    case NodeKind::Leaf:
        return "leaf";
    case NodeKind::Axis:
        return "axis";
    case NodeKind::Oblique:
        return "oblique";
    case NodeKind::Categorical:
        return "categorical";
    }
    return "leaf";
}

NodeKind node_kind_from_str(std::string_view s) {
    if (s == "leaf") {
        return NodeKind::Leaf;
    }
    if (s == "axis") {
        return NodeKind::Axis;
    }
    if (s == "oblique") {
        return NodeKind::Oblique;
    }
    if (s == "categorical") {
        return NodeKind::Categorical;
    }
    throw std::runtime_error("Unsupported node kind in model JSON");
}

} // namespace

std::string model_to_json(const Model& model) {
    std::ostringstream os;
    os.imbue(std::locale::classic());
    os << std::setprecision(17);

    os << "{\n";
    os << "  \"format_version\": " << kJsonFormatVersion << ",\n";
    os << "  \"miter_version\": ";
    write_escaped_string(os, model.miter_version);
    os << ",\n";
    os << "  \"objective\": ";
    write_escaped_string(os, objective_to_str(model.objective));
    os << ",\n";
    os << "  \"init_score\": " << model.init_score << ",\n";
    os << "  \"learning_rate\": " << model.learning_rate << ",\n";
    os << "  \"n_features\": " << model.n_features << ",\n";
    os << "  \"best_iteration\": " << model.best_iteration << ",\n";

    os << "  \"params\": {"
       << "\"n_estimators\": " << model.params.n_estimators
       << ", \"learning_rate\": " << model.params.learning_rate
       << ", \"num_leaves\": " << model.params.num_leaves
       << ", \"max_depth\": " << model.params.max_depth
       << ", \"min_data_in_leaf\": " << model.params.min_data_in_leaf
       << ", \"min_sum_hessian_in_leaf\": " << model.params.min_sum_hessian_in_leaf
       << ", \"lambda_l2\": " << model.params.lambda_l2
       << ", \"min_gain_to_split\": " << model.params.min_gain_to_split
       << ", \"max_bin\": " << model.params.max_bin
       << ", \"max_cat_to_onehot\": " << model.params.max_cat_to_onehot
       << ", \"min_data_per_group\": " << model.params.min_data_per_group
       << ", \"max_categories\": " << model.params.max_categories
       << ", \"cat_smooth\": " << model.params.cat_smooth << ", \"cat_l2\": " << model.params.cat_l2
       << ", \"max_cat_threshold\": " << model.params.max_cat_threshold
       << ", \"m_mult\": " << model.params.m_mult << ", \"k_proj\": " << model.params.k_proj
       << ", \"oblique_max_depth\": " << model.params.oblique_max_depth
       << ", \"early_stopping_rounds\": " << model.params.early_stopping_rounds
       << ", \"time_limit\": " << model.params.time_limit << ", \"n_jobs\": " << model.params.n_jobs
       << ", \"random_state\": " << model.params.random_state << "},\n";

    os << "  \"feature_names\": [";
    for (std::size_t i = 0; i < model.feature_names.size(); ++i) {
        if (i > 0U) {
            os << ", ";
        }
        write_escaped_string(os, model.feature_names[i]);
    }
    os << "],\n";

    os << "  \"feature_kinds\": [";
    for (std::size_t i = 0; i < model.feature_kinds.size(); ++i) {
        if (i > 0U) {
            os << ", ";
        }
        os << static_cast<unsigned int>(model.feature_kinds[i]);
    }
    os << "],\n";

    os << "  \"quantizers\": [";
    for (std::size_t i = 0; i < model.quantizers.size(); ++i) {
        if (i > 0U) {
            os << ", ";
        }
        const auto& q = model.quantizers[i];
        os << "{\"lo\": " << q.lo << ", \"scale\": " << q.scale
           << ", \"is_constant\": " << (q.is_constant ? "true" : "false") << "}";
    }
    os << "],\n";

    os << "  \"cat_features\": [";
    for (std::size_t i = 0; i < model.cat_features.size(); ++i) {
        if (i > 0U) {
            os << ", ";
        }
        os << "[";
        const auto& cats = model.cat_features[i].categories;
        for (std::size_t c = 0; c < cats.size(); ++c) {
            if (c > 0U) {
                os << ", ";
            }
            write_escaped_string(os, cats[c]);
        }
        os << "]";
    }
    os << "],\n";

    os << "  \"trees\": [\n";
    for (std::size_t t = 0; t < model.trees.size(); ++t) {
        const auto& tree = model.trees[t];
        os << "    {\"n_leaves\": " << tree.n_leaves << ", \"nodes\": [\n";
        for (std::size_t n_idx = 0; n_idx < tree.nodes.size(); ++n_idx) {
            const Node& nd = tree.nodes[n_idx];
            os << "      {\"kind\": ";
            write_escaped_string(os, node_kind_to_str(nd.kind));
            if (nd.kind == NodeKind::Leaf) {
                os << ", \"leaf_value\": " << nd.leaf_value;
            } else if (nd.kind == NodeKind::Axis) {
                os << ", \"left_child\": " << nd.left_child
                   << ", \"right_child\": " << nd.right_child << ", \"feature\": " << nd.feature
                   << ", \"threshold_u16\": " << nd.threshold_u16
                   << ", \"nan_go_left\": " << (nd.nan_go_left ? "true" : "false")
                   << ", \"split_gain\": " << nd.split_gain;
            } else if (nd.kind == NodeKind::Categorical) {
                const auto& bs = tree.cat_bitsets[nd.cat_bitset_offset];
                os << ", \"left_child\": " << nd.left_child
                   << ", \"right_child\": " << nd.right_child << ", \"feature\": " << nd.feature
                   << ", \"left_categories\": [";
                bool first = true;
                for (unsigned int code = 0; code < 256U; ++code) {
                    if (cat_bitset_test(bs, static_cast<std::uint8_t>(code))) {
                        if (!first) {
                            os << ", ";
                        }
                        os << code;
                        first = false;
                    }
                }
                os << "]"
                   << ", \"default_go_left\": " << (nd.nan_go_left ? "true" : "false")
                   << ", \"split_gain\": " << nd.split_gain;
            } else if (nd.kind == NodeKind::Oblique) {
                const std::uint8_t k = nd.oblique_k;
                const std::uint32_t* feats =
                    (k <= 3U) ? nd.oblique_features
                              : tree.oblique_overflow[nd.overflow_offset].features.data();
                const std::int16_t* weights =
                    (k <= 3U) ? nd.oblique_weights
                              : tree.oblique_overflow[nd.overflow_offset].weights.data();
                const std::int64_t thr =
                    (k <= 3U) ? static_cast<std::int64_t>(nd.threshold_i32)
                              : tree.oblique_overflow[nd.overflow_offset].threshold_i64;
                os << ", \"left_child\": " << nd.left_child
                   << ", \"right_child\": " << nd.right_child
                   << ", \"k\": " << static_cast<unsigned int>(k) << ", \"features\": [";
                for (std::uint8_t q = 0; q < k; ++q) {
                    if (q > 0U) {
                        os << ", ";
                    }
                    os << feats[q];
                }
                os << "], \"weights\": [";
                for (std::uint8_t q = 0; q < k; ++q) {
                    if (q > 0U) {
                        os << ", ";
                    }
                    os << weights[q];
                }
                os << "], \"threshold\": " << thr
                   << ", \"nan_go_left\": " << (nd.nan_go_left ? "true" : "false")
                   << ", \"split_gain\": " << nd.split_gain;
            }
            os << "}" << (n_idx + 1U < tree.nodes.size() ? "," : "") << "\n";
        }
        os << "    ]}" << (t + 1U < model.trees.size() ? "," : "") << "\n";
    }
    os << "  ]\n";
    os << "}\n";
    return os.str();
}

Model model_from_json(std::string_view json_text) {
    JsonParser parser(json_text);
    const JsonValue root_val = parser.parse();
    const auto& root = root_val.as_object();

    const std::int64_t fmt = require_field(root, "format_version").as_i64();
    if (fmt != kJsonFormatVersion) {
        throw std::runtime_error("Unsupported miter model JSON format_version");
    }

    Model model;
    model.miter_version = require_field(root, "miter_version").as_string();
    model.objective = objective_from_str(require_field(root, "objective").as_string());
    model.init_score = require_field(root, "init_score").as_double();
    model.learning_rate = require_field(root, "learning_rate").as_double();
    const std::int64_t n_feat_i64 = require_field(root, "n_features").as_i64();
    const std::int64_t best_iter_i64 = require_field(root, "best_iteration").as_i64();
    if (n_feat_i64 <= 0 || best_iter_i64 < 0) {
        throw std::runtime_error("Invalid n_features or best_iteration in model JSON");
    }
    model.n_features = static_cast<std::size_t>(n_feat_i64);
    model.best_iteration = static_cast<std::size_t>(best_iter_i64);

    if (const auto it = root.find("params"); it != root.end()) {
        const auto& p = it->second.as_object();
        model.params.n_estimators = static_cast<int>(require_field(p, "n_estimators").as_i64());
        model.params.learning_rate = require_field(p, "learning_rate").as_double();
        model.params.num_leaves = static_cast<int>(require_field(p, "num_leaves").as_i64());
        model.params.max_depth = static_cast<int>(require_field(p, "max_depth").as_i64());
        model.params.min_data_in_leaf =
            static_cast<int>(require_field(p, "min_data_in_leaf").as_i64());
        model.params.min_sum_hessian_in_leaf =
            require_field(p, "min_sum_hessian_in_leaf").as_double();
        model.params.lambda_l2 = require_field(p, "lambda_l2").as_double();
        model.params.min_gain_to_split = require_field(p, "min_gain_to_split").as_double();
        model.params.max_bin = static_cast<int>(require_field(p, "max_bin").as_i64());
        if (const auto mco = p.find("max_cat_to_onehot"); mco != p.end()) {
            model.params.max_cat_to_onehot = static_cast<int>(mco->second.as_i64());
        }
        if (const auto mdg = p.find("min_data_per_group"); mdg != p.end()) {
            model.params.min_data_per_group = static_cast<int>(mdg->second.as_i64());
        }
        if (const auto mc = p.find("max_categories"); mc != p.end()) {
            model.params.max_categories = static_cast<int>(mc->second.as_i64());
        }
        if (const auto cs = p.find("cat_smooth"); cs != p.end()) {
            model.params.cat_smooth = cs->second.as_double();
        }
        if (const auto cl = p.find("cat_l2"); cl != p.end()) {
            model.params.cat_l2 = cl->second.as_double();
        }
        if (const auto mct = p.find("max_cat_threshold"); mct != p.end()) {
            model.params.max_cat_threshold = static_cast<int>(mct->second.as_i64());
        }
        model.params.m_mult = static_cast<int>(require_field(p, "m_mult").as_i64());
        model.params.k_proj = static_cast<int>(require_field(p, "k_proj").as_i64());
        if (const auto omd = p.find("oblique_max_depth"); omd != p.end()) {
            model.params.oblique_max_depth = static_cast<int>(omd->second.as_i64());
        }
        model.params.early_stopping_rounds =
            static_cast<int>(require_field(p, "early_stopping_rounds").as_i64());
        model.params.time_limit = require_field(p, "time_limit").as_double();
        model.params.n_jobs = static_cast<int>(require_field(p, "n_jobs").as_i64());
        model.params.random_state =
            static_cast<std::uint64_t>(require_field(p, "random_state").as_i64());
    }

    for (const auto& fn : require_field(root, "feature_names").as_array()) {
        model.feature_names.push_back(fn.as_string());
    }
    for (const auto& fk : require_field(root, "feature_kinds").as_array()) {
        const std::int64_t kind_i = fk.as_i64();
        if (kind_i != 0 && kind_i != 1) {
            throw std::runtime_error("Invalid feature_kind in model JSON");
        }
        model.feature_kinds.push_back(static_cast<std::uint8_t>(kind_i));
    }
    for (const auto& qv : require_field(root, "quantizers").as_array()) {
        const auto& qo = qv.as_object();
        QuantizerInfo q;
        q.lo = require_field(qo, "lo").as_double();
        q.scale = require_field(qo, "scale").as_double();
        q.is_constant = require_field(qo, "is_constant").as_bool();
        model.quantizers.push_back(q);
    }

    if (const auto cf_it = root.find("cat_features"); cf_it != root.end()) {
        for (const auto& cf_arr : cf_it->second.as_array()) {
            CatFeatureInfo info;
            for (const auto& cat_s : cf_arr.as_array()) {
                info.categories.push_back(cat_s.as_string());
            }
            model.cat_features.push_back(std::move(info));
        }
    } else {
        for (const std::uint8_t fk : model.feature_kinds) {
            if (fk != 0U) {
                throw std::runtime_error("Missing cat_features for categorical model in JSON");
            }
        }
        model.cat_features.resize(model.n_features);
    }

    if (model.feature_names.size() != model.n_features ||
        model.feature_kinds.size() != model.n_features ||
        model.quantizers.size() != model.n_features ||
        model.cat_features.size() != model.n_features) {
        throw std::runtime_error("Feature metadata size mismatch in model JSON");
    }

    for (const auto& tv : require_field(root, "trees").as_array()) {
        const auto& to = tv.as_object();
        Tree tree;
        const std::int64_t nl_i64 = require_field(to, "n_leaves").as_i64();
        if (nl_i64 < 1) {
            throw std::runtime_error("Invalid tree n_leaves in model JSON");
        }
        tree.n_leaves = static_cast<std::uint32_t>(nl_i64);
        const auto& nodes_arr = require_field(to, "nodes").as_array();
        if (nodes_arr.empty()) {
            throw std::runtime_error("Empty tree nodes array in model JSON");
        }
        tree.nodes.reserve(nodes_arr.size());
        for (const auto& nv : nodes_arr) {
            const auto& no = nv.as_object();
            Node nd;
            nd.kind = node_kind_from_str(require_field(no, "kind").as_string());
            if (nd.kind == NodeKind::Leaf) {
                nd.leaf_value = require_field(no, "leaf_value").as_double();
            } else if (nd.kind == NodeKind::Axis) {
                const std::int64_t lc = require_field(no, "left_child").as_i64();
                const std::int64_t rc = require_field(no, "right_child").as_i64();
                const std::int64_t feat = require_field(no, "feature").as_i64();
                const std::int64_t thr = require_field(no, "threshold_u16").as_i64();
                if (lc < 0 || rc < 0 || static_cast<std::size_t>(lc) >= nodes_arr.size() ||
                    static_cast<std::size_t>(rc) >= nodes_arr.size() || feat < 0 ||
                    static_cast<std::size_t>(feat) >= model.n_features || thr < 0 || thr > 65535) {
                    throw std::runtime_error("Out-of-bounds node indices/threshold in model JSON");
                }
                nd.left_child = static_cast<std::uint32_t>(lc);
                nd.right_child = static_cast<std::uint32_t>(rc);
                nd.feature = static_cast<std::uint32_t>(feat);
                nd.threshold_u16 = static_cast<std::uint16_t>(thr);
                nd.nan_go_left = require_field(no, "nan_go_left").as_bool();
                nd.split_gain = require_field(no, "split_gain").as_double();
            } else if (nd.kind == NodeKind::Categorical) {
                const std::int64_t lc = require_field(no, "left_child").as_i64();
                const std::int64_t rc = require_field(no, "right_child").as_i64();
                const std::int64_t feat = require_field(no, "feature").as_i64();
                if (lc < 0 || rc < 0 || static_cast<std::size_t>(lc) >= nodes_arr.size() ||
                    static_cast<std::size_t>(rc) >= nodes_arr.size() || feat < 0 ||
                    static_cast<std::size_t>(feat) >= model.n_features ||
                    model.feature_kinds[static_cast<std::size_t>(feat)] != 1U) {
                    throw std::runtime_error("Invalid categorical node indices in model JSON");
                }
                nd.left_child = static_cast<std::uint32_t>(lc);
                nd.right_child = static_cast<std::uint32_t>(rc);
                nd.feature = static_cast<std::uint32_t>(feat);
                nd.nan_go_left = require_field(no, "default_go_left").as_bool();
                nd.split_gain = require_field(no, "split_gain").as_double();

                CatBitset bs;
                for (const auto& lv : require_field(no, "left_categories").as_array()) {
                    const std::int64_t code = lv.as_i64();
                    if (code < 0 || code > 255) {
                        throw std::runtime_error("Categorical code out of range [0, 255] in JSON");
                    }
                    cat_bitset_set(bs, static_cast<std::uint8_t>(code));
                }
                nd.cat_bitset_offset = static_cast<std::uint32_t>(tree.cat_bitsets.size());
                tree.cat_bitsets.push_back(bs);
            } else if (nd.kind == NodeKind::Oblique) {
                const std::int64_t lc = require_field(no, "left_child").as_i64();
                const std::int64_t rc = require_field(no, "right_child").as_i64();
                const std::int64_t k_i64 = require_field(no, "k").as_i64();
                if (lc < 0 || rc < 0 || static_cast<std::size_t>(lc) >= nodes_arr.size() ||
                    static_cast<std::size_t>(rc) >= nodes_arr.size() || k_i64 < 2 || k_i64 > 8) {
                    throw std::runtime_error("Invalid oblique node indices or k in model JSON");
                }
                const auto k = static_cast<std::uint8_t>(k_i64);
                const auto& feats_arr = require_field(no, "features").as_array();
                const auto& weights_arr = require_field(no, "weights").as_array();
                if (feats_arr.size() != k || weights_arr.size() != k) {
                    throw std::runtime_error("Oblique features/weights array length mismatch");
                }
                const std::int64_t thr = require_field(no, "threshold").as_i64();
                nd.left_child = static_cast<std::uint32_t>(lc);
                nd.right_child = static_cast<std::uint32_t>(rc);
                nd.oblique_k = k;
                nd.nan_go_left = require_field(no, "nan_go_left").as_bool();
                nd.split_gain = require_field(no, "split_gain").as_double();

                if (k <= 3U) {
                    if (thr < -2147483648LL || thr > 2147483647LL) {
                        throw std::runtime_error("Oblique int32 threshold out of range in JSON");
                    }
                    nd.threshold_i32 = static_cast<std::int32_t>(thr);
                    for (std::uint8_t q = 0; q < k; ++q) {
                        const std::int64_t fi = feats_arr[q].as_i64();
                        const std::int64_t wi = weights_arr[q].as_i64();
                        if (fi < 0 || static_cast<std::size_t>(fi) >= model.n_features ||
                            wi < -16384 || wi > 16384) {
                            throw std::runtime_error(
                                "Invalid oblique feature/weight in model JSON");
                        }
                        nd.oblique_features[q] = static_cast<std::uint32_t>(fi);
                        nd.oblique_weights[q] = static_cast<std::int16_t>(wi);
                    }
                } else {
                    ObliqueOverflow ov;
                    ov.threshold_i64 = thr;
                    ov.features.reserve(k);
                    ov.weights.reserve(k);
                    for (std::uint8_t q = 0; q < k; ++q) {
                        const std::int64_t fi = feats_arr[q].as_i64();
                        const std::int64_t wi = weights_arr[q].as_i64();
                        if (fi < 0 || static_cast<std::size_t>(fi) >= model.n_features ||
                            wi < -16384 || wi > 16384) {
                            throw std::runtime_error(
                                "Invalid oblique feature/weight in model JSON");
                        }
                        ov.features.push_back(static_cast<std::uint32_t>(fi));
                        ov.weights.push_back(static_cast<std::int16_t>(wi));
                    }
                    nd.overflow_offset = static_cast<std::uint32_t>(tree.oblique_overflow.size());
                    tree.oblique_overflow.push_back(std::move(ov));
                }
            } else {
                throw std::runtime_error("Unsupported node kind in model JSON");
            }
            tree.nodes.push_back(nd);
        }
        model.trees.push_back(std::move(tree));
    }
    return model;
}

void save_model_json(const Model& model, const std::string& path) {
    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    if (!ofs) {
        throw std::runtime_error("Failed to open model file for writing: " + path);
    }
    const std::string content = model_to_json(model);
    ofs.write(content.data(), static_cast<std::streamsize>(content.size()));
    if (!ofs) {
        throw std::runtime_error("Failed to write model file: " + path);
    }
}

Model load_model_json(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs) {
        throw std::runtime_error("Failed to open model file for reading: " + path);
    }
    const auto size = ifs.tellg();
    if (size < 0) {
        throw std::runtime_error("Failed to determine model file size: " + path);
    }
    ifs.seekg(0, std::ios::beg);
    std::string content(static_cast<std::size_t>(size), '\0');
    if (size > 0 && !ifs.read(content.data(), size)) {
        throw std::runtime_error("Failed to read model file: " + path);
    }
    return model_from_json(content);
}

} // namespace miter
