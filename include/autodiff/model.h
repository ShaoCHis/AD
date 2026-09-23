#pragma once

#include "autodiff/autodiff.h"
#include "autodiff/json.h"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace autodiff {

class CustomOpRegistry {
public:
    void register_op(const std::string& name, std::shared_ptr<const CustomOp> op);
    std::shared_ptr<const CustomOp> find(const std::string& name) const;
private:
    std::map<std::string, std::shared_ptr<const CustomOp>> ops_;
};

// Loads compiled C++ operators before the JSON graph is parsed. Keep this
// object alive longer than the registry, graph and evaluators using its ops.
class PluginManager {
public:
    PluginManager() = default;
    ~PluginManager();
    PluginManager(const PluginManager&) = delete;
    PluginManager& operator=(const PluginManager&) = delete;
    void load(const std::string& path, CustomOpRegistry& registry);
private:
    std::vector<void*> handles_;
};

// A compiled plugin exports: extern "C" void register_autodiff_ops(
//     autodiff::CustomOpRegistry&);

struct NamedExpression {
    std::string id;
    Expr expr;
};
struct DerivativeRequest {
    std::string id, source_name, wrt_name;
    Expr source, wrt, result;
};

class LoadedModel {
public:
    LoadedModel(Json document, const CustomOpRegistry& ops, bool explain_derivatives);
    LoadedModel(const LoadedModel&) = delete;
    LoadedModel& operator=(const LoadedModel&) = delete;
    Graph& graph() { return graph_; }
    const Graph& graph() const { return graph_; }
    const NamedExpression& target() const { return target_; }
    const std::vector<NamedExpression>& outputs() const { return outputs_; }
    const std::vector<DerivativeRequest>& derivatives() const { return derivatives_; }
    const std::map<std::string, double>& input_values() const { return input_values_; }
    Inputs make_inputs(const Evaluator& evaluator) const;
private:
    Json document_;
    const CustomOpRegistry& ops_;
    Graph graph_;
    std::map<std::string, Expr> values_;
    std::set<std::string> variable_names_;
    std::map<std::string, const Json*> definitions_, derivative_specs_;
    std::map<std::string, int> states_;
    std::map<std::string, double> input_values_;
    NamedExpression target_;
    std::vector<NamedExpression> outputs_;
    std::vector<DerivativeRequest> derivatives_;

    void reserve_name(const std::string& id);
    Expr resolve(const std::string& id, int depth = 0);
    Expr resolve_derivative(const std::string& id, int depth);
    Expr parse_expression(const Json& item, int depth);
    static std::string field(const Json& item, const std::string& key);
};

} // namespace autodiff
