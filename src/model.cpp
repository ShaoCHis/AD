#include "autodiff/model.h"

#include <dlfcn.h>
#include <stdexcept>
#include <utility>

namespace autodiff {

void CustomOpRegistry::register_op(const std::string& name, std::shared_ptr<const CustomOp> op) {
    if (name.empty() || !op || !ops_.insert(std::make_pair(name, op)).second)
        throw std::invalid_argument("custom op name must be unique and nonempty: " + name);
}
std::shared_ptr<const CustomOp> CustomOpRegistry::find(const std::string& name) const {
    std::map<std::string, std::shared_ptr<const CustomOp>>::const_iterator it = ops_.find(name);
    if (it == ops_.end()) throw std::invalid_argument("unknown operator: " + name);
    return it->second;
}
PluginManager::~PluginManager() {
    for (std::vector<void*>::reverse_iterator it = handles_.rbegin(); it != handles_.rend(); ++it)
        dlclose(*it);
}
void PluginManager::load(const std::string& path, CustomOpRegistry& registry) {
    if (path.empty()) throw std::invalid_argument("empty plugin path");
    void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) throw std::runtime_error("cannot load plugin " + path + ": " + dlerror());
    dlerror();
    typedef void (*RegisterFunction)(CustomOpRegistry&);
    RegisterFunction entry = reinterpret_cast<RegisterFunction>(dlsym(handle, "register_autodiff_ops"));
    const char* symbol_error = dlerror();
    if (symbol_error) {
        std::string message = "plugin " + path + " has no register_autodiff_ops: " + symbol_error;
        dlclose(handle);
        throw std::runtime_error(message);
    }
    // Store the handle before registration: partial registration may throw;
    // any already-created operator objects must outlive the shared library.
    handles_.push_back(handle);
    entry(registry);
}

std::string LoadedModel::field(const Json& item, const std::string& key) {
    return item.at(key).as_string();
}
void LoadedModel::reserve_name(const std::string& id) {
    if (id.empty() || values_.count(id) || definitions_.count(id) || derivative_specs_.count(id))
        throw std::invalid_argument("duplicate or empty expression id: " + id);
}
LoadedModel::LoadedModel(Json document, const CustomOpRegistry& ops, bool explain_derivatives)
    : document_(std::move(document)), ops_(ops) {
    graph_.enable_derivative_explanations(explain_derivatives);
    document_.as_object();
    const std::vector<Json>& variables = document_.at("variables").as_array();
    for (const Json& item : variables) {
        std::string id = item.as_string();
        reserve_name(id);
        variable_names_.insert(id);
        values_.insert(std::make_pair(id, graph_.variable(id)));
    }
    const std::vector<Json>& defs = document_.at("definitions").as_array();
    std::string target_name;
    for (const Json& item : defs) {
        item.as_object();
        std::string id = field(item, "id");
        reserve_name(id);
        std::string role = item.has("role") ? field(item, "role") : "intermediate";
        if (role == "target") {
            if (!target_name.empty()) throw std::invalid_argument("exactly one target expression is required");
            target_name = id;
        } else if (role != "intermediate") throw std::invalid_argument("invalid role for " + id);
        definitions_.insert(std::make_pair(id, &item));
    }
    if (target_name.empty()) throw std::invalid_argument("missing definition with role=target");
    // Resolve all definitions, even unused ones, so typos and cycles are caught.
    for (const Json& item : defs) resolve(field(item, "id"));
    target_ = NamedExpression{target_name, resolve(target_name)};

    if (document_.has("derivatives")) {
        const std::vector<Json>& requests = document_.at("derivatives").as_array();
        for (const Json& item : requests) {
            item.as_object();
            std::string id = field(item, "id");
            reserve_name(id);
            derivative_specs_.insert(std::make_pair(id, &item));
        }
        for (const Json& item : requests) resolve(field(item, "id"));
        for (const Json& item : requests) {
            std::string id = field(item, "id");
            std::string source = field(item, "of"), wrt = field(item, "wrt");
            derivatives_.push_back(DerivativeRequest{id, source, wrt,
                                                     resolve(source), resolve(wrt), resolve(id)});
        }
    }
    if (document_.has("outputs")) {
        const std::vector<Json>& names = document_.at("outputs").as_array();
        for (const Json& name : names) {
            std::string id = name.as_string();
            outputs_.push_back(NamedExpression{id, resolve(id)});
        }
    } else {
        outputs_.push_back(target_);
        for (const DerivativeRequest& d : derivatives_)
            outputs_.push_back(NamedExpression{d.id, d.result});
    }
    if (outputs_.empty()) throw std::invalid_argument("outputs must not be empty");
    if (document_.has("inputs")) {
        for (const auto& pair : document_.at("inputs").as_object()) {
            if (!variable_names_.count(pair.first))
                throw std::invalid_argument("input is not a declared variable: " + pair.first);
            input_values_.insert(std::make_pair(pair.first, pair.second.as_number()));
        }
    }
}

Expr LoadedModel::resolve(const std::string& id, int depth) {
    if (depth > 512) throw std::invalid_argument("definition chain exceeds 512 levels near " + id);
    std::map<std::string, Expr>::iterator cached = values_.find(id);
    if (cached != values_.end()) return cached->second;
    if (states_[id] == 1) throw std::invalid_argument("cyclic expression definition: " + id);
    std::map<std::string, const Json*>::const_iterator definition = definitions_.find(id);
    if (definition != definitions_.end()) {
        states_[id] = 1;
        Expr result = parse_expression(*definition->second, depth + 1).named(id);
        states_[id] = 2;
        values_.insert(std::make_pair(id, result));
        return result;
    }
    if (derivative_specs_.count(id)) return resolve_derivative(id, depth + 1);
    throw std::invalid_argument("undefined expression reference: " + id);
}
Expr LoadedModel::resolve_derivative(const std::string& id, int depth) {
    if (states_[id] == 1) throw std::invalid_argument("cyclic derivative definition: " + id);
    states_[id] = 1;
    const Json& item = *derivative_specs_.at(id);
    std::string source_id = field(item, "of");
    Expr source = resolve(source_id, depth + 1);
    // Derivatives of the same source share one reverse traversal.
    std::vector<std::string> names(1, id);
    std::vector<Expr> wrts(1, resolve(field(item, "wrt"), depth + 1));
    for (const auto& candidate : derivative_specs_) {
        if (candidate.first == id || states_[candidate.first] != 0 ||
            field(*candidate.second, "of") != source_id) continue;
        states_[candidate.first] = 1;
        names.push_back(candidate.first);
        wrts.push_back(resolve(field(*candidate.second, "wrt"), depth + 1));
    }
    std::vector<Expr> results = source.derivative(wrts);
    for (std::size_t i = 0; i < names.size(); ++i) {
        values_.insert(std::make_pair(names[i], results[i]));
        states_[names[i]] = 2;
    }
    return values_.at(id);
}
Expr LoadedModel::parse_expression(const Json& item, int depth) {
    if (depth > 512) throw std::invalid_argument("expression nesting exceeds 512 levels");
    if (item.type() == Json::Number) return graph_.constant(item.as_number());
    if (item.type() == Json::String) return resolve(item.as_string(), depth + 1);
    item.as_object();
    std::string op = field(item, "op");
    const std::vector<Json>& args = item.at("inputs").as_array();
    std::vector<Expr> inputs;
    inputs.reserve(args.size());
    for (const Json& arg : args) inputs.push_back(parse_expression(arg, depth + 1));
    if (op == "add" || op == "sub" || op == "mul" || op == "div") {
        if (inputs.size() != 2) throw std::invalid_argument(op + " requires 2 inputs");
        if (op == "add") return inputs[0] + inputs[1];
        if (op == "sub") return inputs[0] - inputs[1];
        if (op == "mul") return inputs[0] * inputs[1];
        return inputs[0] / inputs[1];
    }
    if (op == "square" || op == "abs" || op == "exp") {
        if (inputs.size() != 1) throw std::invalid_argument(op + " requires 1 input");
        if (op == "square") return inputs[0].square();
        if (op == "abs") return inputs[0].abs();
        return exp(inputs[0]);
    }
    return graph_.custom(ops_.find(op), inputs);
}
Inputs LoadedModel::make_inputs(const Evaluator& evaluator) const {
    Inputs inputs = evaluator.create_inputs();
    for (const auto& pair : input_values_) {
        std::map<std::string, Expr>::const_iterator it = values_.find(pair.first);
        if (it == values_.end()) throw std::invalid_argument("input is not a variable: " + pair.first);
        inputs.set(it->second, pair.second);
    }
    return inputs;
}

} // namespace autodiff
