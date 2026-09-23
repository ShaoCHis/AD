#include "autodiff/evaluator.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>

namespace autodiff {
namespace {
[[noreturn]] void fail(const char* message) { throw std::invalid_argument(message); }
double sign_value(double x) {
    if (std::isnan(x)) return std::numeric_limits<double>::quiet_NaN();
    return (x > 0) - (x < 0);
}
void children(const Node& n, const std::function<void(NodeId)>& visit) {
    switch (n.op) {
    case Op::Add: case Op::Sub: case Op::Mul: case Op::Div:
        visit(n.a); visit(n.b); break;
    case Op::Square: case Op::Abs: case Op::Exp: case Op::Sign:
        visit(n.a); break;
    case Op::Custom:
        for (NodeId id : n.custom_inputs) visit(id);
        break;
    default: break;
    }
}
} // namespace

void Inputs::set(Expr variable, double value) {
    if (variable.graph_ != graph_ || variable.id_ >= index_by_node_.size()) fail("input variable belongs to another graph");
    std::int32_t i = index_by_node_[variable.id_];
    if (i < 0) fail("input expression is not a reachable variable");
    values_[i] = value;
    supplied_[i] = 1;
}

Evaluator::Evaluator(std::initializer_list<Expr> roots)
    : Evaluator(Span<const Expr>(roots.begin(), roots.size())) {}
Evaluator::Evaluator(Span<const Expr> roots) { compile(roots); }
Evaluator::Evaluator(std::initializer_list<EvaluationQuery> queries)
    : Evaluator(Span<const EvaluationQuery>(queries.begin(), queries.size())) {}
Evaluator::Evaluator(Span<const EvaluationQuery> queries) {
    if (queries.empty()) fail("evaluator requires at least one query");
    Graph* graph = nullptr;
    std::vector<Expr> results(queries.size());
    std::map<NodeId, std::vector<std::size_t>> by_root;
    for (std::size_t i = 0; i < queries.size(); ++i) {
        const EvaluationQuery& q = queries[i];
        Graph& current = q.output.graph();
        if (!graph) graph = &current;
        if (graph != &current) fail("query nodes must belong to one graph");
        if (q.kind == EvaluationQuery::Value) results[i] = q.output;
        else if (q.kind == EvaluationQuery::Derivative) {
            if (&q.root.graph() != graph) fail("root and output must belong to one graph");
            by_root[q.root.id()].push_back(i);
        } else fail("invalid query kind");
    }
    // All d(root)/d(output_i) for the same root share one symbolic reverse pass.
    for (const auto& group : by_root) {
        std::vector<Expr> wrts;
        for (std::size_t index : group.second) wrts.push_back(queries[index].output);
        std::vector<Expr> derivatives = queries[group.second.front()].root.derivative(wrts);
        for (std::size_t i = 0; i < group.second.size(); ++i)
            results[group.second[i]] = derivatives[i];
    }
    compile(results);
}
Expr Evaluator::result_expression(std::size_t index) const {
    if (index >= result_expressions_.size()) fail("query index out of range");
    return result_expressions_[index];
}
bool Evaluator::uses_variable(Expr variable) const {
    if (&variable.graph() != graph_ || variable.id() >= index_by_node_.size())
        fail("variable belongs to another graph");
    return index_by_node_[variable.id()] >= 0 &&
           instructions_[index_by_node_[variable.id()]].node.op == Op::Variable;
}
bool Evaluator::contains(Expr node) const {
    if (&node.graph() != graph_) fail("node belongs to another graph");
    return node.id() < index_by_node_.size() && index_by_node_[node.id()] >= 0;
}
void Evaluator::compile(Span<const Expr> roots) {
    if (roots.empty()) fail("evaluator requires at least one output");
    graph_ = &roots[0].graph();
    for (Expr root : roots) graph_->validate(root);
    result_expressions_.assign(roots.begin(), roots.end());
    index_by_node_.assign(graph_->nodes_.size(), -1);
    std::vector<std::uint8_t> reachable(graph_->nodes_.size());
    std::vector<NodeId> stack;
    for (Expr root : roots) stack.push_back(root.id_);
    while (!stack.empty()) {
        NodeId id = stack.back(); stack.pop_back();
        if (reachable[id]) continue;
        reachable[id] = 1;
        children(graph_->nodes_[id], [&](NodeId child) { stack.push_back(child); });
    }
    // NodeId 的创建顺序天然是拓扑顺序；只编译可达节点，无需再次排序。
    for (NodeId id = 0; id < graph_->nodes_.size(); ++id) {
        if (!reachable[id]) continue;
        const Node& n = graph_->nodes_[id];
        Instruction ins;
        ins.node = n;
        ins.original_id = id;
        if (n.op == Op::Variable) {
            ins.variable_index = static_cast<std::int32_t>(variable_names_.size());
            variable_by_name_.emplace(n.name, ins.variable_index);
            variable_names_.push_back(n.name);
        }
        if (n.op == Op::Add || n.op == Op::Sub || n.op == Op::Mul || n.op == Op::Div ||
            n.op == Op::Square || n.op == Op::Abs || n.op == Op::Exp || n.op == Op::Sign)
            ins.a = index_by_node_[n.a];
        if (n.op == Op::Add || n.op == Op::Sub || n.op == Op::Mul || n.op == Op::Div)
            ins.b = index_by_node_[n.b];
        if (n.op == Op::Custom) {
            for (NodeId child : n.custom_inputs) ins.custom_inputs.push_back(index_by_node_[child]);
            max_custom_arity_ = std::max(max_custom_arity_, n.custom_inputs.size());
        }
        index_by_node_[id] = static_cast<std::int32_t>(instructions_.size());
        instructions_.push_back(std::move(ins));
    }
    for (Expr root : roots) roots_.push_back(index_by_node_[root.id_]);
}
Inputs Evaluator::create_inputs() const {
    Inputs in;
    in.owner_ = this;
    in.graph_ = graph_;
    in.index_by_node_.assign(index_by_node_.size(), -1);
    for (const Instruction& ins : instructions_)
        if (ins.variable_index >= 0) in.index_by_node_[ins.original_id] = ins.variable_index;
    in.values_.resize(variable_names_.size());
    in.supplied_.resize(variable_names_.size());
    return in;
}
Workspace Evaluator::create_workspace() const {
    Workspace ws;
    ws.owner_ = this;
    // 保存每个可达节点的前向值，供多个输出和数值反传复用。
    ws.values_.resize(instructions_.size());
    ws.adjoints_.resize(instructions_.size());
    ws.custom_inputs_.resize(max_custom_arity_);
    ws.custom_grads_.resize(max_custom_arity_);
    return ws;
}
void Evaluator::check(const Inputs& in, const Workspace& ws) const {
    if (in.owner_ != this || in.graph_ != graph_ || in.values_.size() != variable_names_.size())
        fail("inputs do not match evaluator");
    if (ws.owner_ != this) fail("workspace does not match evaluator");
    for (std::uint8_t flag : in.supplied_) if (!flag) fail("missing value for reachable variable");
}
void Evaluator::run(const Inputs& in, Workspace& ws,
                    std::int32_t override_index, double override_value) const {
    // 顺序执行编译好的指令；custom_inputs_ 是预分配的临时缓冲区。
    for (std::size_t i = 0; i < instructions_.size(); ++i) {
        if (static_cast<std::int32_t>(i) == override_index) {
            ws.values_[i] = override_value;
            continue; // Replace the node's output; its parents stay unchanged.
        }
        const auto& ins = instructions_[i];
        const Node& n = ins.node;
        double a = ins.a >= 0 ? ws.values_[ins.a] : 0;
        double b = ins.b >= 0 ? ws.values_[ins.b] : 0;
        switch (n.op) {
        case Op::Variable: ws.values_[i] = in.values_[ins.variable_index]; break;
        case Op::Constant: ws.values_[i] = n.literal; break;
        case Op::Add: ws.values_[i] = a + b; break;
        case Op::Sub: ws.values_[i] = a - b; break;
        case Op::Mul: ws.values_[i] = a * b; break;
        case Op::Div: ws.values_[i] = a / b; break;
        case Op::Square: ws.values_[i] = a * a; break;
        case Op::Abs: ws.values_[i] = std::abs(a); break;
        case Op::Exp: ws.values_[i] = std::exp(a); break;
        case Op::Sign: ws.values_[i] = sign_value(a); break;
        case Op::Custom:
            for (std::size_t j = 0; j < ins.custom_inputs.size(); ++j)
                ws.custom_inputs_[j] = ws.values_[ins.custom_inputs[j]];
            ws.values_[i] = n.custom->forward(Span<const double>(ws.custom_inputs_.data(), ins.custom_inputs.size()));
            break;
        }
    }
}
void Evaluator::evaluate(const Inputs& in, Span<double> out, Workspace& ws) const {
    if (out.size() != roots_.size()) fail("incorrect number of output slots");
    check(in, ws);
    run(in, ws);
    for (std::size_t i = 0; i < roots_.size(); ++i) out[i] = ws.values_[roots_[i]];
}
void Evaluator::evaluate_with_override(const Inputs& in, Expr node, double value,
                                       Span<double> out, Workspace& ws) const {
    if (out.size() != roots_.size() || !contains(node)) fail("invalid override request");
    check(in, ws);
    run(in, ws, index_by_node_[node.id()], value);
    for (std::size_t i = 0; i < roots_.size(); ++i) out[i] = ws.values_[roots_[i]];
}
std::vector<double> Evaluator::evaluate(
    std::initializer_list<std::pair<std::string, double>> named_inputs) const {
    Inputs in = create_inputs();
    for (std::initializer_list<std::pair<std::string, double>>::const_iterator input = named_inputs.begin();
         input != named_inputs.end(); ++input) {
        std::unordered_map<std::string, std::int32_t>::const_iterator it = variable_by_name_.find(input->first);
        if (it == variable_by_name_.end()) fail("unknown input variable name");
        in.values_[it->second] = input->second;
        in.supplied_[it->second] = 1;
    }
    Workspace ws = create_workspace();
    std::vector<double> out(roots_.size());
    evaluate(in, out, ws);
    return out;
}
void Evaluator::backward(std::size_t root_index, const Inputs& in,
                         Span<const Expr> wrts, Span<double> out, Workspace& ws) const {
    if (root_index >= roots_.size() || out.size() != wrts.size()) fail("invalid gradient request");
    // Validate against this plan's immutable snapshot. Reading Graph::nodes_
    // here would race another thread compiling a new derivative on the graph.
    for (Expr x : wrts)
        if (&x.graph() != graph_ || x.id() >= index_by_node_.size())
            fail("gradient node is not in evaluator snapshot");
    check(in, ws);
    run(in, ws);
    std::fill(ws.adjoints_.begin(), ws.adjoints_.end(), 0.0);
    ws.adjoints_[roots_[root_index]] = 1.0;
    auto& v = ws.values_;
    auto& d = ws.adjoints_;
    for (std::size_t end = instructions_.size(); end > 0;) {
        std::size_t i = --end;
        const auto& ins = instructions_[i];
        double dz = d[i];
    // 反向扫描拓扑序，多个入边的梯度累加至同一个 adjoint。
    // 上游梯度为 0 时跳过，可避免自定义算子产生 0 * infinity -> NaN。
        if (dz == 0) continue;
        switch (ins.node.op) {
        case Op::Variable: case Op::Constant: case Op::Sign: break;
        case Op::Add: d[ins.a] += dz; d[ins.b] += dz; break;
        case Op::Sub: d[ins.a] += dz; d[ins.b] -= dz; break;
        case Op::Mul: d[ins.a] += dz * v[ins.b]; d[ins.b] += dz * v[ins.a]; break;
        case Op::Div:
            d[ins.a] += dz / v[ins.b];
            d[ins.b] -= dz * v[ins.a] / (v[ins.b] * v[ins.b]); break;
        case Op::Square: d[ins.a] += dz * 2 * v[ins.a]; break;
        case Op::Abs: d[ins.a] += dz * sign_value(v[ins.a]); break;
        case Op::Exp: d[ins.a] += dz * v[i]; break;
        case Op::Custom: {
            auto arity = ins.custom_inputs.size();
            std::fill_n(ws.custom_grads_.begin(), arity, 0.0);
            for (std::size_t j = 0; j < arity; ++j)
                ws.custom_inputs_[j] = v[ins.custom_inputs[j]];
            ins.node.custom->backward(
                Span<const double>(ws.custom_inputs_.data(), arity), v[i], dz,
                Span<double>(ws.custom_grads_.data(), arity));
            for (std::size_t j = 0; j < arity; ++j) d[ins.custom_inputs[j]] += ws.custom_grads_[j];
            break;
        }
        }
    }
    for (std::size_t i = 0; i < wrts.size(); ++i) {
        auto index = index_by_node_[wrts[i].id_];
        out[i] = index < 0 ? 0.0 : d[index];
    }
}

} // namespace autodiff
