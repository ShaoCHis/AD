#include "autodiff/autodiff.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace autodiff {
namespace {
[[noreturn]] void fail(const char* message) { throw std::invalid_argument(message); }
double sign_value(double x) {
    if (std::isnan(x)) return std::numeric_limits<double>::quiet_NaN();
    return (x > 0) - (x < 0);
}
std::string op_name(Op op) {
    switch (op) {
    case Op::Variable: return "Variable";
    case Op::Constant: return "Constant";
    case Op::Add: return "Add";
    case Op::Sub: return "Sub";
    case Op::Mul: return "Mul";
    case Op::Div: return "Div";
    case Op::Square: return "Square";
    case Op::Abs: return "Abs";
    case Op::Exp: return "Exp";
    case Op::Sign: return "Sign";
    case Op::Custom: return "Custom";
    }
    return "Unknown";
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
std::string dot_escape(const std::string& s) {
    std::string result;
    for (char c : s) {
        if (c == '\\' || c == '"') result += '\\';
        if (c == '\n') { result += "\\n"; continue; }
        result += c;
    }
    return result;
}
} // namespace

Graph& Expr::graph() const {
    if (!graph_) fail("uninitialized expression");
    return *graph_;
}
Expr Expr::square() const { return graph().unary(Op::Square, *this); }
Expr Expr::abs() const { return graph().unary(Op::Abs, *this); }
Expr Expr::pow(int exponent) const {
    auto& g = graph();
    if (exponent == 0) return g.constant(1);
    if (exponent == 1) return *this;
    if (exponent == 2) return square();
    // Convert through int64_t so INT_MIN is safe.
    std::int64_t n = exponent;
    bool inverse = n < 0;
    if (inverse) n = -n;
    Expr result = g.constant(1);
    Expr base = *this;
    while (n) {
        if (n & 1) result = result * base;
        n >>= 1;
        if (n) base = base.square();
    }
    return inverse ? g.constant(1) / result : result;
}
Expr Expr::derivative(Expr wrt) const { return derivative(Span<const Expr>(&wrt, 1))[0]; }
std::vector<Expr> Expr::derivative(std::initializer_list<Expr> wrts) const {
    return derivative(Span<const Expr>(wrts.begin(), wrts.size()));
}
std::vector<Expr> Expr::derivative(Span<const Expr> wrts) const {
    return graph().derivatives(*this, wrts);
}
Expr Expr::named(std::string name) const {
    graph().validate(*this);
    graph().nodes_[id_].name = std::move(name);
    return *this;
}
std::string Expr::explain() const {
    const auto& g = graph();
    g.validate(*this);
    std::function<std::string(NodeId, int)> format = [&](NodeId id, int depth) -> std::string {
        const auto& n = g.nodes_[id];
        if (!n.name.empty()) return n.name;
        if (depth >= 8) return "node#" + std::to_string(id);
        auto a = [&] { return format(n.a, depth + 1); };
        auto b = [&] { return format(n.b, depth + 1); };
        switch (n.op) {
        case Op::Variable: return "var#" + std::to_string(id);
        case Op::Constant: { std::ostringstream s; s << n.literal; return s.str(); }
        case Op::Add: return "(" + a() + " + " + b() + ")";
        case Op::Sub: return "(" + a() + " - " + b() + ")";
        case Op::Mul: return "(" + a() + " * " + b() + ")";
        case Op::Div: return "(" + a() + " / " + b() + ")";
        case Op::Square: return "square(" + a() + ")";
        case Op::Abs: return "abs(" + a() + ")";
        case Op::Exp: return "exp(" + a() + ")";
        case Op::Sign: return "sign(" + a() + ")";
        case Op::Custom: {
            std::string s(n.custom->name()); s += '(';
            for (std::size_t i = 0; i < n.custom_inputs.size(); ++i) {
                if (i) s += ", ";
                s += format(n.custom_inputs[i], depth + 1);
            }
            return s + ')';
        }
        }
        return "?";
    };
    return format(id_, 0);
}

std::size_t Graph::KeyHash::operator()(const Key& k) const noexcept {
    std::size_t h = static_cast<std::size_t>(k.op);
    auto mix = [&](std::uint64_t x) { h ^= std::hash<std::uint64_t>{}(x) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2); };
    mix(k.a); mix(k.b); mix(k.bits);
    return h;
}
void Graph::validate(Expr x) const {
    if (x.graph_ != this || x.id_ >= nodes_.size()) fail("expression belongs to another graph or is invalid");
}
Expr Graph::variable(std::string name) {
    if (name.empty() || variables_.find(name) != variables_.end())
        fail("variable name must be nonempty and unique");
    NodeId id = static_cast<NodeId>(nodes_.size());
    Node n;
    n.op = Op::Variable;
    n.name = name;
    nodes_.push_back(std::move(n));
    variables_.emplace(std::move(name), id);
    return Expr(this, id);
}
Expr Graph::constant(double value) {
    // 以 IEEE 754 原始位作为常量去重键，区分 +0 与 -0；C++11 用 memcpy 取位。
    std::uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value), "double must be 64 bits");
    std::memcpy(&bits, &value, sizeof(bits));
    Key key{Op::Constant, 0, 0, bits};
    std::unordered_map<Key, NodeId, KeyHash>::const_iterator found = interned_.find(key);
    if (found != interned_.end()) return Expr(this, found->second);
    NodeId id = static_cast<NodeId>(nodes_.size());
    Node n;
    n.op = Op::Constant;
    n.literal = value;
    nodes_.push_back(std::move(n));
    interned_.emplace(key, id);
    return Expr(this, id);
}
Expr Graph::unary(Op op, Expr x) {
    validate(x);
    if (op != Op::Square && op != Op::Abs && op != Op::Exp && op != Op::Sign) fail("invalid unary operation");
    const auto& n = nodes_[x.id_];
    if (n.op == Op::Constant) {
        double v = n.literal;
        if (op == Op::Square) return constant(v * v);
        if (op == Op::Abs) return constant(std::abs(v));
        if (op == Op::Exp) return constant(std::exp(v));
        return constant(sign_value(v));
    }
    Key key{op, x.id_, 0, 0};
    std::unordered_map<Key, NodeId, KeyHash>::const_iterator found = interned_.find(key);
    if (found != interned_.end()) return Expr(this, found->second);
    NodeId id = static_cast<NodeId>(nodes_.size());
    Node result;
    result.op = op;
    result.a = x.id_;
    nodes_.push_back(std::move(result));
    interned_.emplace(key, id);
    return Expr(this, id);
}
Expr Graph::binary(Op op, Expr x, Expr y) {
    validate(x); validate(y);
    if (op != Op::Add && op != Op::Sub && op != Op::Mul && op != Op::Div) fail("invalid binary operation");
    // 必须在插入新节点前读出常量：vector 扩容会让旧节点引用失效。
    bool ac = nodes_[x.id_].op == Op::Constant, bc = nodes_[y.id_].op == Op::Constant;
    double av = ac ? nodes_[x.id_].literal : 0, bv = bc ? nodes_[y.id_].literal : 0;
    if (ac && bc) {
        switch (op) {
        case Op::Add: return constant(av + bv);
        case Op::Sub: return constant(av - bv);
        case Op::Mul: return constant(av * bv);
        case Op::Div: return constant(av / bv);
        default: break;
        }
    }
    // 便宜的代数化简可以缩小高阶导数图；这里不做 x*0，避免吞掉 NaN。
    if (op == Op::Add && bc && bv == 0) return x;
    if (op == Op::Add && ac && av == 0) return y;
    if (op == Op::Sub && bc && bv == 0) return x;
    if (op == Op::Mul && bc && bv == 1) return x;
    if (op == Op::Mul && ac && av == 1) return y;
    if (op == Op::Div && bc && bv == 1) return x;
    if ((op == Op::Add || op == Op::Mul) && x.id_ > y.id_) std::swap(x, y);
    Key key{op, x.id_, y.id_, 0};
    std::unordered_map<Key, NodeId, KeyHash>::const_iterator found = interned_.find(key);
    if (found != interned_.end()) return Expr(this, found->second);
    NodeId id = static_cast<NodeId>(nodes_.size());
    Node n;
    n.op = op;
    n.a = x.id_;
    n.b = y.id_;
    nodes_.push_back(std::move(n));
    interned_.emplace(key, id);
    return Expr(this, id);
}
Expr Graph::sign(Expr x) { return unary(Op::Sign, x); }
Expr Graph::custom(std::shared_ptr<const CustomOp> op, Span<const Expr> inputs) {
    if (!op || inputs.empty()) fail("custom operator and its inputs are required");
    std::vector<NodeId> ids;
    ids.reserve(inputs.size());
    for (Expr x : inputs) { validate(x); ids.push_back(x.id()); }
    NodeId id = static_cast<NodeId>(nodes_.size());
    Node n;
    n.op = Op::Custom;
    n.custom = std::move(op);
    n.custom_inputs = std::move(ids);
    nodes_.push_back(std::move(n));
    return Expr(this, id);
}
Expr Graph::custom(std::shared_ptr<const CustomOp> op, std::initializer_list<Expr> inputs) {
    return custom(std::move(op), Span<const Expr>(inputs.begin(), inputs.size()));
}

std::vector<Expr> Graph::derivatives(Expr output, Span<const Expr> wrts) {
    validate(output);
    for (Expr x : wrts) validate(x);
    // 只遍历本次 output 的祖先；后面所有 wrt 共用这一遍逆序传播。
    const std::size_t initial_size = nodes_.size();
    std::vector<std::uint8_t> reachable(initial_size);
    std::vector<NodeId> stack{output.id_};
    while (!stack.empty()) {
        NodeId id = stack.back(); stack.pop_back();
        if (reachable[id]) continue;
        reachable[id] = 1;
        children(nodes_[id], [&](NodeId child) { stack.push_back(child); });
    }
    // adj[id] 表示 d(output)/d(node[id])，元素本身仍是 Expr，
    // 所以可以把梯度图再次传给 derivative() 求高阶导。
    std::vector<Expr> adj(initial_size);
    adj[output.id_] = constant(1);
    std::vector<DerivationStep> steps;
    NodeId current_source = output.id_;
    NodeId current_upstream = adj[output.id_].id_;
    auto add = [&](NodeId id, Expr contribution, const char* rule) {
        if (record_derivations_)
            steps.push_back(DerivationStep{current_source, current_upstream, id,
                                           contribution.id_, rule});
        if (adj[id].graph_) adj[id] = adj[id] + contribution;
        else adj[id] = contribution;
    };
    for (std::size_t end = initial_size; end > 0;) {
        NodeId id = static_cast<NodeId>(--end);
        if (!reachable[id] || !adj[id].graph_) continue;
        // 符号求导会向 nodes_ 追加节点；先复制旧节点，避免 vector 扩容悬空。
        Node n = nodes_[id];
        Expr dz = adj[id], a(this, n.a), b(this, n.b);
        current_source = id;
        current_upstream = dz.id_;
        switch (n.op) {
        case Op::Variable: case Op::Constant: case Op::Sign: break;
        case Op::Add: add(n.a, dz, "dz * 1"); add(n.b, dz, "dz * 1"); break;
        case Op::Sub: add(n.a, dz, "dz * 1"); add(n.b, -dz, "dz * -1"); break;
        case Op::Mul: add(n.a, dz * b, "dz * right"); add(n.b, dz * a, "dz * left"); break;
        // 链式法则：对 a/b 分别传递 dz/b 与 -dz*a/b²。
        case Op::Div: add(n.a, dz / b, "dz / right"); add(n.b, -(dz * a / b.square()), "-dz * left / right^2"); break;
        case Op::Square: add(n.a, dz * (2.0 * a), "dz * 2 * input"); break;
        case Op::Abs: add(n.a, dz * sign(a), "dz * sign(input)"); break;
        // exp 的局部导数就是本节点结果；引用已有节点以复用前向值。
        case Op::Exp: add(n.a, dz * Expr(this, id), "dz * exp(input)"); break;
        case Op::Custom: {
            std::vector<Expr> inputs;
            inputs.reserve(n.custom_inputs.size());
            for (NodeId child : n.custom_inputs) inputs.push_back(Expr(this, child));
            for (std::size_t i = 0; i < inputs.size(); ++i) {
                Expr partial = n.custom->symbolic_partial(*this, inputs, Expr(this, id), i);
                validate(partial);
                add(n.custom_inputs[i], dz * partial, "dz * symbolic_partial(input)");
            }
            break;
        }
        }
    }
    std::vector<Expr> results;
    results.reserve(wrts.size());
    for (Expr x : wrts) results.push_back(adj[x.id_].graph_ ? adj[x.id_] : constant(0));
    if (record_derivations_) {
        std::shared_ptr<const std::vector<DerivationStep>> saved(
            new std::vector<DerivationStep>(std::move(steps)));
        for (std::size_t i = 0; i < wrts.size(); ++i)
            derivations_.push_back(DerivationPass{output.id_, wrts[i].id_,
                                                 results[i].id_, initial_size, saved});
    }
    return results;
}
void Graph::dump_dot(Expr root, std::ostream& out) const {
    validate(root);
    std::vector<std::uint8_t> marked(nodes_.size());
    std::vector<NodeId> stack{root.id_};
    while (!stack.empty()) {
        NodeId id = stack.back(); stack.pop_back();
        if (marked[id]) continue;
        marked[id] = 1;
        children(nodes_[id], [&](NodeId child) { stack.push_back(child); });
    }
    out << "digraph expression {\n  rankdir=LR;\n";
    for (NodeId id = 0; id < nodes_.size(); ++id) {
        if (!marked[id]) continue;
        const Node& n = nodes_[id];
        std::string label = n.op == Op::Custom ? std::string(n.custom->name()) : op_name(n.op);
        if (!n.name.empty()) label = n.name + "\n" + label;
        if (n.op == Op::Constant) { std::ostringstream s; s << n.literal; label += "\n" + s.str(); }
        out << "  n" << id << " [label=\"" << dot_escape(label) << "\"];\n";
        children(n, [&](NodeId child) { out << "  n" << child << " -> n" << id << ";\n"; });
    }
    out << "}\n";
}
void Graph::dump_derivative_dot(Expr source, Expr wrt, Expr derivative,
                                std::ostream& out, const std::string& source_label,
                                const std::string& wrt_label) const {
    validate(source); validate(wrt); validate(derivative);
    const DerivationPass* pass = nullptr;
    for (std::vector<DerivationPass>::const_reverse_iterator it = derivations_.rbegin();
         it != derivations_.rend(); ++it) {
        if (it->source == source.id_ && it->wrt == wrt.id_ && it->result == derivative.id_) {
            pass = &*it;
            break;
        }
    }
    if (!pass) fail("derivative explanation was not recorded; enable it before differentiating");
    // Show only propagation steps whose receiving input still depends on wrt.
    // This keeps, for example, the unrelated y branch out of d(loss)/d(p).
    std::vector<std::uint8_t> on_wrt_path(pass->original_size);
    for (NodeId id = 0; id < pass->original_size; ++id) {
        if (id == wrt.id_) on_wrt_path[id] = 1;
        else children(nodes_[id], [&](NodeId child) {
            if (child < pass->original_size && on_wrt_path[child]) on_wrt_path[id] = 1;
        });
    }
    std::vector<std::uint8_t> marked(nodes_.size());
    std::vector<NodeId> stack{source.id_, derivative.id_};
    for (const DerivationStep& step : *pass->steps)
        if (on_wrt_path[step.input]) stack.push_back(step.contribution);
    while (!stack.empty()) {
        NodeId id = stack.back(); stack.pop_back();
        if (marked[id]) continue;
        marked[id] = 1;
        children(nodes_[id], [&](NodeId child) { stack.push_back(child); });
    }
    std::string source_name = source_label.empty() ?
        (nodes_[source.id_].name.empty() ? "n" + std::to_string(source.id_) : nodes_[source.id_].name) : source_label;
    std::string wrt_name = wrt_label.empty() ?
        (nodes_[wrt.id_].name.empty() ? "n" + std::to_string(wrt.id_) : nodes_[wrt.id_].name) : wrt_label;
    out << "digraph derivative {\n  rankdir=LR;\n  node [style=filled];\n";
    for (NodeId id = 0; id < nodes_.size(); ++id) {
        if (!marked[id]) continue;
        const Node& n = nodes_[id];
        std::string label = "#" + std::to_string(id) + " " +
            (n.op == Op::Custom ? std::string(n.custom->name()) : op_name(n.op));
        if (!n.name.empty()) label += "\n" + n.name;
        if (n.op == Op::Constant) { std::ostringstream s; s << n.literal; label += "\n" + s.str(); }
        const char* color = id >= pass->original_size ? "#d9eeff" : "#eeeeee";
        if (id == source.id_) color = "#ffe5ba";
        if (id == wrt.id_) color = "#dff2cc";
        if (id == derivative.id_) color = "#dfceff";
        out << "  n" << id << " [label=\"" << dot_escape(label)
            << "\", fillcolor=\"" << color << "\"];\n";
        children(n, [&](NodeId child) { out << "  n" << child << " -> n" << id << ";\n"; });
    }
    std::string heading = "d(" + source_name + ")/d(" + wrt_name + ")";
    out << "  summary [shape=note, fillcolor=\"#f5efff\", label=\""
        << dot_escape(heading + "\n" + derivative.explain()) << "\"];\n";
    out << "  n" << source.id_ << " -> summary [style=dashed, color=purple, label=\"differentiate\"];\n";
    if (marked[wrt.id_])
        out << "  n" << wrt.id_ << " -> summary [style=dotted, color=green, label=\"with respect to\"];\n";
    out << "  summary -> n" << derivative.id_ << " [style=dashed, color=purple, label=\"result\"];\n";
    // Dashed propagation notes describe why each contribution was generated.
    std::size_t visible_rule = 0;
    for (std::size_t i = 0; i < pass->steps->size(); ++i) {
        const DerivationStep& step = (*pass->steps)[i];
        if (!on_wrt_path[step.input]) continue;
        const std::string label = "#" + std::to_string(step.node) + " " + op_name(nodes_[step.node].op) +
            " -> input #" + std::to_string(step.input) + "\n" + step.rule +
            "\nupstream #" + std::to_string(step.upstream) +
            "; contribution #" + std::to_string(step.contribution);
        out << "  rule" << visible_rule << " [shape=note, fillcolor=\"#fff7d6\", label=\""
            << dot_escape(label) << "\"];\n";
        out << "  n" << step.node << " -> rule" << visible_rule << " [style=dashed, color=\"#b38423\"];\n";
        out << "  rule" << visible_rule++ << " -> n" << step.contribution << " [style=dotted, color=\"#b38423\"];\n";
    }
    out << "}\n";
}

Expr operator+(Expr a, Expr b) { return a.graph().binary(Op::Add, a, b); }
Expr operator-(Expr a, Expr b) { return a.graph().binary(Op::Sub, a, b); }
Expr operator*(Expr a, Expr b) { return a.graph().binary(Op::Mul, a, b); }
Expr operator/(Expr a, Expr b) { return a.graph().binary(Op::Div, a, b); }
Expr operator-(Expr a) { return a.graph().constant(-1) * a; }
Expr operator+(Expr a, double b) { return a + a.graph().constant(b); }
Expr operator+(double a, Expr b) { return b.graph().constant(a) + b; }
Expr operator-(Expr a, double b) { return a - a.graph().constant(b); }
Expr operator-(double a, Expr b) { return b.graph().constant(a) - b; }
Expr operator*(Expr a, double b) { return a * a.graph().constant(b); }
Expr operator*(double a, Expr b) { return b.graph().constant(a) * b; }
Expr operator/(Expr a, double b) { return a / a.graph().constant(b); }
Expr operator/(double a, Expr b) { return b.graph().constant(a) / b; }
Expr exp(Expr a) { return a.graph().unary(Op::Exp, a); }
Expr abs(Expr a) { return a.abs(); }

void Inputs::set(Expr variable, double value) {
    if (variable.graph_ != graph_ || variable.id_ >= index_by_node_.size()) fail("input variable belongs to another graph");
    std::int32_t i = index_by_node_[variable.id_];
    if (i < 0) fail("input expression is not a reachable variable");
    values_[i] = value;
    supplied_[i] = 1;
}

Evaluator::Evaluator(std::initializer_list<Expr> roots)
    : Evaluator(Span<const Expr>(roots.begin(), roots.size())) {}
Evaluator::Evaluator(Span<const Expr> roots) {
    if (roots.empty()) fail("evaluator requires at least one output");
    graph_ = &roots[0].graph();
    for (Expr root : roots) graph_->validate(root);
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
void Evaluator::run(const Inputs& in, Workspace& ws) const {
    // 顺序执行编译好的指令；custom_inputs_ 是预分配的临时缓冲区。
    for (std::size_t i = 0; i < instructions_.size(); ++i) {
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
    for (Expr x : wrts) graph_->validate(x);
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
