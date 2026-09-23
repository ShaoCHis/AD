#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iosfwd>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace autodiff {

using NodeId = std::uint32_t;
class Graph;
class Evaluator;
class Inputs;

// C++11 没有 std::span：这里提供不拥有内存的连续视图。
// 调用期间，原始数组或 vector 必须保持有效。
template <typename T>
class Span {
public:
    Span(T* data, std::size_t size) : data_(data), size_(size) {}

    template <typename U, std::size_t N>
    Span(std::array<U, N>& a,
         typename std::enable_if<std::is_convertible<U*, T*>::value>::type* = 0)
        : data_(a.data()), size_(N) {}

    template <typename U, std::size_t N>
    Span(const std::array<U, N>& a,
         typename std::enable_if<std::is_convertible<const U*, T*>::value>::type* = 0)
        : data_(a.data()), size_(N) {}

    template <typename U>
    Span(std::vector<U>& v,
         typename std::enable_if<std::is_convertible<U*, T*>::value>::type* = 0)
        : data_(v.data()), size_(v.size()) {}

    template <typename U>
    Span(const std::vector<U>& v,
         typename std::enable_if<std::is_convertible<const U*, T*>::value>::type* = 0)
        : data_(v.data()), size_(v.size()) {}

    T* data() const { return data_; }
    T* begin() const { return data_; }
    T* end() const { return size_ ? data_ + size_ : data_; }
    std::size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    T& operator[](std::size_t i) const { return data_[i]; }

private:
    T* data_;
    std::size_t size_;
};

// 表达式只是 Graph* + NodeId 的轻量句柄，不单独分配计算节点。
class Expr {
public:
    Expr() = default;
    NodeId id() const noexcept { return id_; }
    Graph& graph() const;
    Expr square() const;
    Expr abs() const;
    Expr pow(int exponent) const;
    Expr derivative(Expr wrt) const;
    std::vector<Expr> derivative(std::initializer_list<Expr> wrts) const;
    std::vector<Expr> derivative(Span<const Expr> wrts) const;
    Expr named(std::string name) const;
    std::string explain() const;

private:
    friend class Graph;
    friend class Evaluator;
    friend class Inputs;
    Expr(Graph* graph, NodeId id) : graph_(graph), id_(id) {}
    Graph* graph_ = nullptr;
    NodeId id_ = 0;
};

Expr operator+(Expr a, Expr b);
Expr operator-(Expr a, Expr b);
Expr operator*(Expr a, Expr b);
Expr operator/(Expr a, Expr b);
Expr operator-(Expr a);
Expr operator+(Expr a, double b);
Expr operator+(double a, Expr b);
Expr operator-(Expr a, double b);
Expr operator-(double a, Expr b);
Expr operator*(Expr a, double b);
Expr operator*(double a, Expr b);
Expr operator/(Expr a, double b);
Expr operator/(double a, Expr b);
Expr exp(Expr a);
Expr abs(Expr a);

// 自定义算子须是无副作用且线程安全的：数值 backward 用于立即计算梯度，
// symbolic_partial 生成新的导数表达式，从而支持二阶及更高阶导数。
class CustomOp {
public:
    virtual ~CustomOp() = default;
    virtual const char* name() const = 0;
    // Explicit arity lets the parser reject func_a(x) before evaluation.
    virtual std::size_t arity() const = 0;
    virtual double forward(Span<const double> inputs) const = 0;
    // 把梯度贡献累加到 input_grads；调用方会先清零这个数组。
    virtual void backward(Span<const double> inputs, double output,
                          double output_grad, Span<double> input_grads) const = 0;
    virtual Expr symbolic_partial(Graph& graph, Span<const Expr> inputs,
                                  Expr output, std::size_t input_index) const = 0;
};

enum class Op : std::uint8_t {
    Variable, Constant, Add, Sub, Mul, Div, Square, Abs, Exp, Sign, Custom
};

struct Node {
    Op op = Op::Constant;
    NodeId a = 0, b = 0;
    double literal = 0.0;
    std::string name;
    std::shared_ptr<const CustomOp> custom;
    std::vector<NodeId> custom_inputs;
};

class Graph {
public:
    Graph() = default;
    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;
    Graph(Graph&&) = delete;
    Graph& operator=(Graph&&) = delete;

    Expr variable(std::string name);
    Expr constant(double value);
    Expr custom(std::shared_ptr<const CustomOp> op, Span<const Expr> inputs);
    Expr custom(std::shared_ptr<const CustomOp> op, std::initializer_list<Expr> inputs);
    std::vector<Expr> derivatives(Expr output, Span<const Expr> wrts);
    void dump_dot(Expr root, std::ostream& out) const;
    // Enable before derivative() to retain the propagation rules for DOT.
    void enable_derivative_explanations(bool enabled) { record_derivations_ = enabled; }
    void dump_derivative_dot(Expr source, Expr wrt, Expr derivative, std::ostream& out,
                             const std::string& source_label = "",
                             const std::string& wrt_label = "") const;
    std::size_t node_count() const noexcept { return nodes_.size(); }

private:
    friend class Expr;
    friend class Evaluator;
    friend Expr operator+(Expr, Expr);
    friend Expr operator-(Expr, Expr);
    friend Expr operator*(Expr, Expr);
    friend Expr operator/(Expr, Expr);
    friend Expr exp(Expr);

    Expr unary(Op op, Expr x);
    Expr binary(Op op, Expr x, Expr y);
    void validate(Expr x) const;
    Expr sign(Expr x);
    // 节点按创建顺序存放；一个节点的输入总是更早创建的节点。
    std::vector<Node> nodes_;
    // 内置算子做结构去重（CSE）；自定义算子调用保留独立身份。
    struct Key {
        Op op;
        NodeId a, b;
        std::uint64_t bits;
        bool operator==(const Key& other) const {
            return op == other.op && a == other.a && b == other.b && bits == other.bits;
        }
    };
    struct KeyHash {
        std::size_t operator()(const Key& key) const noexcept;
    };
    std::unordered_map<Key, NodeId, KeyHash> interned_;
    std::unordered_map<std::string, NodeId> variables_;
    struct DerivationStep {
        NodeId node, upstream, input, contribution;
        std::string rule;
    };
    struct DerivationPass {
        NodeId source, wrt, result;
        std::size_t original_size;
        std::shared_ptr<const std::vector<DerivationStep>> steps;
    };
    bool record_derivations_ = false;
    std::vector<DerivationPass> derivations_;
};

class Inputs {
public:
    void set(Expr variable, double value);
private:
    friend class Evaluator;
    const Evaluator* owner_ = nullptr;
    const Graph* graph_ = nullptr;
    std::vector<std::int32_t> index_by_node_;
    std::vector<double> values_;
    std::vector<std::uint8_t> supplied_;
};

class Workspace {
private:
    friend class Evaluator;
    const Evaluator* owner_ = nullptr;
    std::vector<double> values_;
    std::vector<double> adjoints_;
    std::vector<double> custom_inputs_;
    std::vector<double> custom_grads_;
};

// A value query needs only output. A derivative query means d(root)/d(output).
// Both names are graph nodes, so an intermediate expression can be either one.
struct EvaluationQuery {
    enum Kind { Value, Derivative } kind;
    Expr root, output;
    static EvaluationQuery value(Expr output) {
        EvaluationQuery q = {Value, Expr(), output};
        return q;
    }
    static EvaluationQuery derivative(Expr root, Expr output) {
        EvaluationQuery q = {Derivative, root, output};
        return q;
    }
};

// 编译时仅收集输出可达的节点，并将多个输出合并到同一执行计划。
// 图增加新输出后，为新输出重新构造 Evaluator。
class Evaluator {
public:
    explicit Evaluator(std::initializer_list<Expr> roots);
    explicit Evaluator(Span<const Expr> roots);
    explicit Evaluator(std::initializer_list<EvaluationQuery> queries);
    explicit Evaluator(Span<const EvaluationQuery> queries);
    Inputs create_inputs() const;
    Workspace create_workspace() const;
    std::size_t output_count() const noexcept { return roots_.size(); }
    std::size_t instruction_count() const noexcept { return instructions_.size(); }
    Expr result_expression(std::size_t index) const;
    bool uses_variable(Expr variable) const;

    // 预先准备 inputs/outputs/workspace 后，框架的执行过程不再分配堆内存；
    // 每个可达节点仅执行一次，中间结果存入 Workspace。
    void evaluate(const Inputs& inputs, Span<double> outputs, Workspace& ws) const;
    std::vector<double> evaluate(
        std::initializer_list<std::pair<std::string, double>> named_inputs) const;
    // 从指定输出根直接做一次数值反传；wrts 允许包含中间节点，
    // gradient_outputs 的顺序与 wrts 一致。
    void backward(std::size_t root_index, const Inputs& inputs,
                  Span<const Expr> wrts, Span<double> gradient_outputs,
                  Workspace& ws) const;

private:
    struct Instruction {
        Node node;
        NodeId original_id;
        std::int32_t a = -1, b = -1;
        std::vector<std::int32_t> custom_inputs;
        std::int32_t variable_index = -1;
    };
    const Graph* graph_ = nullptr;
    std::vector<Instruction> instructions_;
    std::vector<std::int32_t> index_by_node_;
    std::vector<std::int32_t> roots_;
    std::vector<Expr> result_expressions_;
    std::vector<std::string> variable_names_;
    std::unordered_map<std::string, std::int32_t> variable_by_name_;
    std::size_t max_custom_arity_ = 0;
    void check(const Inputs& inputs, const Workspace& ws) const;
    void compile(Span<const Expr> roots);
    void run(const Inputs& inputs, Workspace& ws) const;
};

} // namespace autodiff
