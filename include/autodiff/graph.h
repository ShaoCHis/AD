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


} // namespace autodiff
