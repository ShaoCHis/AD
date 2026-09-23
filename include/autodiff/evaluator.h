#pragma once

#include "autodiff/graph.h"

namespace autodiff {

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
    bool contains(Expr node) const;

    // 预先准备 inputs/outputs/workspace 后，框架的执行过程不再分配堆内存；
    // 每个可达节点仅执行一次，中间结果存入 Workspace。
    void evaluate(const Inputs& inputs, Span<double> outputs, Workspace& ws) const;
    // Re-evaluate downstream nodes while treating any reachable intermediate
    // node as an independently perturbed input (a graph cut for finite differences).
    void evaluate_with_override(const Inputs& inputs, Expr node, double value,
                                Span<double> outputs, Workspace& ws) const;
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
    void run(const Inputs& inputs, Workspace& ws,
             std::int32_t override_index = -1, double override_value = 0) const;
};

} // namespace autodiff
