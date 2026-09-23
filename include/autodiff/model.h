#pragma once

#include "autodiff/autodiff.h"

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

// Load compiled operators before parsing assignments. This manager must
// outlive every operator instance, graph and evaluator using the plugin.
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

// Plugins export: extern "C" void register_autodiff_ops(CustomOpRegistry&).

// Assignments such as `h=x+y` are parsed first and then resolved as a DAG.
// The symbols map retains every named intermediate expression and variable.
class ExpressionProgram {
public:
    ExpressionProgram(const std::string& source, const CustomOpRegistry& ops);
    ExpressionProgram(const ExpressionProgram&) = delete;
    ExpressionProgram& operator=(const ExpressionProgram&) = delete;

    Graph& graph() { return graph_; }
    const Graph& graph() const { return graph_; }
    Expr at(const std::string& name) const;
    const std::map<std::string, Expr>& symbols() const { return symbols_; }
    bool is_variable(const std::string& name) const { return variables_.count(name) != 0; }
    Inputs make_inputs(const Evaluator& evaluator,
                       const std::map<std::string, double>& values) const;
private:
    struct SyntaxNode;
    typedef std::shared_ptr<SyntaxNode> Syntax;
    struct SyntaxNode {
        enum Kind { Literal, Name, UnaryMinus, Add, Sub, Mul, Div, Call } kind;
        double literal = 0;
        std::string name;
        std::vector<Syntax> args;
        explicit SyntaxNode(Kind k) : kind(k) {}
    };
    class ExpressionParser;
    const CustomOpRegistry& ops_;
    Graph graph_;
    std::map<std::string, Syntax> definitions_;
    std::map<std::string, Expr> symbols_;
    std::set<std::string> variables_;
    std::map<std::string, int> states_;

    Expr resolve(const std::string& name, int depth);
    Expr build(const Syntax& node, int depth);
};

std::string read_expression_file(const std::string& path);

} // namespace autodiff
