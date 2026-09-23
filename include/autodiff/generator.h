#pragma once

#include "autodiff/evaluator.h"

#include <map>
#include <memory>
#include <mutex>
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

// JSON only describes how to generate the static graph. Inputs and queries
// belong to evaluator requests and are intentionally outside this config.
struct GeneratorConfig {
    std::vector<std::string> expressions;
    std::vector<std::string> op_plugins;
};

GeneratorConfig read_generator_json(const std::string& path);

class GraphGenerator {
public:
    // Exactly one graph is loaded per process. Repeated initialization with
    // identical config returns the same object; different config is rejected.
    static GraphGenerator& initialize(const GeneratorConfig& config);
    static GraphGenerator& instance();
    GraphGenerator(const GraphGenerator&) = delete;
    GraphGenerator& operator=(const GraphGenerator&) = delete;
    // Expose only read-only symbols; graph mutations go through compile().
    const ExpressionProgram& program() const { return *program_; }
    // Concurrent calls are serialized while symbolic derivative nodes are
    // added. The returned immutable plan can execute on any number of threads.
    std::shared_ptr<const Evaluator> compile(const std::vector<EvaluationQuery>& queries,
                                              bool explain_dot = false);
    void dump_dot(Expr node, std::ostream& out) const;
    void dump_derivative_dot(Expr root, Expr output, Expr derivative,
                             std::ostream& out, const std::string& root_name,
                             const std::string& output_name) const;
private:
    explicit GraphGenerator(const GeneratorConfig& config);
    GeneratorConfig config_;
    // Destruction is in reverse order: graph, registry, loaded plugin code.
    PluginManager plugins_;
    CustomOpRegistry registry_;
    std::unique_ptr<ExpressionProgram> program_;
    mutable std::mutex graph_mutex_;
    std::map<std::string, std::shared_ptr<const Evaluator>> plans_;
};

} // namespace autodiff
