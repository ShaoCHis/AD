#pragma once

#include "autodiff/generator.h"

#include <iosfwd>
#include <map>
#include <string>
#include <vector>

namespace autodiff {

struct NamedQuery {
    EvaluationQuery request;
    std::string label;
    std::string root_name;
    std::string output_name;
};

struct PostEvaluationOptions {
    bool emit_dot = false;
    std::string dot_dir = "dot";
    bool verify_derivatives = false;
    double fd_step = 1e-5; // second-order-accurate central difference
    double abs_tolerance = 1e-6;
    double rel_tolerance = 1e-4;
};

struct DerivativeCheck {
    std::string label;
    double analytic = 0;
    double numeric = 0;
    double absolute_error = 0;
    double allowed_error = 0;
    bool passed = false;
    std::string reason;
};

// Post-processing is optional. prepare() must precede symbolic differentiation
// because DOT explanations capture the propagation rules when created.
class PostEvaluator {
public:
    explicit PostEvaluator(const PostEvaluationOptions& options);
    void prepare(Graph& graph) const;
    std::vector<DerivativeCheck> run(ExpressionProgram& program,
                                     const Evaluator& evaluator,
                                     Span<const NamedQuery> queries,
                                     const std::map<std::string, double>& input_values,
                                     Span<const double> results) const;
private:
    PostEvaluationOptions options_;
    void write_dot(ExpressionProgram& program, const Evaluator& evaluator,
                   Span<const NamedQuery> queries) const;
    DerivativeCheck verify(ExpressionProgram& program, const NamedQuery& query,
                           double analytic,
                           const std::map<std::string, double>& input_values) const;
};

} // namespace autodiff
