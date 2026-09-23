#include "autodiff/post_evaluator.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <fstream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <sys/stat.h>

namespace autodiff {
namespace {
std::mutex& dot_file_mutex() {
    static std::mutex lock;
    return lock;
}
std::string safe_name(const std::string& label) {
    std::string result;
    for (unsigned char c : label) {
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') || c == '_' || c == '-') result += static_cast<char>(c);
        else result += '_';
    }
    return result;
}
void prepare_directory(const std::string& path) {
    if (path.empty()) throw std::invalid_argument("dot_dir cannot be empty");
    if (mkdir(path.c_str(), 0755) != 0 && errno != EEXIST)
        throw std::runtime_error("cannot create dot_dir: " + path);
    struct stat info;
    if (stat(path.c_str(), &info) != 0 || !S_ISDIR(info.st_mode))
        throw std::runtime_error("dot_dir is not a directory: " + path);
}
} // namespace

PostEvaluator::PostEvaluator(const PostEvaluationOptions& options) : options_(options) {
    if (options_.verify_derivatives &&
        (!std::isfinite(options_.fd_step) || options_.fd_step <= 0 ||
         !std::isfinite(options_.abs_tolerance) || options_.abs_tolerance < 0 ||
         !std::isfinite(options_.rel_tolerance) || options_.rel_tolerance < 0))
        throw std::invalid_argument("finite difference step and tolerances must be finite; step > 0, tolerances >= 0");
}
std::vector<DerivativeCheck> PostEvaluator::run(
    GraphGenerator& generator, const Evaluator& evaluator, Span<const NamedQuery> queries,
    const std::map<std::string, double>& input_values, Span<const double> results) const {
    if (queries.size() != evaluator.output_count() || results.size() != queries.size())
        throw std::invalid_argument("post-evaluator query/result counts differ");
    if (options_.emit_dot) write_dot(generator, evaluator, queries);
    std::vector<DerivativeCheck> checks;
    if (options_.verify_derivatives) {
        for (std::size_t i = 0; i < queries.size(); ++i)
            if (queries[i].request.kind == EvaluationQuery::Derivative)
                checks.push_back(verify(generator, queries[i], results[i], input_values));
    }
    return checks;
}
void PostEvaluator::write_dot(GraphGenerator& generator, const Evaluator& evaluator,
                              Span<const NamedQuery> queries) const {
    // A shared directory may receive post-processing from several threads.
    std::lock_guard<std::mutex> file_guard(dot_file_mutex());
    prepare_directory(options_.dot_dir);
    for (std::size_t i = 0; i < queries.size(); ++i) {
        const NamedQuery& item = queries[i];
        std::string path = options_.dot_dir + "/query_" + std::to_string(i + 1) +
                           "_" + safe_name(item.label) + ".dot";
        std::ofstream out(path.c_str());
        if (!out) throw std::runtime_error("cannot write DOT file: " + path);
        if (item.request.kind == EvaluationQuery::Value)
            generator.dump_dot(item.request.output, out);
        else generator.dump_derivative_dot(item.request.root, item.request.output,
            evaluator.result_expression(i), out, item.root_name, item.output_name);
        if (!out) throw std::runtime_error("failed to write DOT file: " + path);
    }
}
DerivativeCheck PostEvaluator::verify(GraphGenerator& generator, const NamedQuery& query,
                                       double analytic,
                                       const std::map<std::string, double>& input_values) const {
    DerivativeCheck check;
    check.label = query.label;
    check.analytic = analytic;
    const Expr root = query.request.root;
    const Expr output = query.request.output;
    if (!std::isfinite(analytic)) {
        check.reason = "non-finite analytic derivative";
        return check;
    }
    // For a node outside root's ancestry the graph-cut derivative is zero.
    std::shared_ptr<const Evaluator> root_only = generator.compile({EvaluationQuery::value(root)});
    if (!root_only->contains(output)) {
        check.numeric = 0;
    } else {
        // Probe the primal graph, not the derivative graph. The second root
        // gives the original value at the perturbation point.
        std::shared_ptr<const Evaluator> probe = generator.compile(
            {EvaluationQuery::value(root), EvaluationQuery::value(output)});
        Inputs inputs = generator.program().make_inputs(*probe, input_values);
        Workspace workspace = probe->create_workspace();
        double baseline[2];
        probe->evaluate(inputs, Span<double>(baseline, 2), workspace);
        if (!std::isfinite(baseline[0]) || !std::isfinite(baseline[1])) {
            check.reason = "non-finite baseline value";
            return check;
        }
        double step = options_.fd_step * std::max(1.0, std::abs(baseline[1]));
        if (!std::isfinite(step) || !std::isfinite(baseline[1] + step) ||
            !std::isfinite(baseline[1] - step) ||
            baseline[1] + step == baseline[1] || baseline[1] - step == baseline[1]) {
            check.reason = "finite difference step is not representable";
            return check;
        }
        double plus[2], minus[2];
        // Replacing an intermediate node's value keeps all its parents fixed
        // and recomputes descendants: this checks d(root)/d(output), not just
        // gradients with respect to original input variables.
        probe->evaluate_with_override(inputs, output, baseline[1] + step,
                                     Span<double>(plus, 2), workspace);
        probe->evaluate_with_override(inputs, output, baseline[1] - step,
                                     Span<double>(minus, 2), workspace);
        // [f(v+h)-f(v-h)]/(2h) has O(h^2) truncation error.
        check.numeric = (plus[0] - minus[0]) / (2.0 * step);
    }
    if (!std::isfinite(check.numeric)) {
        check.reason = "non-finite finite difference result";
        return check;
    }
    check.absolute_error = std::abs(check.analytic - check.numeric);
    check.allowed_error = options_.abs_tolerance +
        options_.rel_tolerance * std::max(std::abs(check.analytic), std::abs(check.numeric));
    check.passed = check.absolute_error <= check.allowed_error;
    if (!check.passed) check.reason = "difference exceeds tolerance";
    return check;
}

} // namespace autodiff
