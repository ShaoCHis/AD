#include "post_evaluator_flags.h"

#include <gflags/gflags.h>

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

DEFINE_string(config_json, "", "Generator JSON with expressions and optional op_plugins");
DEFINE_string(expr_file, "", "Assignment file (alternative to config_json)");
DEFINE_string(op_plugins, "", "Additional comma-separated custom-operator .so paths");
DEFINE_string(inputs, "", "Runtime variable values, e.g. x=1,y=2");
DEFINE_string(outputs, "", "Value queries, e.g. h,z,t");
DEFINE_string(derivatives, "", "Derivative queries root:output, e.g. t:x,t:h,z:x");

namespace {
std::vector<std::string> split(const std::string& text, char separator) {
    std::vector<std::string> parts;
    if (text.empty()) return parts;
    std::size_t begin = 0;
    while (begin < text.size()) {
        std::size_t end = text.find(separator, begin);
        if (end == std::string::npos) end = text.size();
        std::string value = text.substr(begin, end - begin);
        std::size_t first = value.find_first_not_of(" \t");
        std::size_t last = value.find_last_not_of(" \t");
        if (first == std::string::npos) throw std::invalid_argument("empty query or plugin path");
        parts.push_back(value.substr(first, last - first + 1));
        begin = end + 1;
    }
    if (text.back() == separator) throw std::invalid_argument("trailing separator in flag");
    return parts;
}
std::map<std::string, double> parse_inputs(const std::string& text) {
    std::map<std::string, double> result;
    for (const std::string& part : split(text, ',')) {
        std::size_t eq = part.find('=');
        if (eq == std::string::npos || eq == 0 || eq + 1 == part.size())
            throw std::invalid_argument("expected name=value in --inputs: " + part);
        std::string name = part.substr(0, eq);
        std::string number = part.substr(eq + 1);
        char* end = nullptr;
        errno = 0;
        double value = std::strtod(number.c_str(), &end);
        if (end == number.c_str() || *end != '\0' || errno == ERANGE || !std::isfinite(value))
            throw std::invalid_argument("invalid input value: " + part);
        if (!result.insert(std::make_pair(name, value)).second)
            throw std::invalid_argument("duplicate input name: " + name);
    }
    return result;
}
} // namespace

int main(int argc, char** argv) {
    try {
        gflags::SetUsageMessage("Static graph generator, evaluator, post-evaluator (C++11)");
        gflags::ParseCommandLineFlags(&argc, &argv, true);
        if (argc != 1 || FLAGS_config_json.empty() == FLAGS_expr_file.empty())
            throw std::invalid_argument("select exactly one of --config_json=PATH or --expr_file=PATH");

        // Phase 1: load plugins, parse assignments and keep the name -> node map.
        autodiff::GeneratorConfig config;
        if (!FLAGS_config_json.empty()) config = autodiff::read_generator_json(FLAGS_config_json);
        else config.expressions.push_back(autodiff::read_expression_file(FLAGS_expr_file));
        std::vector<std::string> extra = split(FLAGS_op_plugins, ',');
        config.op_plugins.insert(config.op_plugins.end(), extra.begin(), extra.end());
        autodiff::GraphGenerator& generator = autodiff::GraphGenerator::initialize(config);
        const autodiff::ExpressionProgram& program = generator.program();

        autodiff::PostEvaluator post(autodiff::post_options_from_flags());

        // Phase 2: every query selects its own node(s); there is no target.
        std::vector<autodiff::NamedQuery> selections;
        for (const std::string& name : split(FLAGS_outputs, ','))
            selections.push_back(autodiff::NamedQuery{
                autodiff::EvaluationQuery::value(program.at(name)), name, "", name});
        for (const std::string& item : split(FLAGS_derivatives, ',')) {
            std::size_t colon = item.find(':');
            if (colon == std::string::npos || colon == 0 || colon + 1 == item.size() ||
                item.find(':', colon + 1) != std::string::npos)
                throw std::invalid_argument("expected root:output in --derivatives: " + item);
            std::string root = item.substr(0, colon), output = item.substr(colon + 1);
            selections.push_back(autodiff::NamedQuery{
                autodiff::EvaluationQuery::derivative(program.at(root), program.at(output)),
                "d(" + root + ")/d(" + output + ")", root, output});
        }
        if (selections.empty()) throw std::invalid_argument("select --outputs and/or --derivatives");
        std::vector<autodiff::EvaluationQuery> queries;
        for (const autodiff::NamedQuery& item : selections) queries.push_back(item.request);
        std::shared_ptr<const autodiff::Evaluator> evaluator = generator.compile(queries, post.emits_dot());
        std::map<std::string, double> values = parse_inputs(FLAGS_inputs);
        autodiff::Inputs inputs = program.make_inputs(*evaluator, values);
        autodiff::Workspace workspace = evaluator->create_workspace();
        std::vector<double> results(selections.size());
        evaluator->evaluate(inputs, results, workspace);
        for (std::size_t i = 0; i < selections.size(); ++i)
            std::cout << selections[i].label << " = " << std::setprecision(15) << results[i] << '\n';

        // Phase 3: DOT and numerical checks are independent optional actions.
        std::vector<autodiff::DerivativeCheck> checks = post.run(generator, *evaluator,
            selections, values, results);
        bool failed = false;
        for (const autodiff::DerivativeCheck& item : checks) {
            std::cout << "check " << item.label << ": " << (item.passed ? "PASS" : "FAIL")
                      << " analytic=" << item.analytic << " numeric=" << item.numeric
                      << " error=" << item.absolute_error << " tolerance=" << item.allowed_error;
            if (!item.reason.empty()) std::cout << " (" << item.reason << ')';
            std::cout << '\n';
            failed |= !item.passed;
        }
        return failed ? 2 : 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
