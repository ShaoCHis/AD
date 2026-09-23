#include "autodiff/model.h"

#include <gflags/gflags.h>

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>

DEFINE_string(expr_file, "", "Assignment expression file (required)");
DEFINE_string(op_plugins, "", "Comma-separated custom-operator .so paths, loaded first");
DEFINE_string(inputs, "", "Variable values, e.g. x=1,y=2");
DEFINE_string(outputs, "", "Value queries, e.g. h,z,t");
DEFINE_string(derivatives, "", "Derivative queries root:output, e.g. t:x,t:h,z:x");
DEFINE_bool(emit_dot, false, "Write a DOT graph for every selected value/derivative");
DEFINE_string(dot_dir, "dot", "Existing or new one-level directory for DOT files");

namespace {
struct SelectedQuery {
    autodiff::EvaluationQuery request;
    std::string label, root_name, output_name;
};
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
std::string safe_name(const std::string& id) {
    std::string result;
    for (unsigned char c : id)
        result += (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                  (c >= '0' && c <= '9') || c == '_' || c == '-' ? static_cast<char>(c) : '_';
    return result;
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
void prepare_dot_dir(const std::string& path) {
    if (path.empty()) throw std::invalid_argument("dot_dir cannot be empty");
    if (mkdir(path.c_str(), 0755) != 0 && errno != EEXIST)
        throw std::runtime_error("cannot create dot_dir: " + path);
    struct stat info;
    if (stat(path.c_str(), &info) != 0 || !S_ISDIR(info.st_mode))
        throw std::runtime_error("dot_dir is not a directory: " + path);
}
std::ofstream output_file(const std::string& path) {
    std::ofstream out(path.c_str());
    if (!out) throw std::runtime_error("cannot write DOT file: " + path);
    return out;
}
} // namespace

int main(int argc, char** argv) {
    try {
        gflags::SetUsageMessage("Static autodiff with assignment expressions (C++11)");
        gflags::ParseCommandLineFlags(&argc, &argv, true);
        if (argc != 1 || FLAGS_expr_file.empty())
            throw std::invalid_argument("usage: autodiff_runner --expr_file=PATH --inputs=x=1,y=2 --outputs=h,t --derivatives=t:x");

        // Keep plugins loaded until all registry/model/evaluator objects die.
        autodiff::PluginManager plugins;
        autodiff::CustomOpRegistry registry;
        for (const std::string& path : split(FLAGS_op_plugins, ',')) plugins.load(path, registry);
        autodiff::ExpressionProgram program(autodiff::read_expression_file(FLAGS_expr_file), registry);
        program.graph().enable_derivative_explanations(FLAGS_emit_dot);

        std::vector<SelectedQuery> selections;
        for (const std::string& name : split(FLAGS_outputs, ',')) {
            autodiff::Expr node = program.at(name);
            selections.push_back(SelectedQuery{autodiff::EvaluationQuery::value(node), name, "", name});
        }
        for (const std::string& item : split(FLAGS_derivatives, ',')) {
            std::size_t colon = item.find(':');
            if (colon == std::string::npos || colon == 0 || colon + 1 == item.size() ||
                item.find(':', colon + 1) != std::string::npos)
                throw std::invalid_argument("expected root:output in --derivatives: " + item);
            std::string root = item.substr(0, colon), output = item.substr(colon + 1);
            selections.push_back(SelectedQuery{autodiff::EvaluationQuery::derivative(
                program.at(root), program.at(output)), "d(" + root + ")/d(" + output + ")",
                root, output});
        }
        if (selections.empty()) throw std::invalid_argument("select --outputs and/or --derivatives");
        std::vector<autodiff::EvaluationQuery> queries;
        for (const SelectedQuery& item : selections) queries.push_back(item.request);
        autodiff::Evaluator evaluator(queries);
        autodiff::Inputs inputs = program.make_inputs(evaluator, parse_inputs(FLAGS_inputs));
        autodiff::Workspace workspace = evaluator.create_workspace();
        std::vector<double> values(selections.size());
        evaluator.evaluate(inputs, values, workspace);
        for (std::size_t i = 0; i < selections.size(); ++i)
            std::cout << selections[i].label << " = " << std::setprecision(15) << values[i] << '\n';

        if (FLAGS_emit_dot) {
            prepare_dot_dir(FLAGS_dot_dir);
            for (std::size_t i = 0; i < selections.size(); ++i) {
                const SelectedQuery& item = selections[i];
                const std::string file = FLAGS_dot_dir + "/query_" + std::to_string(i + 1) +
                    "_" + safe_name(item.label) + ".dot";
                std::ofstream out = output_file(file);
                if (item.request.kind == autodiff::EvaluationQuery::Value)
                    program.graph().dump_dot(item.request.output, out);
                else program.graph().dump_derivative_dot(item.request.root, item.request.output,
                    evaluator.result_expression(i), out, item.root_name, item.output_name);
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
