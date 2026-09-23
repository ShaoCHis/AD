#include "autodiff/model.h"

#include <gflags/gflags.h>

#include <cerrno>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>

DEFINE_string(graph_json, "", "Path to the JSON model (required)");
DEFINE_string(op_plugins, "", "Comma-separated custom-operator .so paths; loaded before JSON");
DEFINE_bool(emit_dot, false, "Write target.dot and a DOT explanation for each derivative");
DEFINE_string(dot_dir, "dot", "Existing or new one-level directory for DOT files");

namespace {
std::string safe_name(const std::string& id) {
    std::string result;
    for (unsigned char c : id)
        result += std::isalnum(c) || c == '_' || c == '-' ? static_cast<char>(c) : '_';
    return result;
}
std::ofstream output_file(const std::string& path) {
    std::ofstream out(path.c_str());
    if (!out) throw std::runtime_error("cannot write DOT file: " + path);
    return out;
}
void prepare_dot_dir(const std::string& path) {
    if (path.empty()) throw std::invalid_argument("dot_dir cannot be empty");
    if (mkdir(path.c_str(), 0755) != 0 && errno != EEXIST)
        throw std::runtime_error("cannot create dot_dir: " + path);
    struct stat info;
    if (stat(path.c_str(), &info) != 0 || !S_ISDIR(info.st_mode))
        throw std::runtime_error("dot_dir is not a directory: " + path);
}
void load_plugins(const std::string& paths, autodiff::PluginManager& manager,
                  autodiff::CustomOpRegistry& registry) {
    if (!paths.empty() && paths.back() == ',')
        throw std::invalid_argument("trailing comma in op_plugins");
    std::size_t begin = 0;
    while (begin < paths.size()) {
        std::size_t end = paths.find(',', begin);
        if (end == std::string::npos) end = paths.size();
        if (end == begin) throw std::invalid_argument("empty path in op_plugins");
        manager.load(paths.substr(begin, end - begin), registry);
        begin = end + 1;
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        gflags::SetUsageMessage("JSON static autodiff runner (C++11)");
        gflags::ParseCommandLineFlags(&argc, &argv, true);
        if (argc != 1 || FLAGS_graph_json.empty())
            throw std::invalid_argument("usage: autodiff_runner --graph_json=PATH [--op_plugins=PATH] [--emit_dot]");

        // Declaration order matters: model/registry release plugin objects
        // before PluginManager unloads the shared libraries.
        autodiff::PluginManager plugin_manager;
        autodiff::CustomOpRegistry registry;
        load_plugins(FLAGS_op_plugins, plugin_manager, registry);
        autodiff::LoadedModel model(autodiff::Json::parse_file(FLAGS_graph_json),
                                    registry, FLAGS_emit_dot);

        std::vector<autodiff::Expr> roots;
        for (const autodiff::NamedExpression& output : model.outputs()) roots.push_back(output.expr);
        autodiff::Evaluator evaluator(roots);
        autodiff::Inputs inputs = model.make_inputs(evaluator);
        autodiff::Workspace workspace = evaluator.create_workspace();
        std::vector<double> result(roots.size());
        evaluator.evaluate(inputs, result, workspace);
        for (std::size_t i = 0; i < result.size(); ++i)
            std::cout << model.outputs()[i].id << " = " << std::setprecision(15) << result[i] << '\n';

        if (FLAGS_emit_dot) {
            prepare_dot_dir(FLAGS_dot_dir);
            {
                std::ofstream out = output_file(FLAGS_dot_dir + "/target.dot");
                model.graph().dump_dot(model.target().expr, out);
            }
            const std::vector<autodiff::DerivativeRequest>& derivatives = model.derivatives();
            for (std::size_t i = 0; i < derivatives.size(); ++i) {
                const autodiff::DerivativeRequest& d = derivatives[i];
                std::string file = FLAGS_dot_dir + "/derivative_" + std::to_string(i + 1) +
                                   "_" + safe_name(d.id) + ".dot";
                std::ofstream out = output_file(file);
                model.graph().dump_derivative_dot(d.source, d.wrt, d.result, out,
                                                  d.source_name, d.wrt_name);
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
