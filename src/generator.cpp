#include "autodiff/generator.h"

#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <utility>

namespace autodiff {

void CustomOpRegistry::register_op(const std::string& name, std::shared_ptr<const CustomOp> op) {
    if (name.empty() || !op || !ops_.insert(std::make_pair(name, op)).second)
        throw std::invalid_argument("custom op name must be unique and nonempty: " + name);
}
std::shared_ptr<const CustomOp> CustomOpRegistry::find(const std::string& name) const {
    std::map<std::string, std::shared_ptr<const CustomOp>>::const_iterator it = ops_.find(name);
    if (it == ops_.end()) throw std::invalid_argument("unknown function: " + name);
    return it->second;
}
PluginManager::~PluginManager() {
    for (std::vector<void*>::reverse_iterator it = handles_.rbegin(); it != handles_.rend(); ++it)
        dlclose(*it);
}
void PluginManager::load(const std::string& path, CustomOpRegistry& registry) {
    if (path.empty()) throw std::invalid_argument("empty plugin path");
    void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) throw std::runtime_error("cannot load plugin " + path + ": " + dlerror());
    dlerror();
    typedef void (*RegisterFunction)(CustomOpRegistry&);
    RegisterFunction entry = reinterpret_cast<RegisterFunction>(dlsym(handle, "register_autodiff_ops"));
    const char* symbol_error = dlerror();
    if (symbol_error) {
        std::string message = "plugin " + path + " has no register_autodiff_ops: " + symbol_error;
        dlclose(handle);
        throw std::runtime_error(message);
    }
    handles_.push_back(handle); // outlive any partially registered instances
    entry(registry);
}

namespace {
bool initial(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}
bool continuation(char c) { return initial(c) || (c >= '0' && c <= '9'); }
std::string trim(const std::string& s) {
    std::size_t begin = s.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) return "";
    std::size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(begin, end - begin + 1);
}
} // namespace

class ExpressionProgram::ExpressionParser {
public:
    ExpressionParser(const std::string& source, std::size_t line)
        : source_(source), line_(line), pos_(0), depth_(0) {}
    Syntax parse() {
        Syntax root = additive();
        spaces();
        if (pos_ != source_.size()) error("unexpected token");
        return root;
    }
private:
    const std::string& source_;
    std::size_t line_, pos_;
    int depth_;
    void spaces() {
        while (pos_ < source_.size() && (source_[pos_] == ' ' || source_[pos_] == '\t')) ++pos_;
    }
    bool take(char c) {
        spaces();
        if (pos_ < source_.size() && source_[pos_] == c) { ++pos_; return true; }
        return false;
    }
    void error(const std::string& message) const {
        throw std::invalid_argument("line " + std::to_string(line_) + ", column " +
                                    std::to_string(pos_ + 1) + ": " + message);
    }
    Syntax binary(SyntaxNode::Kind kind, Syntax lhs, Syntax rhs) {
        Syntax n(new SyntaxNode(kind));
        n->args.push_back(lhs);
        n->args.push_back(rhs);
        return n;
    }
    Syntax additive() {
        Syntax lhs = multiplicative();
        while (true) {
            if (take('+')) lhs = binary(SyntaxNode::Add, lhs, multiplicative());
            else if (take('-')) lhs = binary(SyntaxNode::Sub, lhs, multiplicative());
            else return lhs;
        }
    }
    Syntax multiplicative() {
        Syntax lhs = unary();
        while (true) {
            if (take('*')) lhs = binary(SyntaxNode::Mul, lhs, unary());
            else if (take('/')) lhs = binary(SyntaxNode::Div, lhs, unary());
            else return lhs;
        }
    }
    Syntax unary() {
        if (++depth_ > 128) error("expression is nested too deeply");
        if (take('+')) { Syntax value = unary(); --depth_; return value; }
        if (take('-')) {
            Syntax n(new SyntaxNode(SyntaxNode::UnaryMinus));
            n->args.push_back(unary());
            --depth_;
            return n;
        }
        Syntax value = primary();
        --depth_;
        return value;
    }
    Syntax primary() {
        spaces();
        if (++depth_ > 128) error("expression is nested too deeply");
        Syntax result;
        if (take('(')) {
            result = additive();
            if (!take(')')) error("expected ')'");
        } else if (pos_ < source_.size() && initial(source_[pos_])) {
            std::size_t start = pos_++;
            while (pos_ < source_.size() && continuation(source_[pos_])) ++pos_;
            std::string name = source_.substr(start, pos_ - start);
            if (take('(')) {
                result.reset(new SyntaxNode(SyntaxNode::Call));
                result->name = name;
                if (!take(')')) {
                    do { result->args.push_back(additive()); } while (take(','));
                    if (!take(')')) error("expected ')' after function arguments");
                }
            } else {
                result.reset(new SyntaxNode(SyntaxNode::Name));
                result->name = name;
            }
        } else if (pos_ < source_.size() &&
                   (std::isdigit(static_cast<unsigned char>(source_[pos_])) || source_[pos_] == '.')) {
            std::size_t start = pos_;
            if (source_[pos_] == '.') {
                ++pos_;
                if (pos_ >= source_.size() || !std::isdigit(static_cast<unsigned char>(source_[pos_])))
                    error("expected a digit after '.'");
            }
            while (pos_ < source_.size() && std::isdigit(static_cast<unsigned char>(source_[pos_]))) ++pos_;
            if (pos_ < source_.size() && source_[pos_] == '.') {
                ++pos_;
                while (pos_ < source_.size() && std::isdigit(static_cast<unsigned char>(source_[pos_]))) ++pos_;
            }
            if (pos_ < source_.size() && (source_[pos_] == 'e' || source_[pos_] == 'E')) {
                ++pos_;
                if (pos_ < source_.size() && (source_[pos_] == '+' || source_[pos_] == '-')) ++pos_;
                if (pos_ >= source_.size() || !std::isdigit(static_cast<unsigned char>(source_[pos_])))
                    error("expected exponent digits");
                while (pos_ < source_.size() && std::isdigit(static_cast<unsigned char>(source_[pos_]))) ++pos_;
            }
            std::string token = source_.substr(start, pos_ - start);
            errno = 0;
            double value = std::strtod(token.c_str(), nullptr);
            if (errno == ERANGE || !std::isfinite(value)) error("number outside finite double range");
            result.reset(new SyntaxNode(SyntaxNode::Literal));
            result->literal = value;
        } else error("expected number, identifier, or '('");
        --depth_;
        return result;
    }
};

ExpressionProgram::ExpressionProgram(const std::string& source, const CustomOpRegistry& ops)
    : ops_(ops) {
    std::size_t line_number = 0, offset = 0;
    while (offset < source.size()) {
        std::size_t end = source.find('\n', offset);
        if (end == std::string::npos) end = source.size();
        std::string line = source.substr(offset, end - offset);
        ++line_number;
        std::size_t comment = line.find('#');
        if (comment != std::string::npos) line.erase(comment);
        line = trim(line);
        if (!line.empty()) {
            std::size_t pos = 0;
            if (!initial(line[pos]))
                throw std::invalid_argument("line " + std::to_string(line_number) + ": expected assignment name");
            while (pos < line.size() && continuation(line[pos])) ++pos;
            std::string name = line.substr(0, pos);
            std::string rest = trim(line.substr(pos));
            if (rest.empty() || rest[0] != '=')
                throw std::invalid_argument("line " + std::to_string(line_number) + ": expected '=' after " + name);
            if (name == "exp" || name == "abs" || name == "square")
                throw std::invalid_argument("reserved function name cannot be assigned: " + name);
            if (definitions_.count(name)) throw std::invalid_argument("duplicate assignment: " + name);
            definitions_.insert(std::make_pair(name, ExpressionParser(rest.substr(1), line_number).parse()));
        }
        offset = end + 1;
    }
    if (definitions_.empty()) throw std::invalid_argument("expression file contains no assignments");
    // Resolve every definition after parsing all lines: forward references work.
    for (const auto& item : definitions_) resolve(item.first, 0);
}
Expr ExpressionProgram::resolve(const std::string& name, int depth) {
    if (depth > 512) throw std::invalid_argument("definition chain exceeds 512 levels at " + name);
    std::map<std::string, Expr>::const_iterator existing = symbols_.find(name);
    if (existing != symbols_.end()) return existing->second;
    std::map<std::string, Syntax>::const_iterator definition = definitions_.find(name);
    if (definition == definitions_.end()) {
        // A name used on a right-hand side but never assigned is a variable.
        Expr variable = graph_.variable(name);
        symbols_.insert(std::make_pair(name, variable));
        variables_.insert(name);
        return variable;
    }
    if (states_[name] == 1) throw std::invalid_argument("cyclic definition: " + name);
    states_[name] = 1;
    Expr expr = build(definition->second, depth + 1).named(name);
    states_[name] = 2;
    symbols_.insert(std::make_pair(name, expr));
    return expr;
}
Expr ExpressionProgram::build(const Syntax& node, int depth) {
    if (depth > 512) throw std::invalid_argument("expression tree exceeds 512 levels");
    switch (node->kind) {
    case SyntaxNode::Literal: return graph_.constant(node->literal);
    case SyntaxNode::Name: return resolve(node->name, depth + 1);
    case SyntaxNode::UnaryMinus: return -build(node->args[0], depth + 1);
    case SyntaxNode::Add: return build(node->args[0], depth + 1) + build(node->args[1], depth + 1);
    case SyntaxNode::Sub: return build(node->args[0], depth + 1) - build(node->args[1], depth + 1);
    case SyntaxNode::Mul: return build(node->args[0], depth + 1) * build(node->args[1], depth + 1);
    case SyntaxNode::Div: return build(node->args[0], depth + 1) / build(node->args[1], depth + 1);
    case SyntaxNode::Call: {
        std::vector<Expr> args;
        for (const Syntax& child : node->args) args.push_back(build(child, depth + 1));
        if (node->name == "exp" || node->name == "abs" || node->name == "square") {
            if (args.size() != 1) throw std::invalid_argument(node->name + " expects one argument");
            if (node->name == "exp") return exp(args[0]);
            if (node->name == "abs") return args[0].abs();
            return args[0].square();
        }
        std::shared_ptr<const CustomOp> op = ops_.find(node->name);
        if (args.size() != op->arity())
            throw std::invalid_argument("function " + node->name + " expects " +
                                        std::to_string(op->arity()) + " argument(s), got " +
                                        std::to_string(args.size()));
        return graph_.custom(op, args);
    }
    }
    throw std::invalid_argument("invalid expression syntax");
}
Expr ExpressionProgram::at(const std::string& name) const {
    std::map<std::string, Expr>::const_iterator it = symbols_.find(name);
    if (it == symbols_.end()) throw std::invalid_argument("unknown graph symbol: " + name);
    return it->second;
}
Inputs ExpressionProgram::make_inputs(const Evaluator& evaluator,
                                      const std::map<std::string, double>& values) const {
    Inputs input = evaluator.create_inputs();
    for (const auto& item : values) {
        if (!is_variable(item.first)) throw std::invalid_argument("input is not a variable: " + item.first);
        Expr variable = at(item.first);
        if (evaluator.uses_variable(variable)) input.set(variable, item.second);
    }
    return input;
}
std::string read_expression_file(const std::string& path) {
    std::ifstream in(path.c_str(), std::ios::binary);
    if (!in) throw std::invalid_argument("cannot open expression file: " + path);
    in.seekg(0, std::ios::end);
    std::streampos size = in.tellg();
    if (size < 0 || size > static_cast<std::streampos>(16 * 1024 * 1024))
        throw std::invalid_argument("expression file must be at most 16 MiB: " + path);
    in.seekg(0, std::ios::beg);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

namespace {
// The config intentionally accepts only a small JSON schema: two string arrays.
// Expressions remain ordinary assignments rather than a verbose node tree.
class ConfigJsonReader {
public:
    explicit ConfigJsonReader(const std::string& source) : text_(source) {}
    GeneratorConfig parse() {
        GeneratorConfig config;
        expect('{');
        bool expressions_seen = false, plugins_seen = false;
        if (!take('}')) {
            do {
                std::string key = read_string();
                expect(':');
                if (key == "expressions") {
                    if (expressions_seen) error("duplicate expressions key");
                    expressions_seen = true;
                    config.expressions = read_array();
                } else if (key == "op_plugins") {
                    if (plugins_seen) error("duplicate op_plugins key");
                    plugins_seen = true;
                    config.op_plugins = read_array();
                } else error("unknown config key: " + key);
            } while (take(','));
            expect('}');
        }
        space();
        if (pos_ != text_.size()) error("trailing JSON content");
        if (!expressions_seen || config.expressions.empty()) error("expressions must be a nonempty array");
        for (const std::string& expression : config.expressions)
            if (trim(expression).empty()) error("empty assignment");
        for (const std::string& plugin : config.op_plugins)
            if (plugin.empty()) error("empty plugin path");
        return config;
    }
private:
    const std::string& text_;
    std::size_t pos_ = 0;
    void space() {
        while (pos_ < text_.size() && (text_[pos_] == ' ' || text_[pos_] == '\t' ||
               text_[pos_] == '\n' || text_[pos_] == '\r')) ++pos_;
    }
    void error(const std::string& reason) const {
        throw std::invalid_argument("invalid generator JSON at byte " +
                                    std::to_string(pos_) + ": " + reason);
    }
    bool take(char character) {
        space();
        if (pos_ < text_.size() && text_[pos_] == character) { ++pos_; return true; }
        return false;
    }
    void expect(char character) {
        if (!take(character)) error(std::string("expected '") + character + "'");
    }
    // The JSON strings used here are source code or plugin paths. Decode all
    // standard escapes; reject malformed Unicode instead of silently changing it.
    unsigned hex4() {
        if (text_.size() - pos_ < 4) error("incomplete Unicode escape");
        unsigned value = 0;
        for (int i = 0; i < 4; ++i) {
            char c = text_[pos_++];
            unsigned digit = c >= '0' && c <= '9' ? c - '0' :
                             c >= 'a' && c <= 'f' ? c - 'a' + 10 :
                             c >= 'A' && c <= 'F' ? c - 'A' + 10 : 16;
            if (digit > 15) error("invalid Unicode escape");
            value = (value << 4) | digit;
        }
        return value;
    }
    void append_utf8(std::string& out, unsigned codepoint) {
        if (codepoint < 0x80) out += static_cast<char>(codepoint);
        else if (codepoint < 0x800) {
            out += static_cast<char>(0xc0 | (codepoint >> 6));
            out += static_cast<char>(0x80 | (codepoint & 0x3f));
        } else if (codepoint < 0x10000) {
            out += static_cast<char>(0xe0 | (codepoint >> 12));
            out += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f));
            out += static_cast<char>(0x80 | (codepoint & 0x3f));
        } else {
            out += static_cast<char>(0xf0 | (codepoint >> 18));
            out += static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f));
            out += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f));
            out += static_cast<char>(0x80 | (codepoint & 0x3f));
        }
    }
    std::string read_string() {
        expect('"');
        std::string result;
        while (pos_ < text_.size()) {
            unsigned char c = static_cast<unsigned char>(text_[pos_++]);
            if (c == '"') return result;
            if (c < 0x20) error("control character in string");
            if (c != '\\') { result += static_cast<char>(c); continue; }
            if (pos_ >= text_.size()) error("unfinished escape");
            char escaped = text_[pos_++];
            switch (escaped) {
            case '"': case '\\': case '/': result += escaped; break;
            case 'b': result += '\b'; break;
            case 'f': result += '\f'; break;
            case 'n': result += '\n'; break;
            case 'r': result += '\r'; break;
            case 't': result += '\t'; break;
            case 'u': {
                unsigned cp = hex4();
                if (cp >= 0xd800 && cp <= 0xdbff) {
                    if (text_.size() - pos_ < 6 || text_[pos_] != '\\' || text_[pos_ + 1] != 'u')
                        error("missing low surrogate");
                    pos_ += 2;
                    unsigned low = hex4();
                    if (low < 0xdc00 || low > 0xdfff) error("invalid low surrogate");
                    cp = 0x10000 + ((cp - 0xd800) << 10) + (low - 0xdc00);
                } else if (cp >= 0xdc00 && cp <= 0xdfff) error("unpaired low surrogate");
                append_utf8(result, cp);
                break;
            }
            default: error("unknown string escape");
            }
        }
        error("unterminated string");
        return result;
    }
    std::vector<std::string> read_array() {
        std::vector<std::string> result;
        expect('[');
        if (!take(']')) {
            do { result.push_back(read_string()); } while (take(','));
            expect(']');
        }
        return result;
    }
};
} // namespace

GeneratorConfig read_generator_json(const std::string& path) {
    return ConfigJsonReader(read_expression_file(path)).parse();
}

GraphGenerator::GraphGenerator(const GeneratorConfig& config) {
    if (config.expressions.empty()) throw std::invalid_argument("generator needs assignments");
    for (const std::string& path : config.op_plugins) plugins_.load(path, registry_);
    std::string source;
    for (const std::string& assignment : config.expressions) {
        source += assignment;
        source += '\n';
    }
    program_.reset(new ExpressionProgram(source, registry_));
}

} // namespace autodiff
