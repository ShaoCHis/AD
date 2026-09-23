#include "autodiff/json.h"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace autodiff {
namespace {
class Parser {
public:
    explicit Parser(const std::string& text) : text_(text), pos_(0) {}
    Json parse() {
        Json value = parse_value(0);
        whitespace();
        if (pos_ != text_.size()) error("trailing characters");
        return value;
    }
private:
    const std::string& text_;
    std::size_t pos_;
    void error(const char* what) const {
        throw std::invalid_argument(std::string("JSON byte ") + std::to_string(pos_) + ": " + what);
    }
    void whitespace() {
        while (pos_ < text_.size() && (text_[pos_] == ' ' || text_[pos_] == '\n' ||
               text_[pos_] == '\r' || text_[pos_] == '\t')) ++pos_;
    }
    bool take(char c) {
        whitespace();
        if (pos_ < text_.size() && text_[pos_] == c) { ++pos_; return true; }
        return false;
    }
    void expect(char c) { if (!take(c)) error("unexpected character"); }
    bool digit() const { return pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9'; }
    void literal(const char* word) {
        for (const char* p = word; *p; ++p) {
            if (pos_ >= text_.size() || text_[pos_++] != *p) error("invalid literal");
        }
    }
    static void utf8(std::string& out, unsigned code) {
        if (code <= 0x7f) out += static_cast<char>(code);
        else if (code <= 0x7ff) {
            out += static_cast<char>(0xc0 | (code >> 6));
            out += static_cast<char>(0x80 | (code & 0x3f));
        } else if (code <= 0xffff) {
            out += static_cast<char>(0xe0 | (code >> 12));
            out += static_cast<char>(0x80 | ((code >> 6) & 0x3f));
            out += static_cast<char>(0x80 | (code & 0x3f));
        } else {
            out += static_cast<char>(0xf0 | (code >> 18));
            out += static_cast<char>(0x80 | ((code >> 12) & 0x3f));
            out += static_cast<char>(0x80 | ((code >> 6) & 0x3f));
            out += static_cast<char>(0x80 | (code & 0x3f));
        }
    }
    unsigned hex4() {
        if (text_.size() - pos_ < 4) error("short unicode escape");
        unsigned code = 0;
        for (int i = 0; i < 4; ++i) {
            char c = text_[pos_++];
            unsigned n;
            if (c >= '0' && c <= '9') n = c - '0';
            else if (c >= 'a' && c <= 'f') n = c - 'a' + 10;
            else if (c >= 'A' && c <= 'F') n = c - 'A' + 10;
            else error("invalid unicode escape");
            code = code * 16 + n;
        }
        return code;
    }
    std::string string() {
        expect('"');
        std::string value;
        while (pos_ < text_.size()) {
            unsigned char c = static_cast<unsigned char>(text_[pos_++]);
            if (c == '"') return value;
            if (c < 0x20) error("control character in string");
            if (c != '\\') { value += static_cast<char>(c); continue; }
            if (pos_ >= text_.size()) error("unfinished escape");
            char e = text_[pos_++];
            switch (e) {
            case '"': value += '"'; break; case '\\': value += '\\'; break;
            case '/': value += '/'; break; case 'b': value += '\b'; break;
            case 'f': value += '\f'; break; case 'n': value += '\n'; break;
            case 'r': value += '\r'; break; case 't': value += '\t'; break;
            case 'u': {
                unsigned code = hex4();
                if (code >= 0xd800 && code <= 0xdbff) {
                    if (pos_ + 2 > text_.size() || text_.compare(pos_, 2, "\\u") != 0)
                        error("missing low surrogate");
                    pos_ += 2;
                    unsigned low = hex4();
                    if (low < 0xdc00 || low > 0xdfff) error("invalid low surrogate");
                    code = 0x10000 + ((code - 0xd800) << 10) + low - 0xdc00;
                } else if (code >= 0xdc00 && code <= 0xdfff) error("unexpected low surrogate");
                utf8(value, code);
                break;
            }
            default: error("invalid escape");
            }
        }
        error("unterminated string");
        return value;
    }
    double number() {
        std::size_t begin = pos_;
        if (text_[pos_] == '-') ++pos_;
        if (pos_ >= text_.size()) error("incomplete number");
        if (text_[pos_] == '0') ++pos_;
        else {
            if (text_[pos_] < '1' || text_[pos_] > '9') error("invalid number");
            while (digit()) ++pos_;
        }
        if (pos_ < text_.size() && text_[pos_] == '.') {
            ++pos_;
            if (!digit()) error("missing fraction digits");
            while (digit()) ++pos_;
        }
        if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
            ++pos_;
            if (pos_ < text_.size() && (text_[pos_] == '-' || text_[pos_] == '+')) ++pos_;
            if (!digit()) error("missing exponent digits");
            while (digit()) ++pos_;
        }
        std::string digits = text_.substr(begin, pos_ - begin);
        errno = 0;
        double value = std::strtod(digits.c_str(), nullptr);
        if (!std::isfinite(value) || errno == ERANGE) error("number outside finite double range");
        return value;
    }
    Json parse_value(int depth) {
        if (depth > 128) error("nesting exceeds 128 levels");
        whitespace();
        if (pos_ >= text_.size()) error("unexpected end of input");
        char c = text_[pos_];
        if (c == '"') return Json(string());
        if (c == '-' || (c >= '0' && c <= '9')) return Json(number());
        if (c == 't') { literal("true"); return Json(true); }
        if (c == 'f') { literal("false"); return Json(false); }
        if (c == 'n') { literal("null"); return Json(); }
        if (c == '[') {
            ++pos_;
            std::vector<Json> values;
            if (take(']')) return Json(std::move(values));
            do { values.push_back(parse_value(depth + 1)); } while (take(','));
            expect(']');
            return Json(std::move(values));
        }
        if (c == '{') {
            ++pos_;
            std::map<std::string, Json> values;
            if (take('}')) return Json(std::move(values));
            do {
                whitespace();
                if (pos_ >= text_.size() || text_[pos_] != '"') error("object key must be a string");
                std::string key = string();
                expect(':');
                Json value = parse_value(depth + 1);
                if (!values.insert(std::make_pair(key, std::move(value))).second)
                    error("duplicate object key");
            } while (take(','));
            expect('}');
            return Json(std::move(values));
        }
        error("expected JSON value");
        return Json();
    }
};
} // namespace

Json Json::parse(const std::string& text) { return Parser(text).parse(); }
Json Json::parse_file(const std::string& path) {
    std::ifstream in(path.c_str(), std::ios::binary);
    if (!in) throw std::invalid_argument("cannot open JSON file: " + path);
    in.seekg(0, std::ios::end);
    std::streampos length = in.tellg();
    if (length < 0) throw std::invalid_argument("cannot determine JSON file size: " + path);
    if (length > static_cast<std::streampos>(16 * 1024 * 1024))
        throw std::invalid_argument("JSON file exceeds 16 MiB");
    in.seekg(0, std::ios::beg);
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (!in.eof() && in.fail()) throw std::invalid_argument("failed reading JSON file: " + path);
    return parse(content);
}
bool Json::has(const std::string& key) const {
    return as_object().find(key) != object_.end();
}
const Json& Json::at(const std::string& key) const {
    std::map<std::string, Json>::const_iterator it = as_object().find(key);
    if (it == object_.end()) throw std::invalid_argument("missing JSON field: " + key);
    return it->second;
}
const Json& Json::at(std::size_t index) const {
    if (index >= as_array().size()) throw std::invalid_argument("JSON array index out of range");
    return array_[index];
}
const std::string& Json::as_string() const {
    if (type_ != String) throw std::invalid_argument("expected JSON string");
    return string_;
}
double Json::as_number() const {
    if (type_ != Number) throw std::invalid_argument("expected JSON number");
    return number_;
}
bool Json::as_bool() const {
    if (type_ != Boolean) throw std::invalid_argument("expected JSON boolean");
    return boolean_;
}
const std::vector<Json>& Json::as_array() const {
    if (type_ != Array) throw std::invalid_argument("expected JSON array");
    return array_;
}
const std::map<std::string, Json>& Json::as_object() const {
    if (type_ != Object) throw std::invalid_argument("expected JSON object");
    return object_;
}

} // namespace autodiff
