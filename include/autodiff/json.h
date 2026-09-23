#pragma once

#include <cstddef>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace autodiff {

// A small C++11 JSON value/parser used only at graph construction time.
// It rejects duplicate keys, malformed escapes and trailing input.
class Json {
public:
    enum Type { Null, Boolean, Number, String, Array, Object };
    Json() : type_(Null), number_(0), boolean_(false) {}
    explicit Json(bool value) : type_(Boolean), number_(0), boolean_(value) {}
    explicit Json(double value) : type_(Number), number_(value), boolean_(false) {}
    explicit Json(std::string value) : type_(String), number_(0), boolean_(false), string_(std::move(value)) {}
    explicit Json(std::vector<Json> value) : type_(Array), number_(0), boolean_(false), array_(std::move(value)) {}
    explicit Json(std::map<std::string, Json> value)
        : type_(Object), number_(0), boolean_(false), object_(std::move(value)) {}

    static Json parse(const std::string& text);
    static Json parse_file(const std::string& path);
    Type type() const { return type_; }
    bool has(const std::string& key) const;
    const Json& at(const std::string& key) const;
    const Json& at(std::size_t index) const;
    const std::string& as_string() const;
    double as_number() const;
    bool as_bool() const;
    const std::vector<Json>& as_array() const;
    const std::map<std::string, Json>& as_object() const;
private:
    Type type_;
    double number_;
    bool boolean_;
    std::string string_;
    std::vector<Json> array_;
    std::map<std::string, Json> object_;
};

} // namespace autodiff
