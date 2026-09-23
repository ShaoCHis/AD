#include "autodiff/model.h"

#include <memory>
#include <stdexcept>

namespace {
class CubePlus2X final : public autodiff::CustomOp {
public:
    const char* name() const override { return "cube_plus_2x"; }
    std::size_t arity() const override { return 1; }
    double forward(autodiff::Span<const double> in) const override {
        return in[0] * in[0] * in[0] + 2 * in[0];
    }
    void backward(autodiff::Span<const double> in, double, double dy,
                  autodiff::Span<double> dx) const override {
        dx[0] += dy * (3 * in[0] * in[0] + 2);
    }
    autodiff::Expr symbolic_partial(autodiff::Graph&, autodiff::Span<const autodiff::Expr> in,
                                    autodiff::Expr, std::size_t index) const override {
        if (index != 0) throw std::invalid_argument("cube_plus_2x has one input");
        return 3 * in[0].square() + 2;
    }
};
// Binary plugin used by the assignment-language example:
// func_a(x, y) = x*y + x.
class FuncA final : public autodiff::CustomOp {
public:
    const char* name() const override { return "func_a"; }
    std::size_t arity() const override { return 2; }
    double forward(autodiff::Span<const double> in) const override {
        return in[0] * in[1] + in[0];
    }
    void backward(autodiff::Span<const double> in, double, double dz,
                  autodiff::Span<double> grad) const override {
        grad[0] += dz * (in[1] + 1);
        grad[1] += dz * in[0];
    }
    autodiff::Expr symbolic_partial(autodiff::Graph&, autodiff::Span<const autodiff::Expr> in,
                                    autodiff::Expr, std::size_t index) const override {
        if (index == 0) return in[1] + 1;
        if (index == 1) return in[0];
        throw std::invalid_argument("func_a has two inputs");
    }
};
} // namespace

// Register operators before parsing assignments, so calls such as func_a(x,y) resolve.
extern "C" void register_autodiff_ops(autodiff::CustomOpRegistry& registry) {
    registry.register_op("cube_plus_2x", std::make_shared<CubePlus2X>());
    registry.register_op("func_a", std::make_shared<FuncA>());
}
