#include "autodiff/model.h"

#include <memory>
#include <stdexcept>

namespace {
class CubePlus2X final : public autodiff::CustomOp {
public:
    const char* name() const override { return "cube_plus_2x"; }
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
} // namespace

// The stable symbol name is resolved by PluginManager before reading model.json.
extern "C" void register_autodiff_ops(autodiff::CustomOpRegistry& registry) {
    registry.register_op("cube_plus_2x", std::make_shared<CubePlus2X>());
}
