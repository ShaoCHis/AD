#include "autodiff/autodiff.h"

#include <array>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>

using namespace autodiff;

class CubePlus2X final : public CustomOp {
public:
    const char* name() const override { return "CubePlus2X"; }
    // 前向定义 f(x) = x³ + 2x。
    double forward(Span<const double> in) const override {
        double x = in[0];
        return x * x * x + 2 * x;
    }
    void backward(Span<const double> in, double, double dy,
                  Span<double> dx) const override {
        // 数值反传：dx += dy * (3x² + 2)。
        dx[0] += dy * (3 * in[0] * in[0] + 2);
    }
    Expr symbolic_partial(Graph&, Span<const Expr> in,
                          Expr, std::size_t i) const override {
        if (i != 0) throw std::invalid_argument("invalid input index");
        // 返回 Expr 而不是 double，使自定义算子也可以继续求二阶导。
        return 3 * in[0].square() + 2;
    }
};

int main() {
    Graph graph;
    const auto p = graph.variable("p");
    const auto x = graph.variable("x");
    const auto y = graph.variable("y");
    const auto pred = (exp(x * p) + 0.5 * x).named("pred");
    const auto loss = (pred - y).pow(2).named("loss");

    const auto derivatives = loss.derivative({p, pred});
    const auto grad_p = derivatives[0];
    const auto grad_pred = derivatives[1];
    const auto grad_2nd_p = grad_p.derivative(p);
    Evaluator evaluator({loss, grad_p, grad_pred, grad_2nd_p});

    // 快捷接口与题目图片一致；每次调用会临时创建输入和工作区。
    auto result = evaluator.evaluate({{"x", 1.0}, {"y", -0.05}, {"p", 0.5}});
    std::cout << "loss=" << result[0] << ", dloss/dp=" << result[1]
              << ", dloss/dpred=" << result[2] << ", d2loss/dp2=" << result[3] << '\n';

    // 高频调用只初始化一次；循环内更新输入值，不重复分配工作区。
    auto inputs = evaluator.create_inputs();
    auto workspace = evaluator.create_workspace();
    std::array<double, 4> outputs{};
    inputs.set(x, 1.0);
    inputs.set(y, -0.05);
    for (double param : {0.5, 0.6, 0.7}) {
        inputs.set(p, param);
        evaluator.evaluate(inputs, outputs, workspace);
        std::cout << "p=" << param << ", loss=" << outputs[0] << '\n';
    }

    std::array<Expr, 2> wrts{p, pred};
    std::array<double, 2> numeric_grads{};
    evaluator.backward(0, inputs, wrts, numeric_grads, workspace);
    std::cout << "direct backward: dp=" << numeric_grads[0]
              << ", dpred=" << numeric_grads[1] << '\n';

    std::ofstream graph_file("loss.dot");
    graph.dump_dot(loss, graph_file);
    std::cout << "d(loss)/dp: " << grad_p.explain() << '\n';

    auto cube = graph.custom(std::make_shared<CubePlus2X>(), {p});
    auto cube_grad = cube.derivative(p);
    auto cube_second = cube_grad.derivative(p);
    Evaluator custom_eval({cube, cube_grad, cube_second});
    auto v = custom_eval.evaluate({{"p", 2.0}});
    std::cout << "custom f(2)=" << v[0] << ", f'(2)=" << v[1]
              << ", f''(2)=" << v[2] << '\n';
}
