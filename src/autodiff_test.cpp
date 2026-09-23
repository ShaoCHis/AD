#include <cmath>
#include <array>
#include <vector>
#include <memory>
#include <functional>
#include <unordered_set>
#include <iostream>

// ============================================================================
// 自动求导（Automatic Differentiation）—— 前向模式 + 反向模式
//
// 前向模式（Forward Mode）：用「对偶数 Dual Number」的思路，每个变量携带
//   值 val 与一个梯度向量 grad（第 i 个分量为对该变量的偏导）。
//   一次前向求值即可得到函数对所有输入变量的全部偏导数。
//   适合 f: R^n -> R^m 中 m 较大（尤其 m > n）的情形。
//
// 反向模式（Reverse Mode）：构建计算图，正向计算记录每条边的局部偏导，
//   再反向传播累加梯度。
//   适合 f: R^n -> R（标量输出、输入很多）的情形，一次反向即可得到梯度。
// ============================================================================

namespace ad {

// ============================ 前向模式 ====================================
// N：输入变量的个数；grad 保存对所有输入的偏导。
template <std::size_t N>
struct FVar {
    double val;
    std::array<double, N> grad;

    // 常数（对所有变量的偏导均为 0）
    FVar(double v) : val(v) { grad.fill(0.0); }

    // 第 i 个输入变量，df/dx_i = 1
    FVar(double v, std::size_t i) : val(v) {
        grad.fill(0.0);
        grad[i] = 1.0;
    }
};

template <std::size_t N> FVar<N> operator+(FVar<N> a, const FVar<N>& b) {
    a.val += b.val;
    for (std::size_t i = 0; i < N; ++i) a.grad[i] += b.grad[i];
    return a;
}
template <std::size_t N> FVar<N> operator-(FVar<N> a, const FVar<N>& b) {
    a.val -= b.val;
    for (std::size_t i = 0; i < N; ++i) a.grad[i] -= b.grad[i];
    return a;
}
template <std::size_t N> FVar<N> operator*(FVar<N> a, const FVar<N>& b) {
    for (std::size_t i = 0; i < N; ++i)
        a.grad[i] = a.grad[i] * b.val + a.val * b.grad[i];
    a.val *= b.val;
    return a;
}
template <std::size_t N> FVar<N> operator/(FVar<N> a, const FVar<N>& b) {
    for (std::size_t i = 0; i < N; ++i)
        a.grad[i] = (a.grad[i] * b.val - a.val * b.grad[i]) / (b.val * b.val);
    a.val /= b.val;
    return a;
}
template <std::size_t N> FVar<N> operator-(FVar<N> a) {
    a.val = -a.val;
    for (auto& g : a.grad) g = -g;
    return a;
}

template <std::size_t N> FVar<N> sin(FVar<N> a) {
    for (auto& g : a.grad) g *= std::cos(a.val);
    a.val = std::sin(a.val);
    return a;
}
template <std::size_t N> FVar<N> cos(FVar<N> a) {
    for (auto& g : a.grad) g *= -std::sin(a.val);
    a.val = std::cos(a.val);
    return a;
}
template <std::size_t N> FVar<N> tan(FVar<N> a) {
    double c = std::cos(a.val);
    for (auto& g : a.grad) g /= (c * c);
    a.val = std::tan(a.val);
    return a;
}
template <std::size_t N> FVar<N> exp(FVar<N> a) {
    a.val = std::exp(a.val);
    for (auto& g : a.grad) g *= a.val;  // d exp = exp * d
    return a;
}
template <std::size_t N> FVar<N> log(FVar<N> a) {
    for (auto& g : a.grad) g /= a.val;  // d log = 1/x * d
    a.val = std::log(a.val);
    return a;
}
template <std::size_t N> FVar<N> sqrt(FVar<N> a) {
    for (auto& g : a.grad) g *= 0.5 / std::sqrt(a.val);
    a.val = std::sqrt(a.val);
    return a;
}
template <std::size_t N> FVar<N> tanh(FVar<N> a) {
    double t = std::tanh(a.val);
    for (auto& g : a.grad) g *= (1.0 - t * t);
    a.val = t;
    return a;
}
// a^e，e 为常数指数
template <std::size_t N> FVar<N> pow(FVar<N> a, double e) {
    for (auto& g : a.grad) g *= e * std::pow(a.val, e - 1.0);
    a.val = std::pow(a.val, e);
    return a;
}

// ============================ 反向模式 ====================================
struct RNode {
    double val;
    double grad = 0.0;
    // edges: (子节点, 本节点对子节点的局部偏导 ∂this/∂child)
    std::vector<std::pair<std::shared_ptr<RNode>, double>> edges;
    explicit RNode(double v) : val(v) {}
};
using RVar = std::shared_ptr<RNode>;

inline RVar make_var(double v) { return std::make_shared<RNode>(v); }

inline RVar operator+(RVar a, RVar b) {
    auto r = make_var(a->val + b->val);
    r->edges.emplace_back(a, 1.0);
    r->edges.emplace_back(b, 1.0);
    return r;
}
inline RVar operator-(RVar a, RVar b) {
    auto r = make_var(a->val - b->val);
    r->edges.emplace_back(a, 1.0);
    r->edges.emplace_back(b, -1.0);
    return r;
}
inline RVar operator*(RVar a, RVar b) {
    auto r = make_var(a->val * b->val);
    r->edges.emplace_back(a, b->val);  // ∂r/∂a = b
    r->edges.emplace_back(b, a->val);  // ∂r/∂b = a
    return r;
}
inline RVar operator/(RVar a, RVar b) {
    auto r = make_var(a->val / b->val);
    r->edges.emplace_back(a, 1.0 / b->val);
    r->edges.emplace_back(b, -a->val / (b->val * b->val));
    return r;
}
inline RVar operator-(RVar a) {
    auto r = make_var(-a->val);
    r->edges.emplace_back(a, -1.0);
    return r;
}

inline RVar sin(RVar a) {
    auto r = make_var(std::sin(a->val));
    r->edges.emplace_back(a, std::cos(a->val));
    return r;
}
inline RVar cos(RVar a) {
    auto r = make_var(std::cos(a->val));
    r->edges.emplace_back(a, -std::sin(a->val));
    return r;
}
inline RVar tan(RVar a) {
    auto r = make_var(std::tan(a->val));
    double c = std::cos(a->val);
    r->edges.emplace_back(a, 1.0 / (c * c));
    return r;
}
inline RVar exp(RVar a) {
    auto r = make_var(std::exp(a->val));
    r->edges.emplace_back(a, r->val);  // d exp = exp
    return r;
}
inline RVar log(RVar a) {
    auto r = make_var(std::log(a->val));
    r->edges.emplace_back(a, 1.0 / a->val);
    return r;
}
inline RVar sqrt(RVar a) {
    auto r = make_var(std::sqrt(a->val));
    r->edges.emplace_back(a, 0.5 / r->val);
    return r;
}
inline RVar tanh(RVar a) {
    double t = std::tanh(a->val);
    auto r = make_var(t);
    r->edges.emplace_back(a, 1.0 - t * t);
    return r;
}
inline RVar pow(RVar a, double e) {
    auto r = make_var(std::pow(a->val, e));
    r->edges.emplace_back(a, e * std::pow(a->val, e - 1.0));
    return r;
}

// 对以 root 为输出的计算图做一次反向传播，把 root 的梯度种子设为 1。
inline void backward(RVar root) {
    // 1) 拓扑排序（后序遍历）
    std::vector<RVar> topo;
    std::unordered_set<RNode*> visited;
    std::function<void(RVar)> dfs = [&](RVar n) {
        if (visited.count(n.get())) return;
        visited.insert(n.get());
        for (auto& [child, _] : n->edges) dfs(child);
        topo.push_back(n);
    };
    dfs(root);

    // 2) 反向传播
    root->grad = 1.0;
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        for (auto& [child, d] : (*it)->edges) {
            child->grad += d * (*it)->grad;
        }
    }
}

}  // namespace ad

// ============================ 演示与验证 ===================================
using namespace ad;

int main() {
    // 目标函数：f(x, y) = ln(x) + x*y - sin(y)
    // 解析梯度：df/dx = 1/x + y，df/dy = x - cos(y)
    const double x0 = 2.0, y0 = 5.0;
    const double exact_dx = 1.0 / x0 + y0;
    const double exact_dy = x0 - std::cos(y0);

    std::cout << "===== 前向模式 (Forward Mode) =====\n";
    {
        FVar<2> x(x0, 0);   // 变量 x（第 0 维）
        FVar<2> y(y0, 1);   // 变量 y（第 1 维）
        FVar<2> f = log(x) + x * y - sin(y);
        std::cout << "f   = " << f.val << "\n";
        std::cout << "df/dx = " << f.grad[0] << "  (expected " << exact_dx << ")\n";
        std::cout << "df/dy = " << f.grad[1] << "  (expected " << exact_dy << ")\n";
    }

    std::cout << "\n===== 反向模式 (Reverse Mode) =====\n";
    {
        RVar x = make_var(x0);
        RVar y = make_var(y0);
        RVar f = log(x) + x * y - sin(y);
        backward(f);
        std::cout << "f   = " << f->val << "\n";
        std::cout << "df/dx = " << x->grad << "  (expected " << exact_dx << ")\n";
        std::cout << "df/dy = " << y->grad << "  (expected " << exact_dy << ")\n";
    }

    std::cout << "\n===== 数值差分校验 (Numerical Check) =====\n";
    {
        auto f = [](double x, double y) { return std::log(x) + x * y - std::sin(y); };
        double h = 1e-6;
        double ndx = (f(x0 + h, y0) - f(x0 - h, y0)) / (2 * h);
        double ndy = (f(x0, y0 + h) - f(x0, y0 - h)) / (2 * h);
        std::cout << "numeric df/dx = " << ndx << "\n";
        std::cout << "numeric df/dy = " << ndy << "\n";
    }

    return 0;
}