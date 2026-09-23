# Scalar Autodiff (C++11)

一个标量静态求导框架。先用 `Graph` 建立表达式 DAG，再生成导数表达式；`Evaluator` 将多个输出编译为同一份执行计划，适合重复传入不同数据。支持 `+ - * /`、`square()`、整数 `pow()`、`abs()`、`exp()`、自定义算子、一阶和高阶导数，也提供直接数值反向传播。

## 构建

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
./build/autodiff_example
```

没有 CMake 时，可用 C++11 编译器：

```bash
g++ -std=c++11 -O2 -Iinclude src/autodiff.cpp tests/test_autodiff.cpp -o autodiff_tests
./autodiff_tests
g++ -std=c++11 -O2 -Iinclude src/autodiff.cpp examples/main.cpp -o autodiff_example
./autodiff_example
```

代码入口：`include/autodiff/autodiff.h` 定义 API，`src/autodiff.cpp` 实现建图、符号求导和执行；`examples/main.cpp` 展示调用与自定义算子；`tests/test_autodiff.cpp` 验证结果。关键处理处有中文注释。

## 图中样例

```cpp
#include "autodiff/autodiff.h"
using namespace autodiff;

Graph graph;
auto p = graph.variable("p");
auto x = graph.variable("x");
auto y = graph.variable("y");
auto pred = exp(x * p) + 0.5 * x;
auto loss = (pred - y).pow(2);

auto derivatives = loss.derivative({p, pred}); // 一次逆序遍历
auto grad_p = derivatives[0];
auto grad_pred = derivatives[1];
auto grad_2nd_p = grad_p.derivative(p);

Evaluator evaluator({loss, grad_p, grad_pred, grad_2nd_p});
auto result = evaluator.evaluate({{"x", 1}, {"y", -0.05}, {"p", 0.5}});
// result[0..3] = loss, dloss/dp, dloss/dpred, d²loss/dp²
```

与图片相比，变量通过显式 `Graph` 创建：表达式必须属于同一个图，且 `Graph` 的生命周期须覆盖表达式和执行器。构造完成的 `Evaluator` 持有已编译节点与自定义算子的共享所有权，但仍用图来检查传入的 `Expr`，所以图也须覆盖执行器的生命周期。新增节点后如需计算新输出，请构造新的执行器；已有执行器按构造时的图快照工作。

为兼容 C++11，数组视图使用项目提供的 `autodiff::Span<T>`，接收原始指针与长度、`std::array` 或 `std::vector`。`Span` 不持有数据，请保证底层数据在函数调用期间有效。

## 重复执行与数值反传

```cpp
auto inputs = evaluator.create_inputs();
auto ws = evaluator.create_workspace();
std::array<double, 4> outputs{};
inputs.set(x, 1);
inputs.set(y, -0.05);
for (double value : values) {
    inputs.set(p, value);
    evaluator.evaluate(inputs, outputs, ws);
}

std::array<Expr, 2> wrts{p, pred};
std::array<double, 2> grads{};
evaluator.backward(0, inputs, wrts, grads, ws); // 根 0 为 loss
```

`evaluate(inputs, outputs, ws)` 和 `backward(...)` 由框架自身执行时不做堆分配；构造 `Inputs`、`Workspace` 和编译图时会分配内存。每个线程使用单独的 `Inputs`/`Workspace`，多个线程可共享只读执行器；自定义算子实现也须线程安全。快捷接口 `evaluate({{"x", 1}, ...})` 会为方便使用而在每次调用分配内存。

每个可达节点只编译和计算一次；图构建时进行内置节点哈希去重、常量折叠和有限代数化简；执行器剔除不可达节点。值数组按可达节点密集存储，允许在数值反传中复用全部前向值；**当前没有做活跃区间槽位复用或指令融合**。大量节点或深度高阶求导时仍需监控图规模。`Graph` 建图时不可并发修改；构造好执行器后再并发调用只读执行接口。

## 自定义算子

派生 `CustomOp`，实现：

- `forward(inputs)`：计算标量结果；
- `backward(inputs, output, output_grad, input_grads)`：将每个输入的数值梯度**累加**进已清零的 `input_grads`；
- `symbolic_partial(graph, inputs, output, input_index)`：返回该输入的局部导数表达式，用于可再次求导的导数图。

例如 `examples/main.cpp` 给出 `x³ + 2x`；`tests/test_autodiff.cpp` 给出双输入算子。算子对象通过 `std::shared_ptr<const CustomOp>` 传给 `graph.custom(op, {x, y})`，生命周期由图和执行器持有。自定义算子应是无副作用的确定性函数；符号与数值梯度规则应一致。

## 调试与边界约定

`expr.named("label")` 为节点命名；`expr.explain()` 展示有深度上限的表达式；`graph.dump_dot(expr, stream)` 导出依赖图。示例程序在运行目录生成 `loss.dot`，可运行 `dot -Tsvg loss.dot -o loss.svg` 渲染。

`abs(0)` 使用导数 0 的约定。内部 `sign` 节点的导数按几乎处处为 0 处理，因此该点显示的二阶导也为 0，**并非 0 点的经典数学二阶导**。除以零及 `exp` 溢出遵循 `double` 的 IEEE 754 结果。有限代数化简（如 `x + 0 -> x`）不保证完全保持有符号零的 IEEE 754 细节。常量折叠和代数化简不会额外删除 `x * 0`，避免吞掉 `NaN`。请避免在算子不可导点、溢出点依赖数值梯度与符号梯度严格一致。
