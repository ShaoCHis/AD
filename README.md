# Scalar Autodiff (C++11)

一个标量静态求导框架。JSON 命令行程序会先加载 C++ 自定义算子共享库，再从 JSON 建立表达式 DAG、求导并执行。`Evaluator` 把多个输出编译为同一计划，适合重复传入数据。支持 `+ - * /`、`square()`、整数 `pow()`、`abs()`、`exp()`、自定义算子、一阶和高阶导数，也提供直接数值反向传播。原有 C++ 手动建图 API 仍可使用。

## 构建

```bash
# 构建 JSON runner 需要 gflags 的头文件、库与 CMake package config。
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
./build/autodiff_runner --graph_json=examples/model.json --emit_dot=true --dot_dir=dot
./build/autodiff_runner --op_plugins=./build/libexample_custom_ops.so \
  --graph_json=examples/custom_model.json --emit_dot=true --dot_dir=custom_dot
```

其中 `--emit_dot` 默认是 `false`，不生成任何 DOT 文件；设为 `true` 时才输出目标图和每个导数的解释图。`--dot_dir` 需是已有目录或可直接创建的单层目录。gflags 的 `DEFINE_bool`、`DEFINE_string` 与 `ParseCommandLineFlags` 是该开关的实际实现；gflags 官方说明见 [gflags 使用文档](https://gflags.github.io/gflags/)。

如果没有 CMake，且已经安装 gflags，可以在项目根目录直接用 C++11 编译：

```bash
g++ -std=c++11 -O2 -fPIC -shared -Iinclude \
  src/autodiff.cpp src/json.cpp src/model.cpp -ldl -o libscalar_autodiff.so
g++ -std=c++11 -O2 -fPIC -shared -Iinclude \
  examples/custom_ops.cpp -L. -lscalar_autodiff \
  -Wl,-rpath,'$ORIGIN' -o libexample_custom_ops.so
g++ -std=c++11 -O2 -Iinclude runner/main.cpp -L. \
  -lscalar_autodiff -lgflags -ldl -Wl,-rpath,'$ORIGIN' -o autodiff_runner
./autodiff_runner --graph_json=examples/model.json --emit_dot=true
```

原有示例无需 gflags：

```bash
g++ -std=c++11 -O2 -Iinclude src/autodiff.cpp tests/test_autodiff.cpp -o autodiff_tests
./autodiff_tests
g++ -std=c++11 -O2 -Iinclude src/autodiff.cpp examples/main.cpp -o autodiff_example
./autodiff_example
```

代码入口：`include/autodiff/autodiff.h` 与 `src/autodiff.cpp` 是求导内核；`json.h/json.cpp` 严格解析 JSON；`model.h/model.cpp` 注册插件并加载多级表达式；`runner/main.cpp` 使用 gflags；`examples/custom_ops.cpp` 是共享库插件，`examples/model.json` 与 `examples/custom_model.json` 是输入样例。

## JSON 图格式

```json
{
  "variables": ["p", "x", "y"],
  "definitions": [
    {"id": "loss", "role": "target", "op": "square",
     "inputs": [{"op": "sub", "inputs": ["pred", "y"]}]},
    {"id": "pred", "op": "add",
     "inputs": [
       {"op": "exp", "inputs": [{"op": "mul", "inputs": ["x", "p"]}]},
       {"op": "mul", "inputs": [0.5, "x"]}
     ]}
  ],
  "derivatives": [
    {"id": "grad_p", "of": "loss", "wrt": "p"},
    {"id": "grad_pred", "of": "loss", "wrt": "pred"},
    {"id": "grad2_p", "of": "grad_p", "wrt": "p"}
  ],
  "inputs": {"p": 0.5, "x": 1, "y": -0.05},
  "outputs": ["loss", "grad_p", "grad_pred", "grad2_p"]
}
```

- `role: "target"` **恰好出现一次**；其他定义是中间表达式（`role: "intermediate"` 可省略）。`loss` 可以写在它引用的 `pred` 前面；加载器按依赖递归建图，并检查循环与未定义引用。
- `op` 是 `add/sub/mul/div`（恰好两个输入）、`square/abs/exp`（恰好一个输入）或已注册的自定义算子名。`inputs` 元素可为数字、已有变量或表达式 ID、内嵌的 `{ "op": ..., "inputs": [...] }`。
- `derivatives` 可引用变量、中间表达式，也能在 `of` 中引用前一个导数 ID；相同 `of` 的请求会共用一次符号反向遍历。省略 `outputs` 时，输出依次为目标与每个导数；`inputs` 中填写运行时变量值。
- JSON 文件最大 16 MiB；重复键、表达式重名、未定义引用、循环依赖及错误算子参数个数会报错。

### 先加载自定义算子

插件实现 `CustomOp` 的 `forward`、`backward`、`symbolic_partial`，并导出函数：

```cpp
extern "C" void register_autodiff_ops(autodiff::CustomOpRegistry& registry) {
    registry.register_op("cube_plus_2x", std::make_shared<CubePlus2X>());
}
```

启动时传 `--op_plugins=./libexample_custom_ops.so`；多个插件路径以逗号分隔。动态库由 `PluginManager` 加载，并在**读取 JSON 前**完成注册。插件与主程序须使用兼容的 C++ 编译器、ABI 和本框架动态库。运行中的算子对象须无副作用、线程安全。

插件加载和 `mkdir` 使用 POSIX 接口（Linux/macOS）；Windows 需要替换这两处平台实现。

### DOT 微分解释

`--emit_dot=true` 会写 `dot/target.dot`（原目标依赖）与 `dot/derivative_1_grad_p.dot` 等（每个导数一个文件）。微分 DOT 中灰色节点来自求导前的图，蓝色节点是本轮求导新建的表达式，橙色是求导来源，绿色是求导目标变量，紫色是导数结果；浅黄色便笺解释每一步的 `dz × 局部导数`、输入与贡献节点，紫色便笺显示导数标题及公式。只保留与本次 `wrt` 相关的传播步骤。DOT 为文本，可用 `dot -Tsvg dot/derivative_1_grad_p.dot -o grad_p.svg` 渲染；渲染需要另装 Graphviz。

`examples/dot_preview/` 内附基于 `examples/model.json` 的预览 DOT 文件，便于直接查看输出格式。它们是示例资源；运行时生成新文件仍完全由 `--emit_dot` 控制。

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

`expr.named("label")` 为节点命名；`expr.explain()` 展示有深度上限的表达式；C++ API 仍可主动调用 `graph.dump_dot(expr, stream)` 导出依赖图。命令行程序仅在 `--emit_dot=true` 时生成 DOT 文件。

`abs(0)` 使用导数 0 的约定。内部 `sign` 节点的导数按几乎处处为 0 处理，因此该点显示的二阶导也为 0，**并非 0 点的经典数学二阶导**。除以零及 `exp` 溢出遵循 `double` 的 IEEE 754 结果。有限代数化简（如 `x + 0 -> x`）不保证完全保持有符号零的 IEEE 754 细节。常量折叠和代数化简不会额外删除 `x * 0`，避免吞掉 `NaN`。请避免在算子不可导点、溢出点依赖数值梯度与符号梯度严格一致。
