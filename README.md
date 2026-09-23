# Scalar Autodiff（C++11）

使用逐行赋值表达式构造静态计算图，按需查询任意已命名节点的数值及任意两个节点间的导数。表达式支持自定义算子，命令行使用 gflags，求导步骤可按需导出 DOT。手动使用 C++ `Graph` API 的示例仍在 `examples/main.cpp`。

## 表达式文件

`examples/model.expr`：

```text
h = x + y
z = 2*x + exp(x)
t = h - z + func_a(x,y)
```

每行一个赋值；空行和 `#` 后的行内注释会被忽略。未被赋值但出现在右侧的名字（这里的 `x`、`y`）自动成为输入变量；赋值后的名字保存在 `ExpressionProgram::symbols()` 映射中，可通过 `at("h")` 等方式获取节点。定义允许先引用后赋值，解析后检查循环定义和重名。没有固定的 `target`；`h`、`z`、`t` 和 `x` 均可单独查询。

支持括号、数值常量（含小数和科学记数法）、一元正负号，以及按通常优先级计算的 `+ - * /`。函数写作 `exp(x)`、`abs(x)`、`square(x)`，已注册自定义算子写作 `func_a(x,y)`。函数调用必须使用准确的参数个数；未知函数、错误参数个数、错误语法均会在建图时报告。`examples/simple.expr` 展示了无插件的 `t = h - z + x`。

## 构建和运行

需要支持 C++11 的编译器；命令行程序需要 gflags 开发库与 CMake 配置；插件动态加载使用 POSIX 接口（Linux/macOS）。在项目根目录：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure

./build/autodiff_runner \
  --expr_file=examples/model.expr \
  --op_plugins=./build/libexample_custom_ops.so \
  --inputs=x=1,y=2 \
  --outputs=h,z,t \
  --derivatives=t:x,t:h,z:x \
  --emit_dot=true --dot_dir=dot
```

这里 `--outputs` 是要取值的节点名，`--derivatives=root:output` 请求 `d(root)/d(output)`。上例分别得到 `h`、`z`、`t` 的数值与 `d(t)/d(x)`、`d(t)/d(h)`、`d(z)/d(x)`；不指定两类查询中的任何一种会报错。只想取某个中间结果，可仅传 `--outputs=h`。只查询导数，可仅传 `--derivatives=h:y`。`--inputs` 要提供所选查询实际依赖的变量；查询 `h` 时不用为其余不相关变量赋值。要使用无插件的示例，省略 `--op_plugins` 并传 `--expr_file=examples/simple.expr`。

`--emit_dot` 默认为 `false`；设为 `true` 才会在 `--dot_dir` 指定的目录写文件。目录必须已存在或可由程序直接创建。DOT 文本可用 Graphviz 渲染，例如 `dot -Tsvg dot/query_4_d_t__d_x_.dot -o grad.svg`。

如从 `examples` 目录运行，以下命令在项目文件结构下直接编译（需要已安装 gflags）：

```bash
g++ -std=c++11 -O2 -fPIC -shared -I../include \
  ../src/autodiff.cpp ../src/model.cpp -ldl -o libscalar_autodiff.so
g++ -std=c++11 -O2 -fPIC -shared -I../include \
  custom_ops.cpp -L. -lscalar_autodiff \
  -Wl,-rpath,'$ORIGIN' -o libexample_custom_ops.so
g++ -std=c++11 -O2 -I../include ../runner/main.cpp -L. \
  -lscalar_autodiff -lgflags -ldl -Wl,-rpath,'$ORIGIN' -o autodiff_runner
./autodiff_runner --expr_file=model.expr --op_plugins=./libexample_custom_ops.so \
  --inputs=x=1,y=2 --outputs=h,z,t --derivatives=t:x,t:h
```

无需 gflags 也可以编译内核测试和原始 C++ 示例：

```bash
g++ -std=c++11 -O2 -Iinclude src/autodiff.cpp tests/test_autodiff.cpp -o autodiff_tests
./autodiff_tests
g++ -std=c++11 -O2 -Iinclude src/autodiff.cpp examples/main.cpp -o autodiff_example
./autodiff_example
```

## 在 C++ 中查询节点

```cpp
#include "autodiff/model.h"
#include <vector>

using namespace autodiff;
CustomOpRegistry registry;
PluginManager plugins;
plugins.load("./libexample_custom_ops.so", registry); // 在程序建图前注册
ExpressionProgram program(read_expression_file("examples/model.expr"), registry);

Expr h = program.at("h");
Expr t = program.at("t");
Expr x = program.at("x");
std::vector<EvaluationQuery> queries;
queries.push_back(EvaluationQuery::value(h));
queries.push_back(EvaluationQuery::derivative(t, x)); // d(t)/d(x)
Evaluator evaluator(queries);
Inputs inputs = program.make_inputs(evaluator, {{"x", 1.0}, {"y", 2.0}});
Workspace workspace = evaluator.create_workspace();
std::vector<double> results(queries.size());
evaluator.evaluate(inputs, results, workspace);
// results[0] = h，results[1] = d(t)/d(x)
```

`EvaluationQuery::value(output)` 查询节点数值，`EvaluationQuery::derivative(root, output)` 查询 `d(root)/d(output)`。`Evaluator` 只编译所选查询的依赖，在一个计划中计算多个请求；同一 `root` 的符号求导请求合并求解。改变查询时重新构造 `Evaluator`；更换输入数值时复用它和 `Workspace`。`Graph`/`ExpressionProgram` 的生命周期必须覆盖执行器，`PluginManager` 必须比使用插件的对象活得更久。每个线程使用自己的输入和工作区，才能并发使用只读执行器。

## 自定义算子

`examples/custom_ops.cpp` 实现了二元 `func_a(a,b)=a*b+a`。算子继承 `CustomOp` 并实现以下方法：

- `name()`：显示在 DOT 中的名称；`arity()`：规定函数参数个数；
- `forward(inputs)`：计算结果；
- `backward(inputs, output, output_grad, input_grads)`：把各输入的数值梯度累加进 `input_grads`；
- `symbolic_partial(graph, inputs, output, input_index)`：生成相应输入的局部导数表达式，用于解释图和高阶微分。

插件导出 `extern "C" void register_autodiff_ops(autodiff::CustomOpRegistry&)`，以 `registry.register_op("func_a", std::make_shared<FuncA>())` 注册；`--op_plugins` 可以是逗号分隔的多个 `.so` 路径。插件先于表达式加载。新增的 `arity()` 为必需接口，旧插件须按新版头文件重新编译；插件和主程序必须使用兼容的 ABI。自定义算子应是无副作用且线程安全的确定性计算。

## DOT 微分解释

`--emit_dot=true` 为每个选中的数值或导数生成一个 `query_序号_查询名.dot`，`examples/dot_preview/` 有本示例的输出。普通数值文件表示节点的表达式依赖；导数文件同时标明原表达式的图、求导过程中的传播、局部导数和最终导数表达式。灰色节点来自原图；蓝色是新建的求导表达式；橙色表示求导起点；绿色表示求导终点；紫色是导数结果。黄色便笺写明每一步的上游梯度乘局部导数及对结果的贡献；紫色便笺标明所求导数。比如 `d(t)/d(x)` 图中 `func_a` 的局部导数来自插件的 `symbolic_partial()`。

`abs(0)` 的导数约定为 0；`sign` 导数按几乎处处为 0 处理。零点的二阶导不是经典数学二阶导。除零和指数溢出遵循 `double` 的结果。语法不包含比较、条件分支或隐式乘法（写 `2*x`）。
