# Scalar Autodiff（C++11）

按照 **generator → evaluator → post-evaluator** 三个执行阶段组织标量静态计算图。generator 根据简洁 JSON 或赋值表达式构图并加载自定义算子；evaluator 对任意命名节点取值或求导；post-evaluator 根据开关输出解释性 DOT、用二阶精度中心差分检查导数。

| 模块 | 头文件 / 实现 | 职责 |
| --- | --- | --- |
| 图内核 | `graph.h` / `graph.cpp` | 节点、算术、插件节点、符号求导、DOT 图底层操作 |
| generator | `generator.h` / `generator.cpp` | 加载配置和插件、解析赋值表达式、保存名字到节点的映射 |
| evaluator | `evaluator.h` / `evaluator.cpp` | 编译所选查询、执行数值计算与数值反传；提供中间节点扰动接口 |
| post-evaluator | `post_evaluator.h` / `post_evaluator.cpp` | 按开关输出 DOT 和校验导数；无开关时没有额外处理 |
| CLI | `runner/main.cpp`、`runner/post_evaluator_flags.cpp` | 用 gflags 读取配置、输入和查询；post-evaluator 的 gflags 开关集中在后处理适配文件 |

旧的 `autodiff.h` 和 `model.h` 保留为转发头文件，兼容已有示例的 `#include`。

## 定义静态图

`examples/model.json` 只描述**如何建图**：

```json
{
  "expressions": [
    "h = x + y",
    "z = 2*x + exp(x)",
    "t = h - z + func_a(x,y)"
  ]
}
```

可选 `"op_plugins": ["./build/libexample_custom_ops.so"]`，样例见 `examples/model_with_plugin.json`。generator 在解析赋值表达式前加载插件。配置只接收 `expressions`（至少一项）和 `op_plugins`（可选）两个字符串数组；未知键和错误 JSON 会报错。插件路径相对于**运行时工作目录**。也可以直接通过 `--expr_file=examples/model.expr` 加载同样的逐行赋值文本，并用 `--op_plugins` 添加插件；两种图输入方式恰好选择一种。

表达式支持 `+ - * /`、括号、一元正负号、数值、`exp(x)`、`abs(x)`、`square(x)` 和已注册算子 `func_a(x,y)`。函数参数个数、重名、循环定义和语法错误在建图时检查。未在左侧赋值的标识符自动成为输入变量。赋值允许前向引用；所有名称，包括 `h`、`z`、`t`、`x`、`y`，均可通过 `ExpressionProgram::at(name)` 和 `symbols()` 映射查询。没有固定 target。

## 使用 gflags 运行

需安装 C++11 编译器与 gflags 开发包；CMake 构建：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure

./build/autodiff_runner \
  --config_json=examples/model.json \
  --op_plugins=./build/libexample_custom_ops.so \
  --inputs=x=1,y=2 \
  --outputs=h,z,t \
  --derivatives=t:x,t:h,z:x \
  --verify_derivatives=true \
  --emit_dot=true --dot_dir=dot
```

`--outputs=h,z` 查询节点值；`--derivatives=t:x,t:h` 分别查询 `d(t)/d(x)` 和 `d(t)/d(h)`，因此任意中间环节都能充当求导起点或终点。至少指定一类查询。`--inputs` 为请求所需变量提供数值；开启数值验证时，还需提供原表达式及被扰动节点所依赖的输入。也可使用 `--config_json=examples/model_with_plugin.json`，并省略命令行的 `--op_plugins`；仅含内置算子时使用 `--expr_file=examples/simple.expr`。

post-evaluator 的开关彼此独立：

- `--emit_dot=true`：为选中的每项输出一个 `query_*.dot`；默认为 `false`，`--dot_dir=dot` 指定目录。
- `--verify_derivatives=true`：对每个导数做中心差分校验；默认为 `false`。`--fd_step=1e-5` 设置相对于被扰动节点数值的步长，`--fd_abs_tol=1e-6` 与 `--fd_rel_tol=1e-4` 设置误差限。输出 PASS / FAIL；有失败时进程退出码为 2。

中心差分使用 `(f(v+h)-f(v-h))/(2h)`，截断误差阶为 `O(h²)`；这里的“二阶”指**一阶导数的二阶精度数值近似**。对中间节点 `h`，程序将其数值视为独立输入，仅重算依赖它的下游节点，因此验证的是 `d(t)/d(h)` 的图上偏导数。若数学函数在当前点不可导（如某些 `abs` 的尖点）、数值溢出或步长无法表示，校验可能失败或报告原因。校验自定义算子时，会通过其 `forward()` 重算，并与 `symbolic_partial()` 生成的导数比较。

DOT 文件包含原图节点、求导新增节点、传播路径和局部导数说明。可运行 `dot -Tsvg dot/query_4_d_t__d_x_.dot -o grad.svg` 渲染；`examples/dot_preview/` 提供文本预览，渲染需要 Graphviz。导数过程必须在编译求导查询前开启记录：C++ 代码先调用 `post.prepare(program.graph())`，再构造 `Evaluator`。

### 在 examples 目录用 g++ 编译

```bash
cd examples
g++ -std=c++11 -O2 -fPIC -shared -I../include \
  ../src/graph.cpp ../src/evaluator.cpp ../src/generator.cpp ../src/post_evaluator.cpp \
  -ldl -o libscalar_autodiff.so
g++ -std=c++11 -O2 -fPIC -shared -I../include \
  custom_ops.cpp -L. -lscalar_autodiff \
  -Wl,-rpath,'$ORIGIN' -o libexample_custom_ops.so
g++ -std=c++11 -O2 -I../include ../runner/main.cpp ../runner/post_evaluator_flags.cpp -L. \
  -lscalar_autodiff -lgflags -ldl -Wl,-rpath,'$ORIGIN' -o autodiff_runner
./autodiff_runner --expr_file=model.expr --op_plugins=./libexample_custom_ops.so \
  --inputs=x=1,y=2 --outputs=h,z,t --derivatives=t:x,t:h \
  --verify_derivatives=true
```

`examples/main.cpp` 是直接使用 C++ 手动建图的基础样例；仅需编译 `src/graph.cpp src/evaluator.cpp`，无需 gflags。

## 嵌入 C++ 项目

```cpp
#include "autodiff/post_evaluator.h"
using namespace autodiff;

GeneratorConfig config = read_generator_json("examples/model.json");
config.op_plugins.push_back("./build/libexample_custom_ops.so");
GraphGenerator generator(config); // 插件先加载，再建立静态图
ExpressionProgram& program = generator.program();

PostEvaluationOptions options;
options.verify_derivatives = true;
options.emit_dot = true;
PostEvaluator post(options);
post.prepare(program.graph());  // 必须在符号求导前

Expr t = program.at("t");
Expr h = program.at("h");
NamedQuery request{EvaluationQuery::derivative(t, h), "d(t)/d(h)", "t", "h"};
Evaluator evaluator({request.request});
std::map<std::string, double> values{{"x", 1}, {"y", 2}};
Inputs inputs = program.make_inputs(evaluator, values);
Workspace workspace = evaluator.create_workspace();
double result[1];
evaluator.evaluate(inputs, Span<double>(result, 1), workspace);
std::vector<DerivativeCheck> checks = post.run(program, evaluator,
    Span<const NamedQuery>(&request, 1), values, Span<const double>(result, 1));
// result[0] == 1，checks[0].passed == true
```

`GraphGenerator` 持有插件代码、注册表和静态图，必须比 `Evaluator` 活得久。改变查询后重建 `Evaluator`；只改变输入值时复用其编译计划与工作区。每个线程使用单独的输入及工作区。

## 定义插件算子

`examples/custom_ops.cpp` 中的 `func_a(a,b)=a*b+a` 实现了 `CustomOp::name()`、`arity()`、`forward()`、`backward()` 和 `symbolic_partial()`；导出 `extern "C" void register_autodiff_ops(CustomOpRegistry&)`。`arity()` 用于在建图时验证 `func_a(x,y)` 是否传入两个参数。插件需要兼容的 C++ ABI；插件加载和 DOT 目录创建使用 POSIX 接口（Linux/macOS）。
