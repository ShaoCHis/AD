# Scalar Autodiff（C++11）

按照 **generator → evaluator → post-evaluator** 三个执行阶段组织标量静态计算图。generator 在进程中只初始化一次，根据简洁 JSON 或赋值表达式构图并加载自定义算子；evaluator 可在多个线程中对任意命名节点取值或求导；post-evaluator 根据开关输出解释性 DOT、用二阶精度中心差分检查导数。

| 模块 | 头文件 / 实现 | 职责 |
| --- | --- | --- |
| 图内核 | `graph.h` / `graph.cpp` | 节点、算术、插件节点、符号求导、DOT 图底层操作 |
| generator | `generator.h` / `generator.cpp` | 单例加载配置和插件、保存名称映射；加锁编译并缓存不同的求导计划 |
| evaluator | `evaluator.h` / `evaluator.cpp` | 执行只读查询计划、数值计算与数值反传；提供中间节点扰动接口 |
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

DOT 文件包含原图节点、求导新增节点、传播路径和局部导数说明。可运行 `dot -Tsvg dot/query_4_d_t__d_x_.dot -o grad.svg` 渲染；`examples/dot_preview/` 提供文本预览，渲染需要 Graphviz。编译查询时将 `post.emits_dot()` 传给 `generator.compile(...)`，generator 会在同一把锁内开启并记录求导过程；DOT 读取也在锁内进行。

### 在 examples 目录用 g++ 编译

```bash
cd examples
g++ -std=c++11 -O2 -pthread -fPIC -shared -I../include \
  ../src/graph.cpp ../src/evaluator.cpp ../src/generator.cpp ../src/post_evaluator.cpp \
  -ldl -o libscalar_autodiff.so
g++ -std=c++11 -O2 -pthread -fPIC -shared -I../include \
  custom_ops.cpp -L. -lscalar_autodiff \
  -Wl,-rpath,'$ORIGIN' -o libexample_custom_ops.so
g++ -std=c++11 -O2 -pthread -I../include ../runner/main.cpp ../runner/post_evaluator_flags.cpp -L. \
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
GraphGenerator& generator = GraphGenerator::initialize(config); // 初始化一次
const ExpressionProgram& program = generator.program();

PostEvaluationOptions options;
options.verify_derivatives = true;
options.emit_dot = true;
PostEvaluator post(options);

Expr t = program.at("t");
Expr h = program.at("h");
NamedQuery request{EvaluationQuery::derivative(t, h), "d(t)/d(h)", "t", "h"};
std::shared_ptr<const Evaluator> evaluator =
    generator.compile({request.request}, post.emits_dot());
std::map<std::string, double> values{{"x", 1}, {"y", 2}};
Inputs inputs = program.make_inputs(*evaluator, values);
Workspace workspace = evaluator->create_workspace();
double result[1];
evaluator->evaluate(inputs, Span<double>(result, 1), workspace);
std::vector<DerivativeCheck> checks = post.run(generator, *evaluator,
    Span<const NamedQuery>(&request, 1), values, Span<const double>(result, 1));
// result[0] == 1，checks[0].passed == true
```

`GraphGenerator::instance()` 在首次 `initialize(config)` 后取回同一对象；重复传入相同配置返回同一对象，不同配置报错。单例在进程结束前保留图和插件代码。`compile()` 按请求缓存计划；第一次编译新的求导请求会向图追加节点，因此 generator 加锁执行。已编译的 `shared_ptr<const Evaluator>` 是只读的，可供多个线程复用。每个线程分别创建自己的 `Inputs` 和 `Workspace`，并传入自己的变量值；不要跨线程共用这两个可变对象。单例返回的 `program()` 仅供查询名称，不在并发阶段直接修改 `Graph`。DOT 生成与新查询的编译使用同一把锁。自定义算子的 `forward()` / `backward()` / `symbolic_partial()` 应保证线程安全。

例如，将上面的只读 `evaluator` 共享给不同线程时，每个任务自行准备工作区：

```cpp
auto evaluate_one = [&](double x_value) {
    Inputs local_inputs = program.make_inputs(*evaluator, {{"x", x_value}, {"y", 2}});
    Workspace local_workspace = evaluator->create_workspace();
    double local_result[1];
    evaluator->evaluate(local_inputs, Span<double>(local_result, 1), local_workspace);
    return local_result[0];
};
std::thread a([&] { double first = evaluate_one(1); /* 使用 first */ });
std::thread b([&] { double second = evaluate_one(2); /* 使用 second */ });
a.join();
b.join();
```

此段代码还需在调用方包含 `<thread>`。`tests/test_concurrency.cpp` 会让 8 个线程同时请求导数、取值并读取 DOT，检查这些操作的结果。

## 定义插件算子

`examples/custom_ops.cpp` 中的 `func_a(a,b)=a*b+a` 实现了 `CustomOp::name()`、`arity()`、`forward()`、`backward()` 和 `symbolic_partial()`；导出 `extern "C" void register_autodiff_ops(CustomOpRegistry&)`。`arity()` 用于在建图时验证 `func_a(x,y)` 是否传入两个参数。插件需要兼容的 C++ ABI；插件加载和 DOT 目录创建使用 POSIX 接口（Linux/macOS）。
