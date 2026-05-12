# Claude Code ECC Skills 使用方法

## 目录

- [什么是 Skill](#什么是-skill)
- [Skill 的工作原理](#skill-的工作原理)
- [Skill 的分类体系](#skill-的分类体系)
- [调用方式](#调用方式)
- [你的场景推荐 Skill](#你的场景推荐-skill)
- [Skill 调用示例](#skill-调用示例)
- [常用组合拳](#常用组合拳)
- [进阶技巧](#进阶技巧)
- [自定义 Skill](#自定义-skill)
- [注意事项](#注意事项)
- [没有 Skill 但可替代的方案](#没有-skill-但可替代的方案)

---

## 什么是 Skill

### 基本概念

**Skill** 是 Claude Code 生态系统中的专业化子代理（Sub-Agent）系统。每个 skill 本质上是一个**预配置的 AI 角色**，封装了特定领域的：

| 要素 | 说明 | 示例 |
| ---- | ---- | ---- |
| **领域知识** | 该领域的专业知识、术语、常见模式 | `python-review` 内置 PEP 8、类型注解规范 |
| **审查规则** | 自动化检查清单和判断标准 | `security-review` 内置 OWASP Top 10 规则 |
| **最佳实践** | 行业公认的推荐做法 | `pytorch-patterns` 内置 `model.eval()`、`torch.no_grad()` 检查 |
| **工具集** | 该领域专用的分析工具和方法 | `database-reviewer` 可执行查询计划分析 |
| **输出格式** | 结构化的结果呈现方式 | `code-review` 输出分级的问题清单 |

### 为什么需要 Skill

**不用 Skill 的问题：** 每次都要在 prompt 中重复描述规范和要求，且容易遗漏：

```text
# 没有 skill 时，你需要这样写：
请用 PEP 8 标准审查这段代码，检查类型注解是否完整，
检查是否有潜在的 None 引用问题，检查异常处理是否合理，
检查是否有性能问题，检查命名规范...
```

**用了 Skill 之后：**

```text
/python-review
```

一个命令搞定，skill 内置了所有规则，不会遗漏，输出也格式统一。

### Skill 与普通对话的区别

| 维度 | 普通对话 | 使用 Skill |
| ---- | -------- | ---------- |
| **专业性** | 依赖你给出的指令质量 | 内置领域专业规则 |
| **一致性** | 每次输出格式可能不同 | 输出格式标准化 |
| **完整性** | 可能遗漏检查项 | 按清单逐一检查 |
| **效率** | 需要反复描述需求 | 一个命令触发全套流程 |
| **可组合** | 需要手动串联多个任务 | 多个 skill 可串联执行 |

---

## Skill 的工作原理

### 架构概览

```text
用户输入
    │
    ▼
┌──────────────────────────────────────┐
│         Claude Code 核心引擎          │
│                                      │
│  ┌────────────────────────────────┐  │
│  │      Skill 路由器              │  │
│  │  - 分析用户意图                │  │
│  │  - 匹配最佳 skill              │  │
│  │  - 确定调用方式（显式/隐式）   │  │
│  └──────────┬─────────────────────┘  │
│             │                        │
│     ┌───────┼───────┐               │
│     ▼       ▼       ▼               │
│  ┌─────┐ ┌─────┐ ┌─────┐           │
│  │Skill│ │Skill│ │Skill│  ...       │
│  │  A  │ │  B  │ │  C  │           │
│  └──┬──┘ └──┬──┘ └──┬──┘           │
│     │       │       │               │
│     ▼       ▼       ▼               │
│  ┌──────────────────────────┐       │
│  │    子代理执行环境         │       │
│  │  - 独立上下文窗口        │       │
│  │  - 专属工具集            │       │
│  │  - 结构化输出            │       │
│  └──────────────────────────┘       │
└──────────────────────────────────────┘
    │
    ▼
结构化结果 → 返回用户
```

### 执行流程

1. **意图识别**：核心引擎分析你的输入，判断是否需要调用 skill
2. **Skill 选择**：根据语义匹配最合适的 skill（可同时匹配多个）
3. **上下文注入**：将该 skill 的专业知识、规则集注入子代理的上下文
4. **独立执行**：子代理在独立的上下文中执行任务，不影响主对话
5. **结果汇总**：执行结果以结构化格式返回主对话

### 三种触发模式

| 模式 | 触发方式 | 示例 | 适用场景 |
| ---- | -------- | ---- | -------- |
| **显式斜杠** | `/skill-name` | `/code-review` | 你明确知道要用哪个 skill |
| **隐式匹配** | 自然语言描述 | "帮我审查这个文件" | 你不确定用哪个 skill，让系统自动匹配 |
| **多重调用** | 指定多个 skill 串联 | "用 python-review 和 security-review 检查" | 需要多个专业视角 |

---

## Skill 的分类体系

ECC Skills 按功能领域分为以下几大类：

### 代码审查类（Code Review）

分析已有代码的质量、安全、性能问题。

| Skill | 语言/领域 | 检查重点 |
| ---- | --------- | -------- |
| `code-review` | 通用 | 逻辑错误、边界条件、可维护性 |
| `python-review` | Python | PEP 8、类型注解、async/await、GIL |
| `cpp-review` | C++ | 内存安全、RAII、智能指针、UB |
| `go-review` | Go | 并发安全、错误处理、接口设计 |
| `rust-review` | Rust | 所有权、生命周期、unsafe 使用 |
| `kotlin-review` | Kotlin | 协程、空安全、密封类 |
| `swift-review` | Swift | ARC、协议、Swift Concurrency |
| `java-review` | Java | JPA、Spring Boot、并发 |
| `csharp-review` | C# | async、nullable、LINQ |
| `fastapi-review` | FastAPI | 异步、依赖注入、Pydantic |
| `flutter-review` | Flutter | Widget 树、状态管理、性能 |
| `typescript-review` | TypeScript | 类型安全、异步、Node 安全 |

### 安全审查类（Security）

聚焦于安全漏洞和合规问题。

| Skill | 检查范围 |
| ---- | -------- |
| `security-review` | OWASP Top 10、注入、XSS、SSRF、不安全的加密 |
| `silent-failure-hunter` | 被吞掉的异常、空 catch、错误的 fallback 逻辑 |
| `healthcare-review` | PHI 合规、临床安全、医疗数据完整性 |

### 构建修复类（Build Resolver）

专注于编译和构建错误的快速定位和修复。

| Skill | 平台/语言 |
| ---- | --------- |
| `build-fix` | 通用构建错误 |
| `cpp-build-resolver` | C++ / CMake / colcon |
| `pytorch-build-resolver` | PyTorch 运行时 / CUDA |
| `go-build-resolver` | Go / go build |
| `java-build-resolver` | Java / Maven / Gradle |
| `kotlin-build-resolver` | Kotlin / Gradle |
| `dart-build-resolver` | Dart / Flutter |
| `swift-build-resolver` | Swift / Xcode |
| `rust-build-resolver` | Rust / Cargo |

### 测试类（Testing）

| Skill | 领域 |
| ---- | ---- |
| `tdd-workflow` | 测试驱动开发流程 |
| `python-testing` | Pytest 单元测试 |
| `e2e-runner` | Playwright E2E 测试 |
| `test-coverage` | 测试覆盖率分析和提升 |
| `pr-test-analyzer` | PR 测试质量审查 |

### 设计与架构类（Design）

| Skill | 用途 |
| ---- | ---- |
| `plan` | 功能实现计划和任务分解 |
| `architect` | 系统架构设计和技术决策 |
| `code-architect` | 从已有代码推导设计蓝图 |
| `code-explorer` | 追溯代码执行路径、映射架构分层 |
| `api-design` | API 接口设计最佳实践 |
| `design-system` | UI 设计系统构建 |
| `hexagonal-architecture` | 六边形架构（端口-适配器）模式 |

### 工程实践类（Engineering Patterns）

| Skill | 领域 |
| ---- | ---- |
| `tdd-workflow` | 测试驱动开发 |
| `continuous-learning` | AI 持续学习/微调流程 |
| `autonomous-loops` | 自主代理循环 |
| `deployment-patterns` | 部署模式和策略 |
| `backend-patterns` | 后端架构模式 |
| `frontend-patterns` | 前端架构模式 |
| `docker-patterns` | Docker 容器化最佳实践 |
| `redis-patterns` | Redis 缓存和数据结构模式 |
| `postgres-patterns` | PostgreSQL 查询和 Schema 设计 |
| `swiftui-patterns` | SwiftUI 视图和状态管理 |

### 数据库类（Database）

| Skill | 数据库 |
| ---- | ------ |
| `database-reviewer` | PostgreSQL 查询优化、索引、安全 |
| `database-migrations` | 数据库迁移脚本和管理 |
| `postgres-patterns` | PostgreSQL 最佳实践 |
| `redis-patterns` | Redis 缓存和消息队列 |

### 运营与自动化类（Operations）

| Skill | 用途 |
| ---- | ---- |
| `git-workflow` | Git 分支管理、提交整理、PR 流程 |
| `review-pr` | GitHub PR 审查 |
| `update-docs` | 文档和 codemap 更新 |
| `update-codemaps` | 代码结构地图维护 |
| `refactor-cleaner` | 死代码清理、去重 |
| `code-simplifier` | 代码简化和可读性优化 |
| `performance-optimizer` | 性能分析和瓶颈定位 |
| `hookify` | 自定义 hooks 配置和管理 |

---

## 调用方式

### 方式 1: 斜杠命令（精确调用）

最直接的调用方式，适合你明确知道要用哪个 skill：

```text
/code-review
/python-review
/git-workflow
/security-review
/plan
```

**优点：** 精确无误，不会出现匹配偏差
**缺点：** 需要记住 skill 名称

### 方式 2: 自然语言（模糊调用）

直接用自然语言描述需求，系统自动匹配最合适的 skill：

| 你说的话 | 自动匹配的 Skill |
| -------- | --------------- |
| "帮我审查这个文件" | `code-review` |
| "检查安全漏洞" | `security-review` |
| "修复编译错误" | `build-fix` |
| "这个函数怎么写测试" | `python-testing` |
| "清理没用的代码" | `refactor-cleaner` |
| "PR 准备好了帮我看看" | `review-pr` |
| "帮我规划这个功能" | `plan` |

**优点：** 不需要知道 skill 名称，自然交互
**缺点：** 模糊场景下可能匹配到非预期的 skill

### 方式 3: 明确指定 skill 名称（推荐）

在对话中显式指定 skill，兼顾精确和灵活：

```text
用 python-review 审查 Class_9.py
用 git-workflow 提交这些改动
用 security-review + python-review 检查这个 API
先 plan 规划架构，再 code-review 审查实现
```

**优点：** 精确控制 + 灵活组合

### 方式 4: Skill 串联（组合拳）

多个 skill 按顺序执行，形成完整工作流：

```text
# 写完代码后的完整检查流程
用 code-review 审查代码 → 用 security-review 扫描漏洞 → 用 build-fix 确保编译通过

# 重构流程
用 code-simplifier 简化代码 → 用 python-review 审查质量 → 用 python-testing 补充测试
```

---

## 你的场景推荐 Skill

### 深度学习 / PyTorch

你当前的工作目录是 `Py/Projects/DeepLearning/`，以下 skill 直接相关：

| Skill | 用途 | 你的场景 |
| ----- | ---- | -------- |
| `pytorch-patterns` | 训练循环、模型设计、GPU 利用 | 审查 `Class_9.py` 等训练脚本 |
| `python-review` | PEP 8、类型提示、异常处理 | 审查所有 Python 文件 |
| `pytorch-build-resolver` | 张量形状不匹配、CUDA OOM、DataLoader 问题 | 训练报错时快速定位 |
| `python-testing` | 编写模型测试、数据管道测试 | 为数据加载和模型推理写单测 |

### ROS2 机器人

你之前的 ROS2 导航调试场景：

| Skill | 用途 | 你的场景 |
| ----- | ---- | -------- |
| `python-review` | ROS2 Python 节点代码审查 | 审查 launch 文件、节点脚本 |
| `cpp-review` | C++ 控制器/规划器插件审查 | 审查 MPPI、SMAC 等控制器代码 |
| `cpp-build-resolver` | colcon/CMake/ament 编译错误 | 解决 nav2 编译依赖问题 |
| `security-review` | ROS2 网络通信安全审计 | 检查 topic/service 安全配置 |

### 通用工程

| Skill | 用途 | 你的场景 |
| ----- | ---- | -------- |
| `code-review` | 通用代码质量 | 所有语言的基础审查 |
| `code-simplifier` | 消除冗余、提高可读性 | 重构复杂函数 |
| `silent-failure-hunter` | 查找被吞掉的错误 | ROS2 节点静默崩溃排查 |
| `refactor-cleaner` | 死代码清理 | 项目目录整理 |

### Git 工作流

| Skill | 用途 | 你的场景 |
| ----- | ---- | -------- |
| `git-workflow` | 提交规范、分支策略 | 整理 commit 历史 |
| `review-pr` | PR 审查 | 合并前检查 |

### 构建与测试

| Skill | 用途 | 你的场景 |
| ----- | ---- | -------- |
| `build-fix` | 通用构建错误 | pip/conda 依赖问题 |
| `python-testing` | Pytest 单元测试 | 模型推理和数据管道测试 |
| `tdd-workflow` | 测试驱动开发 | 新功能先写测试 |

### 数据库

| Skill | 用途 | 你的场景 |
| ----- | ---- | -------- |
| `database-reviewer` | SQL 优化 | 训练数据查询优化 |
| `postgres-patterns` | PG 最佳实践 | 数据集存储设计 |

### 项目管理

| Skill | 用途 | 你的场景 |
| ----- | ---- | -------- |
| `plan` | 功能规划和任务分解 | 规划 DL 项目实现步骤 |
| `architect` | 系统架构设计 | 设计 ROS2 + DL 系统架构 |
| `update-docs` | 文档自动更新 | 更新项目 README 和文档 |

---

## Skill 调用示例

### 1. 审查深度学习代码

```text
用户: 用 python-review 和 pytorch-patterns 审查 Class_9.py
```

Claude 的审查清单：

**PEP 8 / 代码规范：**

- 行长度是否超过 79 字符
- 命名是否符合 snake_case 规范
- import 顺序是否正确（标准库 → 第三方 → 本地）
- 是否有无用的 import

**类型注解：**

- 函数参数和返回值是否有类型注解
- 泛型类型是否正确使用

**PyTorch 最佳实践：**

- 推理时是否使用了 `model.eval()` + `torch.no_grad()`
- 设备管理是否一致（避免 CPU/GPU 混用）
- 是否正确处理了 `requires_grad`
- DataLoader 的 `num_workers` 和 `pin_memory` 设置是否合理
- 训练循环中 `optimizer.zero_grad()` 位置是否正确
- 是否正确使用 `loss.item()` 而非 `loss` 直接记录

**安全：**

- 模型加载是否使用了 `weights_only=True`（防止 pickle 注入）
- 是否有硬编码的 API 密钥
- 数据路径是否使用硬编码

### 2. 修复 ROS2 编译错误

```text
用户: colcon build 报错 "undefined reference to nav2_core::Controller"
```

Claude 的排查流程：

1. **符号定位：** 搜索 `nav2_core::Controller` 在哪个库中定义
2. **CMakeLists.txt 检查：** 确认 `target_link_libraries` 是否包含对应库
3. **package.xml 检查：** 确认 `<depend>` 声明完整
4. **ament 索引检查：** 确认库已被 ament 正确安装
5. **修复建议：** 给出具体的 CMakeLists.txt 和 package.xml 修改

### 3. 安全审计

```text
用户: 用 security-review 检查 ROS2 节点的网络通信
```

Claude 的审计清单：

**通信安全：**

- topic/service 是否启用了 SROS2 加密
- 是否有明文传输的敏感数据
- DDS 配置是否限制了通信域

**代码安全：**

- 是否有 `os.system()` / `subprocess` 的命令注入风险
- 是否有硬编码的密钥、密码、token
- YAML/XML 解析是否安全
- 第三方依赖是否有已知漏洞

**运行时安全：**

- 节点启动权限是否过高（不应以 root 运行）
- 文件路径访问是否存在目录遍历

### 4. 项目规划

```text
用户: 我想在 ROS2 里加个自定义避障算法，帮我规划一下
```

Claude 的规划输出：

**架构设计：**

- 新插件在 nav2 架构中的位置（Controller 层还是 Planner 层）
- 与现有 costmap、TF 的交互方式
- 参数服务器的配置结构

**文件清单：**

- `include/my_controller/my_controller.hpp` — 插件头文件
- `src/my_controller.cpp` — 插件实现
- `plugins.xml` — 插件注册
- `CMakeLists.txt` — 编译配置
- `config/my_controller_params.yaml` — 参数文件

**实现步骤：**

1. 继承 `nav2_core::Controller` 基类
2. 实现 `configure()`, `activate()`, `deactivate()`, `cleanup()` 生命周期
3. 实现 `computeVelocityCommands()` 核心算法
4. 动态参数回调
5. 单元测试 + 仿真验证

**数据流：**
`costmap → TF → my_controller → cmd_vel`

---

## 常用组合拳

### 开发流程

```text
# 新功能开发全流程
开始 → 用 plan 做规划
     → 写代码
     → 用 code-review 审查
     → 用 security-review 扫描
     → 用 build-fix 确保编译
     → 用 python-testing 补测试
     → 用 test-coverage 查覆盖率
     → 用 git-workflow 整理提交
     → 用 review-pr 最终审查
     → 合入
```

### 快速参考表

| 场景 | Skill 组合 | 说明 |
| ---- | ---------- | ---- |
| 写完新功能 | `code-review` → `security-review` → build | 先审质量 → 审安全 → 审编译 |
| 代码重构 | `code-simplifier` → `code-review` | 先简化 → 再审查 |
| 修复 bug | `build-fix` → `python-review` | 先修编译 → 再审质量 |
| 准备 PR | `git-workflow` → `review-pr` | 先整理提交 → 再自查 |
| 性能优化 | `performance-optimizer` → `code-review` | 先找瓶颈 → 再审视改动 |
| 数据库改动 | `database-reviewer` → `database-migrations` | 先审设计 → 再写迁移 |
| 安全敏感功能 | `security-review` → `silent-failure-hunter` → `code-review` | 三层安全防护 |
| 系统设计 | `plan` → `architect` → `code-architect` | 从抽象到具体 |
| 深度学习项目 | `pytorch-patterns` → `python-review` → `pytorch-build-resolver` | PyTorch 三段式 |
| ROS2 开发 | `cpp-review` → `security-review` → `cpp-build-resolver` | C++ 三段式 |
| 从零开始新模块 | `plan` → `architect` → 写代码 → `code-review` → `python-testing` → `test-coverage` | 全流程 |

---

## 进阶技巧

### 1. 给 Skill 提供充足的上下文

Skill 虽然智能，但需要你提供关键信息才能发挥最大效果：

```text
# ❌ 太模糊
用 code-review 审查一下

# ✅ 提供上下文
用 code-review 审查 Class_9.py，重点看训练循环的 GPU 内存管理，我用的是 RTX 4060 8GB
```

### 2. 组合 skill 时注意顺序

按依赖关系排列，先诊断后治疗：

```text
# ✅ 正确顺序：先审查、再修复
用 python-review 审查 → 根据建议修改 → 用 python-testing 补测试

# ❌ 错误顺序：先写测试、再审查出问题需要回退
用 python-testing 写测试 → 用 python-review 审查 → 发现接口设计问题 → 测试白写
```

### 3. 善用 plan 做前置思考

在写代码之前，用 `plan` 把思路理清，避免返工：

```text
用 plan 帮我规划：如何在 Class_9.py 中实现混合精度训练和梯度累积
```

### 4. 交叉审查提高质量

不同 skill 从不同角度审视同一段代码：

```text
# 三个视角审查同一个 API
用 python-review 检查代码规范
用 security-review 扫描安全风险
用 performance-optimizer 分析潜在瓶颈
```

### 5. 将 skill 嵌入你的工作习惯

```text
# 每天开始工作前
/git-workflow  # 检查分支状态，拉取最新代码

# 每次提交前
/code-review   # 快速自查修改的代码

# 每次 PR 前
/review-pr     # 模拟 PR 审查视角
```

---

## 自定义 Skill

除了内置的 ECC Skills，你还可以通过以下方式扩展：

### 方法 1: CLAUDE.md 项目指令

在项目根目录创建 `CLAUDE.md`，写入项目级别的指令：

```markdown
# 项目指令
- 审查 Python 代码时，额外检查是否符合 ROS2 命名规范
- 所有 topic 名称使用 snake_case
- 优先使用 rclcpp::Node 而非 rclpy
```

Claude 会在每次对话时自动加载这些指令，作为所有 skill 的补充规则。

### 方法 2: Memory 系统

用 `/remember` 保存长期偏好到 memory：

```text
记住：我的 ROS2 项目都在 /home/wheeltec/wheeltec_ros2/src/ 下
```

### 方法 3: 使用 skill-create 创建新 skill

```text
用 skill-create 创建一个 ROS2 包结构审查 skill
```

---

## 注意事项

1. **Skill 不是万能的** — 给出明确的上下文和具体要求，效果更好
2. **最终决定在你** — Skill 给出建议，你需要判断是否采纳
3. **可以不用 skill** — 简单任务直接让我处理就行，skill 主要用于复杂/专业场景
4. **环境已内置** — 这些 skill 已经安装好了，不需要额外安装依赖
5. **上下文窗口有限** — 同时调用过多 skill 可能影响性能，建议每次 2-3 个
6. **Skill 结果需验证** — Skill 给出的是建议，不是真理，尤其在 ROS2 这类复杂系统中
7. **不同语言用不同 skill** — Python 用 `python-review`，C++ 用 `cpp-review`，别用混

---

## 没有 Skill 但可替代的方案

以下场景没有专用 skill，但可以直接用代码/Python 库实现，告诉我需求即可。

---

### PPT / 演示文稿

没有 .pptx 专用 skill，但可以：

| 方案 | 适用场景 | 示例 |
| ---- | -------- | ---- |
| `python-pptx` 脚本 | 生成 .pptx 文件，图表+文字 | "用 python-pptx 给我做个项目汇报 PPT" |
| `ecc:frontend-slides` | 网页版演示（Reveal.js） | "做个网页版的技术分享幻灯片" |
| `ecc:remotion-video-creation` | 视频形式的技术演示 | "给这个模型做个演示视频" |

---

### Markdown 文档

直接用我写就行，不需要 skill：

- "把这段代码文档化"
- "生成 API 文档"
- "写个项目 README"

---

### PDF 报告

| 方案 | 适用场景 |
| ---- | -------- |
| `reportlab` | Python 程序化生成 PDF |
| Markdown → Pandoc 转换 | 技术文档转 PDF |
| Matplotlib / Plotly 图表 | 嵌入数据可视化图表 |

---

### Excel / 数据表格

| 方案 | 适用场景 |
| ---- | -------- |
| `openpyxl` | .xlsx 读写，公式、格式 |
| `pandas` | 数据处理后导出 Excel |
| `csv` 模块 | 简单 CSV 导出 |

---

### 图表 / 可视化

可用 skill `ecc:dashboard-builder`，或直接用 Matplotlib / Plotly 写代码。

---

### LaTeX 论文

没有专用 skill，但可以帮你写 LaTeX 模板、公式、图表代码。

---

### Docker / 部署

没有专用 skill，能用 `Dockerfile` / `docker-compose.yml` 编写和调试：

- "帮我给这个 ROS2 项目写 Dockerfile"
- "写个 docker-compose 启动仿真+导航"

---

### 常用 Python 工具库速查

| 需求 | 库 |
| ---- | -- |
| 生成 PPT | `python-pptx` |
| 生成 PDF | `reportlab` / `fpdf2` |
| 读写 Excel | `openpyxl` / `pandas` |
| 生成 Word | `python-docx` |
| 生成图表 | `matplotlib` / `plotly` |
| 操作图片 | `Pillow` / `opencv-python` |
| 发送邮件 | `smtplib` / `yagmail` |
| 爬网页 | `requests` + `beautifulsoup4` |
| Markdown→PDF | `pandoc` (CLI) |
