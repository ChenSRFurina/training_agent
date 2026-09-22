# 自动 SFT/RL 训练 Agent 实施方案

## 1. 目标与范围

本项目用于编排一次完整的 `agentic_training`：读取训练要求和数据，选择合适的本地命令行训练框架，准备并执行训练，分析训练指标，根据分析结果调整配置并重复训练，最后生成可追溯的报告。

第一阶段支持 SFT，并为 RL 保留任务模型和执行器扩展点。第一阶段不把某个训练框架写死，框架以可注册、可下载、可执行的适配器形式接入。

## 2. 目录与职责

```text
training_agent/
├── agent/
│   ├── model/
│   │   ├── role.md              # Agent 角色、决策边界和流程约束
│   │   ├── base.py              # 模型客户端协议
│   │   ├── openai_compatible.py  # OpenAI Chat Completions 兼容客户端
│   │   └── deepseek.py           # DeepSeek 默认配置封装
│   ├── planner/                  # 训练计划、框架选择和调整策略
│   ├── runner/                   # agentic_training 主循环
│   └── reporter/                 # 日志汇总和最终报告
├── tools/
│   ├── command.py                # 受控命令执行、超时、退出码和输出捕获
│   ├── download.py               # 框架/模型/数据下载
│   ├── search.py                 # 框架资料和适配性搜索接口
│   ├── approval.py               # high-risk-training 风险询问
│   └── hashing.py                # 文件和配置 SHA-256 指纹
├── data_pipeline/
│   ├── inspector.py               # 前 N 行预检和全量统计
│   ├── schema.py                  # 数据格式、任务字段和规范化模型
│   ├── transformer.py             # Agent 提议的可审计数据修改
│   ├── validators.py              # 结构、内容、重复和泄漏检查
│   └── sampling.py                # 稳定采样、脱敏和样本预览
├── data/                         # 输入数据、数据清洗结果和数据说明
├── frame/                        # 已登记或已下载的训练框架
│   ├── registry.yaml             # 框架名称、版本、来源、命令模板和能力
│   └── <framework>/              # 框架源码或本地安装目录
├── origin_model/                 # 初始模型（只读来源目录）
├── outputs/<YYYY_MM_DD_HH-mm>/
│   ├── agent_logs/               # 单文件增量记录 Agent 响应和 action
│   ├── training_logs/
│   │   └── training_attempts_n/
│   │       ├── raw_log/           # 全量终端输出
│   │       └── concise_logs/      # loss、epoch、学习率等结构化指标
│   └── report/                   # 本轮完整报告和必要的结构化摘要
├── tmp/<YYYY_MM_DD_HH-mm>/       # 临时文件，成功或失败收尾时清理
├── training_progress/<YYYY_MM_DD_HH-mm>/
│   └── training_attempts_n/       # checkpoint 和训练产物
├── configs/                      # 默认配置和框架配置模板
├── tests/                        # 单元测试和小规模 dry-run
├── .env.example                  # API 地址、密钥变量名和默认模型示例
└── cli.py                        # 命令行入口
```

每次运行只生成一个顶层时间戳目录，格式固定为 `YYYY_MM_DD_HH-mm`。`training_attempts_n` 从 1 开始，仅在真正启动一次训练命令时递增；规划失败、审批拒绝或 dry-run 不算训练尝试。

## 3. 配置与命令行接口

入口建议为：

```bash
python -m training_agent.cli \
  --model-path ./origin_model/base \
  --data-path ./data/train.jsonl \
  --task sft \
  --max-attempts 3 \
  --high-risk-training=false
```

核心参数：

- `--model-path`：初始模型路径，默认从 `origin_model/` 选择。
- `--data-path`：训练数据路径或数据配置文件。
- `--task`：`sft` 或未来的 `rl`。
- `--max-attempts`：最大训练尝试次数，必须为正整数。
- `--high-risk-training`：布尔值，默认 `false`。为 `false` 时，外部下载和启动训练命令前必须询问；为 `true` 时自动继续。
- `--framework`：可选，强制使用已登记框架；未提供时由 Agent 选择。
- `--dry-run`：只生成计划和命令，不下载、不启动训练。
- `--resume <run-dir>`：从已有运行目录恢复，继续未完成的 attempt。

环境变量只保存凭据和服务默认值，不写入日志：

```text
MODEL_PROVIDER=deepseek|openai_compatible
MODEL_API_KEY=...
MODEL_BASE_URL=https://api.deepseek.com/v1
MODEL_NAME=...
MODEL_TIMEOUT_SECONDS=120
```

`DeepSeek` 和其他 OpenAI 兼容服务都实现同一个 `ModelClient` 协议（消息输入、结构化输出、重试、超时）。DeepSeek 只提供默认 `base_url` 和模型名，不在业务逻辑中分叉调用方式。

## 4. 运行流程

1. **初始化运行目录**：生成时间戳，写入运行元数据、参数、代码版本和环境摘要。
2. **准备数据**：先读取前 `N` 行（默认 20，可由配置覆盖）做低成本预检；Agent 根据任务要求和样本判断格式是否匹配、字段是否完整、输入输出是否有意义。若预检发现疑点，再执行全量检查并生成可审计的修改方案；批准后写出规范化数据。产物写入 `data/` 或本轮 `tmp/`，并在报告中记录原始/修改后指纹和数据统计。
3. **选择框架**：读取 `frame/registry.yaml`，按任务类型、模型架构、硬件和数据格式筛选；必要时调用搜索工具补充候选。选择结果必须包含理由、版本、来源和预计命令。
4. **下载或准备框架**：框架不在本地时，先检查来源和版本，再走风险审批；批准后下载并校验，拒绝则记录并终止本轮或切换已有候选。
5. **生成训练计划**：生成不可变的配置快照，包括模型、数据、超参数、设备、输出目录和命令。命令使用参数数组执行，避免 shell 拼接注入。
6. **启动 attempt**：分配 `training_attempts_n`，将命令、环境摘要和开始时间写入 `agent_logs`；在 `high-risk-training=false` 时，启动训练前再次询问。
7. **实时采集日志**：终端 stdout/stderr 增量写入 `raw_log`；解析 loss、epoch、step、学习率、评估指标和 checkpoint 到 `concise_logs`。原始日志保留完整内容，解析失败的行不得丢弃。
8. **训练后分析**：检查退出码、最后 checkpoint、loss 趋势、过拟合迹象、NaN/梯度异常、资源错误和指标是否达到目标。
9. **调整并循环**：Agent 根据分析提出调整（例如学习率、batch size、epoch、数据比例或框架参数），保存“观察→判断→修改→预期影响”。若达到目标、无可行调整、失败次数耗尽或用户停止，则结束循环；否则回到第 5 步。
10. **生成报告和收尾**：在 `report/` 写入完整 Markdown 报告及机器可读 JSON 摘要，清理本轮 `tmp/`，保留 checkpoints、日志和报告。

## 5. 风险控制

风险判断集中在 `tools/approval.py`，不能由模型回复自行绕过：

- `--high-risk-training=false`：下载外部框架/依赖、下载模型或数据、启动实际训练命令前询问；询问内容包含来源、版本、命令、预计输出目录和风险。拒绝必须写入日志并停止对应动作。
- `--high-risk-training=true`：上述动作自动执行，但仍记录同样的审计信息。
- 本地只读检查、配置生成、日志解析和报告生成无需询问。
- 命令执行器设置超时、工作目录、环境白名单和退出码处理；禁止把未验证的 Agent 文本直接当作 shell 命令。

## 6. 日志和报告规范

`agent_logs/agent.log` 采用增量追加，每条记录至少含：`training_attempts_n`、ISO 时间、事件类型（`action`/`response`/`command`/`error`）、摘要、输入输出引用和退出码。敏感变量统一脱敏。

每个 attempt 的 `raw_log/terminal.log` 保留带时间戳的完整 stdout/stderr；`concise_logs/metrics.jsonl` 每行一个指标事件，字段包括 `step`、`epoch`、`loss`、`eval_loss`、`learning_rate`、`timestamp` 和来源行号。

`report/report.md` 的顺序固定为：

1. **代码或配置变更优先报告**：用醒目的 `#######` 标记，列出文件、变更、原因和影响。
2. 本轮目标、模型、数据和框架版本。
3. 尝试次数、每次训练的起止时间、平均 epoch 和关键指标。
4. 每次 attempt 的配置、日志结论、问题和调整理由。
5. 最终 checkpoint、输出路径、复现命令和已知限制。

## 7. 实现顺序

1. 建立目录、配置模型、时间戳运行上下文和安全日志组件。
2. 实现统一模型客户端，接入 DeepSeek 与任意 OpenAI 兼容端点，并补充脱敏、超时和重试。
3. 实现命令执行器、下载器、审批器和框架 registry；先登记一个可用的本地命令行框架适配器。
4. 实现数据检查、训练计划和单次 attempt runner，完成 raw/concise 日志。
5. 实现指标分析、调整策略、最大尝试次数和 resume。
6. 实现报告生成、tmp 清理和完整 dry-run。
7. 使用模拟命令和小数据集测试全流程，再进行一次真实训练验证。

## 8. 验收标准

- 两类 API 只需切换配置即可调用，业务代码无需改动。
- `--high-risk-training=false` 能在每个外部下载和训练启动点阻断并记录拒绝；`true` 能无交互执行。
- 一次运行的所有日志、配置、checkpoint 和报告都位于同一时间戳目录，且 attempt 编号连续。
- 训练失败、指标解析失败或 Agent 调整失败时仍保留原始日志，并生成可读报告。
- 达到目标或达到 `--max-attempts` 后能确定性结束，临时目录可清理，运行可通过 `--resume` 恢复。
- dry-run 不产生外部副作用，只输出候选框架、训练命令和风险点。

## 9. 数据预检、判断与自动修改

### 9.1 数据契约

运行开始时由 CLI 参数和任务配置形成 `DataContract`。它至少包含：任务类型（SFT/RL）、输入文件格式（JSONL/JSON/CSV/Parquet）、必需字段、字段类型、最大长度、语言或领域要求、是否需要 system/user/assistant 消息结构、验证集比例、去重规则和允许的修改范围。没有显式契约时，SFT 默认接受以下两种输入：

```json
{"instruction": "...", "input": "...", "output": "..."}
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Agent 可以补全 `input` 为空的 instruction 样本，但不能凭空生成答案；不得改变事实内容、标签含义或用户没有授权的字段。

### 9.2 前 N 行预检

`data_pipeline/inspector.py` 负责以流式方式读取前 N 条有效记录，默认 `N=20`。预检必须做到：不把整个大文件载入内存、保留物理行号、捕获解析错误，并将样本脱敏后写入 `agent_logs`。输出 `data_preview.json`，包括：

- 文件类型、编码、总大小、前 N 行成功/失败数和每行错误；
- 推断出的字段集合、类型、嵌套层级和消息 role 分布；
- 空值比例、过长样本、疑似重复、异常 Unicode、对话未闭合和输入/输出长度分布；
- 样本 SHA-256、预检配置、检查器版本和时间戳。

预检只负责发现问题，不直接覆盖原文件。Agent 通过模型客户端得到严格 JSON 结果，格式固定为：

```json
{
  "status": "pass|needs_transform|reject|needs_full_scan",
  "detected_format": "jsonl_messages",
  "issues": [{"code": "missing_output", "severity": "error", "rows": [3]}],
  "transformations": [{"type": "rename_field", "from": "answer", "to": "output", "reason": "..."}],
  "full_scan_required": true,
  "reason": "..."
}
```

模型输出必须通过 JSON Schema 校验；解析失败时使用确定性的规则检查器，不执行模型提出的任意代码或命令。`reject` 表示无法安全修复或缺少训练目标；`needs_full_scan` 表示样本不足以判断，必须进入全量检查。

### 9.3 全量检查和修改计划

只要出现错误级问题、字段推断不一致、预检样本不足或 Agent 明确要求，就执行 `validators.py` 的流式全量扫描。全量扫描写入 `data_full_scan.json`，至少统计总记录、有效/无效记录、字段缺失、类型冲突、重复指纹、token/字符长度分位数、训练验证泄漏候选和类别分布。

`transformer.py` 只允许执行白名单操作：字段重命名/映射、缺省字段填充为空字符串、消息格式转换、首尾空白清理、非法记录隔离、明确规则下的去重、长度截断（必须记录截断位置）和稳定划分 train/validation。所有操作组成版本化 `transform_plan.json`，每个操作包含输入指纹、规则、影响行数、前后示例摘要和风险级别。

修改策略：

1. 可确定且不改变语义的结构转换可自动执行，例如 `answer → output`、单轮 instruction 转 messages。
2. 会丢失内容、推断答案、改变标签、批量截断或删除记录的操作默认暂停并请求用户确认；`--high-risk-training=true` 只跳过外部下载/启动训练询问，不跳过数据语义修改确认。
3. 原始文件永不覆盖。修改结果写入 `tmp/<run>/data/normalized/`，验证通过后复制到 `outputs/<run>/data/` 或配置的持久化目录。
4. 修改后重新运行同一套检查器；若仍有错误级问题，禁止进入训练。

### 9.4 数据版本和回滚

每个版本包含 `manifest.json`：原始路径、原始 SHA-256、修改后 SHA-256、契约、检查器/转换器版本、变更计数和父版本。训练计划只引用已验证版本的绝对路径和指纹。恢复运行时先校验指纹；不一致则停止并要求重新预检。任何转换都可通过 manifest 的父版本回滚。

## 10. 各模块的实现规格

### 10.1 运行上下文和状态机

`runner/context.py` 创建 `RunContext`：`run_id`、`run_dir`、`tmp_dir`、当前状态、attempt 编号、配置快照和取消事件。状态只允许按以下图转换：

```text
INIT → DATA_PREVIEW → DATA_VALIDATED → FRAMEWORK_READY → PLAN_READY
PLAN_READY → WAITING_APPROVAL → RUNNING → ANALYZING
ANALYZING → ADJUSTING → PLAN_READY
ANALYZING → COMPLETED | FAILED | STOPPED
```

每次状态变化先追加日志，再原子写入 `state.json`；进程中断后通过 state 和 manifest 恢复，不依赖内存。

### 10.2 模型客户端

`ModelClient` 提供 `complete(messages, response_schema, temperature, timeout)`。实现层负责 API key、base URL、重试退避、请求 ID、限时和 JSON 输出校验；业务层只依赖协议。提示词分为系统约束、任务契约、脱敏数据预览和输出 schema，禁止把密钥、完整训练数据或未脱敏日志发送到模型。

### 10.3 框架 registry 和适配器

`frame/registry.yaml` 每项包含名称、版本、来源 URL、许可证、支持任务/模型、安装检查命令、训练命令模板、指标解析器、checkpoint 目录规则和硬件要求。适配器接口至少包括 `detect()`, `validate_config()`, `build_command()`, `parse_metrics()`, `find_checkpoint()`。命令模板渲染后必须经过参数白名单和路径校验。

### 10.4 命令、下载和审批

命令执行使用 `subprocess.Popen` 的参数数组，不启用 shell；stdout/stderr 分流写入日志，支持超时、SIGTERM 后 SIGKILL、取消和退出码。下载器支持固定版本、临时目录下载、校验和、原子移动；禁止无版本的 latest。审批器返回 `approved/rejected`，并记录用户选择、动作摘要和时间。

### 10.5 训练分析和调整

指标解析器输出统一事件模型；分析器计算初始/最终 loss、最佳 eval 指标、epoch、吞吐、峰值显存和异常窗口。调整器只能从配置白名单选择变更，限制单次变化幅度并避免重复尝试；每次输出 `adjustment_n.json`，包括证据、假设、改动、预期和回滚值。没有可靠指标时不得自动声称训练成功。

### 10.6 报告器

报告器读取 manifest、state、所有 attempt 指标和 agent log 生成 Markdown/JSON，不直接依赖模型记忆。报告必须区分事实、规则判断和模型建议；任何未执行的建议标为 `not_applied`。

## 11. 代码执行模型的逐步实现清单

实现模型应按以下顺序提交小步变更，每步运行对应测试后再继续：

1. 创建包结构、配置类、`RunContext`、时间戳工具和 manifest/hash 工具。
2. 创建统一事件日志、脱敏器和原子文件写入工具。
3. 实现 `DataContract`、JSONL/JSON/CSV 读取器、前 N 行预检和确定性 validators。
4. 实现模型客户端及结构化响应校验；用 mock client 测试缺字段、非法 JSON 和超时。
5. 实现数据转换白名单、transform plan、前后校验、版本 manifest 和回滚。
6. 实现 registry、适配器协议、参数数组命令构造和 dry-run。
7. 实现下载/审批/命令执行器，测试拒绝、取消、超时、非零退出码和校验失败。
8. 实现单次 attempt、raw/concise 日志和 checkpoint 发现。
9. 实现分析、调整、最大尝试次数、状态持久化和 resume。
10. 实现报告、tmp 清理和端到端模拟训练；最后再接入真实框架和小数据集。

每步不得把未实现的功能伪装成成功；遇到不可恢复错误要保留现场、写明状态并返回非零退出码。

## 12. 新增验收标准

- 仅凭前 N 行即可识别常见 SFT 格式；判断不确定时会自动升级到全量扫描。
- 数据有缺字段、字段别名或消息格式差异时，Agent 能提出结构化修改计划；不能安全修复时阻止训练。
- 原始数据永不覆盖，修改前后均有 SHA-256、manifest、影响行数和可回滚版本。
- 数据语义修改、删除、截断和答案生成与外部下载/启动训练的风险控制分开，不能因 `--high-risk-training=true` 自动放行语义风险操作。
- 修改后的数据必须重新验证通过才可进入框架选择和训练计划。
- 断电或进程中止后，`--resume` 能从最后一个持久化状态继续，且不会重复执行已完成的转换或训练 attempt。

## 13. 当前代码实现状态

截至当前版本，以下 MVP 能力已经落地：

- SFT 数据契约、JSONL/JSON/CSV 流式预检和全量扫描；错误数量和截断状态分开统计。
- `instruction`/`answer`/`output` 到 `messages` 的确定性结构转换；转换前后验证、源数据保留、原子写出和 SHA-256。
- DeepSeek/OpenAI 兼容模型客户端、可选的结构化数据判断、模型建议操作白名单和本地确定性回退。
- 独立进程组命令执行、实时 raw log、超时和子进程树清理。
- loss、eval loss、epoch、step 和 learning rate 的结构化解析，以及 `observed`/`unverified`/`failed` 结果区分。
- `local` 框架登记、风险审批、运行 manifest、训练报告和 13 个回归测试。

以下能力仍明确未完成，不能在当前版本中声称已支持：具体训练框架的自动下载与适配、模型驱动的超参数调整、checkpoint 自动发现与恢复、`--resume` 断点续跑、RL 数据契约和多轮实验比较。
