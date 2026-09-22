# Training Agent

这是一个本地训练编排 MVP。当前版本先把数据验证、可审计规范化、风险审批和本地命令执行做成可靠闭环；框架 registry、自动下载和基于训练指标的多轮调整仍需在此基础上接入。

## 运行 dry-run

从仓库外层目录执行：

```bash
PYTHONPATH="$PWD" python -m training_agent.cli \
  --model-path ./training_agent/origin_model/model \
  --data-path ./training_agent/data/train.jsonl \
  --task sft \
  --dry-run
```

dry-run 会创建时间戳运行目录，执行前 N 行预检、必要的全量扫描、JSONL 规范化、转换后验证、manifest/hash 和报告，但不会启动训练。

加上 `--model-judge` 后，前 N 行预览会发送给 `.env` 中配置的 DeepSeek 或 OpenAI 兼容模型，模型只能返回结构化的数据判断；实际转换仍由本地白名单转换器完成。未配置模型时默认使用确定性检查器，不会发起网络请求。

## 执行本地训练命令

训练命令通过 `shlex` 解析为参数数组，不经过 shell。`--high-risk-training=false`（默认值）会在启动前询问；自动化环境可显式传 `true`。

```bash
PYTHONPATH="$PWD" python -m training_agent.cli \
  --model-path ./training_agent/origin_model/model \
  --data-path ./training_agent/data/train.jsonl \
  --train-command "python train.py --model ./training_agent/origin_model/model" \
  --max-attempts 2 \
  --train-timeout 86400 \
  --checkpoint-path ./outputs/final \
  --require-checkpoint \
  --high-risk-training=false
```

没有 `--train-command` 时，非 dry-run 会失败并生成失败报告，不会假装训练成功。训练 stdout/stderr 写入 `training_logs/training_attempts_n/raw_log/terminal.log`，可识别的 loss、epoch、step 和学习率写入 `concise_logs/metrics.jsonl`。
命令退出码为 0 但没有可解析 loss 时，运行状态为 `COMPLETED_UNVERIFIED`；需要 checkpoint 时用 `--require-checkpoint` 强制校验输出路径。

当前 registry 内置 `local` 适配器，表示用户显式提供的本地命令：

```bash
--framework local --train-command "python train.py ..."
```

后续接入具体训练框架时，应在 `frame/registry.py` 增加版本固定的适配器，而不是把命令直接拼接到 CLI 中。

## 数据约束

SFT 输入支持 JSONL、JSON 和 CSV 的读取；JSONL 规范化当前支持：

- `messages` 对话格式；
- `instruction` + `output`；
- `instruction` + `answer`（自动映射为 output）。

缺少答案、非法 role、空数据和解析错误会阻止训练。原始数据不会覆盖，规范化数据写入本轮 `tmp`，并在 manifest 中记录 SHA-256。
