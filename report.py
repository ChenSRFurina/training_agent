from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def write_report(output_dir: Path, *, status: str, reason: str, attempts: list[dict[str, Any]]) -> None:
    report_dir = output_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    metrics: list[dict[str, Any]] = []
    for item in attempts:
        concise = item.get("concise_log")
        if concise and Path(str(concise)).exists():
            with Path(str(concise)).open(encoding="utf-8") as stream:
                metrics.extend(json.loads(line) for line in stream if line.strip())
    losses = [float(item["loss"]) for item in metrics if isinstance(item.get("loss"), (int, float)) and math.isfinite(float(item["loss"]))]
    epochs = [float(item["epoch"]) for item in metrics if isinstance(item.get("epoch"), (int, float)) and math.isfinite(float(item["epoch"]))]
    payload = {"status": status, "reason": reason, "attempts": attempts, "metrics": {"best_loss": min(losses) if losses else None, "last_epoch": max(epochs) if epochs else None, "metric_events": len(metrics)}}
    (report_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Agentic Training Report", "", "####### 代码或配置变更优先报告", "", "本次运行使用当前工作区代码，训练执行状态见下方。", "", f"- 状态：`{status}`", f"- 原因：{reason}", f"- 尝试次数：{len(attempts)}", f"- 最佳 loss：`{min(losses) if losses else 'unavailable'}`", f"- 最后 epoch：`{max(epochs) if epochs else 'unavailable'}`", ""]
    for item in attempts:
        lines.extend([f"## training_attempts_{item['attempt']}", "", f"- 退出码：`{item['returncode']}`", f"- 原始日志：`{item['raw_log']}`", ""])
    adjustment_dir = output_dir / "adjustments"
    if adjustment_dir.exists():
        lines.extend(["## 调整记录", ""])
        for path in sorted(adjustment_dir.glob("adjustment_*.json")):
            try:
                adjustment = json.loads(path.read_text(encoding="utf-8"))
                lines.append(f"- {path.name}: {adjustment.get('reason')}；changes={adjustment.get('changes', {})}")
            except json.JSONDecodeError:
                lines.append(f"- {path.name}: 无法解析")
        lines.append("")
    (report_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
