from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

from .config import parse_args
from .context import RunContext
from .agent.data_judge import judge
from .agent.model.deepseek import DeepSeekClient
from .agent.model.openai_compatible import OpenAICompatibleClient
from .data_pipeline.inspector import inspect, full_scan
from .data_pipeline.schema import DataContract
from .data_pipeline.transformer import transform_any
from .hashing import sha256_file
from .report import write_report
from .frame.registry import build_command, select as select_framework
from .frame.downloader import ensure_installed
from .frame.adjustments import propose
from .tools.approval import approve
from .tools.command import run_command
from .tools.metrics import analyze_metrics


def _judge_data(settings, preview: dict) -> dict:
    if not settings.model_judge:
        return judge(preview)
    client_cls = DeepSeekClient if settings.model_provider == "deepseek" else OpenAICompatibleClient
    client = client_cls(base_url=settings.model_base_url, model=settings.model_name)
    return judge(preview, client)


def _execute_training(context: RunContext, settings) -> int:
    framework = select_framework(settings.framework, settings.task, settings.train_command, settings.registry_path)
    framework_dir = ensure_installed(framework, settings.framework_root, settings.high_risk_training, approve)
    manifest = {}
    manifest_path = context.output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data_for_training = manifest.get("normalized_data_path", str(settings.data_path.resolve()))
    parameters = dict(settings.training_params)
    context.write_manifest(framework=framework.name, framework_version=framework.version, framework_dir=str(framework_dir.resolve()), training_params=parameters)
    context.transition("FRAMEWORK_READY")
    attempts: list[dict[str, object]] = []
    previous_summary = context.output_dir / "report" / "summary.json"
    if previous_summary.exists():
        try:
            previous = json.loads(previous_summary.read_text(encoding="utf-8"))
            if isinstance(previous.get("attempts"), list):
                attempts.extend(item for item in previous["attempts"] if isinstance(item, dict))
        except json.JSONDecodeError:
            context.agent_log.write("error", "could not read previous report attempts during resume")
    exit_code = 0
    start_attempt = context.attempt + 1
    if start_attempt > settings.max_attempts:
        raise ValueError("resume run already exhausted --max-attempts")
    for number in range(start_attempt, settings.max_attempts + 1):
        context.attempt = number
        context.persist()
        raw, _ = context.attempt_dirs(number)
        values = {
            "model_path": settings.model_path.resolve(),
            "data_path": data_for_training,
            "output_dir": (context.progress_dir / f"training_attempts_{number}").resolve(),
            "run_dir": context.output_dir.resolve(),
            "framework_dir": framework_dir.resolve(),
            "attempt": number,
        }
        command = build_command(framework, settings.train_command, values, parameters)
        if not approve("启动训练命令: " + " ".join(shlex.quote(arg) for arg in command), settings.high_risk_training):
            context.agent_log.write("response", "training approval rejected", attempt=number)
            context.transition("STOPPED")
            write_report(context.output_dir, status="STOPPED", reason="training approval rejected", attempts=attempts)
            return 2
        context.transition("RUNNING")
        raw_log = raw / "terminal.log"
        returncode = run_command(command, Path.cwd(), raw_log, settings.train_timeout)
        concise_log = raw.parent / "concise_logs" / "metrics.jsonl"
        analysis = analyze_metrics(raw_log, concise_log, returncode)
        checkpoint_exists = bool(settings.checkpoint_path and settings.checkpoint_path.exists())
        analysis["checkpoint_exists"] = checkpoint_exists
        item = {"attempt": number, "returncode": returncode, "raw_log": str(raw_log), "concise_log": str(concise_log), "metric_count": analysis["metric_count"], "analysis": analysis}
        attempts.append(item)
        context.agent_log.write("command", "training command finished", attempt=number, command=command, returncode=returncode, analysis=analysis)
        if returncode == 0 and settings.require_checkpoint and not checkpoint_exists:
            context.transition("FAILED")
            write_report(context.output_dir, status="FAILED", reason="training exited successfully but checkpoint verification failed", attempts=attempts)
            return 1
        if returncode == 0 and settings.adaptive_training and settings.target_loss is not None and analysis.get("last_loss") is not None and float(analysis["last_loss"]) > settings.target_loss:
            if number < settings.max_attempts:
                adjustment = propose(analysis, parameters, framework.parameter_flags, settings.target_loss)
                adjustment_path = context.output_dir / "adjustments" / f"adjustment_{number}.json"
                adjustment_path.parent.mkdir(parents=True, exist_ok=True)
                adjustment_path.write_text(json.dumps({"attempt": number, "applied": adjustment.applied, "changes": adjustment.changes, "reason": adjustment.reason, "evidence": adjustment.evidence}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
                context.agent_log.write("action", "metric target adjustment proposed", attempt=number, applied=adjustment.applied, changes=adjustment.changes, reason=adjustment.reason)
                if adjustment.applied:
                    parameters.update(adjustment.changes)
                    context.transition("ADJUSTING")
                    continue
            context.transition("FAILED")
            write_report(context.output_dir, status="FAILED", reason=f"observed loss did not meet target {settings.target_loss}", attempts=attempts)
            return 1
        if returncode == 0 and analysis["outcome"] == "unverified":
            context.transition("COMPLETED_UNVERIFIED")
            write_report(context.output_dir, status="COMPLETED_UNVERIFIED", reason="command exited successfully but metrics were insufficient to verify training", attempts=attempts)
            return 2
        if returncode == 0:
            context.transition("COMPLETED")
            write_report(context.output_dir, status="COMPLETED", reason="training command exited successfully", attempts=attempts)
            return 0
        if number < settings.max_attempts and settings.adaptive_training:
            adjustment = propose(analysis, parameters, framework.parameter_flags, settings.target_loss)
            adjustment_path = context.output_dir / "adjustments" / f"adjustment_{number}.json"
            adjustment_path.parent.mkdir(parents=True, exist_ok=True)
            adjustment_path.write_text(json.dumps({"attempt": number, "applied": adjustment.applied, "changes": adjustment.changes, "reason": adjustment.reason, "evidence": adjustment.evidence}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            context.agent_log.write("action", "training adjustment proposed", attempt=number, applied=adjustment.applied, changes=adjustment.changes, reason=adjustment.reason)
            if adjustment.applied:
                parameters.update(adjustment.changes)
                context.transition("ADJUSTING")
                continue
            context.transition("FAILED")
            write_report(context.output_dir, status="FAILED", reason=f"no safe adaptive adjustment: {adjustment.reason}", attempts=attempts)
            return 1
        if number < settings.max_attempts:
            context.transition("FAILED")
            write_report(context.output_dir, status="FAILED", reason="training failed and adaptive training was disabled", attempts=attempts)
            return 1
        if number == settings.max_attempts:
            context.transition("FAILED")
            write_report(context.output_dir, status="FAILED", reason="training command failed and attempts were exhausted", attempts=attempts)
            exit_code = 1
    return exit_code


def main(argv: list[str] | None = None) -> int:
    settings = parse_args(argv)
    if not settings.data_path.exists():
        raise FileNotFoundError(settings.data_path)
    if not settings.model_path.exists():
        raise FileNotFoundError(settings.model_path)
    if settings.resume:
        context = RunContext.resume(settings)
        manifest_path = context.output_dir / "manifest.json"
        if not manifest_path.exists():
            raise ValueError("resume run has no manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("data_sha256") != sha256_file(settings.data_path):
            raise ValueError("resume input data hash does not match the original run")
        if context.state in {"COMPLETED", "STOPPED"}:
            raise ValueError(f"run is not resumable from state {context.state}")
        exit_code = _execute_training(context, settings)
        print(f"run_id={context.run_id}")
        print(f"output_dir={context.output_dir}")
        return exit_code
    context = RunContext(settings)
    context.write_manifest(data_path=str(settings.data_path.resolve()), data_sha256=sha256_file(settings.data_path), model_path=str(settings.model_path.resolve()), task=settings.task)
    contract = DataContract(task=settings.task)
    context.transition("DATA_PREVIEW")
    preview = inspect(settings.data_path, contract, settings.preview_rows)
    (context.output_dir / "data_preview.json").write_text(json.dumps(preview, ensure_ascii=False, indent=2), encoding="utf-8")
    context.agent_log.write("response", "data preview complete", preview={key: value for key, value in preview.items() if key != "samples"})
    judge_result = _judge_data(settings, preview)
    (context.output_dir / "data_judgement.json").write_text(json.dumps(judge_result, ensure_ascii=False, indent=2), encoding="utf-8")
    context.agent_log.write("response", "data judgement complete", judgement=judge_result)
    scan = None
    if preview["full_scan_required"]:
        scan = full_scan(settings.data_path, contract)
        (context.output_dir / "data_full_scan.json").write_text(json.dumps(scan, ensure_ascii=False, indent=2), encoding="utf-8")
        context.agent_log.write("response", "full data scan complete", scan=scan)
    if preview["status"] == "reject" or judge_result["status"] == "reject" or (scan and scan.get("error_count", 0) > 0):
        context.transition("FAILED")
        reason = "data preview rejected the input" if preview["status"] == "reject" else "full data scan found invalid records"
        write_report(context.output_dir, status="FAILED", reason=reason, attempts=[])
        raise ValueError(f"{reason}; see data_preview.json/data_full_scan.json")
    context.transition("DATA_VALIDATED")
    if settings.data_path.suffix.lower() in {".jsonl", ".json", ".csv"}:
        normalized = context.tmp_dir / "data" / "normalized.jsonl"
        try:
            result = transform_any(settings.data_path, normalized)
        except (ValueError, json.JSONDecodeError) as exc:
            context.transition("FAILED")
            write_report(context.output_dir, status="FAILED", reason=f"data normalization failed: {exc}", attempts=[])
            raise
        (context.output_dir / "data_transform.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        context.agent_log.write("action", "normalized data", transform=result)
        context.write_manifest(normalized_data_path=str(normalized.resolve()), normalized_data_sha256=sha256_file(normalized))
        normalized_scan = full_scan(normalized, contract)
        (context.output_dir / "normalized_full_scan.json").write_text(json.dumps(normalized_scan, ensure_ascii=False, indent=2), encoding="utf-8")
        if normalized_scan.get("error_count", 0) > 0:
            context.transition("FAILED")
            write_report(context.output_dir, status="FAILED", reason="normalized data failed validation", attempts=[])
            raise ValueError("normalized data failed validation; see normalized_full_scan.json")
    if settings.dry_run:
        if settings.framework or settings.train_command:
            framework = select_framework(settings.framework, settings.task, settings.train_command, settings.registry_path)
            context.write_manifest(framework=framework.name, framework_version=framework.version)
        context.transition("COMPLETED")
        write_report(context.output_dir, status="COMPLETED", reason="dry-run completed; no training was started", attempts=[])
    else:
        exit_code = _execute_training(context, settings)
    print(f"run_id={context.run_id}")
    print(f"output_dir={context.output_dir}")
    return exit_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"training-agent error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
