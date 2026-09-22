from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    model_path: Path
    data_path: Path
    task: str = "sft"
    max_attempts: int = 1
    high_risk_training: bool = False
    framework: str | None = None
    dry_run: bool = False
    preview_rows: int = 20
    output_root: Path = Path("outputs")
    tmp_root: Path = Path("tmp")
    progress_root: Path = Path("training_progress")
    model_provider: str = "openai_compatible"
    model_base_url: str | None = None
    model_name: str | None = None
    train_command: str | None = None
    train_timeout: int | None = None
    model_judge: bool = False
    checkpoint_path: Path | None = None
    require_checkpoint: bool = False
    resume: Path | None = None


def _bool(value: str) -> bool:
    value = value.lower().strip()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def parse_args(argv: list[str] | None = None) -> Settings:
    parser = argparse.ArgumentParser(description="Agentic training orchestrator")
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--data-path", required=True, type=Path)
    parser.add_argument("--task", choices=["sft", "rl"], default="sft")
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument("--high-risk-training", type=_bool, default=False)
    parser.add_argument("--framework")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preview-rows", type=int, default=20)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--tmp-root", type=Path, default=Path("tmp"))
    parser.add_argument("--progress-root", type=Path, default=Path("training_progress"))
    parser.add_argument("--model-provider", choices=["deepseek", "openai_compatible"],
                        default=os.getenv("MODEL_PROVIDER", "openai_compatible"))
    parser.add_argument("--model-base-url", default=os.getenv("MODEL_BASE_URL"))
    parser.add_argument("--model-name", default=os.getenv("MODEL_NAME"))
    parser.add_argument("--train-command", help="explicit local training command; parsed without a shell")
    parser.add_argument("--train-timeout", type=int)
    parser.add_argument("--model-judge", action="store_true", help="send the redacted preview to the configured model for a structured data decision")
    parser.add_argument("--checkpoint-path", type=Path, help="path to the expected final checkpoint")
    parser.add_argument("--require-checkpoint", action="store_true", help="fail if checkpoint-path is absent after a zero exit")
    parser.add_argument("--resume", type=Path, help="resume a previous run directory after verifying its input manifest")
    ns = parser.parse_args(argv)
    if ns.max_attempts < 1 or ns.preview_rows < 1:
        parser.error("--max-attempts and --preview-rows must be positive")
    if ns.train_timeout is not None and ns.train_timeout < 1:
        parser.error("--train-timeout must be positive")
    return Settings(**vars(ns))
