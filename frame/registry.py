from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from typing import Any


@dataclass(frozen=True)
class FrameworkSpec:
    name: str
    version: str
    tasks: tuple[str, ...]
    source: str = "local"
    sha256: str | None = None
    install_dir: str | None = None
    command_template: tuple[str, ...] = ()
    parameter_flags: dict[str, str] = field(default_factory=dict)

    @property
    def is_remote(self) -> bool:
        return self.source.startswith(("http://", "https://"))


BUILTIN: dict[str, FrameworkSpec] = {
    "local": FrameworkSpec(
        "local", "user-supplied", ("sft", "rl"),
        source="local", command_template=(),
    ),
}


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError as exc:
        stripped = "\n".join(line.split("#", 1)[0].strip() for line in path.read_text(encoding="utf-8").splitlines() if line.split("#", 1)[0].strip())
        if stripped in {"", "frameworks: []"}:
            return {}
        raise RuntimeError("registry.yaml requires PyYAML; install it or use the built-in local framework") from exc
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def load_registry(path: Path | None = None) -> dict[str, FrameworkSpec]:
    result = dict(BUILTIN)
    if path is None:
        path = Path(__file__).with_name("registry.yaml")
    if not path.exists():
        return result
    data = _load_yaml(path)
    for raw in data.get("frameworks", []):
        if not isinstance(raw, dict) or not raw.get("name"):
            raise ValueError("each registry framework needs a name")
        name = str(raw["name"])
        tasks = tuple(str(task) for task in raw.get("tasks", []))
        command = raw.get("command_template", [])
        if isinstance(command, str):
            command = shlex.split(command)
        if not isinstance(command, list) or any(not isinstance(item, str) for item in command):
            raise ValueError(f"framework {name} command_template must be a list of strings")
        flags = raw.get("parameter_flags", {})
        if not isinstance(flags, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in flags.items()):
            raise ValueError(f"framework {name} parameter_flags must be a string mapping")
        source = str(raw.get("source", "local"))
        if source.startswith(("http://", "https://")) and not raw.get("sha256"):
            raise ValueError(f"remote framework {name} must specify sha256")
        result[name] = FrameworkSpec(
            name=name,
            version=str(raw.get("version", "unversioned")),
            tasks=tasks,
            source=source,
            sha256=str(raw["sha256"]) if raw.get("sha256") else None,
            install_dir=str(raw["install_dir"]) if raw.get("install_dir") else None,
            command_template=tuple(command),
            parameter_flags=dict(flags),
        )
    return result


def select(name: str | None, task: str, command: str | None, registry_path: Path | None = None) -> FrameworkSpec:
    selected = name or ("local" if command else None)
    if not selected:
        raise ValueError("no framework selected; provide --framework or --train-command")
    registry = load_registry(registry_path)
    if selected not in registry:
        raise ValueError(f"framework '{selected}' is not registered")
    spec = registry[selected]
    if task not in spec.tasks:
        raise ValueError(f"framework '{selected}' does not support task '{task}'")
    if not command and not spec.command_template:
        raise ValueError(f"framework '{selected}' has no command_template; provide --train-command")
    return spec


def build_command(spec: FrameworkSpec, explicit: str | None, values: dict[str, object], parameters: dict[str, object]) -> list[str]:
    if explicit:
        command = shlex.split(explicit)
    else:
        command = list(spec.command_template)
    if not command:
        raise ValueError(f"framework '{spec.name}' produced an empty command")
    rendered = [Template(token).safe_substitute({key: str(value) for key, value in values.items()}) for token in command]
    for name, value in parameters.items():
        flag = spec.parameter_flags.get(name)
        if flag is None:
            continue
        if flag in rendered:
            raise ValueError(f"framework '{spec.name}' declares parameter flag {flag} both in template and parameter_flags")
        rendered.extend([flag, str(value)])
    return rendered
