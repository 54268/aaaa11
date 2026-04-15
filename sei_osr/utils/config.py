from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config = deepcopy(config)
    config["_config_path"] = str(path)
    config["_project_root"] = str(path.parent.parent.resolve())
    return resolve_relative_paths(config)


def resolve_relative_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    root = Path(config["_project_root"])

    def _resolve(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: _resolve(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_resolve(v) for v in value]
        if isinstance(value, str):
            if value.startswith("./") or value.startswith("../"):
                return str((root / value).resolve())
        return value

    return _resolve(config)
