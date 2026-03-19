from __future__ import annotations

import json
import tempfile
from pathlib import Path
from threading import Lock
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class JsonFileStore:
    def __init__(self) -> None:
        self._lock = Lock()

    def ensure_parent(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

    def read_model(self, path: Path, model_type: type[T], default: T) -> T:
        if not path.exists():
            self.ensure_parent(path)
            self.write_model(path, default)
            return default

        with path.open("r", encoding="utf-8") as handle:
            return model_type.model_validate(json.load(handle))

    def write_model(self, path: Path, model: BaseModel) -> None:
        self.ensure_parent(path)
        payload = model.model_dump(mode="json", exclude_none=True)
        with self._lock:
            with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
                temp_name = handle.name
            Path(temp_name).replace(path)

    def read_text(self, path: Path, default: str = "") -> str:
        if not path.exists():
            self.ensure_parent(path)
            path.write_text(default, encoding="utf-8")
            return default
        return path.read_text(encoding="utf-8")
