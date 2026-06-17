from __future__ import annotations

import json
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


TESTS_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = TESTS_DIR.parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_MODEL = os.getenv("RMTOOLS_TEST_MODEL", "").strip() or "openai/gpt-4o-mini"


def data_path(*parts: str) -> Path:
    return TESTS_DIR / "data" / Path(*parts)


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def make_live_instance(model: str = DEFAULT_MODEL):
    from rmtools.ai import AI_Instance

    key = require_env("OPENROUTER_API_KEY")
    return AI_Instance(openrouter_api_key=key, model=model or DEFAULT_MODEL)


def make_offline_instance(model: str = "openai/gpt-4o-mini"):
    from rmtools.ai import AI_Instance

    return AI_Instance(openrouter_api_key="offline", model=model)


def print_header(title: str) -> None:
    print()
    print("=" * len(title))
    print(title)
    print("=" * len(title))


def dump_json(label: str, value: Any) -> None:
    print(f"{label}:")
    print(json.dumps(value, indent=2, sort_keys=True))


@contextmanager
def temp_workspace(prefix: str = "rmtools-tests-") -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix=prefix) as tmpdir:
        yield Path(tmpdir)


def assert_contains(text: str, needle: str) -> None:
    if needle not in text:
        raise AssertionError(f"Expected {needle!r} in {text!r}")


def assert_nonempty_text(value: Any) -> str:
    if not isinstance(value, str):
        raise AssertionError(f"Expected a string response, got {type(value).__name__}")
    if not value.strip():
        raise AssertionError("Expected a non-empty string response")
    return value


def assert_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AssertionError(f"Expected a dict response, got {type(value).__name__}")
    return value
