#!/usr/bin/env python3
"""Synchronize .env with example.env defaults.

This script keeps `.env` aligned with the latest defaults in `example.env`.
Existing variables keep their current values, while keys missing from `.env`
are copied from `example.env` to keep the file up to date. Keys that only
exist in `.env` are preserved under a "Custom variables" section.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent
EXAMPLE_ENV = ROOT_DIR / "example.env"
ENV_FILE = ROOT_DIR / ".env"


def read_env_file(path: Path) -> tuple[Dict[str, str], List[str]]:
    variables: Dict[str, str] = {}
    lines: List[str] = []

    if not path.exists():
        return variables, lines

    for line in path.read_text().splitlines():
        lines.append(line)
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = line.split("=", 1)
        variables[key.strip()] = value
    return variables, lines


def main() -> None:
    if not EXAMPLE_ENV.exists():
        raise SystemExit("example.env not found; cannot synchronize.")

    example_vars, example_lines = read_env_file(EXAMPLE_ENV)
    env_vars, env_lines = read_env_file(ENV_FILE)

    if not env_lines:
        ENV_FILE.write_text("\n".join(example_lines) + "\n")
        print("Created .env from example.env")
        return

    updated_lines: List[str] = []

    for line in example_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            updated_lines.append(line)
            continue

        key, example_value = line.split("=", 1)
        current_value = env_vars.get(key)
        value_to_write = current_value if current_value is not None else example_value
        updated_lines.append(f"{key}={value_to_write}")

    custom_entries = [key for key in env_vars if key not in example_vars]
    if custom_entries:
        updated_lines.extend(
            [
                "",
                "# Custom variables preserved from existing .env",
                *[f"{key}={env_vars[key]}" for key in custom_entries],
            ]
        )

    ENV_FILE.write_text("\n".join(updated_lines) + "\n")

    print("Synchronized .env to include any new variables from example.env")


if __name__ == "__main__":
    main()
