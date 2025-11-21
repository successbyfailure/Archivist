#!/usr/bin/env python3
"""Synchronize .env with example.env defaults.

If new variables are added to example.env, this script appends them to
.env with their default values while preserving existing customizations.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
EXAMPLE_ENV = ROOT_DIR / "example.env"
ENV_FILE = ROOT_DIR / ".env"


def read_env_file(path: Path) -> Tuple[Dict[str, str], List[str]]:
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

    new_entries = []
    for key, default_value in example_vars.items():
        if key not in env_vars:
            new_entries.append(f"{key}={default_value}")

    if not new_entries:
        print(".env already contains all variables from example.env")
        return

    updated_content = "\n".join(env_lines + ["", *new_entries]) + "\n"
    ENV_FILE.write_text(updated_content)
    print(f"Added {len(new_entries)} new variable(s) to .env:")
    for entry in new_entries:
        print(f"  {entry}")


if __name__ == "__main__":
    main()
