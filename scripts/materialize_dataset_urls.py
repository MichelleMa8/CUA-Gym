#!/usr/bin/env python3
"""Materialize CUA-Gym dataset endpoint placeholders.

Downloaded CUA-Gym task artifacts intentionally use placeholders such as
__CUA_GYM_GMAIL_URL__ instead of hard-coded hosted endpoints. Deploy the
corresponding CUA-Gym-Hub applications yourself, set the environment variables
listed in url_variables.json, and run this script before executing setup or
reward files.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
import json
from urllib.parse import urlparse


TEXT_SUFFIXES = {".json", ".md", ".py", ".sh", ".txt", ".yaml", ".yml"}


def load_env_file(path: Path | None) -> dict[str, str]:
    values: dict[str, str] = {}
    if path is None:
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def replacement_map(manifest_path: Path, env_file: Path | None, use_defaults: bool) -> dict[str, str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    variables = manifest.get("variables") or {}
    env_values = {**os.environ, **load_env_file(env_file)}
    replacements: dict[str, str] = {}
    missing: list[str] = []

    for placeholder, spec in variables.items():
        env_name = spec["env"]
        value = env_values.get(env_name)
        if value is None and spec.get("kind") == "host" and env_name.endswith("_HOST"):
            url_value = env_values.get(env_name[: -len("_HOST")] + "_URL")
            if url_value:
                value = urlparse(url_value).netloc or url_value
        if value is None and use_defaults:
            value = spec.get("default")
        if value is None:
            missing.append(env_name)
            continue
        replacements[placeholder] = value.rstrip("/") if spec.get("kind") == "url" else value

    if missing:
        missing_list = "\n".join(f"  {name}=..." for name in sorted(set(missing)))
        raise SystemExit(
            "Missing endpoint environment variables. Set them in your shell or .env file:\n"
            f"{missing_list}\n\n"
            "Use --use-hosted-defaults only for smoke tests against the release-hosted endpoints."
        )
    return replacements


def materialize(root: Path, replacements: dict[str, str]) -> tuple[int, int]:
    touched_files = 0
    replacements_made = 0
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
            continue
        try:
            original = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        updated = original
        for placeholder, value in replacements.items():
            count = updated.count(placeholder)
            if count:
                updated = updated.replace(placeholder, value)
                replacements_made += count
        if updated != original:
            path.write_text(updated, encoding="utf-8")
            touched_files += 1
    return touched_files, replacements_made


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_dir",
        help="Directory containing extracted CUA-Gym task bundles and url_variables.json.",
    )
    parser.add_argument(
        "--manifest",
        help="Path to url_variables.json. Defaults to <dataset_dir>/url_variables.json.",
    )
    parser.add_argument("--env-file", help="Optional .env file containing CUA_GYM_* variables.")
    parser.add_argument(
        "--output",
        help="Optional output directory. If set, copy dataset_dir there before materializing.",
    )
    parser.add_argument(
        "--use-hosted-defaults",
        action="store_true",
        help="Use release-hosted xlang.ai defaults when variables are missing.",
    )
    args = parser.parse_args()

    source = Path(args.dataset_dir).expanduser().resolve()
    target = Path(args.output).expanduser().resolve() if args.output else source
    if args.output:
        if target.exists():
            raise SystemExit(f"Refusing to overwrite existing output directory: {target}")
        shutil.copytree(source, target)

    manifest_path = Path(args.manifest).expanduser().resolve() if args.manifest else target / "url_variables.json"
    replacements = replacement_map(
        manifest_path,
        Path(args.env_file).expanduser().resolve() if args.env_file else None,
        args.use_hosted_defaults,
    )
    touched_files, replacements_made = materialize(target, replacements)
    print(f"Materialized {replacements_made} placeholders across {touched_files} files under {target}")


if __name__ == "__main__":
    main()
