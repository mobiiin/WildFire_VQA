#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from typing import Any, Dict, List, Tuple


def find_json_files(root_dir: str) -> List[str]:
    json_files: List[str] = []
    for root, _, files in os.walk(root_dir):
        for name in files:
            if name.lower().endswith(".json"):
                json_files.append(os.path.join(root, name))
    return sorted(json_files)


def normalize_prefix(prefix: str) -> str:
    return prefix.strip().rstrip("/")


def replace_prefix_in_string(value: str, old_prefix: str, new_prefix: str) -> Tuple[str, bool]:
    old = normalize_prefix(old_prefix)
    new = normalize_prefix(new_prefix)

    if value == old:
        return new, True

    if value.startswith(old + "/"):
        return new + value[len(old):], True

    return value, False


def update_paths_recursive(obj: Any, old_prefix: str, new_prefix: str) -> Tuple[Any, int]:
    changed = 0

    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            new_v, c = update_paths_recursive(v, old_prefix, new_prefix)
            out[k] = new_v
            changed += c
        return out, changed

    if isinstance(obj, list):
        out_list: List[Any] = []
        for item in obj:
            new_item, c = update_paths_recursive(item, old_prefix, new_prefix)
            out_list.append(new_item)
            changed += c
        return out_list, changed

    if isinstance(obj, str):
        new_s, did = replace_prefix_in_string(obj, old_prefix, new_prefix)
        return new_s, (1 if did else 0)

    return obj, 0


def collect_paths_recursive(obj: Any, key_path: str = "") -> List[Tuple[str, str]]:
    collected: List[Tuple[str, str]] = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            child_key = f"{key_path}.{k}" if key_path else k
            collected.extend(collect_paths_recursive(v, child_key))
        return collected

    if isinstance(obj, list):
        for i, item in enumerate(obj):
            child_key = f"{key_path}[{i}]"
            collected.extend(collect_paths_recursive(item, child_key))
        return collected

    if isinstance(obj, str):
        if obj.startswith("/"):
            collected.append((key_path, obj))

    return collected


def file_access_reason(path: str) -> str:
    if not os.path.exists(path):
        return "missing (path does not exist)"

    if os.path.isdir(path):
        return "is a directory (expected a file)"

    if not os.path.isfile(path):
        return "not a regular file"

    if not os.access(path, os.R_OK):
        return "permission denied (not readable)"

    try:
        with open(path, "rb") as f:
            f.read(1)
    except PermissionError as e:
        return f"permission denied: {e}"
    except OSError as e:
        return f"os error: {e}"
    except Exception as e:
        return f"unexpected error: {e}"

    return ""


def validate_paths_in_json_file(json_path: str) -> List[Dict[str, str]]:
    with open(json_path, "r") as f:
        data = json.load(f)

    found = collect_paths_recursive(data)
    issues: List[Dict[str, str]] = []

    seen = set()
    for key_ref, path in found:
        if path in seen:
            continue
        seen.add(path)

        reason = file_access_reason(path)
        if reason:
            issues.append({
                "json_file": json_path,
                "json_key": key_ref,
                "path": path,
                "reason": reason,
            })

    return issues


def update_json_file_in_place(json_path: str, old_prefix: str, new_prefix: str) -> int:
    with open(json_path, "r") as f:
        data = json.load(f)

    updated, changed = update_paths_recursive(data, old_prefix, new_prefix)

    if changed > 0:
        with open(json_path, "w") as f:
            json.dump(updated, f, indent=2)

    return changed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update image path prefixes in JSON files under response_v16 and validate changed paths."
    )
    parser.add_argument(
        "--root-dir",
        default="./response_v16",
        help="Directory containing JSON files (default: ./response_v16)",
    )
    parser.add_argument(
        "--old-prefix",
        default="/scratch/mhabibp/Downloads/flame3333",
        help="Old absolute path prefix to replace",
    )
    parser.add_argument(
        "--new-prefix",
        default="/project/fafghah/iswinlab/flame3333",
        help="New absolute path prefix",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate paths; do not modify JSON files",
    )
    args = parser.parse_args()

    root_dir = args.root_dir
    old_prefix = args.old_prefix
    new_prefix = args.new_prefix

    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Root directory does not exist: {root_dir}")

    json_files = find_json_files(root_dir)
    if not json_files:
        print(f"No JSON files found under: {root_dir}")
        return

    print(f"Discovered {len(json_files)} JSON files under {root_dir}")

    changed_files = 0
    total_replacements = 0

    if not args.validate_only:
        for jf in json_files:
            count = update_json_file_in_place(jf, old_prefix, new_prefix)
            if count > 0:
                changed_files += 1
                total_replacements += count
                print(f"[UPDATED] {jf} (replacements={count})")

        print("\nUpdate summary:")
        print(f"  Files changed: {changed_files}")
        print(f"  Total replacements: {total_replacements}")
    else:
        print("Skipping update step (--validate-only set)")

    print("\nValidating file accessibility for all absolute paths found in JSON...")

    all_issues: List[Dict[str, str]] = []
    for jf in json_files:
        try:
            issues = validate_paths_in_json_file(jf)
            all_issues.extend(issues)
        except Exception as e:
            all_issues.append(
                {
                    "json_file": jf,
                    "json_key": "<json_read>",
                    "path": "<n/a>",
                    "reason": f"failed to parse/read json: {e}",
                }
            )

    if not all_issues:
        print("[OK] No missing/inaccessible file paths detected.")
        return

    print(f"[WARN] Found {len(all_issues)} missing/inaccessible paths:\n")
    for issue in all_issues:
        print(f"json_file: {issue['json_file']}")
        print(f"json_key : {issue['json_key']}")
        print(f"path     : {issue['path']}")
        print(f"reason   : {issue['reason']}")
        print("-" * 80)


if __name__ == "__main__":
    main()
