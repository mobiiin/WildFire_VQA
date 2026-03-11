#!/usr/bin/env python3

import argparse
import json
import os
from typing import Any, Dict, Iterator, List, Optional, Tuple


def find_json_files(directory: str, recursive: bool = False) -> List[str]:
    if recursive:
        out: List[str] = []
        for root, _, files in os.walk(directory):
            for name in files:
                if name.lower().endswith(".json"):
                    out.append(os.path.join(root, name))
        return sorted(out)

    return sorted(
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.lower().endswith(".json") and os.path.isfile(os.path.join(directory, name))
    )


def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def iter_records(obj: Any) -> Iterator[Tuple[Dict[str, Any], str]]:
    if isinstance(obj, dict) and obj.get("type") == "checkpoint" and isinstance(obj.get("items"), dict):
        for checkpoint_key, entry in obj["items"].items():
            if not isinstance(entry, dict):
                continue
            meta = entry.get("meta", {})
            if isinstance(meta, dict):
                yield meta, str(checkpoint_key)
        return

    if isinstance(obj, list):
        for idx, row in enumerate(obj):
            if isinstance(row, dict):
                qid = row.get("question_id")
                label = str(qid) if qid is not None else f"row_{idx}"
                yield row, label


def resolve_path(path_value: Optional[str], json_file: str) -> Optional[str]:
    if not path_value:
        return None
    p = str(path_value).strip()
    if not p:
        return None
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(os.path.dirname(json_file), p))


def check_file(path: Optional[str]) -> Tuple[bool, str]:
    if path is None:
        return False, "empty_path"
    if not os.path.exists(path):
        return False, "missing"
    if not os.path.isfile(path):
        return False, "not_a_file"
    if not os.access(path, os.R_OK):
        return False, "not_readable"
    return True, "ok"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Check rgb_path and thermal_path in JSON files and report missing/unreadable paths."
    )
    ap.add_argument(
        "--dir",
        default=".",
        help="Directory containing JSON files (default: current directory).",
    )
    ap.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan subdirectories for JSON files.",
    )
    args = ap.parse_args()

    target_dir = os.path.abspath(args.dir)
    if not os.path.isdir(target_dir):
        print(f"ERROR: Not a directory: {target_dir}")
        return 2

    json_files = find_json_files(target_dir, recursive=args.recursive)
    if not json_files:
        print(f"No JSON files found in: {target_dir}")
        return 0

    issues = 0
    checked_records = 0

    for json_file in json_files:
        try:
            obj = load_json(json_file)
        except Exception as exc:
            issues += 1
            print(f"[JSON_READ_ERROR] file={json_file} error={repr(exc)}")
            continue

        for record, rec_id in iter_records(obj):
            checked_records += 1
            for field in ("rgb_path", "thermal_path"):
                raw = record.get(field)
                resolved = resolve_path(raw, json_file)
                ok, reason = check_file(resolved)
                if not ok:
                    issues += 1
                    print(
                        f"[BAD_PATH] json={json_file} record={rec_id} field={field} "
                        f"raw={repr(raw)} resolved={repr(resolved)} reason={reason}"
                    )

    print(
        f"Done. json_files={len(json_files)} records_checked={checked_records} issues_found={issues}"
    )
    return 1 if issues > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
