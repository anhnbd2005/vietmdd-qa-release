#!/usr/bin/env python3
"""
Rewrite VietMDD logical audio IDs in a JSONL file to verified absolute WAV paths.

Example
-------
python map_vietmdd_audio_paths.py \
  --input qa_model_facing.jsonl \
  --output qa_model_facing_server.jsonl \
  --dataset-root /home/voice/data/voice/datasets/VietMDD \
  --expect-unique 3181 \
  --mapping-out vietmdd_audio_mapping.jsonl

Input:
    "audio": ["vietmdd_audio_1071"]

Output:
    "audio": ["/home/voice/data/voice/datasets/VietMDD/audio/1071.wav"]

Safety:
- Never modifies the input file.
- Fails closed if an audio reference is unrecognized.
- Fails if any mapped WAV does not physically exist.
- Preserves audio list order.
- Can enforce the expected number of unique audio files.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

LOGICAL_RE = re.compile(r"^vietmdd_audio_(\d+)$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Map VietMDD logical audio IDs to verified absolute WAV paths."
    )
    p.add_argument("--input", required=True, type=Path, help="Input JSONL.")
    p.add_argument("--output", required=True, type=Path, help="Output JSONL.")
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/voice/data/voice/datasets/VietMDD"),
        help="VietMDD dataset root. Default: /home/voice/data/voice/datasets/VietMDD",
    )
    p.add_argument(
        "--expect-unique",
        type=int,
        default=3181,
        help="Expected unique mapped WAV files. Use 0 to disable. Default: 3181.",
    )
    p.add_argument(
        "--mapping-out",
        type=Path,
        default=None,
        help="Optional JSONL audit mapping: logical_id -> absolute_path.",
    )
    p.add_argument(
        "--check-only",
        action="store_true",
        help="Validate mapping without writing the rewritten JSONL.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting --output / --mapping-out if they already exist.",
    )
    return p.parse_args()


def map_logical_audio(logical_id: str, dataset_root: Path) -> Path:
    """
    Convert:
        vietmdd_audio_0000 -> <dataset_root>/audio/0.wav
        vietmdd_audio_1071 -> <dataset_root>/audio/1071.wav

    Leading zeroes belong to the logical ID, while the physical filename is
    normalized to its integer form.
    """
    m = LOGICAL_RE.fullmatch(logical_id)
    if not m:
        raise ValueError(f"Unrecognized VietMDD audio reference: {logical_id!r}")

    numeric_id = int(m.group(1))
    return (dataset_root / "audio" / f"{numeric_id}.wav").resolve()


def fail(msg: str) -> "NoReturn":
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(2)


def atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        delete=False,
        dir=path.parent,
        prefix=path.name + ".tmp.",
    ) as tmp:
        tmp_path = Path(tmp.name)
        for row in rows:
            tmp.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    os.replace(tmp_path, path)


def atomic_write_mapping(path: Path, mapping: dict[str, str]) -> None:
    rows = [
        {"logical_audio_id": logical_id, "absolute_audio_path": abs_path}
        for logical_id, abs_path in sorted(mapping.items())
    ]
    atomic_write_jsonl(path, rows)


def main() -> int:
    args = parse_args()

    input_path = args.input.resolve()
    output_path = args.output.resolve()
    dataset_root = args.dataset_root.resolve()
    audio_dir = dataset_root / "audio"

    if not input_path.is_file():
        fail(f"Input JSONL does not exist: {input_path}")

    if input_path == output_path:
        fail("Refusing to overwrite the input JSONL. Use a different --output path.")

    if not dataset_root.is_dir():
        fail(f"Dataset root does not exist: {dataset_root}")

    if not audio_dir.is_dir():
        fail(f"Audio directory does not exist: {audio_dir}")

    if not args.check_only and output_path.exists() and not args.force:
        fail(f"Output already exists: {output_path}. Use --force to replace it.")

    if args.mapping_out is not None:
        mapping_out = args.mapping_out.resolve()
        if mapping_out == input_path:
            fail("--mapping-out cannot overwrite the input JSONL.")
        if mapping_out.exists() and not args.force:
            fail(f"Mapping output already exists: {mapping_out}. Use --force to replace it.")
    else:
        mapping_out = None

    rows: list[dict] = []
    mapping: dict[str, str] = {}
    physical_to_logical: dict[str, str] = {}

    total_rows = 0
    total_audio_refs = 0
    missing_files: list[tuple[str, str]] = []

    with input_path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            if not raw.strip():
                continue

            try:
                row = json.loads(raw)
            except json.JSONDecodeError as e:
                fail(f"Invalid JSON at line {line_no}: {e}")

            if not isinstance(row, dict):
                fail(f"Line {line_no}: JSON value must be an object.")

            audio = row.get("audio")
            if not isinstance(audio, list) or not audio:
                fail(f"Line {line_no}: 'audio' must be a non-empty list.")

            mapped_audio: list[str] = []

            for pos, ref in enumerate(audio):
                if not isinstance(ref, str) or not ref:
                    fail(
                        f"Line {line_no}, audio[{pos}]: expected non-empty string, got {ref!r}"
                    )

                try:
                    wav_path = map_logical_audio(ref, dataset_root)
                except ValueError as e:
                    fail(f"Line {line_no}, audio[{pos}]: {e}")

                wav_str = str(wav_path)

                previous = mapping.get(ref)
                if previous is not None and previous != wav_str:
                    fail(
                        f"Non-deterministic mapping for {ref!r}: "
                        f"{previous!r} vs {wav_str!r}"
                    )
                mapping[ref] = wav_str

                other_logical = physical_to_logical.get(wav_str)
                if other_logical is not None and other_logical != ref:
                    fail(
                        "Two logical IDs map to the same physical WAV: "
                        f"{other_logical!r}, {ref!r} -> {wav_str!r}"
                    )
                physical_to_logical[wav_str] = ref

                if not wav_path.is_file():
                    missing_files.append((ref, wav_str))

                mapped_audio.append(wav_str)
                total_audio_refs += 1

            # Preserve audio order; mutate only the audio field in the copied row.
            row["audio"] = mapped_audio
            rows.append(row)
            total_rows += 1

    if missing_files:
        print("\nFirst missing mappings:", file=sys.stderr)
        for logical_id, path in missing_files[:20]:
            print(f"  {logical_id} -> {path}", file=sys.stderr)
        fail(f"{len(missing_files)} referenced WAV mappings do not exist physically.")

    unique_count = len(mapping)

    if args.expect_unique and unique_count != args.expect_unique:
        fail(
            f"Unique mapped WAV count mismatch: got {unique_count}, "
            f"expected {args.expect_unique}."
        )

    print("VietMDD audio mapping preflight: PASS")
    print(f"Input JSONL        : {input_path}")
    print(f"Dataset root       : {dataset_root}")
    print(f"Audio directory    : {audio_dir}")
    print(f"Rows               : {total_rows}")
    print(f"Audio references   : {total_audio_refs}")
    print(f"Unique logical IDs : {unique_count}")
    print(f"Unique WAV paths   : {len(physical_to_logical)}")
    print(f"Missing WAV files  : {len(missing_files)}")

    print("\nSample mappings:")
    for logical_id in sorted(mapping)[:10]:
        print(f"  {logical_id} -> {mapping[logical_id]}")

    if args.check_only:
        print("\nCHECK-ONLY: no files written.")
        return 0

    atomic_write_jsonl(output_path, rows)
    print(f"\nRewritten JSONL    : {output_path}")

    if mapping_out is not None:
        atomic_write_mapping(mapping_out, mapping)
        print(f"Mapping audit      : {mapping_out}")

    print("\nDONE. Input JSONL was not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
