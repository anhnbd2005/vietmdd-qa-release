#!/usr/bin/env python3
"""
Convert VietMDD model-facing JSONL from `audio: [path, ...]` to exactly one
audio path per row.

Rules
-----
1 audio:
    [".../audio/1071.wav"]
    -> ".../audio/1071.wav"

2 audios:
    [".../audio/1.wav", ".../audio/2.wav"]
    -> ".../audio_pairs_beep/0001_0002.wav"

The merged WAV is:
    audio_1 + BEEP + audio_2

Audio order is preserved. Questions/choices/answers/type_id/operator are not
modified.

Default merged audio format:
    mono, 16 kHz, PCM s16le
Default separator:
    1000 Hz sine beep, 0.25 s, volume=0.25

Requires:
    ffmpeg

Example:
    python3 make_vietmdd_single_audio.py \
      --input qa_model_facing_server.jsonl \
      --output qa_model_facing_single_audio.jsonl \
      --merged-dir /home/voice/data/voice/datasets/VietMDD/audio_pairs_beep \
      --manifest audio_pair_merge_manifest.jsonl \
      --expect-rows 21927 \
      --expect-two-audio-rows 9203 \
      --workers 4
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import NoReturn


def fail(msg: str) -> NoReturn:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collapse VietMDD audio lists to one path; merge 2-audio rows with a beep."
    )
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--merged-dir", required=True, type=Path)
    p.add_argument("--manifest", type=Path, default=None)

    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--beep-frequency", type=float, default=1000.0)
    p.add_argument("--beep-duration", type=float, default=0.25)
    p.add_argument("--beep-volume", type=float, default=0.25)
    p.add_argument("--pad-width", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)

    p.add_argument("--expect-rows", type=int, default=21927)
    p.add_argument("--expect-two-audio-rows", type=int, default=9203)

    p.add_argument(
        "--check-only",
        action="store_true",
        help="Validate JSONL and source WAV paths; do not merge or write outputs.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Regenerate merged WAVs and allow replacing output/manifest.",
    )
    return p.parse_args()


def numeric_stem(path_str: str) -> int:
    stem = Path(path_str).stem
    if not stem.isdigit():
        raise ValueError(
            f"Expected numeric WAV filename like 1071.wav, got: {path_str!r}"
        )
    return int(stem)


def pair_filename(a: str, b: str, width: int) -> str:
    ia = numeric_stem(a)
    ib = numeric_stem(b)
    return f"{ia:0{width}d}_{ib:0{width}d}.wav"


def run_ffmpeg_merge(
    a: Path,
    b: Path,
    out: Path,
    *,
    sample_rate: int,
    beep_frequency: float,
    beep_duration: float,
    beep_volume: float,
    force: bool,
) -> tuple[str, str]:
    """
    Returns (status, output_path), where status is CREATED or REUSED.
    """
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.is_file() and out.stat().st_size > 44 and not force:
        return "REUSED", str(out)

    tmp = out.with_name(out.stem + ".tmp.wav")
    if tmp.exists():
        tmp.unlink()

    # Inputs are normalized before concat so ffmpeg can concatenate safely even
    # when source WAV metadata differs slightly.
    filt = (
        f"[0:a]aresample={sample_rate},"
        f"aformat=sample_fmts=s16:channel_layouts=mono[a0];"
        f"[1:a]volume={beep_volume},aresample={sample_rate},"
        f"aformat=sample_fmts=s16:channel_layouts=mono[beep];"
        f"[2:a]aresample={sample_rate},"
        f"aformat=sample_fmts=s16:channel_layouts=mono[a1];"
        f"[a0][beep][a1]concat=n=3:v=0:a=1[out]"
    )

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(a),
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency={beep_frequency}:sample_rate={sample_rate}:duration={beep_duration}",
        "-i",
        str(b),
        "-filter_complex",
        filt,
        "-map",
        "[out]",
        "-c:a",
        "pcm_s16le",
        str(tmp),
    ]

    try:
        cp = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise

    if cp.returncode != 0:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(
            f"ffmpeg failed for {a} + {b}\n"
            f"returncode={cp.returncode}\n"
            f"stderr={cp.stderr.strip()}"
        )

    if not tmp.is_file() or tmp.stat().st_size <= 44:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"ffmpeg produced invalid output: {tmp}")

    os.replace(tmp, out)
    return "CREATED", str(out)


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


def main() -> int:
    args = parse_args()

    if shutil.which("ffmpeg") is None:
        fail("ffmpeg was not found in PATH.")

    if args.workers < 1:
        fail("--workers must be >= 1")
    if args.sample_rate <= 0:
        fail("--sample-rate must be > 0")
    if args.beep_duration <= 0:
        fail("--beep-duration must be > 0")
    if args.beep_frequency <= 0:
        fail("--beep-frequency must be > 0")
    if not (0 < args.beep_volume <= 1.0):
        fail("--beep-volume must be in (0, 1].")

    input_path = args.input.resolve()
    output_path = args.output.resolve()
    merged_dir = args.merged_dir.resolve()
    manifest_path = args.manifest.resolve() if args.manifest else None

    if not input_path.is_file():
        fail(f"Input JSONL does not exist: {input_path}")

    if input_path == output_path:
        fail("Refusing to overwrite input JSONL; choose a different --output.")

    if not args.check_only:
        if output_path.exists() and not args.force:
            fail(f"Output already exists: {output_path}. Use --force to replace it.")
        if manifest_path and manifest_path.exists() and not args.force:
            fail(f"Manifest already exists: {manifest_path}. Use --force to replace it.")

    rows: list[dict] = []
    pair_to_output: dict[tuple[str, str], str] = {}
    pair_first_row: dict[tuple[str, str], str] = {}

    row_count = 0
    single_rows = 0
    two_rows = 0
    source_audio_refs = 0
    unique_source_audio: set[str] = set()

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
            if not isinstance(audio, list):
                fail(
                    f"Line {line_no}: expected 'audio' to be a list before normalization; "
                    f"got {type(audio).__name__}."
                )
            if len(audio) not in (1, 2):
                fail(f"Line {line_no}: expected 1 or 2 audio paths, got {len(audio)}.")

            checked: list[str] = []
            for pos, p in enumerate(audio):
                if not isinstance(p, str) or not p:
                    fail(f"Line {line_no}, audio[{pos}]: invalid path {p!r}")
                pp = Path(p)
                if not pp.is_absolute():
                    fail(
                        f"Line {line_no}, audio[{pos}]: expected mapped absolute server path, "
                        f"got {p!r}"
                    )
                if not pp.is_file():
                    fail(f"Line {line_no}, audio[{pos}]: WAV does not exist: {p}")
                if pp.suffix.lower() != ".wav":
                    fail(f"Line {line_no}, audio[{pos}]: not a WAV path: {p}")
                # Also proves the naming rule can be applied.
                try:
                    numeric_stem(p)
                except ValueError as e:
                    fail(f"Line {line_no}, audio[{pos}]: {e}")

                checked.append(str(pp.resolve()))
                unique_source_audio.add(str(pp.resolve()))
                source_audio_refs += 1

            row_id = str(row.get("id", f"line_{line_no}"))

            if len(checked) == 1:
                row["audio"] = checked[0]
                single_rows += 1
            else:
                a, b = checked
                name = pair_filename(a, b, args.pad_width)
                merged_path = (merged_dir / name).resolve()

                pair = (a, b)
                existing = pair_to_output.get(pair)
                if existing is not None and existing != str(merged_path):
                    fail(f"Internal pair mapping inconsistency for {pair}")

                # Guard against filename collision from two different ordered pairs.
                for other_pair, other_out in pair_to_output.items():
                    if other_out == str(merged_path) and other_pair != pair:
                        fail(
                            "Merged filename collision:\n"
                            f"  {other_pair} -> {other_out}\n"
                            f"  {pair} -> {merged_path}"
                        )

                pair_to_output[pair] = str(merged_path)
                pair_first_row.setdefault(pair, row_id)
                row["audio"] = str(merged_path)
                two_rows += 1

            rows.append(row)
            row_count += 1

    if args.expect_rows and row_count != args.expect_rows:
        fail(f"Row count mismatch: got {row_count}, expected {args.expect_rows}.")

    if args.expect_two_audio_rows and two_rows != args.expect_two_audio_rows:
        fail(
            f"Two-audio row count mismatch: got {two_rows}, "
            f"expected {args.expect_two_audio_rows}."
        )

    print("VietMDD single-audio preflight: PASS")
    print(f"Input rows              : {row_count}")
    print(f"1-audio rows            : {single_rows}")
    print(f"2-audio rows            : {two_rows}")
    print(f"Source audio references : {source_audio_refs}")
    print(f"Unique source WAVs      : {len(unique_source_audio)}")
    print(f"Unique ordered pairs    : {len(pair_to_output)}")
    print(f"Merged directory        : {merged_dir}")
    print(f"Beep                    : {args.beep_frequency:g} Hz, {args.beep_duration:g} s")
    print(f"Merged format           : mono {args.sample_rate} Hz PCM s16le")

    print("\nSample pair outputs:")
    for (a, b), out in list(sorted(pair_to_output.items()))[:10]:
        print(f"  {Path(a).name} + BEEP + {Path(b).name} -> {Path(out).name}")

    if args.check_only:
        print("\nCHECK-ONLY: no merged WAVs or JSONL were written.")
        return 0

    merged_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for (a, b), out in sorted(pair_to_output.items()):
            jobs.append(
                ex.submit(
                    run_ffmpeg_merge,
                    Path(a),
                    Path(b),
                    Path(out),
                    sample_rate=args.sample_rate,
                    beep_frequency=args.beep_frequency,
                    beep_duration=args.beep_duration,
                    beep_volume=args.beep_volume,
                    force=args.force,
                )
            )

        created = 0
        reused = 0
        completed = 0
        total = len(jobs)

        try:
            for fut in as_completed(jobs):
                status, _ = fut.result()
                completed += 1
                if status == "CREATED":
                    created += 1
                else:
                    reused += 1

                if completed == 1 or completed % 100 == 0 or completed == total:
                    print(
                        f"\rMerged pairs: {completed}/{total} "
                        f"(created={created}, reused={reused})",
                        end="",
                        flush=True,
                    )
        except Exception as e:
            print()
            fail(str(e))

    print()

    # Final existence gate before publishing JSONL.
    missing_merged = [
        out for out in pair_to_output.values()
        if not Path(out).is_file() or Path(out).stat().st_size <= 44
    ]
    if missing_merged:
        for p in missing_merged[:20]:
            print(f"Missing/invalid merged WAV: {p}", file=sys.stderr)
        fail(f"{len(missing_merged)} merged WAV files are missing or invalid.")

    atomic_write_jsonl(output_path, rows)

    if manifest_path is not None:
        manifest_rows = []
        for (a, b), out in sorted(pair_to_output.items()):
            manifest_rows.append(
                {
                    "audio_a": a,
                    "audio_b": b,
                    "merged_audio": out,
                    "merged_filename": Path(out).name,
                    "first_row_id": pair_first_row[(a, b)],
                    "separator": {
                        "kind": "sine_beep",
                        "frequency_hz": args.beep_frequency,
                        "duration_seconds": args.beep_duration,
                        "volume": args.beep_volume,
                    },
                    "output_format": {
                        "sample_rate_hz": args.sample_rate,
                        "channels": 1,
                        "codec": "pcm_s16le",
                    },
                }
            )
        atomic_write_jsonl(manifest_path, manifest_rows)

    # Verify published JSONL has scalar audio everywhere.
    scalar_audio_rows = 0
    with output_path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            if not raw.strip():
                continue
            x = json.loads(raw)
            if not isinstance(x.get("audio"), str):
                fail(f"Post-write verification failed at output line {line_no}.")
            scalar_audio_rows += 1

    if scalar_audio_rows != row_count:
        fail(
            f"Post-write row count mismatch: got {scalar_audio_rows}, expected {row_count}."
        )

    print("\nDONE")
    print(f"Output JSONL             : {output_path}")
    if manifest_path:
        print(f"Pair manifest            : {manifest_path}")
    print(f"All audio fields scalar  : YES ({scalar_audio_rows}/{row_count})")
    print(f"Created merged WAVs      : {created}")
    print(f"Reused merged WAVs       : {reused}")
    print("Question/answer/type semantics were not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
