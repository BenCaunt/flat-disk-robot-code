#!/usr/bin/env python3
"""Recover motor command scalar paths in Rerun recordings.

Some recordings contain motor command values under ``/status/*`` but not under
the explicit ``/commands/*`` paths used by the dashboard/blueprint. This script
copies the existing scalar chunks to the command paths while preserving the
original timelines, then merges the recovered chunks back into a new recording.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pyarrow as pa
import rerun as rr
from rerun.recording import load_recording


MAPPINGS = {
    "/status/motor1_percent": ("/commands/motor1_percent", "/telemetry/motor1_percent"),
    "/status/motor2_percent": ("/commands/motor2_percent", "/telemetry/motor2_percent"),
    "/status/motor1_us": ("/commands/motor1_us", "/telemetry/motor1_us"),
    "/status/motor2_us": ("/commands/motor2_us", "/telemetry/motor2_us"),
    "/status/motor_commands": ("/commands/count",),
    "/status/motor_command_errors": ("/commands/errors",),
}

JSON_KEY_MAPPINGS = {
    "motor1_percent": ("/commands/motor1_percent", "/telemetry/motor1_percent"),
    "motor2_percent": ("/commands/motor2_percent", "/telemetry/motor2_percent"),
    "motor1_us": ("/commands/motor1_us", "/telemetry/motor1_us"),
    "motor2_us": ("/commands/motor2_us", "/telemetry/motor2_us"),
    "motor_commands": ("/commands/count",),
    "motor_command_errors": ("/commands/errors",),
}

RERUN_KIND = b"rerun:kind"
RERUN_KIND_CONTROL = b"control"
RERUN_KIND_INDEX = b"index"
RERUN_INDEX_NAME = b"rerun:index_name"
RERUN_ENTITY_PATH = b"rerun:entity_path"

SCALAR_FIELD = pa.field(
    "Scalars:scalars",
    pa.list_(pa.float64()),
    metadata={
        b"rerun:component": b"Scalars:scalars",
        b"rerun:archetype": b"rerun.archetypes.Scalars",
        b"rerun:component_type": b"rerun.components.Scalar",
        b"rerun:kind": b"data",
    },
)


def is_data_field(field: pa.Field) -> bool:
    metadata = field.metadata or {}
    if metadata.get(RERUN_KIND) in {RERUN_KIND_CONTROL, RERUN_KIND_INDEX}:
        return False
    return RERUN_INDEX_NAME not in metadata


def retarget_batch(batch: pa.RecordBatch, target_entity: str) -> pa.RecordBatch:
    fields: list[pa.Field] = []
    for field in batch.schema:
        metadata = dict(field.metadata or {})
        if is_data_field(field):
            metadata[RERUN_ENTITY_PATH] = target_entity.encode("utf-8")
        fields.append(field.with_metadata(metadata))
    schema = pa.schema(fields, metadata=batch.schema.metadata)
    return pa.record_batch([batch.column(i) for i in range(batch.num_columns)], schema=schema)


def to_scalar(value: object) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def textlog_column_name(batch: pa.RecordBatch) -> str | None:
    for field in batch.schema:
        metadata = field.metadata or {}
        if metadata.get(b"rerun:component") == b"TextLog:text" or field.name == "TextLog:text":
            return field.name
    return None


def json_recovery_batches(chunk: object, target_key: str, target_entity: str) -> list[pa.RecordBatch]:
    batch = chunk.to_record_batch()
    text_column_name = textlog_column_name(batch)
    if text_column_name is None:
        return []

    text_column = batch.column(text_column_name)
    selected_rows: list[int] = []
    values: list[list[float]] = []
    for row_idx in range(batch.num_rows):
        entries = text_column[row_idx].as_py()
        if not entries:
            continue
        try:
            status = json.loads(entries[0])
        except (TypeError, json.JSONDecodeError):
            continue
        scalar = to_scalar(status.get(target_key))
        if scalar is None:
            continue
        selected_rows.append(row_idx)
        values.append([scalar])

    if not selected_rows:
        return []

    take_indices = pa.array(selected_rows, type=pa.int64())
    columns: list[pa.Array] = []
    fields: list[pa.Field] = []
    for field in batch.schema:
        if is_data_field(field):
            continue
        columns.append(batch.column(field.name).take(take_indices))
        fields.append(field)

    scalar_metadata = dict(SCALAR_FIELD.metadata or {})
    scalar_metadata[RERUN_ENTITY_PATH] = target_entity.encode("utf-8")
    fields.append(SCALAR_FIELD.with_metadata(scalar_metadata))
    columns.append(pa.array(values, type=pa.list_(pa.float64())))
    return [pa.record_batch(columns, schema=pa.schema(fields, metadata=batch.schema.metadata))]


def default_output_path(input_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{input_path.stem}.recovered.rrd"


def recover_file(input_path: Path, output_path: Path, overwrite: bool) -> int:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} exists; pass --overwrite to replace it")

    recording = load_recording(input_path)
    chunks = list(recording.chunks())
    existing_entities = {str(chunk.entity_path) for chunk in chunks}
    planned_entities = set(existing_entities)

    recovery_batches: list[pa.RecordBatch] = []
    recovered_entities: set[str] = set()
    for chunk in chunks:
        for target in MAPPINGS.get(str(chunk.entity_path), ()):
            if target in planned_entities:
                continue
            recovery_batches.append(retarget_batch(chunk.to_record_batch(), target))
            planned_entities.add(target)
            recovered_entities.add(target)

    for chunk in chunks:
        if str(chunk.entity_path) != "/status/json":
            continue
        for key, targets in JSON_KEY_MAPPINGS.items():
            for target in targets:
                if target in planned_entities:
                    continue
                batches = json_recovery_batches(chunk, key, target)
                if not batches:
                    continue
                recovery_batches.extend(batches)
                planned_entities.add(target)
                recovered_entities.add(target)

    if not recovery_batches:
        print(f"{input_path}: no missing motor command paths found")
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rerun_motor_recovery_") as tmpdir:
        recovery_path = Path(tmpdir) / "motor_commands.rrd"
        rr.init(
            recording.application_id(),
            recording_id=recording.recording_id(),
            spawn=False,
        )
        rr.save(recovery_path)
        try:
            for batch in recovery_batches:
                rr.send_record_batch(batch)
        finally:
            rr.disconnect()

        rerun_cli = Path(sys.executable).with_name("rerun")
        subprocess.run(
            [
                str(rerun_cli),
                "rrd",
                "merge",
                "-o",
                str(output_path),
                str(input_path),
                str(recovery_path),
            ],
            check=True,
        )

    print(f"{input_path}: recovered {len(recovered_entities)} paths -> {output_path}")
    return len(recovered_entities)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rrd", nargs="+", type=Path, help="Input .rrd recording(s) to repair.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("captures/recovered"),
        help="Directory for repaired recordings when --output is not used.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output path for a single input recording.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing recovered recordings.")
    args = parser.parse_args()

    if args.output is not None and len(args.rrd) != 1:
        parser.error("--output can only be used with one input recording")

    total = 0
    for input_path in args.rrd:
        output_path = args.output if args.output is not None else default_output_path(input_path, args.output_dir)
        total += recover_file(input_path, output_path, args.overwrite)
    print(f"Recovered {total} missing paths total.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
