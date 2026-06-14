"""Plan a Qwen VLM DPO training job from materialized navigation preferences."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
from time import gmtime, monotonic, strftime
from typing import Any, Iterable

from .qwen_tool_training import DEFAULT_FORBIDDEN_MODEL_TOKENS


QWEN_DPO_TRAINING_JOB_SCHEMA = "flatdisk.qwen_dpo_training_job.v1"
QWEN_DPO_TRAINING_RESULT_SCHEMA = "flatdisk.qwen_dpo_training_result.v1"
DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_REQUIRED_PACKAGES = [
    "accelerate",
    "datasets",
    "peft",
    "pillow",
    "torch",
    "transformers",
    "trl",
]


def plan_qwen_dpo_training(
    input_path: Path,
    *,
    output_dir: Path,
    model_id: str = DEFAULT_MODEL_ID,
    adapter_output_dir: Path | None = None,
    max_steps: int = 100,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 5e-6,
    beta: float = 0.1,
    require_existing_images: bool = True,
) -> dict[str, Any]:
    qwen_manifest = _resolve_qwen_training_manifest(input_path)
    manifest = json.loads(qwen_manifest.read_text(encoding="utf-8"))
    manifest["_manifest_path"] = str(qwen_manifest)
    dpo_path = _resolve_manifest_path(
        manifest,
        "qwen_dpo_messages_jsonl",
        default=qwen_manifest.parent / "qwen_dpo_messages.jsonl",
    )
    records = _read_jsonl_if_exists(dpo_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_output_dir = adapter_output_dir or output_dir / "adapter"
    train_script_path = output_dir / "train_qwen_dpo_trl.py"
    job_path = output_dir / "qwen_dpo_training_job.json"
    validation = _validate_dpo_records(
        records,
        require_existing_images=require_existing_images,
        expected_count=_optional_int(manifest.get("dpo_preference_count")),
        dpo_path=dpo_path,
    )
    launch_argv = _launch_argv(
        train_script_path=train_script_path,
        dataset_path=dpo_path,
        model_id=model_id,
        adapter_output_dir=adapter_output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        beta=beta,
    )
    _write_train_script(train_script_path)
    train_script_sha256 = _sha256_file(train_script_path)
    job = {
        "schema": QWEN_DPO_TRAINING_JOB_SCHEMA,
        "created_at": strftime("%Y-%m-%dT%H:%M:%SZ", gmtime()),
        "status": "ready" if not validation["blockers"] else "not_ready",
        "blockers": validation["blockers"],
        "warnings": validation["warnings"],
        "input": str(input_path),
        "qwen_tool_training_manifest": str(qwen_manifest),
        "qwen_dpo_messages_jsonl": str(dpo_path),
        "output_dir": str(output_dir),
        "adapter_output_dir": str(adapter_output_dir),
        "train_script": str(train_script_path),
        "train_script_sha256": train_script_sha256,
        "launch_argv": launch_argv,
        "launch_command": _argv_to_command(launch_argv),
        "training_method": "offline_dpo",
        "trainer": "trl.DPOTrainer",
        "model_id": model_id,
        "required_packages": DEFAULT_REQUIRED_PACKAGES,
        "runtime": {
            "python_entrypoint": str(train_script_path),
            "launcher": "accelerate",
            "dependency_check": "importlib.util.find_spec without importing GPU training libraries",
            "required_packages": DEFAULT_REQUIRED_PACKAGES,
        },
        "training_args": {
            "max_steps": max_steps,
            "per_device_train_batch_size": per_device_train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "beta": beta,
            "remove_unused_columns": False,
        },
        "dataset": validation["dataset"],
        "audit": {
            "explicit_prompt_preference_format": True,
            "vlm_image_column": "images",
            "policy_input_only": True,
            "evaluator_labels_excluded": True,
            "require_existing_images": require_existing_images,
        },
    }
    job_path.write_text(json.dumps(job, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return job


def run_qwen_dpo_training_job(
    job_input: Path,
    *,
    result_dir: Path | None = None,
    dry_run: bool = False,
    check_dependencies: bool = True,
    timeout_s: float | None = None,
    launch_command: str | None = None,
    tail_chars: int = 4000,
) -> dict[str, Any]:
    job_path = _resolve_dpo_training_job(job_input)
    job = json.loads(job_path.read_text(encoding="utf-8"))
    result_dir = result_dir or Path(str(job.get("output_dir") or job_path.parent))
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / "qwen_dpo_training_result.json"
    launch_argv = _job_launch_argv(job, launch_command_override=launch_command)
    command = _argv_to_command(launch_argv)
    blockers = _training_job_blockers(job, job_path=job_path, check_dependencies=check_dependencies)
    if not launch_argv:
        blockers.append("missing launch_command")

    result: dict[str, Any] = {
        "schema": QWEN_DPO_TRAINING_RESULT_SCHEMA,
        "created_at": strftime("%Y-%m-%dT%H:%M:%SZ", gmtime()),
        "completed_at": None,
        "status": "not_ready",
        "dry_run": dry_run,
        "job_manifest": str(job_path),
        "result_path": str(result_path),
        "model_id": job.get("model_id"),
        "sample_count": (job.get("dataset") or {}).get("sample_count"),
        "adapter_output_dir": job.get("adapter_output_dir"),
        "launch_command": command,
        "launch_argv": launch_argv,
        "returncode": None,
        "stdout_tail": "",
        "stderr_tail": "",
        "duration_s": None,
        "blockers": blockers,
        "dependency_check": _dependency_check_payload(job, enabled=check_dependencies),
    }
    if blockers:
        result["completed_at"] = strftime("%Y-%m-%dT%H:%M:%SZ", gmtime())
        _write_result(result_path, result)
        return result
    if dry_run:
        result["status"] = "dry_run"
        result["completed_at"] = strftime("%Y-%m-%dT%H:%M:%SZ", gmtime())
        _write_result(result_path, result)
        return result

    start = monotonic()
    try:
        completed = subprocess.run(
            launch_argv,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_s,
        )
        result["returncode"] = completed.returncode
        result["stdout_tail"] = _tail(completed.stdout, tail_chars)
        result["stderr_tail"] = _tail(completed.stderr, tail_chars)
        result["status"] = "complete" if completed.returncode == 0 else "failed"
    except subprocess.TimeoutExpired as exc:
        result["status"] = "failed"
        result["blockers"] = [f"training command timed out after {timeout_s} second(s)"]
        result["stdout_tail"] = _tail(_decode_timeout_output(exc.stdout), tail_chars)
        result["stderr_tail"] = _tail(_decode_timeout_output(exc.stderr), tail_chars)
    finally:
        result["duration_s"] = round(monotonic() - start, 3)
        result["completed_at"] = strftime("%Y-%m-%dT%H:%M:%SZ", gmtime())
    _write_result(result_path, result)
    return result


def _resolve_qwen_training_manifest(input_path: Path) -> Path:
    path = input_path.expanduser()
    if path.is_file() and path.name == "qwen_tool_training_manifest.json":
        return path
    candidates = [
        path / "qwen_tool_training_manifest.json",
        path / "qwen_tool_training" / "qwen_tool_training_manifest.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"could not find qwen_tool_training_manifest.json under {input_path}"
    )


def _resolve_manifest_path(manifest: dict[str, Any], key: str, *, default: Path) -> Path:
    value = manifest.get(key)
    if not value:
        return default
    path = Path(str(value))
    if path.exists():
        return path
    manifest_path = Path(str(manifest["_manifest_path"]))
    if path.is_absolute():
        relocated = _relocated_absolute_path(path, local_qwen_training_dir=manifest_path.parent)
        if relocated is not None and relocated.exists():
            return relocated
        local_name_candidate = manifest_path.parent / path.name
        return local_name_candidate if local_name_candidate.exists() else path
    candidates = [
        manifest_path.parent / path,
        Path(str(manifest.get("output_dir") or manifest_path.parent)) / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return default if default.exists() else path


def _resolve_dpo_training_job(job_input: Path) -> Path:
    path = job_input.expanduser()
    if path.is_file() and path.name == "qwen_dpo_training_job.json":
        return path
    candidates = [
        path / "qwen_dpo_training_job.json",
        path / "qwen_dpo_training" / "qwen_dpo_training_job.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"could not find qwen_dpo_training_job.json under {job_input}")


def _relocated_absolute_path(path: Path, *, local_qwen_training_dir: Path) -> Path | None:
    parts = path.parts
    if "qwen_tool_training" not in parts:
        return None
    index = len(parts) - 1 - list(reversed(parts)).index("qwen_tool_training")
    tail = parts[index + 1 :]
    return local_qwen_training_dir / Path(*tail) if tail else local_qwen_training_dir


def _validate_dpo_records(
    records: list[dict[str, Any]],
    *,
    require_existing_images: bool,
    expected_count: int | None,
    dpo_path: Path,
) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    if not dpo_path.exists():
        blockers.append(f"missing qwen_dpo_messages_jsonl: {dpo_path}")
    if not records:
        blockers.append("qwen_dpo_messages_jsonl contains no preference records")
    if expected_count is not None and expected_count != len(records):
        blockers.append(f"dpo_preference_count mismatch: manifest={expected_count}, jsonl={len(records)}")

    missing_images: list[str] = []
    forbidden_hits: Counter[str] = Counter()
    malformed_count = 0
    image_reference_count = 0
    for record in records:
        if (
            not isinstance(record.get("prompt"), list)
            or not isinstance(record.get("chosen"), list)
            or not isinstance(record.get("rejected"), list)
        ):
            malformed_count += 1
        image_paths = _record_image_paths(record)
        image_reference_count += len(image_paths)
        missing_images.extend(str(path) for path in image_paths if not Path(path).exists())
        for token in _forbidden_tokens([record.get("prompt"), record.get("chosen"), record.get("rejected")]):
            forbidden_hits[token] += 1
    if malformed_count:
        blockers.append(f"{malformed_count} DPO record(s) are missing prompt/chosen/rejected list columns")
    if require_existing_images and missing_images:
        blockers.append(f"{len(missing_images)} DPO image reference(s) are missing")
    if forbidden_hits:
        blockers.append("DPO messages contain forbidden privileged token(s): " + ", ".join(sorted(forbidden_hits)))
    if image_reference_count == 0:
        warnings.append(
            "DPO records contain no image references; this is unexpected for Qwen VLM navigation preferences"
        )

    return {
        "blockers": blockers,
        "warnings": warnings,
        "dataset": {
            "sample_count": len(records),
            "expected_sample_count": expected_count,
            "image_reference_count": image_reference_count,
            "missing_image_count": len(missing_images),
            "missing_images": sorted(set(missing_images)),
            "forbidden_model_token_hits": dict(sorted(forbidden_hits.items())),
        },
    }


def _record_image_paths(record: dict[str, Any]) -> list[str]:
    values = record.get("images")
    if not isinstance(values, list):
        values = record.get("image_paths")
    if not isinstance(values, list):
        return []
    return [str(value) for value in values if str(value)]


def _forbidden_tokens(payloads: Iterable[Any]) -> list[str]:
    text = json.dumps(list(payloads), sort_keys=True, default=str).lower()
    return [token for token in DEFAULT_FORBIDDEN_MODEL_TOKENS if token.lower() in text]


def _training_job_blockers(
    job: dict[str, Any],
    *,
    job_path: Path,
    check_dependencies: bool,
) -> list[str]:
    blockers: list[str] = []
    if job.get("schema") != QWEN_DPO_TRAINING_JOB_SCHEMA:
        blockers.append(f"unexpected job schema: {job.get('schema')}")
    if job.get("status") != "ready":
        blockers.append(f"training job is not ready: {job.get('status')}")
    train_script = _job_path(job, "train_script", relative_to=job_path.parent)
    if train_script is None or not train_script.exists():
        blockers.append(f"missing train_script: {job.get('train_script')}")
    dataset_path = _job_path(job, "qwen_dpo_messages_jsonl", relative_to=job_path.parent)
    if dataset_path is None or not dataset_path.exists():
        blockers.append(f"missing qwen_dpo_messages_jsonl: {job.get('qwen_dpo_messages_jsonl')}")
    if check_dependencies:
        missing_packages = _missing_required_packages(job)
        if missing_packages:
            blockers.append("missing required training package(s): " + ", ".join(missing_packages))
    return blockers


def _job_path(job: dict[str, Any], key: str, *, relative_to: Path) -> Path | None:
    value = job.get(key)
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else relative_to / path if (relative_to / path).exists() else path


def _dependency_check_payload(job: dict[str, Any], *, enabled: bool) -> dict[str, Any]:
    required = [str(package) for package in job.get("required_packages", [])]
    missing = _missing_required_packages(job) if enabled else []
    return {
        "enabled": enabled,
        "required_packages": required,
        "missing_packages": missing,
    }


def _missing_required_packages(job: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for package in [str(value) for value in job.get("required_packages", [])]:
        if importlib.util.find_spec(_import_module_for_package(package)) is None:
            missing.append(package)
    return missing


def _import_module_for_package(package: str) -> str:
    return {"pillow": "PIL"}.get(package.lower(), package.replace("-", "_"))


def _read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _launch_argv(
    *,
    train_script_path: Path,
    dataset_path: Path,
    model_id: str,
    adapter_output_dir: Path,
    max_steps: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    beta: float,
) -> list[str]:
    return [
        "accelerate",
        "launch",
        str(train_script_path),
        "--dataset",
        str(dataset_path),
        "--model-id",
        model_id,
        "--output-dir",
        str(adapter_output_dir),
        "--max-steps",
        str(max_steps),
        "--per-device-train-batch-size",
        str(per_device_train_batch_size),
        "--gradient-accumulation-steps",
        str(gradient_accumulation_steps),
        "--learning-rate",
        str(learning_rate),
        "--beta",
        str(beta),
    ]


def _argv_to_command(argv: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in argv)


def _job_launch_argv(job: dict[str, Any], *, launch_command_override: str | None) -> list[str]:
    if launch_command_override:
        return shlex.split(launch_command_override)
    launch_argv = job.get("launch_argv")
    if isinstance(launch_argv, list) and all(isinstance(part, str) and part for part in launch_argv):
        return [str(part) for part in launch_argv]
    launch_command = str(job.get("launch_command") or "")
    return shlex.split(launch_command) if launch_command else []


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_train_script(path: Path) -> None:
    path.write_text(_TRAIN_SCRIPT, encoding="utf-8")


def _write_result(path: Path, result: dict[str, Any]) -> None:
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _tail(text: str, chars: int) -> str:
    return text[-chars:] if chars > 0 and len(text) > chars else text


def _decode_timeout_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


_TRAIN_SCRIPT = '''"""Run TRL DPO over flatdisk Qwen VLM preference records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from trl import DPOConfig, DPOTrainer


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    args = parser.parse_args()

    records = read_jsonl(args.dataset)
    for record in records:
        record["images"] = [load_image(path) for path in record.get("images", [])]
    dataset = Dataset.from_list(records)
    model = AutoModelForImageTextToText.from_pretrained(args.model_id, torch_dtype="auto", device_map="auto")
    processor = AutoProcessor.from_pretrained(args.model_id)
    training_args = DPOConfig(
        output_dir=str(args.output_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        beta=args.beta,
        remove_unused_columns=False,
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=processor,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))


if __name__ == "__main__":
    main()
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="qwen_tool_training dir or qwen_tool_training_manifest.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--adapter-output-dir", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--allow-missing-images", action="store_true")
    parser.add_argument("--fail-on-not-ready", action="store_true")
    return parser.parse_args()


def parse_run_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or dry-run a planned Qwen DPO training job.")
    parser.add_argument("--job", type=Path, required=True, help="qwen_dpo_training dir or qwen_dpo_training_job.json")
    parser.add_argument("--result-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-dependency-check", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=None)
    parser.add_argument("--launch-command", default=None, help="Override launch_command; intended for tests or manual recovery.")
    parser.add_argument("--tail-chars", type=int, default=4000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    job = plan_qwen_dpo_training(
        args.input,
        output_dir=args.output_dir,
        model_id=args.model_id,
        adapter_output_dir=args.adapter_output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        beta=args.beta,
        require_existing_images=not args.allow_missing_images,
    )
    print(
        json.dumps(
            {
                "status": job["status"],
                "sample_count": job["dataset"]["sample_count"],
                "missing_image_count": job["dataset"]["missing_image_count"],
                "forbidden_model_token_hits": job["dataset"]["forbidden_model_token_hits"],
                "job_path": str(Path(job["output_dir"]) / "qwen_dpo_training_job.json"),
                "train_script": job["train_script"],
                "launch_command": job["launch_command"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.fail_on_not_ready and job["status"] != "ready":
        return 2
    return 0


def run_main() -> int:
    args = parse_run_args()
    result = run_qwen_dpo_training_job(
        args.job,
        result_dir=args.result_dir,
        dry_run=args.dry_run,
        check_dependencies=not args.skip_dependency_check,
        timeout_s=args.timeout_s,
        launch_command=args.launch_command,
        tail_chars=args.tail_chars,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "returncode": result["returncode"],
                "blockers": result["blockers"],
                "result_path": result["result_path"],
                "launch_command": result["launch_command"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if result["status"] in {"complete", "dry_run"}:
        return 0
    if result["status"] == "not_ready":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
