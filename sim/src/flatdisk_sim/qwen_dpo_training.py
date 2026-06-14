"""Plan a Qwen VLM DPO training job from materialized navigation preferences."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from time import gmtime, strftime
from typing import Any, Iterable

from .qwen_tool_training import DEFAULT_FORBIDDEN_MODEL_TOKENS


QWEN_DPO_TRAINING_JOB_SCHEMA = "flatdisk.qwen_dpo_training_job.v1"
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
    launch_command = _launch_command(
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
        "launch_command": launch_command,
        "training_method": "offline_dpo",
        "trainer": "trl.DPOTrainer",
        "model_id": model_id,
        "required_packages": DEFAULT_REQUIRED_PACKAGES,
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


def _launch_command(
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
) -> str:
    return (
        "accelerate launch "
        f"{train_script_path} "
        f"--dataset {dataset_path} "
        f"--model-id {model_id} "
        f"--output-dir {adapter_output_dir} "
        f"--max-steps {max_steps} "
        f"--per-device-train-batch-size {per_device_train_batch_size} "
        f"--gradient-accumulation-steps {gradient_accumulation_steps} "
        f"--learning-rate {learning_rate} "
        f"--beta {beta}"
    )


def _write_train_script(path: Path) -> None:
    path.write_text(_TRAIN_SCRIPT, encoding="utf-8")


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


if __name__ == "__main__":
    raise SystemExit(main())
