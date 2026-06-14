# 2026-06-13 Open-Vocab Nav Research Loop

Seed experiment for Qwen planner prompt variants using the real robot-compatible
tool surface in THOR.

Dry-run:

```bash
uv run --project sim flatdisk-sim-research-loop \
  --config experiments/2026-06-13-open-vocab-nav-research-loop/qwen_prompt_sweep.json \
  --output-dir sim/scratch/open_vocab_nav_research_loop \
  --dry-run
```

Topomap-memory sweep dry-run:

```bash
uv run --project sim flatdisk-sim-research-loop \
  --config experiments/2026-06-13-open-vocab-nav-research-loop/qwen_topomap_memory_sweep.json \
  --output-dir sim/scratch/open_vocab_nav_research_loop \
  --dry-run
```

Runpod/Linux sweep dry-run:

```bash
uv run --project sim flatdisk-sim-research-loop \
  --config experiments/2026-06-13-open-vocab-nav-research-loop/qwen_strategy_sweep_runpod_linux.json \
  --output-dir sim/scratch/open_vocab_nav_research_loop \
  --dry-run
```

Execute locally with a Qwen OpenAI-compatible endpoint already running:

```bash
uv run --project sim flatdisk-sim-research-loop \
  --config experiments/2026-06-13-open-vocab-nav-research-loop/qwen_prompt_sweep.json \
  --output-dir sim/scratch/open_vocab_nav_research_loop \
  --preflight-endpoints \
  --parallelism 1
```

The configs intentionally use only model-based Qwen variants. Research configs
default to `strict_model_based=true`, so hard-coded scripted runners and
THOR-derived semantic topomap terms are rejected unless a debug config explicitly
opts out. Scripted runners are reserved for smoke tests and should not be
counted as general open-vocabulary results.

`qwen_topomap_memory_sweep.json` is local/Apple oriented and uses the MLX Qwen
and Florence paths. `qwen_topomap_memory_sweep_runpod_linux.json` is the
minimal Runpod/Linux sibling, and `qwen_strategy_sweep_runpod_linux.json` is the
current generated-strategy sweep. The Linux configs expect
`Qwen/Qwen3-VL-8B-Instruct` behind an OpenAI-compatible endpoint at
`127.0.0.1:8000` and use `florence-transformers` for `visual_servo_object`.
A Runpod worker can either connect to a pre-existing endpoint or set
`START_QWEN_SERVER=1` / `--start-qwen-server` to start a local vLLM endpoint
before preflight.

Variants can opt into non-motion topomap memory by adding
`topomap_memory_map_dir` and `topomap_memory_use_clip`. Keep
`topomap_memory_allow_semantic_terms` off for strict policy runs unless the saved
semantic terms came from a real-robot-compatible labeling process.

`topomap_memory_map_dir` supports `{episode}` templating. The topomap sweep
expects maps such as `sim/scratch/semantic_topomaps/living_room_sofa_clip`.
Run with `--preflight-endpoints` before launching THOR; this checks both the
Qwen endpoint and configured topomap maps/CLIP embeddings. The
`qwen-strategy-runpod-linux-v4` Warmhub queue includes conditional build guards
for missing CLIP topomap artifacts in fixture, preflight, and topomap-memory
slices, without rebuilding existing maps.

Use `--preflight-only` when a worker should validate endpoint plus topomap
fixtures and stop before launching any THOR episodes.

Runpod launch command dry-run:

```bash
uv run --project sim flatdisk-sim-runpod-launch-task \
  --task qwen-strategy-runpod-linux-v4-topomap-fixtures \
  --agent agent-name \
  --all-commands \
  --start-qwen-server \
  --terminate-after 4h
```

Add `--launch` only after the generated `runpodctl pod create` command is
reviewed, the current git ref is pushed, and Runpod credentials plus `wh` CLI
availability are confirmed.

Runpod dispatch dry-run for multiple planned trial slices:

```bash
uv run --project sim flatdisk-sim-runpod-dispatch \
  --name-prefix qwen-strategy-runpod-linux-v4-run- \
  --tag trial-slice \
  --max-workers 2 \
  --start-qwen-server \
  --terminate-after 4h
```

Generated strategy sweep:

```bash
uv run --project sim flatdisk-sim-generate-nav-strategy-sweep \
  --base-config experiments/2026-06-13-open-vocab-nav-research-loop/qwen_topomap_memory_sweep_runpod_linux.json \
  --output experiments/2026-06-13-open-vocab-nav-research-loop/qwen_strategy_sweep_runpod_linux.json \
  --experiment-id open_vocab_nav_qwen_strategy_runpod_linux_v1
```

The generated sweep keeps all variants model-based (`qwen`), avoids static
object/color examples in actor rules, and keeps semantic topomap term routing
off. It relies on the default strict research-loop gate. Use
`flatdisk-sim-research-warmhub task-plan-config` to seed planned Runpod
trial-slice tasks from it.

Executed sweeps write `training_export/policy_steps.jsonl`,
`training_export/episode_rollouts.jsonl`, `training_export/rollout_groups.jsonl`,
`training_export/trajectory_preferences.jsonl`, and
`training_export/policy_dataset_v1/`. These are intended as the bridge to
offline SFT/GRPO/PPO work: policy sample records contain only model-facing
inputs and outputs, while hidden THOR distance/success is stored in separate
evaluator reward/label fields.

Before training, run the Qwen tool-training materializer, which joins
`policy_dataset_v1/policy_samples.jsonl` with
`policy_dataset_v1/evaluator_labels.jsonl`, validates referenced image paths,
filters unsafe or rejected actions, emits Qwen-compatible multimodal SFT JSONL,
and audits that no privileged evaluator fields leak into model-facing messages.
Treat PPO/GRPO as a later step once grouped rollouts, reward normalization, and
on-policy collection are explicit.

```bash
uv run --project sim flatdisk-sim-prepare-qwen-tool-training \
  --input sim/scratch/open_vocab_nav_research_loop/<run>/training_export/policy_dataset_v1 \
  --output-dir sim/scratch/open_vocab_nav_research_loop/<run>/qwen_tool_training
```
